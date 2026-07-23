"""Phrase-level, transposition-invariant melody contour matching.

Unlike the frame matcher, each candidate lyric phrase is normalized
independently.  This preserves the phrase-wide melody shape while removing one
global singer/key offset.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .pitch_extractor import PitchFeatures


@dataclass(frozen=True)
class PhraseMatch:
    line_index: int
    start_frame: int
    end_frame: int
    cost: float
    pitch_cost: float
    slope_cost: float
    range_penalty: float
    path_length: int
    segment_coverage: float


def match_lyric_phrases(
    query: PitchFeatures,
    reference: PitchFeatures,
    lrc_lines: list[dict],
    config: dict,
) -> list[PhraseMatch]:
    settings = config["phrase_matching"]
    points = int(settings["contour_points"])
    query_contour = phrase_contour(query.pitch, points)
    if query_contour is None:
        return []
    query_voiced_seconds = float(np.count_nonzero(query.voiced)) * float(config["pitch"]["hop_seconds"])
    active_settings = settings
    if query_voiced_seconds < float(settings.get("segmental_min_voiced_seconds", 0.0)):
        active_settings = dict(settings)
        active_settings["segment_coverages"] = [1.0]
    hop = float(config["pitch"]["hop_seconds"])
    matches: list[PhraseMatch] = []
    for line in lrc_lines:
        start_frame = max(0, int(np.floor(float(line["start_time"]) / hop)))
        end_frame = min(
            len(reference.pitch),
            int(np.ceil(float(line.get("end_time", line["start_time"])) / hop)),
        )
        if end_frame - start_frame < 2:
            continue
        reference_contour = phrase_contour(reference.pitch[start_frame:end_frame], points)
        if reference_contour is None:
            continue
        score = segmental_contour_dtw(query_contour, reference_contour, active_settings)
        matches.append(
            PhraseMatch(
                line_index=int(line["index"]),
                start_frame=start_frame,
                end_frame=end_frame - 1,
                **score,
            )
        )
    return sorted(matches, key=lambda item: item.cost)


def phrase_contour(pitch: np.ndarray, points: int) -> np.ndarray | None:
    """Interpolate a voiced phrase and remove one phrase-wide pitch offset."""
    values = np.asarray(pitch, dtype=np.float32)
    voiced_indices = np.flatnonzero(np.isfinite(values))
    if voiced_indices.size < 3:
        return None
    first, last = int(voiced_indices[0]), int(voiced_indices[-1])
    if last <= first:
        return None
    positions = np.linspace(first, last, points, dtype=np.float32)
    contour = np.interp(positions, voiced_indices, values[voiced_indices]).astype(np.float32)
    contour -= float(np.median(contour))
    # A small median filter suppresses vibrato and isolated octave artifacts
    # without flattening note-to-note movement.
    if contour.size >= 5:
        contour = np.asarray(
            [np.median(contour[max(0, index - 2) : min(len(contour), index + 3)]) for index in range(len(contour))],
            dtype=np.float32,
        )
    return contour


def contour_dtw(query: np.ndarray, reference: np.ndarray, settings: dict) -> dict[str, float | int]:
    result = _single_contour_dtw(query, reference, settings)
    return {
        **result,
        "segment_coverage": 1.0,
    }


def segmental_contour_dtw(query: np.ndarray, reference: np.ndarray, settings: dict) -> dict[str, float | int]:
    coverages = sorted({float(value) for value in settings.get("segment_coverages", [1.0])}, reverse=True)
    candidates: list[dict[str, float | int]] = []
    for coverage in coverages:
        coverage = min(1.0, max(0.4, coverage))
        width = max(8, int(round(len(reference) * coverage)))
        maximum_start = len(reference) - width
        starts = {0, maximum_start}
        if maximum_start > 0:
            starts.add(maximum_start // 2)
        for start in sorted(starts):
            segment = reference[start : start + width]
            positions = np.linspace(0, len(segment) - 1, len(reference), dtype=np.float32)
            normalized = np.interp(positions, np.arange(len(segment)), segment).astype(np.float32)
            normalized -= float(np.median(normalized))
            result = contour_dtw(query, normalized, settings)
            result["cost"] = float(result["cost"]) + (
                1.0 - coverage
            ) * float(settings.get("partial_coverage_penalty", 0.0))
            result["segment_coverage"] = coverage
            candidates.append(result)
    return min(candidates, key=lambda result: float(result["cost"]))


def _single_contour_dtw(query: np.ndarray, reference: np.ndarray, settings: dict) -> dict[str, float | int]:
    query_slope = np.gradient(query)
    reference_slope = np.gradient(reference)
    raw_pitch = np.abs(query[:, None] - reference[None, :])
    pitch_distance = np.minimum(
        np.minimum(raw_pitch, np.abs(raw_pitch - 12.0) + float(settings["octave_penalty"])),
        float(settings["pitch_distance_cap"]),
    )
    slope_distance = np.minimum(
        np.abs(query_slope[:, None] - reference_slope[None, :]),
        float(settings["slope_distance_cap"]),
    )
    local = (
        float(settings["pitch_weight"]) * pitch_distance
        + float(settings["slope_weight"]) * slope_distance
    ).astype(np.float32)
    accumulated = librosa.sequence.dtw(
        C=local,
        global_constraints=True,
        band_rad=float(settings["band_radius"]),
        backtrack=False,
    )
    path = np.asarray(_trace_path(accumulated), dtype=np.int32)
    pitch_cost = float(np.mean(pitch_distance[path[:, 0], path[:, 1]]))
    slope_cost = float(np.mean(slope_distance[path[:, 0], path[:, 1]]))
    query_range = float(np.ptp(query))
    reference_range = float(np.ptp(reference))
    range_penalty = min(
        abs(query_range - reference_range),
        float(settings["range_difference_cap"]),
    )
    path_cost = float(accumulated[-1, -1] / max(1, len(path)))
    cost = path_cost + float(settings["range_weight"]) * range_penalty
    return {
        "cost": cost,
        "pitch_cost": pitch_cost,
        "slope_cost": slope_cost,
        "range_penalty": range_penalty,
        "path_length": int(len(path)),
    }


def _trace_path(accumulated: np.ndarray) -> list[tuple[int, int]]:
    i, j = accumulated.shape[0] - 1, accumulated.shape[1] - 1
    path: list[tuple[int, int]] = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = int(
                np.argmin(
                    (
                        accumulated[i - 1, j - 1],
                        accumulated[i - 1, j],
                        accumulated[i, j - 1],
                    )
                )
            )
            if move == 0:
                i, j = i - 1, j - 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return path
