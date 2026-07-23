from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from .pitch_extractor import PitchFeatures


@dataclass(frozen=True)
class DtwResult:
    normalized_cost: float
    raw_normalized_cost: float
    start_frame: int
    end_frame: int
    path: list[tuple[int, int]]
    speed_ratio: float
    paired_voiced_seconds: float
    query_voiced_coverage: float


def warm_dtw() -> None:
    """Compile librosa's internal DTW recurrence during service startup."""
    librosa.sequence.dtw(C=np.zeros((2, 2), dtype=np.float32), subseq=True, backtrack=False)


def subsequence_dtw(query: PitchFeatures, reference: PitchFeatures, config: dict) -> DtwResult | None:
    if len(query.time) == 0 or len(reference.time) == 0:
        return None
    local = _cost_matrix(query, reference, config["matching"])
    n, m = local.shape
    # librosa's DTW recurrence is JIT-compiled internally.  We keep the
    # accumulated matrix to evaluate several valid end columns below.
    dtw_cost = librosa.sequence.dtw(C=local, subseq=True, backtrack=False)
    accumulated = np.full((n + 1, m + 1), np.inf, dtype=dtw_cost.dtype)
    accumulated[0, :] = 0.0
    accumulated[1:, 1:] = dtw_cost
    candidates = np.argsort(accumulated[n, 1:]) + 1
    for end_column in candidates[: min(64, m)]:
        path = _trace_path(accumulated, n, int(end_column))
        if not path:
            continue
        start_frame, end_frame = path[0][1], path[-1][1]
        reference_span = max(1, end_frame - start_frame + 1)
        query_span = max(1, path[-1][0] - path[0][0] + 1)
        speed_ratio = reference_span / query_span
        if not (float(config["matching"]["min_speed_ratio"]) <= speed_ratio <= float(config["matching"]["max_speed_ratio"])):
            continue
        raw_cost = float(accumulated[n, end_column] / max(1, len(path)))
        effective_cost, paired_seconds, query_coverage = _path_quality(query, reference, path, local, config)
        if not np.isfinite(effective_cost):
            continue
        return DtwResult(
            normalized_cost=effective_cost,
            raw_normalized_cost=raw_cost,
            start_frame=start_frame,
            end_frame=end_frame,
            path=path,
            speed_ratio=float(speed_ratio),
            paired_voiced_seconds=paired_seconds,
            query_voiced_coverage=query_coverage,
        )
    return None


def _cost_matrix(query: PitchFeatures, reference: PitchFeatures, config: dict) -> np.ndarray:
    """Vectorized equivalent of the old per-frame Python distance function."""
    matching = config
    q_voiced = query.voiced[:, None]
    r_voiced = reference.voiced[None, :]
    both_voiced = q_voiced & r_voiced
    voiced_mismatch = q_voiced ^ r_voiced
    confidence = np.maximum(0.15, query.confidence[:, None])

    q_pitch = np.nan_to_num(query.relative_pitch, nan=0.0)[:, None]
    r_pitch = np.nan_to_num(reference.relative_pitch, nan=0.0)[None, :]
    raw_difference = np.abs(q_pitch - r_pitch)
    pitch_distance = np.minimum(
        np.minimum(raw_difference, np.abs(raw_difference - 12.0) + float(matching["octave_penalty"])),
        float(matching["pitch_distance_cap"]),
    )
    q_delta = np.nan_to_num(query.delta_pitch, nan=0.0)[:, None]
    r_delta = np.nan_to_num(reference.delta_pitch, nan=0.0)[None, :]
    delta_distance = np.minimum(np.abs(q_delta - r_delta), 12.0)
    onset_distance = np.minimum(np.abs(query.onset_strength[:, None] - reference.onset_strength[None, :]), 1.0)

    matrix = float(matching["onset_weight"]) * onset_distance
    matrix += both_voiced * confidence * (
        float(matching["relative_pitch_weight"]) * pitch_distance
        + float(matching["delta_pitch_weight"]) * delta_distance
    )
    matrix += voiced_mismatch * float(matching["voiced_weight"]) * float(matching["voiced_mismatch_penalty"])
    matrix += (~q_voiced & ~r_voiced) * float(matching["silence_silence_penalty"])
    return np.asarray(matrix, dtype=np.float32)


def _path_quality(query: PitchFeatures, reference: PitchFeatures, path: list[tuple[int, int]], local: np.ndarray, config: dict) -> tuple[float, float, float]:
    query_indices = np.asarray([point[0] for point in path], dtype=np.int32)
    reference_indices = np.asarray([point[1] for point in path], dtype=np.int32)
    paired = query.voiced[query_indices] & reference.voiced[reference_indices]
    if not np.any(paired):
        return float("inf"), 0.0, 0.0
    paired_query_indices = query_indices[paired]
    confidence = np.maximum(0.15, query.confidence[paired_query_indices])
    paired_cost = local[query_indices[paired], reference_indices[paired]]
    effective_cost = float(np.average(paired_cost, weights=confidence))
    paired_seconds = float(np.count_nonzero(paired)) * float(config["pitch"]["hop_seconds"])
    total_query_voiced = int(np.count_nonzero(query.voiced))
    coverage = len(np.unique(paired_query_indices)) / max(1, total_query_voiced)
    return effective_cost, paired_seconds, float(coverage)


def _trace_path(accumulated: np.ndarray, i: int, j: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        options = (accumulated[i - 1, j - 1], accumulated[i - 1, j], accumulated[i, j - 1])
        move = int(np.argmin(options))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path
