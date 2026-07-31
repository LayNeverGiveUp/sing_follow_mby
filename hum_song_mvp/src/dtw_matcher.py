from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa
from numba import njit

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
    zeros = np.zeros(2, dtype=np.float32)
    _cost_matrix_kernel(
        np.zeros(2, dtype=np.bool_),
        zeros,
        zeros,
        zeros,
        zeros,
        np.zeros(2, dtype=np.bool_),
        zeros,
        zeros,
        zeros,
        1.0,
        6.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )


def subsequence_dtw(query: PitchFeatures, reference: PitchFeatures, config: dict) -> DtwResult | None:
    results = subsequence_dtw_nbest(query, reference, config, max_candidates=1)
    return results[0] if results else None


def subsequence_dtw_nbest(
    query: PitchFeatures,
    reference: PitchFeatures,
    config: dict,
    max_candidates: int | None = None,
) -> list[DtwResult]:
    """Return distinct full-song alignment locations in DTW endpoint order.

    A subsequence DTW minimum is normally surrounded by many nearly identical
    endpoints.  Returning those adjacent endpoints as N-best candidates would
    create false ambiguity, so paths are suppressed by temporal overlap and
    centre distance before they leave the matcher.
    """
    if len(query.time) == 0 or len(reference.time) == 0:
        return []
    local = _cost_matrix(query, reference, config["matching"])
    n, m = local.shape
    # librosa's DTW recurrence is JIT-compiled internally.  We keep the
    # accumulated matrix to evaluate several valid end columns below.
    dtw_cost = librosa.sequence.dtw(C=local, subseq=True, backtrack=False)
    settings = config.get("position_resolution", {})
    requested = max_candidates if max_candidates is not None else int(settings.get("max_candidates", 5))
    if requested <= 0:
        return []
    endpoint_limit = min(int(settings.get("max_endpoints_to_trace", 64)), m)
    primary_limit = min(int(settings.get("primary_endpoints_to_trace", 64)), m)
    min_separation_frames = max(
        1,
        int(round(float(settings.get("min_temporal_separation_seconds", 1.5)) / float(config["pitch"]["hop_seconds"]))),
    )
    max_overlap = float(settings.get("max_candidate_overlap", 0.50))
    ranked_endpoints = np.argsort(dtw_cost[-1, :])
    results: list[DtwResult] = []
    # Preserve the legacy Top-1 exactly: it searched the 64 lowest accumulated
    # endpoints in order until it found a path with a valid speed ratio.
    for end_column in ranked_endpoints[:primary_limit]:
        result = _result_from_endpoint(query, reference, config, local, dtw_cost, n, int(end_column))
        if result is not None:
            results.append(result)
            break
    if not results or requested == 1:
        return results

    # Only after Top-1 is fixed do we inspect endpoints from other time regions.
    # Skipping endpoints close to an accepted path prevents hundreds of nearly
    # identical backtracks without changing the primary result.
    traced = 0
    for end_column in ranked_endpoints:
        if traced >= endpoint_limit or len(results) >= requested:
            break
        if any(abs(int(end_column) + 1 - result.end_frame) < min_separation_frames for result in results):
            continue
        traced += 1
        result = _result_from_endpoint(query, reference, config, local, dtw_cost, n, int(end_column))
        if result is None:
            continue
        if any(_same_location(result, existing, min_separation_frames, max_overlap) for existing in results):
            continue
        results.append(result)
    return results


def _result_from_endpoint(
    query: PitchFeatures,
    reference: PitchFeatures,
    config: dict,
    local: np.ndarray,
    accumulated: np.ndarray,
    query_frames: int,
    end_column: int,
) -> DtwResult | None:
    path = _trace_path(accumulated, query_frames - 1, end_column)
    if not path:
        return None
    start_frame, end_frame = path[0][1], path[-1][1]
    reference_span = max(1, end_frame - start_frame + 1)
    query_span = max(1, path[-1][0] - path[0][0] + 1)
    speed_ratio = reference_span / query_span
    if not (float(config["matching"]["min_speed_ratio"]) <= speed_ratio <= float(config["matching"]["max_speed_ratio"])):
        return None
    raw_cost = float(accumulated[query_frames - 1, end_column] / max(1, len(path)))
    effective_cost, paired_seconds, query_coverage = _path_quality(query, reference, path, local, config)
    if not np.isfinite(effective_cost):
        return None
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


def _same_location(first: DtwResult, second: DtwResult, min_separation_frames: int, max_overlap: float) -> bool:
    first_center = (first.start_frame + first.end_frame) / 2.0
    second_center = (second.start_frame + second.end_frame) / 2.0
    if abs(first_center - second_center) < min_separation_frames:
        return True
    overlap = max(0, min(first.end_frame, second.end_frame) - max(first.start_frame, second.start_frame) + 1)
    shorter = max(1, min(first.end_frame - first.start_frame + 1, second.end_frame - second.start_frame + 1))
    return overlap / shorter > max_overlap


def _cost_matrix(query: PitchFeatures, reference: PitchFeatures, config: dict) -> np.ndarray:
    """Compute the same frame costs without allocating full-size temporaries."""
    return _cost_matrix_kernel(
        query.voiced,
        query.confidence,
        query.relative_pitch,
        query.delta_pitch,
        query.onset_strength,
        reference.voiced,
        reference.relative_pitch,
        reference.delta_pitch,
        reference.onset_strength,
        float(config["octave_penalty"]),
        float(config["pitch_distance_cap"]),
        float(config["onset_weight"]),
        float(config["relative_pitch_weight"]),
        float(config["delta_pitch_weight"]),
        float(config["voiced_weight"]),
        float(config["voiced_mismatch_penalty"]),
        float(config["silence_silence_penalty"]),
    )


@njit(cache=False, nogil=True)
def _cost_matrix_kernel(
    query_voiced: np.ndarray,
    query_confidence: np.ndarray,
    query_pitch: np.ndarray,
    query_delta: np.ndarray,
    query_onset: np.ndarray,
    reference_voiced: np.ndarray,
    reference_pitch: np.ndarray,
    reference_delta: np.ndarray,
    reference_onset: np.ndarray,
    octave_penalty: float,
    pitch_distance_cap: float,
    onset_weight: float,
    relative_pitch_weight: float,
    delta_pitch_weight: float,
    voiced_weight: float,
    voiced_mismatch_penalty: float,
    silence_silence_penalty: float,
) -> np.ndarray:
    matrix = np.empty((len(query_voiced), len(reference_voiced)), dtype=np.float32)
    for query_index in range(len(query_voiced)):
        confidence = max(0.15, float(query_confidence[query_index]))
        query_pitch_value = float(query_pitch[query_index]) if np.isfinite(query_pitch[query_index]) else 0.0
        query_delta_value = float(query_delta[query_index]) if np.isfinite(query_delta[query_index]) else 0.0
        for reference_index in range(len(reference_voiced)):
            onset_distance = min(
                abs(float(query_onset[query_index]) - float(reference_onset[reference_index])),
                1.0,
            )
            cost = onset_weight * onset_distance
            if query_voiced[query_index] and reference_voiced[reference_index]:
                reference_pitch_value = (
                    float(reference_pitch[reference_index])
                    if np.isfinite(reference_pitch[reference_index])
                    else 0.0
                )
                raw_difference = abs(query_pitch_value - reference_pitch_value)
                pitch_distance = min(
                    raw_difference,
                    abs(raw_difference - 12.0) + octave_penalty,
                    pitch_distance_cap,
                )
                reference_delta_value = (
                    float(reference_delta[reference_index])
                    if np.isfinite(reference_delta[reference_index])
                    else 0.0
                )
                delta_distance = min(abs(query_delta_value - reference_delta_value), 12.0)
                cost += confidence * (
                    relative_pitch_weight * pitch_distance
                    + delta_pitch_weight * delta_distance
                )
            elif query_voiced[query_index] != reference_voiced[reference_index]:
                cost += voiced_weight * voiced_mismatch_penalty
            else:
                cost += silence_silence_penalty
            matrix[query_index, reference_index] = cost
    return matrix


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
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 or j == 0:
            break
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
