from __future__ import annotations

import numpy as np


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    result = np.full(f0_hz.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(f0_hz) & (f0_hz > 0)
    result[valid] = 69.0 + 12.0 * np.log2(f0_hz[valid] / 440.0)
    return result


def clean_pitch(
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    *,
    min_confidence: float,
    min_midi: float,
    max_midi: float,
    max_gap_frames: int,
    median_filter_frames: int,
    octave_jump_semitones: float,
) -> np.ndarray:
    """Return MIDI pitch with only short internal gaps repaired."""
    pitch = hz_to_midi(np.asarray(f0_hz, dtype=np.float32))
    confidence = np.asarray(confidence, dtype=np.float32)
    pitch[~np.isfinite(confidence) | (confidence < min_confidence)] = np.nan
    pitch[(pitch < min_midi) | (pitch > max_midi)] = np.nan
    pitch = _repair_octave_spikes(pitch, octave_jump_semitones)
    pitch = _repair_short_gaps(pitch, max_gap_frames)
    pitch = _repair_octave_spikes(pitch, octave_jump_semitones)
    return _median_smooth(pitch, median_filter_frames)


def local_relative_pitch(pitch: np.ndarray, window_frames: int) -> np.ndarray:
    result = np.full(pitch.shape, np.nan, dtype=np.float32)
    radius = max(1, window_frames // 2)
    for index, value in enumerate(pitch):
        if not np.isfinite(value):
            continue
        window = pitch[max(0, index - radius) : min(len(pitch), index + radius + 1)]
        valid = window[np.isfinite(window)]
        if valid.size:
            result[index] = value - np.median(valid)
    return result


def pitch_delta(pitch: np.ndarray) -> np.ndarray:
    result = np.zeros(pitch.shape, dtype=np.float32)
    for index in range(1, len(pitch)):
        if np.isfinite(pitch[index]) and np.isfinite(pitch[index - 1]):
            result[index] = float(np.clip(pitch[index] - pitch[index - 1], -12.0, 12.0))
    return result


def _repair_short_gaps(values: np.ndarray, maximum: int) -> np.ndarray:
    result = values.copy()
    index = 0
    while index < len(result):
        if np.isfinite(result[index]):
            index += 1
            continue
        start = index
        while index < len(result) and not np.isfinite(result[index]):
            index += 1
        end = index
        if start == 0 or end == len(result) or end - start > maximum:
            continue
        left, right = result[start - 1], result[end]
        if np.isfinite(left) and np.isfinite(right):
            result[start:end] = np.linspace(left, right, end - start + 2, dtype=np.float32)[1:-1]
    return result


def _repair_octave_spikes(values: np.ndarray, jump: float) -> np.ndarray:
    result = values.copy()
    for index in range(1, len(result) - 1):
        current = result[index]
        if not np.isfinite(current):
            continue
        previous = index - 1
        following = index + 1
        while previous >= 0 and not np.isfinite(result[previous]):
            previous -= 1
        while following < len(result) and not np.isfinite(result[following]):
            following += 1
        if previous < 0 or following >= len(result):
            continue
        before, after = result[previous], result[following]
        if abs(current - before) > jump and abs(current - after) > jump and abs(before - after) < 3.0:
            candidates = (current - 12.0, current + 12.0)
            result[index] = min(candidates, key=lambda value: abs(value - (before + after) / 2.0))
    return result


def _median_smooth(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return values
    result = values.copy()
    radius = size // 2
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        window = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        valid = window[np.isfinite(window)]
        if valid.size:
            result[index] = np.median(valid)
    return result
