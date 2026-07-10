from __future__ import annotations

import audioop
import math
import struct
from typing import Iterable, List, Optional


def coarse_pcm_features(chunk: bytes, sample_width: int = 2, frame_size: int = 640) -> List[float]:
    """Convert raw PCM bytes into coarse normalized energy features.

    This is a placeholder for streaming pitch/chroma extraction. It gives binary
    chunks a deterministic representation so the API can be exercised before
    real music feature extraction is wired in.
    """
    if not chunk:
        return []

    step = frame_size * sample_width
    values: List[float] = []
    for offset in range(0, len(chunk), step):
        frame = chunk[offset : offset + step]
        if len(frame) < sample_width:
            continue
        rms = audioop.rms(frame, sample_width)
        if rms <= 0:
            values.append(0.0)
        else:
            values.append(min(127.0, rms / 256.0))
    return values


class StreamingPitchExtractor:
    """Small streaming monophonic pitch extractor for PCM16 microphone input.

    This is intentionally dependency-free so the demo can run anywhere. It is
    good enough to close the microphone-to-DTW loop, but production should
    replace it with a stronger pitch/chroma extractor.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 80,
        hop_ms: int = 40,
        min_freq: float = 80.0,
        max_freq: float = 900.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.frame_size = max(256, int(sample_rate * frame_ms / 1000))
        self.hop_size = max(128, int(sample_rate * hop_ms / 1000))
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(self.frame_size - 1, int(sample_rate / min_freq))
        self._samples: List[float] = []

    def append_pcm16(self, chunk: bytes) -> List[float]:
        if not chunk:
            return []

        self._samples.extend(_pcm16_to_float(chunk))
        features: List[float] = []
        while len(self._samples) >= self.frame_size:
            frame = self._samples[: self.frame_size]
            midi = _estimate_midi_pitch(frame, self.sample_rate, self.min_lag, self.max_lag)
            if midi is not None:
                features.append(midi)
            del self._samples[: self.hop_size]
        return features


def extract_pitch_features_from_pcm16(pcm: bytes, sample_rate: int) -> List[float]:
    extractor = StreamingPitchExtractor(sample_rate=sample_rate)
    return extractor.append_pcm16(pcm)


def _pcm16_to_float(chunk: bytes) -> List[float]:
    if len(chunk) % 2:
        chunk = chunk[:-1]
    if not chunk:
        return []
    count = len(chunk) // 2
    samples = struct.unpack(f"<{count}h", chunk)
    return [sample / 32768.0 for sample in samples]


def _estimate_midi_pitch(
    frame: List[float],
    sample_rate: int,
    min_lag: int,
    max_lag: int,
    min_rms: float = 0.01,
    min_correlation: float = 0.35,
) -> Optional[float]:
    rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
    if rms < min_rms:
        return None

    mean = sum(frame) / len(frame)
    centered = [(sample - mean) * _hann(index, len(frame)) for index, sample in enumerate(frame)]
    energy = sum(sample * sample for sample in centered)
    if energy <= 1e-9:
        return None

    best_lag = 0
    best_score = 0.0
    for lag in range(min_lag, max_lag + 1):
        score = 0.0
        for index in range(0, len(centered) - lag):
            score += centered[index] * centered[index + lag]
        normalized = score / energy
        if normalized > best_score:
            best_score = normalized
            best_lag = lag

    if best_lag <= 0 or best_score < min_correlation:
        return None

    freq = sample_rate / best_lag
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def _hann(index: int, size: int) -> float:
    if size <= 1:
        return 1.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * index / (size - 1))


def normalize_features(values: Iterable[float]) -> List[float]:
    """Normalize a feature contour by removing absolute pitch/energy offset."""
    seq = [float(value) for value in values]
    if not seq:
        return []
    mean = sum(seq) / len(seq)
    return [value - mean for value in seq]


def stabilize_pitch_contour(values: Iterable[float]) -> List[float]:
    """Reduce common pitch-tracker artifacts before melody matching.

    The demo extractor often sees octave/harmonic spikes in original-song
    prompts with accompaniment. Human microphone input is usually smoother, so
    matching raw contours makes those reference spikes count as melody. This
    keeps the broad contour but removes isolated jumps and light jitter.
    """
    seq = [float(value) for value in values if 35.0 <= float(value) <= 90.0]
    if len(seq) < 3:
        return seq

    repaired = seq[:]
    for index in range(1, len(seq) - 1):
        previous_value = seq[index - 1]
        value = seq[index]
        next_value = seq[index + 1]
        if (
            abs(value - previous_value) > 10.0
            and abs(value - next_value) > 10.0
            and abs(previous_value - next_value) < 6.0
        ):
            repaired[index] = (previous_value + next_value) / 2.0

    if len(repaired) < 5:
        return repaired

    smoothed: List[float] = []
    for index in range(len(repaired)):
        start = max(0, index - 1)
        end = min(len(repaired), index + 2)
        window = sorted(repaired[start:end])
        smoothed.append(window[len(window) // 2])
    return smoothed
