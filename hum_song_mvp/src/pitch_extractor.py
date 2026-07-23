from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .pitch_postprocess import clean_pitch, local_relative_pitch, pitch_delta


@dataclass(frozen=True)
class PitchFeatures:
    time: np.ndarray
    pitch: np.ndarray
    relative_pitch: np.ndarray
    delta_pitch: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray
    onset_strength: np.ndarray

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def extract_features(samples: np.ndarray, config: dict) -> PitchFeatures:
    audio = config["audio"]
    pitch_config = config["pitch"]
    sr = int(audio["sample_rate"])
    hop = max(1, int(round(float(pitch_config["hop_seconds"]) * sr)))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        samples,
        fmin=float(pitch_config["fmin_hz"]),
        fmax=float(pitch_config["fmax_hz"]),
        sr=sr,
        frame_length=int(pitch_config["frame_length"]),
        hop_length=hop,
    )
    confidence = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
    cleaned = clean_pitch(
        np.asarray(f0, dtype=np.float32), confidence,
        min_confidence=float(pitch_config["min_confidence"]),
        min_midi=float(pitch_config["min_midi"]),
        max_midi=float(pitch_config["max_midi"]),
        max_gap_frames=int(pitch_config["max_gap_frames"]),
        median_filter_frames=int(pitch_config["median_filter_frames"]),
        octave_jump_semitones=float(pitch_config["octave_jump_semitones"]),
    )
    onset = librosa.onset.onset_strength(y=samples, sr=sr, hop_length=hop)
    onset = _resample_to_length(onset, len(cleaned))
    if onset.size and float(np.max(onset)) > 0:
        onset = onset / float(np.max(onset))
    return PitchFeatures(
        time=librosa.times_like(cleaned, sr=sr, hop_length=hop).astype(np.float32),
        pitch=cleaned,
        relative_pitch=local_relative_pitch(cleaned, int(pitch_config["local_median_frames"])),
        delta_pitch=pitch_delta(cleaned),
        voiced=np.isfinite(cleaned),
        confidence=confidence,
        onset_strength=onset.astype(np.float32),
    )


def _resample_to_length(values: np.ndarray, length: int) -> np.ndarray:
    if length == 0:
        return np.empty(0, dtype=np.float32)
    if len(values) == length:
        return np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros(length, dtype=np.float32)
    positions = np.linspace(0, len(values) - 1, length)
    return np.interp(positions, np.arange(len(values)), values).astype(np.float32)
