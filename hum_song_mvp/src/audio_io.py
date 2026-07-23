from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def load_mono_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {path.suffix}. Use WAV, MP3, M4A, AAC, FLAC, or OGG.")
    samples, _ = librosa.load(path, sr=sample_rate, mono=True)
    if samples.size == 0:
        raise ValueError(f"Audio file has no samples: {path}")
    return np.asarray(samples, dtype=np.float32)


def trim_outer_silence(samples: np.ndarray, top_db: float) -> tuple[np.ndarray, int, int]:
    if samples.size == 0:
        return samples, 0, 0
    trimmed, indices = librosa.effects.trim(samples, top_db=top_db)
    return np.asarray(trimmed, dtype=np.float32), int(indices[0]), int(indices[1])
