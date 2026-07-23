"""Bridge the offline humming MVP into the browser WebSocket service."""
from __future__ import annotations

import os
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import librosa
import numpy as np

from hum_song_mvp.src.config import load_config
from hum_song_mvp.src.recognize import recognize_samples


logger = logging.getLogger(__name__)


class HumMvpRecognizer:
    def __init__(self, database_dir: Path | None = None) -> None:
        configured = os.getenv("HUM_SONG_MVP_DATABASE_DIR")
        default_dir = Path(__file__).resolve().parents[1] / "hum_song_mvp" / "data" / "database"
        self.database_dir = database_dir or (Path(configured) if configured else default_dir)
        self.config = load_config()

    def recognize_pcm16(self, chunks: list[bytes], sample_rate: int) -> dict[str, Any]:
        total_started = perf_counter()
        if not chunks:
            raise ValueError("No microphone audio was received")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        decode_started = perf_counter()
        pcm = b"".join(chunks)
        if len(pcm) < 2:
            raise ValueError("No valid PCM16 samples were received")
        samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.float32) / 32768.0
        pcm_decode_ms = (perf_counter() - decode_started) * 1000
        target_rate = int(self.config["audio"]["sample_rate"])
        resample_started = perf_counter()
        if sample_rate != target_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_rate)
        resample_ms = (perf_counter() - resample_started) * 1000
        payload = recognize_samples(samples, self.database_dir, self.config)
        payload["matched"] = payload["accepted"]
        payload["confidence"] = payload["score"]
        payload["song_name"] = payload["song_id"]
        payload["feature_mode"] = "hum_song_mvp"
        diagnostics = payload.get("diagnostics", {})
        stages = diagnostics.setdefault("stage_ms", {})
        stages["pcm_decode"] = round(pcm_decode_ms, 1)
        stages["resample"] = round(resample_ms, 1)
        stages["service_total"] = round((perf_counter() - total_started) * 1000, 1)
        logger.info(
            "hum_mvp_result accepted=%s reason=%s song=%s input=%.2fs trimmed=%.2fs voiced=%.2fs range=%.2fst rms=%.1fdB candidates=%s",
            payload["accepted"], payload.get("reason"), payload.get("song_id"),
            diagnostics.get("input_duration_seconds", 0.0), diagnostics.get("trimmed_duration_seconds", 0.0),
            diagnostics.get("voiced_seconds", 0.0), diagnostics.get("pitch_range_semitones", 0.0),
            diagnostics.get("input_rms_dbfs", 0.0), diagnostics.get("candidates", []),
        )
        return payload


hum_mvp_recognizer = HumMvpRecognizer()
