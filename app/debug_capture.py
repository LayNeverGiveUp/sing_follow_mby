"""Persist complete WebSocket inputs and recognition results for debugging."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave


@dataclass(frozen=True)
class DebugCapture:
    case_id: str
    directory: Path
    audio_path: Path
    request_path: Path
    result_path: Path


def capture_pcm16(
    root: Path,
    chunks: list[bytes],
    sample_rate: int,
    metadata: dict[str, Any] | None = None,
) -> DebugCapture:
    """Save the exact mono PCM16 stream received by the WebSocket service."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    pcm = b"".join(chunks)
    pcm = pcm[: len(pcm) - (len(pcm) % 2)]
    if not pcm:
        raise ValueError("No valid PCM16 samples were received")

    captured_at = datetime.now().astimezone()
    case_id = f"{captured_at:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory = root / f"{captured_at:%Y-%m-%d}" / case_id
    directory.mkdir(parents=True, exist_ok=False)
    audio_path = directory / "input.wav"
    request_path = directory / "request.json"
    result_path = directory / "result.json"

    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)

    request = {
        "case_id": case_id,
        "captured_at": captured_at.isoformat(timespec="milliseconds"),
        "format": "pcm_s16le",
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width_bytes": 2,
        "chunk_count": len(chunks),
        "pcm_bytes": len(pcm),
        "duration_seconds": round(len(pcm) / (sample_rate * 2.0), 3),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "metadata": metadata or {},
    }
    _write_json(request_path, request)
    return DebugCapture(case_id, directory, audio_path, request_path, result_path)


def write_capture_result(capture: DebugCapture, payload: dict[str, Any]) -> None:
    _write_json(capture.result_path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
