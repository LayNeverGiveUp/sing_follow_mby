from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import os
from time import monotonic, sleep
import uuid
import wave

import numpy as np
import requests


@dataclass(frozen=True)
class AsrWord:
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None


@dataclass(frozen=True)
class AsrTranscript:
    text: str
    source: str
    confidence: float | None = None
    words: tuple[AsrWord, ...] = ()


class DisabledLyricsAsr:
    enabled = False
    status = "disabled"

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> AsrTranscript | None:
        return None


class VolcengineLyricsAsr:
    """Synchronous adapter for Volcengine recording-file big-model ASR 2.0."""

    SUBMIT_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
    QUERY_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
    DEFAULT_RESOURCE_ID = "volc.seedasr.auc"
    SUCCESS = "20000000"
    PROCESSING = {"20000001", "20000002"}

    def __init__(
        self,
        access_token: str | None,
        app_id: str | None,
        user_id: str = "hum-song-followup-mvp",
        resource_id: str = DEFAULT_RESOURCE_ID,
        submit_endpoint: str = SUBMIT_ENDPOINT,
        query_endpoint: str = QUERY_ENDPOINT,
        request_timeout: float = 20.0,
        poll_interval: float = 0.5,
        max_wait: float = 20.0,
    ) -> None:
        self.access_token = access_token
        self.app_id = app_id
        self.user_id = user_id
        self.resource_id = resource_id
        self.submit_endpoint = submit_endpoint
        self.query_endpoint = query_endpoint
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    @property
    def enabled(self) -> bool:
        return bool(self.access_token and self.app_id)

    @property
    def status(self) -> str:
        if not self.access_token or not self.app_id:
            return "missing_credentials"
        return "volcengine_auc"

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> AsrTranscript | None:
        if not self.enabled or samples.size == 0:
            return None
        pcm = _float_samples_to_pcm16(samples)
        if not pcm:
            return None

        task_id = str(uuid.uuid4())
        wav_data = _build_pcm16_wav(pcm, sample_rate)
        self._submit(task_id, base64.b64encode(wav_data).decode("ascii"))
        payload = self._wait_for_result(task_id)

        text = _extract_text(payload).strip()
        words = _extract_words(payload)
        return AsrTranscript(text=text, source="volcengine_auc", words=words) if text else None

    def _headers(self, task_id: str, *, submit: bool) -> dict[str, str]:
        headers = {
            "X-Api-App-Key": self.app_id or "",
            "X-Api-Access-Key": self.access_token or "",
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": task_id,
        }
        if submit:
            headers["X-Api-Sequence"] = "-1"
        return headers

    def _submit(self, task_id: str, audio_data: str) -> None:
        payload = {
            "user": {"uid": self.user_id},
            "audio": {"data": audio_data},
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": False,
                "enable_ddc": True,
                "enable_lid": True,
            },
        }
        response = requests.post(
            self.submit_endpoint,
            headers=self._headers(task_id, submit=True),
            json=payload,
            timeout=self.request_timeout,
        )
        self._require_status(response, expected={self.SUCCESS}, action="submit")

    def _wait_for_result(self, task_id: str) -> dict:
        deadline = monotonic() + self.max_wait
        while True:
            response = requests.post(
                self.query_endpoint,
                headers=self._headers(task_id, submit=False),
                json={},
                timeout=self.request_timeout,
            )
            status = response.headers.get("X-Api-Status-Code")
            if status == self.SUCCESS:
                payload = response.json()
                return payload if isinstance(payload, dict) else {}
            if status not in self.PROCESSING:
                self._require_status(response, expected=self.PROCESSING | {self.SUCCESS}, action="query")
            if monotonic() >= deadline:
                raise TimeoutError(f"Volcengine ASR query timed out after {self.max_wait:.1f}s")
            sleep(self.poll_interval)

    @staticmethod
    def _require_status(response, *, expected: set[str], action: str) -> None:
        status = response.headers.get("X-Api-Status-Code")
        if response.status_code < 400 and status in expected:
            return
        message = response.headers.get("X-Api-Message") or response.reason or "unknown error"
        log_id = response.headers.get("X-Tt-Logid")
        suffix = f" (log_id={log_id})" if log_id else ""
        raise RuntimeError(f"Volcengine ASR {action} failed: status={status}, message={message}{suffix}")


def get_lyrics_asr(config: dict | None = None):
    settings = (config or {}).get("lyrics_asr", {})
    if not bool(settings.get("enabled", True)):
        return DisabledLyricsAsr()
    provider = os.getenv("HUM_LYRICS_ASR_PROVIDER", str(settings.get("provider", "volcengine_auc"))).lower()
    if provider not in {"volcengine", "volcengine_auc"}:
        return DisabledLyricsAsr()
    return VolcengineLyricsAsr(
        access_token=os.getenv("VOLCENGINE_ASR_ACCESS_TOKEN") or os.getenv("VOLCENGINE_ASR_ACCESS_KEY"),
        app_id=os.getenv("VOLCENGINE_ASR_APP_ID"),
        user_id=os.getenv("VOLCENGINE_ASR_USER_ID", "hum-song-followup-mvp"),
        resource_id=os.getenv("VOLCENGINE_ASR_RESOURCE_ID", VolcengineLyricsAsr.DEFAULT_RESOURCE_ID),
        submit_endpoint=os.getenv("VOLCENGINE_ASR_SUBMIT_ENDPOINT", VolcengineLyricsAsr.SUBMIT_ENDPOINT),
        query_endpoint=os.getenv("VOLCENGINE_ASR_QUERY_ENDPOINT", VolcengineLyricsAsr.QUERY_ENDPOINT),
        request_timeout=float(settings.get("request_timeout_seconds", 20.0)),
        poll_interval=float(settings.get("poll_interval_seconds", 0.5)),
        max_wait=float(settings.get("max_wait_seconds", 20.0)),
    )


def _float_samples_to_pcm16(samples: np.ndarray) -> bytes:
    finite = np.nan_to_num(np.asarray(samples, dtype=np.float32))
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    if peak <= 1e-8:
        return b""
    gain = min(8.0, 0.85 / peak) if peak < 0.85 else 1.0
    normalized = np.clip(finite * gain, -1.0, 1.0)
    return (normalized * 32767.0).astype("<i2").tobytes()


def _build_pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def _extract_text(payload: dict) -> str:
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"]
    return ""


def _extract_words(payload: dict) -> tuple[AsrWord, ...]:
    result = payload.get("result") if isinstance(payload, dict) else None
    utterances = result.get("utterances") if isinstance(result, dict) else None
    if not isinstance(utterances, list):
        return ()
    words: list[AsrWord] = []
    for utterance in utterances:
        values = utterance.get("words") if isinstance(utterance, dict) else None
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict) or not str(value.get("text", "")).strip():
                continue
            try:
                start_time = float(value["start_time"]) / 1000.0
                end_time = float(value["end_time"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            confidence_value = value.get("confidence")
            confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else None
            words.append(
                AsrWord(
                    text=str(value["text"]),
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                )
            )
    return tuple(words)
