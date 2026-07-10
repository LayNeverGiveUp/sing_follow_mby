from __future__ import annotations

import audioop
import gzip
import json
import os
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from websockets.sync.client import connect


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    source: str


class AsrEngine:
    """Optional ASR adapter.

    Set VOLCENGINE_ASR_API_KEY to enable Volcengine AUC HTTP ASR. Because the
    official AUC API consumes public audio URLs, set
    VOLCENGINE_ASR_PUBLIC_BASE_URL when transcribing local browser recordings.
    Set SONG_FOLLOWUP_ASR_MODEL to use local faster-whisper. When neither is
    set, the service can still accept client-provided ASR text.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        language: str = "zh",
        auc: Optional["VolcengineAucClient"] = None,
        volcengine: Optional["VolcengineAsrClient"] = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.auc = auc
        self.volcengine = volcengine
        self._model = None
        self._load_error: Optional[str] = None
        self._last_detail: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.auc and self.auc.enabled) or bool(self.volcengine and self.volcengine.enabled) or bool(self.model_name)

    @property
    def status(self) -> str:
        if self._load_error:
            return f"unavailable:{self._load_error}"
        if self.auc and self.auc.enabled:
            return self.auc.status
        if self.volcengine and self.volcengine.enabled:
            return "volcengine_ws"
        if not self.model_name:
            return "client_or_disabled"
        return "faster_whisper"

    @property
    def last_detail(self) -> Optional[str]:
        return self._last_detail

    def transcribe_pcm16(self, pcm_chunks: Iterable[bytes], sample_rate: int) -> Optional[TranscriptionResult]:
        self._last_detail = None
        pcm = b"".join(pcm_chunks)
        if not pcm:
            self._last_detail = "empty_audio"
            return None

        if self.auc and self.auc.enabled:
            try:
                text = self.auc.transcribe_pcm16(pcm, sample_rate)
            except Exception as exc:  # pragma: no cover - depends on remote ASR
                self._load_error = f"volcengine_auc:{exc}"
                text = None
            if text:
                self._last_detail = "volcengine_auc:text"
                return TranscriptionResult(text=text, source="volcengine_auc")
            self._last_detail = self.auc.last_detail or "volcengine_auc:no_text"
            if not self.model_name and not (self.volcengine and self.volcengine.enabled):
                return None

        if self.volcengine and self.volcengine.enabled:
            try:
                text = self.volcengine.transcribe_pcm16(pcm, sample_rate)
            except Exception as exc:  # pragma: no cover - depends on remote ASR
                self._load_error = f"volcengine:{exc}"
                text = None
            if text:
                self._last_detail = "volcengine_ws:text"
                return TranscriptionResult(text=text, source="volcengine_ws")

        if not self.model_name:
            return None

        model = self._load_model()
        if model is None:
            return None

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "recording.wav"
            _write_wav_pcm16(wav_path, pcm, sample_rate)
            segments, _ = model.transcribe(str(wav_path), language=self.language, vad_filter=True)
            text = "".join(segment.text for segment in segments).strip()

        if not text:
            self._last_detail = "faster_whisper:no_text"
            return None
        self._last_detail = "faster_whisper:text"
        return TranscriptionResult(text=text, source="faster_whisper")

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self._load_error = "missing faster_whisper"
            return None

        try:
            self._model = WhisperModel(self.model_name, device="auto", compute_type="auto")
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            self._load_error = str(exc)
            return None
        return self._model


def get_asr_engine() -> AsrEngine:
    return AsrEngine(
        model_name=os.getenv("SONG_FOLLOWUP_ASR_MODEL"),
        language=os.getenv("SONG_FOLLOWUP_ASR_LANGUAGE", "zh"),
        auc=VolcengineAucClient.from_env(),
        volcengine=VolcengineAsrClient.from_env(),
    )


def _write_wav_pcm16(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def prepare_pcm16_for_asr(pcm: bytes, sample_rate: int, frame_ms: int = 30) -> bytes:
    """Trim obvious silence and normalize level before cloud ASR submission."""
    if len(pcm) < 2:
        return b""

    sample_width = 2
    frame_bytes = max(sample_width, int(sample_rate * frame_ms / 1000) * sample_width)
    frames = [pcm[offset : offset + frame_bytes] for offset in range(0, len(pcm), frame_bytes)]
    rms_values = [audioop.rms(frame, sample_width) for frame in frames if len(frame) >= sample_width]
    if not rms_values:
        return b""

    voiced = [value > 120 for value in rms_values]
    if any(voiced):
        first = next(index for index, item in enumerate(voiced) if item)
        last = len(voiced) - 1 - next(index for index, item in enumerate(reversed(voiced)) if item)
        pad = max(1, int(120 / frame_ms))
        start = max(0, first - pad) * frame_bytes
        end = min(len(pcm), (last + pad + 1) * frame_bytes)
        pcm = pcm[start:end]

    return _normalize_pcm16(pcm)


def _normalize_pcm16(pcm: bytes, target_peak: int = 24000) -> bytes:
    peak = audioop.max(pcm, 2) if pcm else 0
    if peak <= 0:
        return pcm
    factor = min(8.0, target_peak / peak)
    if factor <= 1.05:
        return pcm
    return audioop.mul(pcm, 2, factor)


class VolcengineAucClient:
    SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
    QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
    DEFAULT_RESOURCE_ID = "volc.seedasr.auc"

    def __init__(
        self,
        api_key: Optional[str],
        public_base_url: Optional[str],
        upload_dir: Path,
        resource_id: str = DEFAULT_RESOURCE_ID,
        poll_interval_s: float = 0.5,
        max_polls: int = 20,
    ) -> None:
        self.api_key = api_key
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.upload_dir = upload_dir
        self.resource_id = resource_id
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self.last_detail: Optional[str] = None
        self._last_status_code: Optional[str] = None

    @classmethod
    def from_env(cls) -> "VolcengineAucClient":
        root = Path(os.getenv("SONG_FOLLOWUP_DATA_DIR", "data"))
        return cls(
            api_key=os.getenv("VOLCENGINE_ASR_API_KEY"),
            public_base_url=os.getenv("VOLCENGINE_ASR_PUBLIC_BASE_URL"),
            upload_dir=root / "asr_uploads",
            resource_id=os.getenv("VOLCENGINE_ASR_RESOURCE_ID", cls.DEFAULT_RESOURCE_ID),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def status(self) -> str:
        if not self.api_key:
            return "client_or_disabled"
        if not self.public_base_url:
            return "volcengine_auc_missing_public_base_url"
        return "volcengine_auc"

    def transcribe_pcm16(self, pcm: bytes, sample_rate: int) -> Optional[str]:
        self.last_detail = None
        if not self.api_key:
            self.last_detail = "volcengine_auc:missing_api_key"
            return None
        if not self.public_base_url:
            self.last_detail = "volcengine_auc:missing_public_base_url"
            return None

        pcm = prepare_pcm16_for_asr(pcm, sample_rate)
        if not pcm:
            self.last_detail = "volcengine_auc:empty_after_preprocess"
            return None

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.wav"
        wav_path = self.upload_dir / filename
        _write_wav_pcm16(wav_path, pcm, sample_rate)
        audio_url = f"{self.public_base_url}/static/asr/{filename}"
        return self.transcribe_url(audio_url, audio_format="wav", sample_rate=sample_rate)

    def transcribe_url(self, audio_url: str, audio_format: str = "mp3", sample_rate: int = 16000) -> Optional[str]:
        request_id = str(uuid.uuid4())
        payload = {
            "user": {"uid": "song-followup-api"},
            "audio": {
                "url": audio_url,
                "format": audio_format,
                "codec": "raw",
                "rate": sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": False,
                "enable_ddc": False,
                "enable_speaker_info": False,
                "enable_channel_split": False,
                "show_utterances": False,
                "vad_segment": False,
                "sensitive_words_filter": "",
            },
        }
        self._post_json(self.SUBMIT_URL, request_id, payload)
        for _ in range(self.max_polls):
            result = self._post_json(self.QUERY_URL, request_id, {})
            text = _extract_text(result)
            if text:
                self.last_detail = f"volcengine_auc:{self._last_status_code or 'ok'}:text"
                return text
            if self._last_status_code == "20000003":
                self.last_detail = "volcengine_auc:20000003:no_valid_speech"
                return None
            time.sleep(self.poll_interval_s)
        self.last_detail = "volcengine_auc:timeout_no_text"
        return None

    def _post_json(self, url: str, request_id: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key or "",
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Request-Id": request_id,
                "X-Api-Sequence": "-1",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=15) as response:
                status_code = response.headers.get("x-api-status-code")
                message = response.headers.get("x-api-message", "")
                raw = response.read()
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        self._last_status_code = status_code
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
        if status_code in {"20000001", "20000003"}:
            self.last_detail = f"volcengine_auc:{status_code}:{message}"
            return parsed
        if status_code and status_code != "20000000":
            raise RuntimeError(f"status {status_code}: {message}")
        self.last_detail = f"volcengine_auc:{status_code or 'unknown'}:{message}"
        return parsed


class VolcengineAsrClient:
    ENDPOINT = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    DEFAULT_RESOURCE_ID = "volc.bigasr.sauc.duration"
    TARGET_SAMPLE_RATE = 16000
    PROTOCOL_VERSION = 0b0001
    HEADER_SIZE_WORDS = 0b0001
    SERIALIZATION_NONE = 0b0000
    SERIALIZATION_JSON = 0b0001
    COMPRESSION_GZIP = 0b0001
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_REQUEST = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    SERVER_ERROR_RESPONSE = 0b1111
    NO_SEQUENCE = 0b0000
    LAST_PACKET_NO_SEQUENCE = 0b0010

    def __init__(
        self,
        access_key: Optional[str],
        app_key: Optional[str],
        app_id: str,
        resource_id: str = DEFAULT_RESOURCE_ID,
        endpoint: str = ENDPOINT,
        connect_timeout: float = 8.0,
    ) -> None:
        self.access_key = access_key
        self.app_key = app_key
        self.app_id = app_id
        self.resource_id = resource_id
        self.endpoint = endpoint
        self.connect_timeout = connect_timeout

    @classmethod
    def from_env(cls) -> "VolcengineAsrClient":
        return cls(
            access_key=os.getenv("VOLCENGINE_ASR_ACCESS_KEY") or os.getenv("VOLCENGINE_ASR_ACCESS_TOKEN"),
            app_key=os.getenv("VOLCENGINE_ASR_APP_KEY") or os.getenv("VOLCENGINE_ASR_API_KEY"),
            app_id=os.getenv("VOLCENGINE_ASR_APP_ID", "song-followup-api"),
            resource_id=os.getenv("VOLCENGINE_ASR_RESOURCE_ID", cls.DEFAULT_RESOURCE_ID),
            endpoint=os.getenv("VOLCENGINE_ASR_ENDPOINT", cls.ENDPOINT),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.access_key and self.app_key)

    def transcribe_pcm16(self, pcm: bytes, sample_rate: int) -> Optional[str]:
        if not self.enabled:
            return None

        pcm = _ensure_pcm16_sample_rate(pcm, sample_rate, self.TARGET_SAMPLE_RATE)
        if not pcm:
            return None

        request_id = str(uuid.uuid4())
        headers = self._headers(request_id)
        request_payload = {
            "user": {"uid": self.app_id},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": self.TARGET_SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": True,
            },
        }

        final_text = ""
        with connect(self.endpoint, additional_headers=headers, open_timeout=self.connect_timeout) as websocket:
            websocket.send(_build_full_client_request(request_payload))
            response = _parse_response(websocket.recv())
            final_text = response.text or final_text

            for chunk, is_last in _iter_pcm_chunks(pcm, self.TARGET_SAMPLE_RATE):
                websocket.send(_build_audio_request(chunk, is_last=is_last))
                response = _parse_response(websocket.recv())
                final_text = response.text or final_text
                if response.is_final:
                    break

        return final_text.strip() or None

    def _headers(self, request_id: str) -> dict:
        headers = {
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Connect-Id": request_id,
        }
        if self.app_key:
            headers["X-Api-App-Key"] = self.app_key
        if self.access_key:
            headers["X-Api-Access-Key"] = self.access_key
        return headers


@dataclass(frozen=True)
class _VolcengineResponse:
    text: str
    is_final: bool


def _build_full_client_request(payload: dict) -> bytes:
    body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return _header(
        message_type=VolcengineAsrClient.FULL_CLIENT_REQUEST,
        flags=VolcengineAsrClient.NO_SEQUENCE,
        serialization=VolcengineAsrClient.SERIALIZATION_JSON,
        compression=VolcengineAsrClient.COMPRESSION_GZIP,
    ) + len(body).to_bytes(4, "big", signed=True) + body


def _build_audio_request(pcm: bytes, is_last: bool) -> bytes:
    body = gzip.compress(pcm)
    flags = VolcengineAsrClient.LAST_PACKET_NO_SEQUENCE if is_last else VolcengineAsrClient.NO_SEQUENCE
    return _header(
        message_type=VolcengineAsrClient.AUDIO_ONLY_REQUEST,
        flags=flags,
        serialization=VolcengineAsrClient.SERIALIZATION_NONE,
        compression=VolcengineAsrClient.COMPRESSION_GZIP,
    ) + len(body).to_bytes(4, "big", signed=True) + body


def _header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    return bytes(
        [
            (VolcengineAsrClient.PROTOCOL_VERSION << 4) | VolcengineAsrClient.HEADER_SIZE_WORDS,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0,
        ]
    )


def _parse_response(message) -> _VolcengineResponse:
    if isinstance(message, str):
        payload = json.loads(message)
        return _VolcengineResponse(text=_extract_text(payload), is_final=True)

    data = bytes(message)
    if len(data) < 8:
        return _VolcengineResponse(text="", is_final=False)

    header_size = (data[0] & 0x0F) * 4
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    serialization = data[2] >> 4
    compression = data[2] & 0x0F
    offset = header_size

    if message_type == VolcengineAsrClient.SERVER_ERROR_RESPONSE:
        if len(data) < offset + 8:
            raise RuntimeError("Volcengine ASR returned malformed error response")
        code = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        size = int.from_bytes(data[offset + 4 : offset + 8], "big", signed=False)
        payload = _decode_payload(data[offset + 8 : offset + 8 + size], serialization, compression)
        raise RuntimeError(f"Volcengine ASR error {code}: {payload}")

    if message_type != VolcengineAsrClient.FULL_SERVER_RESPONSE:
        return _VolcengineResponse(text="", is_final=flags == VolcengineAsrClient.LAST_PACKET_NO_SEQUENCE)

    if len(data) < offset + 4:
        return _VolcengineResponse(text="", is_final=False)
    size = int.from_bytes(data[offset : offset + 4], "big", signed=True)
    payload = _decode_payload(data[offset + 4 : offset + 4 + size], serialization, compression)
    parsed = json.loads(payload.decode("utf-8")) if isinstance(payload, bytes) else payload
    return _VolcengineResponse(text=_extract_text(parsed), is_final=_is_final_response(parsed, flags))


def _decode_payload(payload: bytes, serialization: int, compression: int):
    if compression == VolcengineAsrClient.COMPRESSION_GZIP:
        payload = gzip.decompress(payload)
    if serialization == VolcengineAsrClient.SERIALIZATION_JSON:
        return json.loads(payload.decode("utf-8"))
    return payload


def _extract_text(payload) -> str:
    if not isinstance(payload, dict):
        return ""
    result = payload.get("result")
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text
    payload_text = payload.get("text")
    return payload_text if isinstance(payload_text, str) else ""


def _is_final_response(payload, flags: int) -> bool:
    if flags == VolcengineAsrClient.LAST_PACKET_NO_SEQUENCE:
        return True
    if not isinstance(payload, dict):
        return False
    result = payload.get("result")
    if isinstance(result, dict):
        return bool(result.get("is_final") or result.get("final"))
    return bool(payload.get("is_final") or payload.get("final"))


def _iter_pcm_chunks(pcm: bytes, sample_rate: int, chunk_ms: int = 200):
    chunk_size = max(2, int(sample_rate * chunk_ms / 1000) * 2)
    total = len(pcm)
    if total == 0:
        return
    for offset in range(0, total, chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        yield chunk, offset + chunk_size >= total


def _ensure_pcm16_sample_rate(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, source_rate, target_rate, None)
    return converted
