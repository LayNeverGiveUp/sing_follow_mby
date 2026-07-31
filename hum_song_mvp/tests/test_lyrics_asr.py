from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import wave

import numpy as np

from src.lyrics_asr import VolcengineLyricsAsr


class FakeResponse:
    def __init__(self, status: str, payload: dict | None = None, message: str = "OK"):
        self.status_code = 200
        self.headers = {
            "X-Api-Status-Code": status,
            "X-Api-Message": message,
            "X-Tt-Logid": "test-log-id",
        }
        self._payload = payload or {}
        self.reason = message

    def json(self):
        return self._payload


def make_client(tmp_path: Path) -> VolcengineLyricsAsr:
    return VolcengineLyricsAsr(
        access_token="token-value",
        app_id="app-id-value",
        poll_interval=0.0,
    )


def test_auc_asr_requires_application_credentials(tmp_path):
    assert make_client(tmp_path).enabled
    assert not VolcengineLyricsAsr("token", None).enabled
    assert not VolcengineLyricsAsr(None, "app").enabled


def test_auc_asr_submits_base64_wav_and_polls_result(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        if url.endswith("/submit"):
            wav_data = base64.b64decode(json["audio"]["data"])
            with wave.open(BytesIO(wav_data), "rb") as audio:
                assert audio.getnchannels() == 1
                assert audio.getframerate() == 16000
                assert audio.getsampwidth() == 2
            return FakeResponse("20000000")
        return FakeResponse(
            "20000000",
            {
                "result": {
                    "text": "一杯敬朝阳一杯敬月光",
                    "utterances": [
                        {
                            "words": [
                                {"text": "一杯", "start_time": 100, "end_time": 500, "confidence": 0.9},
                                {"text": "敬", "start_time": 600, "end_time": 800, "confidence": 0.8},
                            ]
                        }
                    ],
                }
            },
        )

    client = make_client(tmp_path)
    monkeypatch.setattr(client.session, "post", fake_post)
    transcript = client.transcribe(np.ones(16000, dtype=np.float32) * 0.1, 16000)

    assert transcript is not None
    assert transcript.text == "一杯敬朝阳一杯敬月光"
    assert transcript.source == "volcengine_auc"
    assert len(transcript.words) == 2
    assert transcript.words[0].text == "一杯"
    assert transcript.words[0].start_time == 0.1
    assert transcript.words[0].end_time == 0.5
    assert len(calls) == 2
    assert calls[0][1]["X-Api-App-Key"] == "app-id-value"
    assert calls[0][1]["X-Api-Access-Key"] == "token-value"
    assert calls[0][1]["X-Api-Resource-Id"] == "volc.seedasr.auc"
    assert calls[0][1]["X-Api-Sequence"] == "-1"
    assert calls[0][2]["audio"]["data"]


def test_auc_asr_waits_while_task_is_processing(tmp_path, monkeypatch):
    statuses = iter(["20000000", "20000001", "20000002", "20000000"])

    def fake_post(url, *, headers, json, timeout):
        status = next(statuses)
        payload = {"result": {"text": "歌词"}} if status == "20000000" and url.endswith("/query") else {}
        return FakeResponse(status, payload)

    client = make_client(tmp_path)
    monkeypatch.setattr(client.session, "post", fake_post)
    transcript = client.transcribe(np.ones(8000, dtype=np.float32) * 0.1, 16000)

    assert transcript is not None
    assert transcript.text == "歌词"
