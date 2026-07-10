import gzip
import json
import unittest
from unittest import mock

from app.asr.engine import (
    AsrEngine,
    TranscriptionResult,
    VolcengineAucClient,
    VolcengineAsrClient,
    _build_audio_request,
    _build_full_client_request,
    _parse_response,
    prepare_pcm16_for_asr,
)


class VolcengineAsrClientTest(unittest.TestCase):
    def test_prepare_pcm16_for_asr_normalizes_quiet_audio(self) -> None:
        import audioop
        import math
        import struct

        frames = []
        sample_rate = 16000
        for index in range(sample_rate):
            sample = int(800 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
            frames.append(struct.pack("<h", sample))
        pcm = b"".join(frames)

        prepared = prepare_pcm16_for_asr(pcm, sample_rate)

        self.assertGreater(audioop.max(prepared, 2), audioop.max(pcm, 2))
        self.assertGreater(len(prepared), 0)

    def test_auc_status_requires_public_base_url(self) -> None:
        client = VolcengineAucClient(api_key="key", public_base_url=None, upload_dir=__import__("pathlib").Path("/tmp"))

        self.assertEqual(client.status, "volcengine_auc_missing_public_base_url")

    def test_auc_transcribe_url_uses_submit_then_query(self) -> None:
        class FakeAuc(VolcengineAucClient):
            def __init__(self):
                super().__init__(api_key="key", public_base_url="https://example.com", upload_dir=__import__("pathlib").Path("/tmp"))
                self.calls = []

            def _post_json(self, url, request_id, payload):
                self.calls.append((url, request_id, payload))
                if url == self.QUERY_URL:
                    return {"result": {"text": "刚刚还在想你"}}
                return {}

        client = FakeAuc()

        text = client.transcribe_url("https://example.com/audio.mp3")

        self.assertEqual(text, "刚刚还在想你")
        self.assertEqual(client.calls[0][0], client.SUBMIT_URL)
        self.assertEqual(client.calls[1][0], client.QUERY_URL)

    def test_auc_processing_status_is_not_error(self) -> None:
        class Headers:
            def get(self, key, default=None):
                values = {"x-api-status-code": "20000001", "x-api-message": "Processing"}
                return values.get(key, default)

        class Response:
            headers = Headers()

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return b"{}"

        client = VolcengineAucClient(api_key="key", public_base_url="https://example.com", upload_dir=__import__("pathlib").Path("/tmp"))
        with mock.patch("app.asr.engine.urlrequest.urlopen", return_value=Response()):
            self.assertEqual(client._post_json(client.QUERY_URL, "request-id", {}), {})

    def test_auc_no_valid_speech_status_is_terminal_no_text(self) -> None:
        class FakeAuc(VolcengineAucClient):
            def __init__(self):
                super().__init__(api_key="key", public_base_url="https://example.com", upload_dir=__import__("pathlib").Path("/tmp"))

            def _post_json(self, url, request_id, payload):
                if url == self.QUERY_URL:
                    self._last_status_code = "20000003"
                    return {"result": {"text": ""}}
                return {}

        client = FakeAuc()

        self.assertIsNone(client.transcribe_url("https://example.com/audio.wav", audio_format="wav"))
        self.assertEqual(client.last_detail, "volcengine_auc:20000003:no_valid_speech")

    def test_asr_engine_uses_volcengine_without_local_model(self) -> None:
        class FakeVolcengine:
            enabled = True

            def transcribe_pcm16(self, pcm, sample_rate):
                self.called = (pcm, sample_rate)
                return "背上所有的梦与想"

        fake = FakeVolcengine()
        engine = AsrEngine(model_name=None, volcengine=fake)

        result = engine.transcribe_pcm16([b"\x00\x00"], 16000)

        self.assertEqual(result, TranscriptionResult(text="背上所有的梦与想", source="volcengine_ws"))
        self.assertEqual(fake.called, (b"\x00\x00", 16000))

    def test_headers_use_runtime_credentials(self) -> None:
        client = VolcengineAsrClient(
            access_key="access-key",
            app_key="app-key",
            app_id="app",
            resource_id="resource",
        )

        headers = client._headers("request-id")

        self.assertEqual(headers["X-Api-App-Key"], "app-key")
        self.assertEqual(headers["X-Api-Access-Key"], "access-key")
        self.assertEqual(headers["X-Api-Resource-Id"], "resource")
        self.assertEqual(headers["X-Api-Request-Id"], "request-id")

    def test_builds_gzip_json_full_client_request(self) -> None:
        packet = _build_full_client_request({"audio": {"format": "pcm"}})

        self.assertEqual(packet[:4], bytes([0x11, 0x10, 0x11, 0x00]))
        size = int.from_bytes(packet[4:8], "big", signed=True)
        payload = json.loads(gzip.decompress(packet[8 : 8 + size]).decode("utf-8"))

        self.assertEqual(payload["audio"]["format"], "pcm")

    def test_builds_last_audio_request(self) -> None:
        packet = _build_audio_request(b"\x00\x01", is_last=True)

        self.assertEqual(packet[:4], bytes([0x11, 0x22, 0x01, 0x00]))
        size = int.from_bytes(packet[4:8], "big", signed=True)
        self.assertEqual(gzip.decompress(packet[8 : 8 + size]), b"\x00\x01")

    def test_parses_server_text_response(self) -> None:
        payload = gzip.compress(json.dumps({"result": {"text": "背上所有的梦与想", "is_final": True}}).encode())
        packet = bytes([0x11, 0x90, 0x11, 0x00]) + len(payload).to_bytes(4, "big", signed=True) + payload

        response = _parse_response(packet)

        self.assertEqual(response.text, "背上所有的梦与想")
        self.assertTrue(response.is_final)


if __name__ == "__main__":
    unittest.main()
