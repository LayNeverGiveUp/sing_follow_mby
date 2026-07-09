from pathlib import Path
import math
import struct
import unittest

from app.audio.features import StreamingPitchExtractor
from app.matching.catalog import CatalogStore
from app.matching.engine import RealtimeMatcher


class RealtimeMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.catalog = CatalogStore(base_dir / "data" / "catalog").load("kids_songs_v1")

    def test_matches_twinkle_demo_features(self) -> None:
        matcher = RealtimeMatcher(self.catalog)
        matcher.append_features([60, 60, 67])
        matcher.append_features([67, 69, 69, 67])

        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, "twinkle_twinkle")
        self.assertGreaterEqual(result.confidence, 0.9)
        self.assertEqual(result.reply_audio, "twinkle_next.wav")

    def test_low_information_input_falls_back(self) -> None:
        matcher = RealtimeMatcher(self.catalog)
        matcher.append_features([60])

        result = matcher.finalize()

        self.assertFalse(result.matched)
        self.assertEqual(result.handoff_type, "fallback")

    def test_matches_mao_buyi_catalog_demo_features(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        catalog = CatalogStore(base_dir / "data" / "catalog").load("mao_buyi_v1")
        matcher = RealtimeMatcher(catalog)
        matcher.append_features([60, 62, 63])
        matcher.append_features([67, 65, 63])
        matcher.append_features([62, 60])

        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, "mao_buyi_xiaochou")
        self.assertEqual(result.song_name, "消愁")
        self.assertEqual(result.reply_audio, "mao_buyi_xiaochou_next.wav")

    def test_synthetic_pcm_pitch_matches_mao_buyi_catalog(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        catalog = CatalogStore(base_dir / "data" / "catalog").load("mao_buyi_v1")
        matcher = RealtimeMatcher(catalog)
        extractor = StreamingPitchExtractor(sample_rate=16000)

        for midi in [60, 62, 63, 67, 65, 63, 62, 60]:
            pcm = _sine_pcm16(_midi_to_hz(midi), sample_rate=16000, duration_ms=130)
            for offset in range(0, len(pcm), 2048):
                matcher.append_features(extractor.append_pcm16(pcm[offset : offset + 2048]))

        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, "mao_buyi_xiaochou")


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _sine_pcm16(freq: float, sample_rate: int, duration_ms: int) -> bytes:
    frames = []
    total = int(sample_rate * duration_ms / 1000)
    for index in range(total):
        sample = int(12000 * math.sin(2.0 * math.pi * freq * index / sample_rate))
        frames.append(struct.pack("<h", sample))
    return b"".join(frames)


if __name__ == "__main__":
    unittest.main()
