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

    def test_matches_mao_buyi_catalog_segment_features(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        catalog = CatalogStore(base_dir / "data" / "catalog").load("mao_buyi_v1")
        expected = catalog.segments[0]
        matcher = RealtimeMatcher(catalog)
        midpoint = max(1, len(expected.features) // 2)
        matcher.append_features(expected.features[:midpoint])
        matcher.append_features(expected.features[midpoint:])

        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, expected.song_id)
        self.assertEqual(result.line_id, expected.line_id)
        self.assertEqual(result.reply_audio, expected.reply_audio)

    def test_matches_mao_buyi_segment_with_octave_spikes(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        catalog = CatalogStore(base_dir / "data" / "catalog").load("mao_buyi_v1")
        expected = catalog.segments[0]
        noisy_features = list(expected.features)
        for index in range(4, len(noisy_features), 12):
            noisy_features[index] += 12.0
        matcher = RealtimeMatcher(catalog)

        matcher.append_features(noisy_features)
        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, expected.song_id)
        self.assertEqual(result.line_id, expected.line_id)

    def test_asr_text_recall_then_audio_rerank(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        catalog = CatalogStore(base_dir / "data" / "catalog").load("mao_buyi_v1")
        expected = catalog.segments[1]
        matcher = RealtimeMatcher(catalog)

        matcher.set_transcript(expected.text)
        matcher.append_features(expected.features)
        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, expected.song_id)
        self.assertEqual(result.line_id, expected.line_id)
        self.assertEqual(result.asr_transcript, expected.text)
        self.assertEqual(result.recall_source, "asr_text_recall")
        self.assertLessEqual(result.candidate_count, 16)

    def test_synthetic_pcm_pitch_matches_kids_catalog(self) -> None:
        matcher = RealtimeMatcher(self.catalog)
        extractor = StreamingPitchExtractor(sample_rate=16000)

        for midi in [60, 60, 67, 67, 69, 69, 67]:
            pcm = _sine_pcm16(_midi_to_hz(midi), sample_rate=16000, duration_ms=130)
            for offset in range(0, len(pcm), 2048):
                matcher.append_features(extractor.append_pcm16(pcm[offset : offset + 2048]))

        result = matcher.finalize()

        self.assertTrue(result.matched)
        self.assertEqual(result.song_id, "twinkle_twinkle")


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
