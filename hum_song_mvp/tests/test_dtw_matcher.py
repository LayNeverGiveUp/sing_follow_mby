import unittest

import numpy as np

from src.config import load_config
from src.dtw_matcher import subsequence_dtw
from src.pitch_postprocess import local_relative_pitch
from src.pitch_extractor import PitchFeatures


def features(values):
    values = np.asarray(values, dtype=np.float32)
    voiced = np.isfinite(values)
    relative = local_relative_pitch(values, 5)
    delta = np.zeros_like(values)
    delta[1:] = np.nan_to_num(values[1:] - values[:-1])
    return PitchFeatures(
        time=np.arange(len(values), dtype=np.float32) * 0.025,
        pitch=values,
        relative_pitch=relative,
        delta_pitch=delta,
        voiced=voiced,
        confidence=np.where(voiced, 1.0, 0.0).astype(np.float32),
        onset_strength=np.zeros_like(values),
    )


class DtwMatcherTest(unittest.TestCase):
    def test_finds_phrase_inside_full_song(self):
        config = load_config()
        query = features([60, 62, 64, 67, 65, 64, 62, 60])
        reference = features([50, 51, 52, 60, 62, 64, 67, 65, 64, 62, 60, 55, 56])
        result = subsequence_dtw(query, reference, config)
        self.assertIsNotNone(result)
        assert result
        self.assertLessEqual(result.start_frame, 4)
        self.assertGreaterEqual(result.end_frame, 9)
        self.assertLess(result.normalized_cost, 0.5)

    def test_accepts_octave_error_with_penalty(self):
        config = load_config()
        query = features([60, 62, 64, 67, 65, 64, 62, 60])
        reference = features([48, 50, 52, 55, 53, 52, 50, 48])
        result = subsequence_dtw(query, reference, config)
        self.assertIsNotNone(result)
        assert result
        # A whole-phrase octave transposition is intentionally normalized away.
        self.assertLess(result.normalized_cost, 0.2)


if __name__ == "__main__":
    unittest.main()
