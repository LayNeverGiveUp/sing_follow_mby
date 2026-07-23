import unittest

import numpy as np

from src.pitch_postprocess import clean_pitch, hz_to_midi, local_relative_pitch


class PitchPostprocessTest(unittest.TestCase):
    def test_hz_to_midi(self):
        midi = hz_to_midi(np.array([440.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(float(midi[0]), 69.0, places=3)
        self.assertTrue(np.isnan(midi[1]))

    def test_repairs_short_gap_and_octave_spike(self):
        hz = np.array([261.63, 261.63, 0.0, 523.25, 261.63, 261.63], dtype=np.float32)
        confidence = np.ones_like(hz)
        pitch = clean_pitch(
            hz, confidence, min_confidence=0.2, min_midi=36, max_midi=96,
            max_gap_frames=1, median_filter_frames=1, octave_jump_semitones=8,
        )
        self.assertTrue(np.all(np.isfinite(pitch)))
        self.assertLess(abs(float(pitch[3] - pitch[2])), 3.0)

    def test_keeps_long_silence_missing(self):
        hz = np.array([261.63, 0.0, 0.0, 0.0, 261.63], dtype=np.float32)
        pitch = clean_pitch(
            hz, np.ones_like(hz), min_confidence=0.2, min_midi=36, max_midi=96,
            max_gap_frames=2, median_filter_frames=1, octave_jump_semitones=8,
        )
        self.assertTrue(np.isnan(pitch[2]))

    def test_relative_pitch_uses_local_median(self):
        relative = local_relative_pitch(np.array([60.0, 62.0, 64.0], dtype=np.float32), 3)
        self.assertAlmostEqual(float(relative[1]), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
