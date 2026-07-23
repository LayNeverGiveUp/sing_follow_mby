import unittest

from src.lyric_mapper import map_lyrics


class LyricMapperTest(unittest.TestCase):
    def setUp(self):
        self.lines = [
            {"index": 0, "start_time": 1.0, "end_time": 3.0, "text": "first"},
            {"index": 1, "start_time": 3.0, "end_time": 6.0, "text": "second"},
        ]

    def test_maps_to_next_line(self):
        result = map_lyrics(self.lines, 3.5)
        self.assertEqual(result["current_lyric_index"], 1)
        self.assertIsNone(result["next_lyric_index"])

    def test_before_first_line_uses_first_line(self):
        result = map_lyrics(self.lines, 0.4)
        self.assertEqual(result["current_lyric_index"], 0)
        self.assertEqual(result["next_lyric_index"], 1)

    def test_boundary_tolerance_keeps_completed_line_current(self):
        result = map_lyrics(self.lines, 3.08, end_boundary_tolerance_seconds=0.25)
        self.assertEqual(result["current_lyric_index"], 0)
        self.assertEqual(result["next_lyric_index"], 1)


if __name__ == "__main__":
    unittest.main()
