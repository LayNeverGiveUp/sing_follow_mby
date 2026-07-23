import tempfile
import unittest
from pathlib import Path

from src.lrc_parser import parse_lrc


class LrcParserTest(unittest.TestCase):
    def test_parses_multiple_timestamps_and_end_times(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.lrc"
            path.write_text("[00:01.20][00:05.20]same\n[00:09.00]last\n", encoding="utf-8")
            lines = parse_lrc(path, 12.0)
        self.assertEqual([line["start_time"] for line in lines], [1.2, 5.2, 9.0])
        self.assertEqual(lines[0]["end_time"], 5.2)
        self.assertEqual(lines[-1]["end_time"], 12.0)

    def test_rejects_lrc_without_timestamped_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.lrc"
            path.write_text("no timing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no timestamped"):
                parse_lrc(path, 10.0)


if __name__ == "__main__":
    unittest.main()
