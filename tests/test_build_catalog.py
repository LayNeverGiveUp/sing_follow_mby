import csv
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from tools.build_catalog import load_lyrics_for_song


class BuildCatalogTest(unittest.TestCase):
    def test_builds_catalog_from_wav_and_timestamp_csv(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            lyrics_dir = tmp_path / "lyrics"
            prompts_dir = tmp_path / "prompts"
            lyrics_out = tmp_path / "lyrics_out"
            catalog_out = tmp_path / "catalog.json"
            audio_dir.mkdir()
            lyrics_dir.mkdir()

            write_sine_wav(audio_dir / "mao_buyi_xiaochou.wav")
            with (lyrics_dir / "mao_buyi_xiaochou.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["line_id", "start_ms", "end_ms", "text"])
                writer.writeheader()
                writer.writerow({"line_id": "line_001", "start_ms": 0, "end_ms": 900, "text": "test line"})

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/build_catalog.py",
                    "--audio-dir",
                    str(audio_dir),
                    "--lyrics-dir",
                    str(lyrics_dir),
                    "--catalog-out",
                    str(catalog_out),
                    "--prompts-out",
                    str(prompts_dir),
                    "--lyrics-out",
                    str(lyrics_out),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Built 1 segments", result.stdout)
            self.assertTrue(catalog_out.exists())
            self.assertTrue((prompts_dir / "mao_buyi_xiaochou_line_001_prompt.wav").exists())
            self.assertTrue((lyrics_out / "mao_buyi_xiaochou.txt").exists())

    def test_builds_catalog_from_wav_and_lrc(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            lyrics_dir = tmp_path / "lyrics"
            prompts_dir = tmp_path / "prompts"
            lyrics_out = tmp_path / "lyrics_out"
            catalog_out = tmp_path / "catalog.json"
            audio_dir.mkdir()
            lyrics_dir.mkdir()

            write_sine_wav(audio_dir / "mao_buyi_xiaochou.wav")
            (lyrics_dir / "mao_buyi_xiaochou.lrc").write_text(
                "[00:00.00]first line\n[00:00.90]second line\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/build_catalog.py",
                    "--audio-dir",
                    str(audio_dir),
                    "--lyrics-dir",
                    str(lyrics_dir),
                    "--catalog-out",
                    str(catalog_out),
                    "--prompts-out",
                    str(prompts_dir),
                    "--lyrics-out",
                    str(lyrics_out),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Built 2 segments", result.stdout)
            self.assertTrue((prompts_dir / "mao_buyi_xiaochou_line_001_prompt.wav").exists())
            self.assertTrue((prompts_dir / "mao_buyi_xiaochou_line_002_prompt.wav").exists())

    def test_mp3_input_reports_ffmpeg_requirement_when_missing(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            lyrics_dir = tmp_path / "lyrics"
            prompts_dir = tmp_path / "prompts"
            lyrics_out = tmp_path / "lyrics_out"
            catalog_out = tmp_path / "catalog.json"
            audio_dir.mkdir()
            lyrics_dir.mkdir()

            (audio_dir / "mao_buyi_xiaochou.mp3").write_bytes(b"not a real mp3")
            (lyrics_dir / "mao_buyi_xiaochou.lrc").write_text("[00:00.00]first line\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/build_catalog.py",
                    "--audio-dir",
                    str(audio_dir),
                    "--lyrics-dir",
                    str(lyrics_dir),
                    "--catalog-out",
                    str(catalog_out),
                    "--prompts-out",
                    str(prompts_dir),
                    "--lyrics-out",
                    str(lyrics_out),
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )

            if result.returncode != 0:
                self.assertTrue(
                    "ffmpeg" in result.stderr.lower()
                    or "invalid data" in result.stderr.lower()
                    or "decodeerror" in result.stderr.lower()
                )

    def test_chinese_song_name_files_are_accepted(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            lyrics_dir = tmp_path / "lyrics"
            prompts_dir = tmp_path / "prompts"
            lyrics_out = tmp_path / "lyrics_out"
            catalog_out = tmp_path / "catalog.json"
            audio_dir.mkdir()
            lyrics_dir.mkdir()

            write_sine_wav(audio_dir / "消愁.wav")
            (lyrics_dir / "消愁.lrc").write_text(
                "[00:00.00]first line\n[00:00.90]second line\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/build_catalog.py",
                    "--audio-dir",
                    str(audio_dir),
                    "--lyrics-dir",
                    str(lyrics_dir),
                    "--catalog-out",
                    str(catalog_out),
                    "--prompts-out",
                    str(prompts_dir),
                    "--lyrics-out",
                    str(lyrics_out),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Built 2 segments", result.stdout)

    def test_embedded_lrc_is_preferred_over_sidecar_lrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lyrics_dir = tmp_path / "lyrics"
            lyrics_dir.mkdir()
            (lyrics_dir / "消愁.lrc").write_text("[00:00.00]sidecar line\n", encoding="utf-8")

            with mock.patch("tools.build_catalog.extract_embedded_lrc_text", return_value="[00:00.00]embedded line\n"):
                segments, source = load_lyrics_for_song(
                    audio_path=tmp_path / "消愁.mp3",
                    lyrics_dir=lyrics_dir,
                    song_id="mao_buyi_xiaochou",
                    song_name="消愁",
                    audio_duration_ms=2000,
                )

            self.assertEqual(source, "embedded_id3")
            self.assertEqual(segments[0].text, "embedded line")


def write_sine_wav(path: Path) -> None:
    sample_rate = 16000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(int(sample_rate * 2.0)):
            freq = 261.63 if index < sample_rate else 293.66
            sample = int(12000 * math.sin(2.0 * math.pi * freq * index / sample_rate))
            wav.writeframes(struct.pack("<h", sample))


if __name__ == "__main__":
    unittest.main()
