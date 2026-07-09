from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio.features import extract_pitch_features_from_pcm16

CATALOG_SONGS = [
    ("mao_buyi_xiaochou", "消愁"),
    ("mao_buyi_like_me", "像我这样的人"),
    ("mao_buyi_summer", "盛夏"),
    ("mao_buyi_unstained", "不染"),
    ("mao_buyi_ordinary_day", "平凡的一天"),
    ("mao_buyi_meat_and_vegetable", "一荤一素"),
    ("mao_buyi_borrow", "借"),
    ("mao_buyi_murmur", "呓语"),
    ("mao_buyi_no_question", "无问"),
    ("mao_buyi_muma_city", "牧马城市"),
    ("mao_buyi_yichengshanlu", "一程山路"),
]

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac")


@dataclass(frozen=True)
class LyricSegment:
    line_id: str
    start_ms: int
    end_ms: int
    text: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mao Buyi matching catalog from licensed wav + lyric timestamps.")
    parser.add_argument("--catalog-id", default="mao_buyi_v1")
    parser.add_argument("--audio-dir", default="data/source_audio/mao_buyi_v1")
    parser.add_argument("--lyrics-dir", default="data/source_lyrics/mao_buyi_v1")
    parser.add_argument("--catalog-out", default="data/catalog/mao_buyi_v1.json")
    parser.add_argument("--prompts-out", default="data/prompts/mao_buyi_v1")
    parser.add_argument("--lyrics-out", default="data/lyrics/mao_buyi_v1")
    parser.add_argument("--min-feature-count", type=int, default=3)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    lyrics_dir = Path(args.lyrics_dir)
    prompts_out = Path(args.prompts_out)
    lyrics_out = Path(args.lyrics_out)
    prompts_out.mkdir(parents=True, exist_ok=True)
    lyrics_out.mkdir(parents=True, exist_ok=True)

    songs = []
    for song_id, song_name in CATALOG_SONGS:
        audio_path = find_audio_path(audio_dir, song_id, song_name)
        lyric_path = find_lyric_path(lyrics_dir, song_id, song_name)
        if audio_path is None or lyric_path is None:
            continue

        wav = read_audio_as_pcm16(audio_path)
        lyric_segments = parse_lyrics(lyric_path, int(wav["duration_ms"]))
        copy_lyrics_for_reference(lyric_segments, lyrics_out / f"{song_id}.txt")

        segments = []
        for index, lyric in enumerate(lyric_segments, start=1):
            pcm = slice_pcm(wav["pcm"], wav["sample_rate"], lyric.start_ms, lyric.end_ms)
            features = [round(value, 3) for value in extract_pitch_features_from_pcm16(pcm, wav["sample_rate"])]
            if len(features) < args.min_feature_count:
                continue

            prompt_audio = f"{song_id}_{lyric.line_id}_prompt.wav"
            write_wav_pcm16(prompts_out / prompt_audio, pcm, wav["sample_rate"])
            segments.append(
                {
                    "line_id": lyric.line_id or f"line_{index}",
                    "text": lyric.text,
                    "start_ms": lyric.start_ms,
                    "end_ms": lyric.end_ms,
                    "features": features,
                    "reply_audio": f"{song_id}_next.wav",
                    "prompt_audio": prompt_audio,
                    "lyrics_file": f"{song_id}.txt",
                }
            )

        if segments:
            songs.append({"song_id": song_id, "song_name": song_name, "segments": segments})

    if not songs:
        raise SystemExit(
            "No songs were built. Put wav/mp3/m4a/aac/flac files in data/source_audio/mao_buyi_v1/ "
            "and timestamp csv or lrc files in data/source_lyrics/mao_buyi_v1/."
        )

    catalog = {
        "catalog_id": args.catalog_id,
        "reply_base_path": "/static/replies",
        "prompt_base_path": f"/static/prompts/{args.catalog_id}",
        "lyrics_base_path": f"data/lyrics/{args.catalog_id}",
        "songs": songs,
    }

    out_path = Path(args.catalog_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {sum(len(song['segments']) for song in songs)} segments from {len(songs)} songs -> {out_path}")


def find_audio_path(audio_dir: Path, song_id: str, song_name: str) -> Optional[Path]:
    for stem in (song_id, song_name):
        for suffix in SUPPORTED_AUDIO_EXTENSIONS:
            path = audio_dir / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def read_audio_as_pcm16(path: Path) -> Dict[str, object]:
    if path.suffix.lower() == ".wav":
        return read_wav_pcm16(path)
    decoded = decode_with_miniaudio(path)
    if decoded is not None:
        return decoded
    return decode_with_ffmpeg(path)


def decode_with_miniaudio(path: Path) -> Optional[Dict[str, object]]:
    try:
        import miniaudio
    except ImportError:
        return None

    sound = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=16000,
    )
    pcm = sound.samples.tobytes()
    duration_ms = int((len(sound.samples) / sound.nchannels) * 1000 / sound.sample_rate)
    return {"pcm": pcm, "sample_rate": sound.sample_rate, "duration_ms": duration_ms}


def decode_with_ffmpeg(path: Path) -> Dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"{path} requires miniaudio or ffmpeg for decoding. Install project requirements, "
            "install ffmpeg, or convert the file to 16-bit PCM WAV."
        )

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return read_wav_pcm16(wav_path)


def read_wav_pcm16(path: Path) -> Dict[str, object]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV")
    if channels == 1:
        pcm = frames
    elif channels == 2:
        pcm = stereo_to_mono_pcm16(frames)
    else:
        raise ValueError(f"{path} must be mono or stereo WAV")

    return {"pcm": pcm, "sample_rate": sample_rate, "duration_ms": int(frame_count * 1000 / sample_rate)}


def stereo_to_mono_pcm16(pcm: bytes) -> bytes:
    import struct

    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    mono = []
    for index in range(0, len(samples), 2):
        mono.append(int((samples[index] + samples[index + 1]) / 2))
    return struct.pack(f"<{len(mono)}h", *mono)


def write_wav_pcm16(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def slice_pcm(pcm: bytes, sample_rate: int, start_ms: int, end_ms: int) -> bytes:
    start = int(sample_rate * start_ms / 1000) * 2
    end = int(sample_rate * end_ms / 1000) * 2
    return pcm[max(0, start) : max(start, end)]


def find_lyric_path(lyrics_dir: Path, song_id: str, song_name: str) -> Optional[Path]:
    for stem in (song_id, song_name):
        for suffix in (".csv", ".lrc"):
            path = lyrics_dir / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def parse_lyrics(path: Path, audio_duration_ms: int) -> List[LyricSegment]:
    if path.suffix.lower() == ".csv":
        return parse_lyric_csv(path)
    if path.suffix.lower() == ".lrc":
        return parse_lrc(path, audio_duration_ms)
    raise ValueError(f"Unsupported lyric format: {path}")


def parse_lyric_csv(path: Path) -> List[LyricSegment]:
    rows: List[LyricSegment] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"start_ms", "end_ms", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for index, row in enumerate(reader, start=1):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            line_id = (row.get("line_id") or f"line_{index:03d}").strip()
            rows.append(
                LyricSegment(
                    line_id=line_id,
                    start_ms=int(float(row["start_ms"])),
                    end_ms=int(float(row["end_ms"])),
                    text=text,
                )
            )
    return rows


def parse_lrc(path: Path, audio_duration_ms: int, default_last_line_ms: int = 5000) -> List[LyricSegment]:
    timestamp_re = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
    entries: List[tuple[int, str]] = []

    for raw_line in read_text_with_fallback(path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matches = list(timestamp_re.finditer(line))
        if not matches:
            continue
        text = timestamp_re.sub("", line).strip()
        if not text:
            continue
        for match in matches:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            fraction = match.group(3) or "0"
            if len(fraction) == 1:
                millis = int(fraction) * 100
            elif len(fraction) == 2:
                millis = int(fraction) * 10
            else:
                millis = int(fraction[:3])
            entries.append(((minutes * 60 + seconds) * 1000 + millis, text))

    entries.sort(key=lambda item: item[0])
    segments: List[LyricSegment] = []
    for index, (start_ms, text) in enumerate(entries):
        if index + 1 < len(entries):
            end_ms = entries[index + 1][0]
        else:
            end_ms = min(audio_duration_ms, start_ms + default_last_line_ms)
        if end_ms <= start_ms:
            continue
        segments.append(
            LyricSegment(
                line_id=f"line_{index + 1:03d}",
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )
    return segments


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def copy_lyrics_for_reference(segments: Iterable[LyricSegment], path: Path) -> None:
    lines = [f"{item.start_ms},{item.end_ms},{item.text}" for item in segments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
