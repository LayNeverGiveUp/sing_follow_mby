from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .audio_io import SUPPORTED_AUDIO_EXTENSIONS, load_mono_audio
from .config import load_config
from .lrc_parser import parse_lrc
from .pitch_extractor import extract_features
from .vocal_separator import separate_vocals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-singer humming recognition database.")
    parser.add_argument("--songs-dir", required=True, type=Path)
    parser.add_argument("--lyrics-dir", type=Path, help="Optional directory containing <song_id>.lrc files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--separated-vocals-dir", type=Path, default=Path("data/separated_vocals"))
    parser.add_argument("--separation-mode", choices=("none", "audio-separator", "demucs"), default="none")
    args = parser.parse_args()
    config = load_config(args.config)
    build_database(
        args.songs_dir, args.output_dir, args.separated_vocals_dir, args.separation_mode, config, args.lyrics_dir
    )


def build_database(
    songs_dir: Path,
    output_dir: Path,
    separated_vocals_dir: Path,
    separation_mode: str,
    config: dict,
    lyrics_dir: Path | None = None,
) -> list[Path]:
    if not songs_dir.is_dir():
        raise FileNotFoundError(f"Songs directory does not exist: {songs_dir}")
    audio_files = sorted(path for path in songs_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not audio_files:
        raise ValueError(f"No supported audio files found in {songs_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    failures: list[str] = []
    for audio_path in audio_files:
        try:
            created.append(
                _build_song(audio_path, songs_dir, output_dir, separated_vocals_dir, separation_mode, config, lyrics_dir)
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            failures.append(f"{audio_path.name}: {exc}")
    if failures:
        raise RuntimeError("Database build failed:\n" + "\n".join(failures))
    print(f"Built {len(created)} song database entries in {output_dir}")
    return created


def _build_song(
    audio_path: Path,
    songs_dir: Path,
    output_dir: Path,
    separated_vocals_dir: Path,
    separation_mode: str,
    config: dict,
    lyrics_dir: Path | None,
) -> Path:
    song_id = audio_path.stem
    lrc_path = (lyrics_dir or songs_dir) / f"{song_id}.lrc"
    if not lrc_path.exists():
        raise FileNotFoundError(f"Missing matching LRC file: {lrc_path}")
    vocal_path = separate_vocals(audio_path, separated_vocals_dir, separation_mode, str(config["separator"]["model"]))
    samples = load_mono_audio(vocal_path, int(config["audio"]["sample_rate"]))
    features = extract_features(samples, config)
    if not np.any(features.voiced):
        raise ValueError("F0 extraction produced no voiced frames; provide a clean vocal recording or check F0 range")
    duration = len(samples) / float(config["audio"]["sample_rate"])
    lrc_lines = filter_lyric_lines(parse_lrc(lrc_path, duration), song_id, duration)
    lrc_lines = refine_line_end_times(lrc_lines, features, config, duration)
    if not lrc_lines:
        raise ValueError(f"LRC contains no usable lyric lines after metadata filtering: {lrc_path}")
    features_name = f"{song_id}_features.npz"
    np.savez_compressed(output_dir / features_name, **features.to_npz_dict())
    metadata = {
        "song_id": song_id,
        "audio_path": str(audio_path.resolve()),
        "vocal_path": str(vocal_path.resolve()),
        "duration": round(duration, 3),
        "feature_hop_seconds": float(config["pitch"]["hop_seconds"]),
        "features_file": features_name,
        "lrc_lines": lrc_lines,
    }
    metadata_path = output_dir / f"{song_id}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def filter_lyric_lines(lines: list[dict], song_id: str, duration: float) -> list[dict]:
    """Remove common title/credit entries while retaining their timing context."""
    credit_prefixes = (
        "词", "作词", "曲", "作曲", "编曲", "制作人", "音乐总监", "音乐统筹", "乐队", "键盘", "鼓手",
        "吉他", "贝斯", "和声", "打击乐", "电脑工程", "混音", "母带", "录音", "lyricist", "composer",
        "arranger", "producer", "鼓", "琵琶", "弦乐", "人声编辑", "配唱制作人", "录音师", "音频编辑",
        "乐器", "管乐", "vocal editing", "drums", "pipa", "strings",
    )
    kept = []
    for line in lines:
        text = str(line["text"]).strip()
        compact = re.sub(r"\s+", "", text).lower()
        if not compact:
            continue
        # A timestamp after the actual vocal file is a version mismatch, not a
        # playable lyric.  Keeping it creates empty test clips and corrupts
        # the following-line mapping.
        if float(line["start_time"]) >= duration:
            continue
        if compact.startswith(song_id.lower()) and ("-" in compact or "－" in compact):
            continue
        if any(compact.startswith(prefix) for prefix in credit_prefixes):
            continue
        kept.append(dict(line))
    for index, line in enumerate(kept):
        line["index"] = index
        next_start = kept[index + 1]["start_time"] if index + 1 < len(kept) else duration
        line["end_time"] = round(min(float(next_start), duration), 3)
    return kept


def refine_line_end_times(lines: list[dict], features, config: dict, duration: float) -> list[dict]:
    """Replace nominal LRC ends with the final sustained vocal run in each line.

    LRC timestamps describe starts reliably, but commonly leave instrumental
    gaps at the end of a line.  The nominal end is retained for auditing while
    ``end_time`` becomes the playable vocal end used for test clips.
    """
    if not lines:
        return []
    settings = config["lyric_segmentation"]
    hop = float(config["pitch"]["hop_seconds"])
    min_frames = max(1, int(np.ceil(float(settings["min_voiced_run_seconds"]) / hop)))
    max_gap_frames = max(1, int(np.floor(float(settings["max_intra_phrase_silence_seconds"]) / hop)))
    padding = float(settings["vocal_end_padding_seconds"])
    minimum_nominal_duration = float(settings["min_nominal_line_duration_to_trim_seconds"])
    minimum_trim = float(settings["min_trailing_silence_to_trim_seconds"])
    times = np.asarray(features.time, dtype=np.float32)
    voiced = np.asarray(features.voiced, dtype=bool)
    refined = []
    for source in lines:
        line = dict(source)
        start = float(line["start_time"])
        nominal_end = min(float(line["end_time"]), duration)
        line["nominal_end_time"] = round(nominal_end, 3)
        start_frame = int(np.searchsorted(times, start, side="left"))
        end_frame = int(np.searchsorted(times, nominal_end, side="left"))
        phrase_end = _first_phrase_voiced_end(voiced[start_frame:end_frame], min_frames, max_gap_frames)
        if phrase_end is None:
            line["end_time"] = round(nominal_end, 3)
            line["vocal_end_source"] = "nominal_lrc_fallback"
        else:
            frame = start_frame + phrase_end - 1
            vocal_end = min(nominal_end, float(times[frame]) + hop + padding)
            if nominal_end - start >= minimum_nominal_duration and nominal_end - vocal_end >= minimum_trim:
                line["end_time"] = round(max(start, vocal_end), 3)
                line["vocal_end_source"] = "f0_long_silence_trim"
            else:
                line["end_time"] = round(nominal_end, 3)
                line["vocal_end_source"] = "nominal_lrc_normal_duration"
        refined.append(line)
    return refined


def _first_phrase_voiced_end(values: np.ndarray, min_frames: int, max_gap_frames: int) -> int | None:
    """Return the end of the first voiced phrase, ignoring later bleed/artifacts."""
    run_start = None
    phrase_end = None
    for index, is_voiced in enumerate(values):
        if is_voiced and run_start is None:
            run_start = index
        if not is_voiced and run_start is not None:
            if index - run_start >= min_frames:
                if phrase_end is not None and run_start - phrase_end > max_gap_frames:
                    return phrase_end
                phrase_end = index
            run_start = None
    if run_start is not None and len(values) - run_start >= min_frames:
        if phrase_end is not None and run_start - phrase_end > max_gap_frames:
            return phrase_end
        phrase_end = len(values)
    return phrase_end


if __name__ == "__main__":
    main()
