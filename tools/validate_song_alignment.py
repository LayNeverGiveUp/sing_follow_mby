"""Validate that a timestamped LRC belongs to the supplied song audio."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hum_song_mvp.src.alignment_validator import (
    corrected_lrc_text,
    transcript_from_dict,
    transcript_to_dict,
    validate_song_alignment,
    write_review_bundle,
)
from hum_song_mvp.src.audio_io import load_mono_audio
from hum_song_mvp.src.build_database import filter_lyric_lines
from hum_song_mvp.src.config import load_config
from hum_song_mvp.src.lrc_parser import parse_lrc
from hum_song_mvp.src.lyrics_asr import get_lyrics_asr
from hum_song_mvp.src.vocal_separator import separate_vocals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use singing ASR timestamps to verify that an LRC and song audio are the same version."
    )
    parser.add_argument("--audio", required=True, type=Path, help="Original song audio used for human review clips")
    parser.add_argument("--lrc", required=True, type=Path, help="Timestamped LRC to validate")
    parser.add_argument("--vocal", type=Path, help="Optional pre-separated vocal track used for ASR")
    parser.add_argument(
        "--separation-mode",
        choices=("none", "audio-separator", "demucs"),
        default="none",
        help="How to obtain vocals when --vocal is omitted",
    )
    parser.add_argument(
        "--separated-vocals-dir",
        type=Path,
        default=ROOT / "data" / "source_vocals" / "mao_buyi_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to data/alignment_reports/<song-name>",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--transcript-json", type=Path, help="Reuse a previously saved timed ASR result")
    parser.add_argument("--asr-max-wait-seconds", type=float, default=180.0)
    parser.add_argument("--review-clips", type=int, default=8)
    parser.add_argument("--no-corrected-lrc", action="store_true")
    args = parser.parse_args()

    for path, label in ((args.audio, "audio"), (args.lrc, "LRC")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    config = load_config(args.config)
    sample_rate = int(config["audio"]["sample_rate"])
    output_dir = args.output_dir or ROOT / "data" / "alignment_reports" / args.audio.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vocal is not None:
        if not args.vocal.is_file():
            raise FileNotFoundError(f"Vocal file does not exist: {args.vocal}")
        vocal_path = args.vocal
        input_mode = "pre_separated_vocal"
    else:
        vocal_path = separate_vocals(
            args.audio,
            args.separated_vocals_dir,
            args.separation_mode,
            str(config["separator"]["model"]),
        )
        input_mode = "mixed_audio_as_asr_input" if args.separation_mode == "none" else args.separation_mode

    vocal_samples = load_mono_audio(vocal_path, sample_rate)
    duration = vocal_samples.size / float(sample_rate)
    lrc_lines = filter_lyric_lines(parse_lrc(args.lrc, duration), args.audio.stem, duration)
    if not lrc_lines:
        raise ValueError("LRC contains no usable lyric lines after filtering metadata and credits")

    asr_started = perf_counter()
    if args.transcript_json:
        transcript = transcript_from_dict(json.loads(args.transcript_json.read_text(encoding="utf-8")))
        asr_mode = "saved_transcript"
    else:
        asr_config = deepcopy(config)
        asr_config["lyrics_asr"]["max_wait_seconds"] = args.asr_max_wait_seconds
        asr_config["lyrics_asr"]["request_timeout_seconds"] = max(
            float(asr_config["lyrics_asr"].get("request_timeout_seconds", 20.0)), 60.0
        )
        asr = get_lyrics_asr(asr_config)
        if not asr.enabled:
            raise RuntimeError(f"Lyrics ASR is not available: {asr.status}")
        transcript = asr.transcribe(vocal_samples, sample_rate)
        if transcript is None:
            raise RuntimeError("Lyrics ASR returned no text")
        asr_mode = asr.status
    asr_seconds = perf_counter() - asr_started
    transcript_path = output_dir / "asr_transcript.json"
    transcript_path.write_text(
        json.dumps(transcript_to_dict(transcript), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    validation_started = perf_counter()
    report = validate_song_alignment(
        lrc_lines,
        transcript,
        duration,
        config.get("alignment_validation", {}),
    )
    report["song_id"] = args.audio.stem
    report["inputs"] = {
        "audio_path": str(args.audio.resolve()),
        "vocal_path": str(vocal_path.resolve()),
        "lrc_path": str(args.lrc.resolve()),
        "input_mode": input_mode,
        "asr_mode": asr_mode,
    }
    report["runtime_seconds"] = {
        "asr": round(asr_seconds, 3),
        "alignment": round(perf_counter() - validation_started, 3),
    }
    report_path = output_dir / "alignment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    corrected_path = None
    if report.get("can_auto_correct") and not args.no_corrected_lrc:
        corrected_path = output_dir / f"{args.audio.stem}.corrected.lrc"
        corrected_path.write_text(corrected_lrc_text(lrc_lines, report), encoding="utf-8")

    review_samples = load_mono_audio(args.audio, sample_rate)
    manifest = write_review_bundle(
        output_dir,
        review_samples,
        sample_rate,
        lrc_lines,
        report,
        args.review_clips,
    )
    summary = {
        "song_id": args.audio.stem,
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "metrics": report["metrics"],
        "time_mapping": report.get("time_mapping"),
        "can_auto_correct": report.get("can_auto_correct", False),
        "alignment_report": str(report_path),
        "corrected_lrc": str(corrected_path) if corrected_path else None,
        "review_page": str(output_dir / "review.html"),
        "review_clip_count": len(manifest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["verdict"] == "warning":
        raise SystemExit(1)
    if report["verdict"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
