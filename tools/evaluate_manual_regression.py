"""Evaluate the fixed manual recording set with fixture, live, or disabled ASR."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = ROOT / "hum_song_mvp"
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from src.audio_io import load_mono_audio  # noqa: E402
from src.config import load_config  # noqa: E402
from src.lyrics_asr import AsrTranscript, DisabledLyricsAsr, get_lyrics_asr  # noqa: E402
from src.recognize import recognize_samples  # noqa: E402


class FixtureLyricsAsr:
    enabled = True
    status = "fixture"

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        return AsrTranscript(self.text, "fixture", 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ten-line manual regression dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "evaluation" / "manual_test_sets" / "manual_regression_v1",
    )
    parser.add_argument("--database-dir", type=Path, default=MVP_ROOT / "data" / "database")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asr-mode", choices=("fixture", "live", "disabled"), default="fixture")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    config = load_config(args.config)
    metadata_by_song = _load_metadata(args.database_dir)
    record_paths = sorted(
        path
        for path in args.dataset_dir.glob("*.json")
        if not path.name.startswith("regression_report_")
    )
    if not record_paths:
        raise SystemExit(f"No regression records found in {args.dataset_dir}")

    rows = []
    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_song = str(record["expected"]["song_id"])
        expected_index = int(record["expected"]["lyric_index"])
        lines = {int(line["index"]): line for line in metadata_by_song[expected_song]["lrc_lines"]}
        expected_current = lines[expected_index]["text"]
        expected_next = lines.get(expected_index + 1, {}).get("text")
        asr = _lyrics_asr(args.asr_mode, config, expected_current)
        samples = load_mono_audio(args.dataset_dir / record["audio_file"], int(config["audio"]["sample_rate"]))
        result = recognize_samples(samples, args.database_dir, config, asr)
        output_correct = bool(
            result.get("accepted")
            and result.get("song_id") == expected_song
            and result.get("current_lyric_text") == expected_current
            and result.get("next_lyric_text") == expected_next
        )
        row = {
            "case_id": record["case_id"],
            "expected_song_id": expected_song,
            "expected_lyric_index": expected_index,
            "accepted": bool(result.get("accepted")),
            "actual_song_id": result.get("song_id"),
            "actual_lyric_index": result.get("current_lyric_index"),
            "output_correct": output_correct,
            "reason": result.get("reason"),
            "hybrid_route": result.get("diagnostics", {}).get("hybrid_route"),
            "lyrics_asr": result.get("diagnostics", {}).get("lyrics_asr"),
            "result": result,
        }
        rows.append(row)
        verdict = "PASS" if output_correct else "FAIL"
        print(
            f"{verdict} {record['case_id']}: "
            f"expected={expected_song}#{expected_index} "
            f"actual={result.get('song_id')}#{result.get('current_lyric_index')} "
            f"reason={result.get('reason')}"
        )

    summary = {
        "schema_version": 1,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "asr_mode": args.asr_mode,
        "total": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "correct": sum(row["output_correct"] for row in rows),
        "incorrect": sum(not row["output_correct"] for row in rows),
        "rows": rows,
    }
    report_path = args.report or args.dataset_dir / f"regression_report_{args.asr_mode}.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(
        f"Summary: correct={summary['correct']}/{summary['total']} "
        f"accepted={summary['accepted']}/{summary['total']} report={report_path}"
    )
    raise SystemExit(0 if summary["correct"] == summary["total"] else 1)


def _load_metadata(database_dir: Path) -> dict[str, dict]:
    metadata_by_song = {}
    for path in database_dir.glob("*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata_by_song[str(metadata["song_id"])] = metadata
    return metadata_by_song


def _lyrics_asr(mode: str, config: dict, expected_text: str):
    if mode == "fixture":
        return FixtureLyricsAsr(expected_text)
    if mode == "live":
        return get_lyrics_asr(config)
    return DisabledLyricsAsr()


if __name__ == "__main__":
    main()
