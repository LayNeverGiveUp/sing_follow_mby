"""Evaluate manually labeled external vocal segments against the humming MVP."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hum_song_mvp.src.config import load_config
from hum_song_mvp.src.recognize import recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate consecutively labeled external vocal segments.")
    parser.add_argument("--segments-dir", required=True, type=Path)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--first-line-index", required=True, type=int)
    parser.add_argument("--database-dir", type=Path, default=ROOT / "hum_song_mvp" / "data" / "database")
    args = parser.parse_args()

    metadata = json.loads((args.database_dir / f"{args.song_id}.json").read_text(encoding="utf-8"))
    lines = metadata["lrc_lines"]
    paths = sorted(args.segments_dir.glob("segment_*.wav"))
    if not paths:
        raise FileNotFoundError(f"No segment_*.wav files found in {args.segments_dir}")
    config = load_config()
    rows = []
    for offset, path in enumerate(paths):
        expected_index = args.first_line_index + offset
        if expected_index >= len(lines):
            break
        started = perf_counter()
        result = recognize(path, args.database_dir, config)
        latency_ms = round((perf_counter() - started) * 1000, 1)
        expected = lines[expected_index]
        actual_index = result.get("current_lyric_index")
        correct = bool(result.get("accepted") and result.get("song_id") == args.song_id and actual_index == expected_index)
        row = {
            "segment": path.name,
            "expected_song_id": args.song_id,
            "expected_current_index": expected_index,
            "expected_current_text": expected["text"],
            "accepted": bool(result.get("accepted")),
            "actual_song_id": result.get("song_id"),
            "actual_current_index": actual_index,
            "actual_current_text": result.get("current_lyric_text"),
            "correct": correct,
            "score": result.get("score"),
            "top2_margin": result.get("top2_margin"),
            "reason": result.get("reason"),
            "latency_ms": latency_ms,
            "diagnostics": result.get("diagnostics"),
        }
        rows.append(row)
        print(f"{path.name}: expected={expected_index} actual={actual_index} accepted={row['accepted']} correct={correct} score={row['score']}")
    summary = {
        "total": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 4),
        "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1),
    }
    (args.segments_dir / "evaluation.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.segments_dir / "evaluation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "diagnostics"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "diagnostics"} for row in rows])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
