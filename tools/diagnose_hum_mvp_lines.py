"""Batch-diagnose per-line humming recognition against generated vocal clips."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hum_song_mvp.src.config import load_config
from hum_song_mvp.src.recognize import recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose song/current-line/next-line accuracy for each vocal test clip.")
    parser.add_argument("--database-dir", type=Path, default=ROOT / "hum_song_mvp" / "data" / "database")
    parser.add_argument("--queries-dir", type=Path, default=ROOT / "data" / "queries" / "mao_buyi_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "evaluation" / "hum_mvp_line_diagnosis")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--song-id", help="Optional song ID/name to diagnose")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many sorted query clips (for resumable batches)")
    parser.add_argument("--limit", type=int, help="Process at most this many clips (for resumable batches)")
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = []
    for metadata_path in sorted(args.database_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        song_id = metadata["song_id"]
        if args.song_id and song_id != args.song_id:
            continue
        lines = metadata["lrc_lines"]
        for line in lines[:-1]:
            expected_index = int(line["index"])
            query_path = args.queries_dir / song_id / f"line_{expected_index:03d}.wav"
            if not query_path.exists() or query_path.stat().st_size == 0:
                continue
            jobs.append((metadata, line, query_path))

    jobs = jobs[max(args.offset, 0):]
    if args.limit is not None:
        jobs = jobs[: max(args.limit, 0)]
    rows = []
    for metadata, line, query_path in jobs:
        song_id = metadata["song_id"]
        lines = metadata["lrc_lines"]
        expected_index = int(line["index"])
        query_duration = round(float(sf.info(query_path).duration), 3)
        started = perf_counter()
        try:
            result = recognize(query_path, args.database_dir, config)
            error = None
        except Exception as exc:  # Keep the whole diagnostic run alive.
            result = {}
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = round((perf_counter() - started) * 1000, 1)
        expected_next = lines[expected_index + 1]
        actual_song = result.get("song_id")
        actual_current = result.get("current_lyric_index")
        actual_next = result.get("next_lyric_index")
        accepted = bool(result.get("accepted"))
        song_correct = accepted and actual_song == song_id
        current_correct = song_correct and actual_current == expected_index
        next_correct = current_correct and actual_next == expected_index + 1
        matched_start = result.get("matched_start_time")
        matched_end = result.get("matched_end_time")
        expected_start = round(float(line["start_time"]), 3)
        expected_end = round(float(line["end_time"]), 3)
        if error:
            outcome = "error"
        elif not accepted:
            outcome = "rejected"
        elif not song_correct:
            outcome = "wrong_song"
        elif not current_correct:
            outcome = "wrong_line"
        else:
            outcome = "correct"
        rows.append({
            "query_file": str(query_path.relative_to(ROOT)),
            "expected_song_id": song_id,
            "expected_current_index": expected_index,
            "expected_current_text": line["text"],
            "expected_start_time": expected_start,
            "expected_end_time": expected_end,
            "expected_next_index": expected_index + 1,
            "expected_next_text": expected_next["text"],
            "query_duration_seconds": query_duration,
            "accepted": accepted,
            "actual_song_id": actual_song,
            "actual_current_index": actual_current,
            "actual_current_text": result.get("current_lyric_text"),
            "actual_next_index": actual_next,
            "actual_next_text": result.get("next_lyric_text"),
            "matched_start_time": matched_start,
            "matched_end_time": matched_end,
            "matched_start_offset_seconds": _offset(matched_start, expected_start),
            "matched_end_offset_seconds": _offset(matched_end, expected_end),
            "song_correct": song_correct,
            "current_correct": current_correct,
            "next_correct": next_correct,
            "score": result.get("score"),
            "top2_margin": result.get("top2_margin"),
            "reason": result.get("reason"),
            "outcome": outcome,
            "latency_ms": latency_ms,
            "error": error,
        })
        print(f"{len(rows):02d} {song_id} line {expected_index:02d}: accepted={accepted} current={actual_current} next={actual_next} score={result.get('score')}")

    if not rows:
        raise SystemExit("No query clips found. Run tools/build_mvp_test_queries.py first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.offset:03d}" if args.limit is not None or args.offset else ""
    csv_path = args.output_dir / f"line_results{suffix}.csv"
    json_path = args.output_dir / f"line_results{suffix}.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}\nJSON: {json_path}")


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    summary = {
        "total": total,
        "accepted": sum(row["accepted"] for row in rows),
        "song_correct": sum(row["song_correct"] for row in rows),
        "current_line_correct": sum(row["current_correct"] for row in rows),
        "next_line_correct": sum(row["next_correct"] for row in rows),
        "accept_rate": round(sum(row["accepted"] for row in rows) / total, 4),
        "song_accuracy": round(sum(row["song_correct"] for row in rows) / total, 4),
        "current_line_accuracy": round(sum(row["current_correct"] for row in rows) / total, 4),
        "next_line_accuracy": round(sum(row["next_correct"] for row in rows) / total, 4),
        "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / total, 1),
    }
    for song_id in sorted({row["expected_song_id"] for row in rows}):
        song_rows = [row for row in rows if row["expected_song_id"] == song_id]
        summary.setdefault("by_song", {})[song_id] = _group_summary(song_rows)
    summary["outcomes"] = {key: sum(row["outcome"] == key for row in rows) for key in ("correct", "wrong_line", "wrong_song", "rejected", "error")}
    return summary


def _group_summary(rows: list[dict]) -> dict:
    total = len(rows)
    return {
        "total": total,
        "correct": sum(row["outcome"] == "correct" for row in rows),
        "wrong_line": sum(row["outcome"] == "wrong_line" for row in rows),
        "wrong_song": sum(row["outcome"] == "wrong_song" for row in rows),
        "rejected": sum(row["outcome"] == "rejected" for row in rows),
        "error": sum(row["outcome"] == "error" for row in rows),
        "next_line_accuracy": round(sum(row["next_correct"] for row in rows) / total, 4),
        "mean_matched_end_offset_seconds": _mean([row["matched_end_offset_seconds"] for row in rows]),
    }


def _offset(actual: float | None, expected: float) -> float | None:
    return round(float(actual) - expected, 3) if actual is not None else None


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 3) if present else None


if __name__ == "__main__":
    main()
