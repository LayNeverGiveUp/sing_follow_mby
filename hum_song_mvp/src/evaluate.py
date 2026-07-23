from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .config import load_config
from .recognize import recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a labelled humming-query manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSON array of query annotations")
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Evaluation manifest must be a non-empty JSON array")
    config = load_config(args.config)
    rows = []
    for item in manifest:
        if not isinstance(item, dict) or "audio" not in item:
            raise ValueError("Each manifest item must contain an audio path")
        started = perf_counter()
        result = recognize(Path(item["audio"]), args.database_dir, config)
        result["latency_ms"] = round((perf_counter() - started) * 1000, 1)
        result["expected"] = item
        rows.append(result)
    summary = _summarize(rows)
    print(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2))


def _summarize(rows: list[dict]) -> dict:
    known = [row for row in rows if row["expected"].get("accepted", True)]
    unknown = [row for row in rows if not row["expected"].get("accepted", True)]
    song_correct = sum(row["accepted"] and row["song_id"] == row["expected"].get("song_id") for row in known)
    lyric_correct = sum(
        row["accepted"] and row["current_lyric_index"] == row["expected"].get("current_lyric_index")
        for row in known if "current_lyric_index" in row["expected"]
    )
    lyric_total = sum("current_lyric_index" in row["expected"] for row in known)
    next_correct = sum(
        row["accepted"] and row["next_lyric_index"] == row["expected"].get("next_lyric_index")
        for row in known if "next_lyric_index" in row["expected"]
    )
    next_total = sum("next_lyric_index" in row["expected"] for row in known)
    rejected_unknown = sum(not row["accepted"] for row in unknown)
    return {
        "query_count": len(rows),
        "song_top1_accuracy": _ratio(song_correct, len(known)),
        "lyric_line_accuracy": _ratio(lyric_correct, lyric_total),
        "next_line_accuracy": _ratio(next_correct, next_total),
        "unknown_rejection_accuracy": _ratio(rejected_unknown, len(unknown)),
        "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


if __name__ == "__main__":
    main()
