"""Replay the fixed recordings through the same WebSocket path as the browser."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean, median
import wave

from websockets.asyncio.client import connect


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "manual_test_sets" / "manual_regression_v1"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure warm-service end-to-result RT.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime-match")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = []
    record_paths = sorted(
        path
        for path in args.dataset_dir.glob("*.json")
        if not path.name.startswith("regression_report_")
    )
    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        sample_rate, pcm = read_pcm16(args.dataset_dir / record["audio_file"])
        async with connect(args.url, max_size=4 * 1024 * 1024) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "matcher_mode": "hum_song_mvp",
                        "format": "pcm_s16le",
                        "sample_rate": sample_rate,
                        "input_source": "service_rt_regression",
                    }
                )
            )
            ack = json.loads(await websocket.recv())
            if ack.get("type") != "ack":
                raise RuntimeError(f"Unexpected service response: {ack}")
            for offset in range(0, len(pcm), 8192):
                await websocket.send(pcm[offset : offset + 8192])
            await websocket.send(json.dumps({"type": "end"}))
            result = json.loads(await websocket.recv())
        latency = result.get("latency_ms", {})
        row = {
            "case_id": record["case_id"],
            "accepted": bool(result.get("accepted")),
            "song_id": result.get("song_id"),
            "current_lyric_index": result.get("current_lyric_index"),
            "end_to_result_ms": float(latency.get("end_to_result", 0)),
            "service_total_ms": result.get("diagnostics", {}).get("stage_ms", {}).get("service_total"),
            "result": result,
        }
        rows.append(row)
        print(
            f"{row['case_id']}: end_to_result={row['end_to_result_ms']:.0f}ms "
            f"accepted={row['accepted']} song={row['song_id']}#{row['current_lyric_index']}"
        )

    values = sorted(row["end_to_result_ms"] for row in rows)
    summary = {
        "total": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "average_ms": round(mean(values), 1),
        "median_ms": round(median(values), 1),
        "p95_ms": round(values[max(0, int((len(values) * 0.95 + 0.999999)) - 1)], 1),
        "rows": rows,
    }
    print(
        f"Summary: average={summary['average_ms']:.1f}ms "
        f"median={summary['median_ms']:.1f}ms p95={summary['p95_ms']:.1f}ms"
    )
    if args.report:
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_pcm16(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        return source.getframerate(), source.readframes(source.getnframes())


if __name__ == "__main__":
    asyncio.run(main())
