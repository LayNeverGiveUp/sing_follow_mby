from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter

import websockets

DEMO_FEATURES = {
    "消愁": [60, 62, 63, 67, 65, 63, 62, 60],
    "像我这样的人": [64, 64, 65, 67, 65, 64, 62, 60],
    "盛夏": [55, 60, 62, 64, 67, 64, 62, 60],
    "不染": [67, 69, 70, 72, 70, 69, 67, 65],
    "平凡的一天": [60, 60, 62, 64, 62, 60, 57, 60],
    "一荤一素": [62, 65, 64, 62, 60, 62, 59, 57],
    "借": [57, 60, 62, 60, 57, 55, 57, 60],
    "呓语": [65, 64, 62, 60, 62, 64, 65, 67],
    "无问": [60, 63, 65, 67, 68, 67, 65, 63],
    "牧马城市": [59, 62, 64, 66, 64, 62, 61, 59]
}


async def main() -> None:
    uri = os.getenv("SONG_FOLLOWUP_WS", "ws://127.0.0.1:8000/v1/realtime-match")
    song_name = os.getenv("MAO_DEMO_SONG", "消愁")
    features = DEMO_FEATURES.get(song_name)
    if features is None:
        raise SystemExit(f"Unknown MAO_DEMO_SONG={song_name!r}. Available: {', '.join(DEMO_FEATURES)}")

    async with websockets.connect(uri) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "catalog_id": "mao_buyi_v1",
                    "sample_rate": 16000,
                    "format": "pcm_s16le",
                },
                ensure_ascii=False,
            )
        )
        print(await websocket.recv())

        for offset in range(0, len(features), 3):
            await websocket.send(
                json.dumps(
                    {"type": "demo_features", "values": features[offset : offset + 3]},
                    ensure_ascii=False,
                )
            )
            await asyncio.sleep(0.05)

        started = perf_counter()
        await websocket.send(json.dumps({"type": "end"}))
        print(await websocket.recv())
        print(f"client end-to-result ms={int((perf_counter() - started) * 1000)}")


if __name__ == "__main__":
    asyncio.run(main())
