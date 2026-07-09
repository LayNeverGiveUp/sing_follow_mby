from __future__ import annotations

import asyncio
import json
from time import perf_counter

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8000/v1/realtime-match"
    async with websockets.connect(uri) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "catalog_id": "kids_songs_v1",
                    "sample_rate": 16000,
                    "format": "pcm_s16le",
                }
            )
        )
        print(await websocket.recv())

        for chunk in ([60, 60, 67], [67, 69], [69, 67]):
            await websocket.send(json.dumps({"type": "demo_features", "values": chunk}))
            await asyncio.sleep(0.05)

        started = perf_counter()
        await websocket.send(json.dumps({"type": "end"}))
        print(await websocket.recv())
        print(f"client end-to-result ms={int((perf_counter() - started) * 1000)}")


if __name__ == "__main__":
    asyncio.run(main())
