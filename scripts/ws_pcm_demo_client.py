from __future__ import annotations

import asyncio
import json
import math
import struct
from time import perf_counter

import websockets


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def sine_pcm16(freq: float, sample_rate: int, duration_ms: int) -> bytes:
    frames = []
    total = int(sample_rate * duration_ms / 1000)
    for index in range(total):
        sample = int(12000 * math.sin(2.0 * math.pi * freq * index / sample_rate))
        frames.append(struct.pack("<h", sample))
    return b"".join(frames)


async def main() -> None:
    uri = "ws://127.0.0.1:8000/v1/realtime-match"
    sample_rate = 16000
    melody = [60, 62, 63, 67, 65, 63, 62, 60]

    async with websockets.connect(uri) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "catalog_id": "mao_buyi_v1",
                    "sample_rate": sample_rate,
                    "format": "pcm_s16le",
                }
            )
        )
        print(await websocket.recv())

        for midi in melody:
            pcm = sine_pcm16(midi_to_hz(midi), sample_rate=sample_rate, duration_ms=130)
            for offset in range(0, len(pcm), 2048):
                await websocket.send(pcm[offset : offset + 2048])
                await asyncio.sleep(0.005)

        started = perf_counter()
        await websocket.send(json.dumps({"type": "end"}))
        print(await websocket.recv())
        print(f"client end-to-result ms={int((perf_counter() - started) * 1000)}")


if __name__ == "__main__":
    asyncio.run(main())
