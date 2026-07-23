from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.hum_recognizer import hum_mvp_recognizer
from hum_song_mvp.src.dtw_matcher import warm_dtw

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
app = FastAPI(title="Hum Song Follow-up MVP", version="0.2.0")
app.mount("/static/queries", StaticFiles(directory=DATA_DIR / "queries"), name="queries")
app.mount("/demo", StaticFiles(directory=BASE_DIR / "app" / "web", html=True), name="demo")


@app.on_event("startup")
def warm_matcher_runtime() -> None:
    warm_dtw()
    logger.info("hum_mvp_dtw_warmed")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hum-song-followup-mvp"}


@app.get("/v1/hum-mvp/test-queries")
def hum_mvp_test_queries() -> dict[str, list[dict[str, Any]]]:
    """Return every playable static vocal clip used by one-click testing."""
    database_dir = hum_mvp_recognizer.database_dir
    minimum_duration = float(hum_mvp_recognizer.config["confidence"]["min_voiced_seconds"])
    items: list[dict[str, Any]] = []
    for metadata_path in sorted(database_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        song_id = metadata["song_id"]
        lines = metadata["lrc_lines"]
        song_dir = DATA_DIR / "queries" / "mao_buyi_v1" / song_id
        for line in lines[:-1]:
            index = int(line["index"])
            if float(line["end_time"]) - float(line["start_time"]) < minimum_duration:
                continue
            query_file = song_dir / f"line_{index:03d}.wav"
            next_line = lines[index + 1]
            next_file = song_dir / f"line_{index + 1:03d}.wav"
            if not (query_file.exists() and next_file.exists()):
                continue
            encoded_song_id = quote(song_id)
            query_version = query_file.stat().st_mtime_ns
            next_version = next_file.stat().st_mtime_ns
            items.append(
                {
                    "song_id": song_id,
                    "current_lyric_index": index,
                    "current_lyric_text": line["text"],
                    "next_lyric_index": next_line["index"],
                    "next_lyric_text": next_line["text"],
                    "query_audio_url": f"/static/queries/mao_buyi_v1/{encoded_song_id}/line_{index:03d}.wav?v={query_version}",
                    "next_audio_url": f"/static/queries/mao_buyi_v1/{encoded_song_id}/line_{index + 1:03d}.wav?v={next_version}",
                }
            )
    if not items:
        raise ValueError("No humming-MVP test clips are available. Run tools/build_mvp_test_queries.py first.")
    return {"items": items}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo/")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "app" / "web" / "favicon.svg", media_type="image/svg+xml")


@app.websocket("/v1/realtime-match")
async def realtime_match(websocket: WebSocket) -> None:
    """Receive a complete PCM16 phrase and run the non-streaming humming matcher."""
    await websocket.accept()
    session_started = perf_counter()
    sample_rate = 16000
    pcm_chunks: list[bytes] = []
    started = False
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                if not started:
                    await websocket.send_json({"type": "error", "message": "send a hum_song_mvp start message first"})
                    continue
                pcm_chunks.append(message["bytes"])
                continue
            if message.get("text") is None:
                await websocket.send_json({"type": "error", "message": "unsupported websocket message"})
                continue
            payload = _parse_json_message(message["text"])
            msg_type = payload.get("type")
            if msg_type == "start":
                if payload.get("matcher_mode") != "hum_song_mvp":
                    await websocket.send_json({"type": "error", "message": "only matcher_mode=hum_song_mvp is supported"})
                    continue
                if payload.get("format", "pcm_s16le") != "pcm_s16le":
                    await websocket.send_json({"type": "error", "message": "only pcm_s16le is supported"})
                    continue
                sample_rate = int(payload.get("sample_rate", 16000))
                pcm_chunks = []
                started = True
                await websocket.send_json({"type": "ack", "sample_rate": sample_rate, "feature_mode": "hum_song_mvp", "asr_mode": "not_used"})
                continue
            if msg_type != "end" or not started:
                await websocket.send_json({"type": "error", "message": "send start, PCM16 binary chunks, then end"})
                continue
            end_received = perf_counter()
            logger.info("hum_mvp_request sample_rate=%s chunks=%s pcm_bytes=%s", sample_rate, len(pcm_chunks), sum(map(len, pcm_chunks)))
            result = hum_mvp_recognizer.recognize_pcm16(pcm_chunks, sample_rate)
            sent_at = perf_counter()
            result.update(
                {
                    "type": "result",
                    "latency_ms": {
                        "upload_duration": int((end_received - session_started) * 1000),
                        "end_to_result": int((sent_at - end_received) * 1000),
                        "total_session": int((sent_at - session_started) * 1000),
                    },
                }
            )
            await websocket.send_json(result)
            await websocket.close()
            return
    except WebSocketDisconnect:
        return
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _parse_json_message(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("websocket JSON messages must be objects")
    return payload
