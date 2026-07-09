from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.audio.features import StreamingPitchExtractor
from app.matching.catalog import CatalogStore
from app.matching.engine import RealtimeMatcher

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Song Followup API", version="0.1.0")
app.mount("/static/replies", StaticFiles(directory=DATA_DIR / "replies"), name="replies")
app.mount("/static/prompts", StaticFiles(directory=DATA_DIR / "prompts"), name="prompts")
app.mount("/demo", StaticFiles(directory=BASE_DIR / "app" / "web", html=True), name="demo")

catalog_store = CatalogStore(DATA_DIR / "catalog")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "song-followup-api"}


@app.get("/v1/catalog/{catalog_id}")
def catalog(catalog_id: str) -> Dict[str, Any]:
    loaded = catalog_store.load(catalog_id)
    return {
        "catalog_id": loaded.catalog_id,
        "songs": [
            {
                "song_id": segment.song_id,
                "song_name": segment.song_name,
                "line_id": segment.line_id,
                "text": segment.text,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "features": segment.features,
                "prompt_audio_url": f"/static/prompts/{catalog_id}/{segment.prompt_audio}" if segment.prompt_audio else None,
            }
            for segment in loaded.segments
        ],
    }


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo/")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "app" / "web" / "favicon.svg", media_type="image/svg+xml")


@app.websocket("/v1/realtime-match")
async def realtime_match(websocket: WebSocket) -> None:
    await websocket.accept()
    session_started = perf_counter()
    stream_started = None
    matcher = None
    sample_rate = 16000
    pitch_extractor = None

    try:
        while True:
            message = await websocket.receive()
            if message.get("text") is not None:
                payload = _parse_json_message(message["text"])
                msg_type = payload.get("type")

                if msg_type == "start":
                    catalog_id = str(payload.get("catalog_id", "kids_songs_v1"))
                    sample_rate = int(payload.get("sample_rate", 16000))
                    if payload.get("format", "pcm_s16le") != "pcm_s16le":
                        await websocket.send_json({"type": "error", "message": "only pcm_s16le is supported"})
                        continue
                    matcher = RealtimeMatcher(catalog_store.load(catalog_id))
                    pitch_extractor = StreamingPitchExtractor(sample_rate=sample_rate)
                    stream_started = perf_counter()
                    await websocket.send_json(
                        {
                            "type": "ack",
                            "catalog_id": catalog_id,
                            "sample_rate": sample_rate,
                            "feature_mode": "pitch_midi",
                        }
                    )
                    continue

                if matcher is None:
                    await websocket.send_json({"type": "error", "message": "send start message first"})
                    continue

                if msg_type == "demo_features":
                    values = payload.get("values", [])
                    if not isinstance(values, list):
                        await websocket.send_json({"type": "error", "message": "values must be a list"})
                        continue
                    matcher.append_features([float(value) for value in values])
                    continue

                if msg_type == "end":
                    end_received = perf_counter()
                    result = matcher.finalize()
                    await websocket.send_json(
                        _result_payload(
                            result=result,
                            stream_started=stream_started or session_started,
                            end_received=end_received,
                            session_started=session_started,
                            host=_http_origin(websocket),
                            feature_count=matcher.feature_count,
                        )
                    )
                    await websocket.close()
                    return

                await websocket.send_json({"type": "error", "message": f"unknown message type: {msg_type}"})

            elif message.get("bytes") is not None:
                if matcher is None:
                    await websocket.send_json({"type": "error", "message": "send start message first"})
                    continue
                if stream_started is None:
                    stream_started = perf_counter()
                if pitch_extractor is None:
                    pitch_extractor = StreamingPitchExtractor(sample_rate=sample_rate)
                features = pitch_extractor.append_pcm16(message["bytes"])
                matcher.append_features(features)
            else:
                await websocket.send_json({"type": "error", "message": "unsupported websocket message"})

    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _parse_json_message(text: str) -> Dict[str, Any]:
    import json

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("websocket JSON messages must be objects")
    return payload


def _http_origin(websocket: WebSocket) -> str:
    scheme = "https" if websocket.url.scheme == "wss" else "http"
    host = websocket.url.hostname or "127.0.0.1"
    port = f":{websocket.url.port}" if websocket.url.port else ""
    return f"{scheme}://{host}{port}"


def _result_payload(
    result,
    stream_started: float,
    end_received: float,
    session_started: float,
    host: str,
    feature_count: int,
) -> Dict[str, Any]:
    sent_at = perf_counter()
    reply_url = None
    if result.reply_audio:
        reply_url = f"{host}/static/replies/{result.reply_audio}"
    prompt_url = None
    if result.prompt_audio:
        prompt_url = f"{host}/static/prompts/mao_buyi_v1/{result.prompt_audio}"

    return {
        "type": "result",
        "matched": result.matched,
        "song_id": result.song_id,
        "song_name": result.song_name,
        "confidence": result.confidence,
        "matched_line": result.matched_line,
        "line_id": result.line_id,
        "handoff_type": result.handoff_type,
        "reply_audio_url": reply_url,
        "prompt_audio_url": prompt_url,
        "lyrics_file": result.lyrics_file,
        "feature_count": feature_count,
        "latency_ms": {
            "stream_duration": int((end_received - stream_started) * 1000),
            "end_to_result": int((sent_at - end_received) * 1000),
            "matcher_processing": result.processing_ms,
            "total_session": int((sent_at - session_started) * 1000),
        },
    }
