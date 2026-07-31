from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from app.debug_capture import DebugCapture, capture_pcm16, write_capture_result
from app.hum_recognizer import hum_mvp_recognizer
from app.regression_dataset import build_plan_payload, load_plan, resolve_case, save_case
from hum_song_mvp.src.audio_io import trim_outer_silence
from hum_song_mvp.src.dtw_matcher import warm_dtw
from hum_song_mvp.src.phrase_matcher import warm_phrase_dtw
from hum_song_mvp.src.recognize import warm_recognition_database

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEBUG_RECORDINGS_DIR = Path(os.getenv("HUM_SONG_DEBUG_RECORDINGS_DIR", DATA_DIR / "debug_recordings"))
QUERIES_DIR = DATA_DIR / "queries"
REGRESSION_DATASET_DIR = DATA_DIR / "evaluation" / "manual_test_sets"
REGRESSION_PLAN = load_plan(BASE_DIR / "assets" / "manual_regression_v1.json")
# Runtime-only media is intentionally not committed. Creating the mount point
# here keeps a fresh clone bootable even when optional one-click clips are absent.
QUERIES_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Hum Song Follow-up MVP", version="0.2.0")
app.mount("/static/queries", StaticFiles(directory=QUERIES_DIR), name="queries")
app.mount("/demo", StaticFiles(directory=BASE_DIR / "app" / "web", html=True), name="demo")


@app.on_event("startup")
def warm_matcher_runtime() -> None:
    sample_rate = int(hum_mvp_recognizer.config["audio"]["sample_rate"])
    warmup = np.zeros(sample_rate, dtype=np.float32)
    positions = np.arange(sample_rate // 2, dtype=np.float32)
    warmup[sample_rate // 4 : 3 * sample_rate // 4] = 0.1 * np.sin(
        2.0 * np.pi * 220.0 * positions / sample_rate
    )
    trim_outer_silence(warmup, float(hum_mvp_recognizer.config["audio"]["trim_top_db"]))
    warm_dtw()
    warm_phrase_dtw()
    warm_recognition_database(hum_mvp_recognizer.database_dir, hum_mvp_recognizer.config)
    logger.info("hum_mvp_matcher_warmed")


@app.get("/health")
def health() -> dict[str, Any]:
    song_count = len(list(hum_mvp_recognizer.database_dir.glob("*.json")))
    return {
        "status": "ok" if song_count else "degraded",
        "ready": song_count > 0,
        "service": "hum-song-followup-mvp",
        "lyrics_asr": hum_mvp_recognizer.lyrics_asr.status,
        "song_count": song_count,
    }


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
    return {"items": items}


@app.get("/v1/hum-mvp/regression-plan")
def hum_mvp_regression_plan() -> dict[str, Any]:
    """Return the fixed ten-line recording plan and its persisted progress."""
    return build_plan_payload(
        REGRESSION_PLAN,
        hum_mvp_recognizer.database_dir,
        QUERIES_DIR,
        REGRESSION_DATASET_DIR,
    )


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
    start_metadata: dict[str, Any] = {}
    debug_capture: DebugCapture | None = None
    regression_case: dict[str, Any] | None = None
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
                start_metadata = {
                    "input_source": str(payload.get("input_source", "unknown"))[:64],
                    "catalog_id": str(payload.get("catalog_id", ""))[:128],
                    "expected_song_id": payload.get("expected_song_id"),
                    "expected_lyric_index": payload.get("expected_lyric_index"),
                    "user_agent": websocket.headers.get("user-agent", "")[:512],
                }
                regression_case = None
                if start_metadata["input_source"] == "manual_regression":
                    plan_id = str(payload.get("test_plan_id", ""))
                    case_id = str(payload.get("test_case_id", ""))
                    regression_case = resolve_case(REGRESSION_PLAN, plan_id, case_id)
                    start_metadata.update(
                        {
                            "test_plan_id": REGRESSION_PLAN["plan_id"],
                            "test_case_id": regression_case["case_id"],
                            "expected_song_id": regression_case["song_id"],
                            "expected_lyric_index": int(regression_case["lyric_index"]),
                        }
                    )
                started = True
                await websocket.send_json(
                    {
                        "type": "ack",
                        "sample_rate": sample_rate,
                        "feature_mode": "hum_song_mvp",
                        "asr_mode": hum_mvp_recognizer.lyrics_asr.status,
                    }
                )
                continue
            if msg_type != "end" or not started:
                await websocket.send_json({"type": "error", "message": "send start, PCM16 binary chunks, then end"})
                continue
            end_received = perf_counter()
            try:
                debug_capture = capture_pcm16(DEBUG_RECORDINGS_DIR, pcm_chunks, sample_rate, start_metadata)
            except (OSError, ValueError) as exc:
                logger.exception("hum_mvp_debug_capture_failed error=%s", exc)
            logger.info(
                "hum_mvp_request case=%s source=%s sample_rate=%s chunks=%s pcm_bytes=%s",
                debug_capture.case_id if debug_capture else None,
                start_metadata.get("input_source"),
                sample_rate,
                len(pcm_chunks),
                sum(map(len, pcm_chunks)),
            )
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
            if debug_capture is not None and regression_case is not None:
                try:
                    audio_path, _ = save_case(
                        REGRESSION_DATASET_DIR,
                        REGRESSION_PLAN,
                        regression_case,
                        debug_capture,
                        result,
                    )
                    result.update(
                        {
                            "test_dataset_saved": True,
                            "test_plan_id": REGRESSION_PLAN["plan_id"],
                            "test_case_id": regression_case["case_id"],
                            "test_audio_file": str(audio_path.relative_to(BASE_DIR)),
                        }
                    )
                except OSError as exc:
                    logger.exception(
                        "hum_mvp_regression_save_failed case=%s error=%s",
                        regression_case["case_id"],
                        exc,
                    )
                    result["test_dataset_saved"] = False
            if debug_capture is not None:
                result["debug_case_id"] = debug_capture.case_id
                result["debug_saved"] = _save_debug_result(debug_capture, result)
            await websocket.send_json(result)
            await websocket.close()
            return
    except WebSocketDisconnect:
        return
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error_payload: dict[str, Any] = {"type": "error", "message": str(exc)}
        if debug_capture is not None and regression_case is not None:
            try:
                audio_path, _ = save_case(
                    REGRESSION_DATASET_DIR,
                    REGRESSION_PLAN,
                    regression_case,
                    debug_capture,
                    error_payload,
                )
                error_payload.update(
                    {
                        "test_dataset_saved": True,
                        "test_plan_id": REGRESSION_PLAN["plan_id"],
                        "test_case_id": regression_case["case_id"],
                        "test_audio_file": str(audio_path.relative_to(BASE_DIR)),
                    }
                )
            except OSError as save_exc:
                logger.exception(
                    "hum_mvp_regression_save_failed case=%s error=%s",
                    regression_case["case_id"],
                    save_exc,
                )
                error_payload["test_dataset_saved"] = False
        if debug_capture is not None:
            error_payload["debug_case_id"] = debug_capture.case_id
            error_payload["debug_saved"] = _save_debug_result(debug_capture, error_payload)
        await websocket.send_json(error_payload)
        await websocket.close()


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _parse_json_message(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("websocket JSON messages must be objects")
    return payload


def _save_debug_result(capture: DebugCapture, payload: dict[str, Any]) -> bool:
    try:
        write_capture_result(capture, payload)
    except OSError as exc:
        logger.exception("hum_mvp_debug_result_failed case=%s error=%s", capture.case_id, exc)
        return False
    logger.info("hum_mvp_debug_saved case=%s directory=%s", capture.case_id, capture.directory)
    return True
