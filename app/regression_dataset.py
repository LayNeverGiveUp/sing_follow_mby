"""Manage the fixed ten-line manual recording regression dataset."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import quote

from app.debug_capture import DebugCapture


_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


def load_plan(plan_path: Path) -> dict[str, Any]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_id = str(payload.get("plan_id", ""))
    if not _SAFE_ID.fullmatch(plan_id):
        raise ValueError("invalid regression plan id")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("regression plan must contain items")
    seen: set[str] = set()
    for item in items:
        case_id = str(item.get("case_id", ""))
        if not _SAFE_ID.fullmatch(case_id) or case_id in seen:
            raise ValueError(f"invalid or duplicate regression case id: {case_id}")
        seen.add(case_id)
    return payload


def resolve_case(plan: dict[str, Any], plan_id: str, case_id: str) -> dict[str, Any]:
    if plan_id != plan["plan_id"]:
        raise ValueError("unknown regression plan")
    match = next((item for item in plan["items"] if item["case_id"] == case_id), None)
    if match is None:
        raise ValueError("unknown regression case")
    return match


def build_plan_payload(
    plan: dict[str, Any],
    database_dir: Path,
    queries_dir: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    metadata_by_song: dict[str, dict[str, Any]] = {}
    for path in database_dir.glob("*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata_by_song[str(metadata["song_id"])] = metadata

    plan_dir = dataset_root / plan["plan_id"]
    items: list[dict[str, Any]] = []
    for ordinal, configured in enumerate(plan["items"], start=1):
        song_id = str(configured["song_id"])
        lyric_index = int(configured["lyric_index"])
        metadata = metadata_by_song.get(song_id)
        if metadata is None:
            raise ValueError(f"regression song is not in the database: {song_id}")
        lines = {int(line["index"]): line for line in metadata["lrc_lines"]}
        current = lines.get(lyric_index)
        following = lines.get(lyric_index + 1)
        if current is None or following is None:
            raise ValueError(f"regression lyric line is invalid: {song_id}#{lyric_index}")
        encoded_song = quote(song_id)
        case_id = str(configured["case_id"])
        result_path = plan_dir / f"{case_id}.json"
        recording_path = plan_dir / f"{case_id}.wav"
        reference_path = queries_dir / "mao_buyi_v1" / song_id / f"line_{lyric_index:03d}.wav"
        if not reference_path.exists():
            raise ValueError(f"regression reference clip is missing: {song_id}#{lyric_index}")
        items.append(
            {
                **configured,
                "ordinal": ordinal,
                "lyric_text": current["text"],
                "next_lyric_index": int(following["index"]),
                "next_lyric_text": following["text"],
                "reference_audio_url": (
                    f"/static/queries/mao_buyi_v1/{encoded_song}/line_{lyric_index:03d}.wav"
                ),
                "recorded": result_path.exists() and recording_path.exists(),
            }
        )
    return {
        "schema_version": plan["schema_version"],
        "plan_id": plan["plan_id"],
        "title": plan["title"],
        "description": plan["description"],
        "recorded_count": sum(bool(item["recorded"]) for item in items),
        "total_count": len(items),
        "items": items,
    }


def save_case(
    dataset_root: Path,
    plan: dict[str, Any],
    configured_case: dict[str, Any],
    capture: DebugCapture,
    result: dict[str, Any],
) -> tuple[Path, Path]:
    plan_dir = dataset_root / str(plan["plan_id"])
    plan_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(configured_case["case_id"])
    audio_path = plan_dir / f"{case_id}.wav"
    result_path = plan_dir / f"{case_id}.json"

    temporary_audio = audio_path.with_suffix(".wav.tmp")
    shutil.copyfile(capture.audio_path, temporary_audio)
    temporary_audio.replace(audio_path)

    record = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "case_id": case_id,
        "saved_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "expected": {
            "song_id": configured_case["song_id"],
            "lyric_index": int(configured_case["lyric_index"]),
        },
        "audio_file": audio_path.name,
        "debug_case_id": capture.case_id,
        "result": result,
    }
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_result.replace(result_path)
    return audio_path, result_path
