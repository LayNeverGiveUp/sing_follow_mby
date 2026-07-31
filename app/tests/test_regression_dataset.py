from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.debug_capture import capture_pcm16
from app.regression_dataset import build_plan_payload, resolve_case, save_case


def sample_plan() -> dict:
    return {
        "schema_version": 1,
        "plan_id": "manual_test",
        "title": "Manual test",
        "description": "Test plan",
        "items": [
            {
                "case_id": "case_01",
                "song_id": "song",
                "lyric_index": 0,
                "purpose": "baseline",
            }
        ],
    }


def prepare_database(tmp_path: Path) -> tuple[Path, Path]:
    database_dir = tmp_path / "database"
    queries_dir = tmp_path / "queries"
    database_dir.mkdir()
    reference_dir = queries_dir / "mao_buyi_v1" / "song"
    reference_dir.mkdir(parents=True)
    (reference_dir / "line_000.wav").write_bytes(b"reference")
    metadata = {
        "song_id": "song",
        "lrc_lines": [
            {"index": 0, "text": "current"},
            {"index": 1, "text": "next"},
        ],
    }
    (database_dir / "song.json").write_text(json.dumps(metadata), encoding="utf-8")
    return database_dir, queries_dir


def test_plan_progress_changes_after_capture_is_saved(tmp_path: Path) -> None:
    plan = sample_plan()
    database_dir, queries_dir = prepare_database(tmp_path)
    dataset_root = tmp_path / "datasets"
    payload = build_plan_payload(plan, database_dir, queries_dir, dataset_root)
    assert payload["recorded_count"] == 0
    assert payload["items"][0]["lyric_text"] == "current"
    assert payload["items"][0]["next_lyric_text"] == "next"

    capture = capture_pcm16(tmp_path / "debug", [b"\x00\x00" * 160], 16000)
    save_case(dataset_root, plan, plan["items"][0], capture, {"accepted": False})

    payload = build_plan_payload(plan, database_dir, queries_dir, dataset_root)
    assert payload["recorded_count"] == 1
    assert payload["items"][0]["recorded"] is True
    saved = json.loads((dataset_root / "manual_test" / "case_01.json").read_text(encoding="utf-8"))
    assert saved["expected"] == {"song_id": "song", "lyric_index": 0}
    assert saved["result"]["accepted"] is False


def test_resolve_case_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown regression case"):
        resolve_case(sample_plan(), "manual_test", "missing")
