"""Prepare and validate the minimal runtime included in a fresh clone."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "hum_song_mvp" / "data" / "database"
RUNTIME_DIRECTORIES = (
    ROOT / "data" / "queries",
    ROOT / "data" / "debug_recordings",
    ROOT / "data" / "evaluation",
)


def main() -> None:
    for directory in RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(DATABASE_DIR.glob("*.json"))
    if not metadata_paths:
        raise SystemExit(f"Runtime database is missing: {DATABASE_DIR}")

    errors: list[str] = []
    song_ids: list[str] = []
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{metadata_path.name}: invalid JSON ({exc})")
            continue
        missing = {"song_id", "features_file", "feature_hop_seconds", "lrc_lines"} - set(metadata)
        if missing:
            errors.append(f"{metadata_path.name}: missing {', '.join(sorted(missing))}")
            continue
        feature_path = DATABASE_DIR / str(metadata["features_file"])
        if not feature_path.is_file():
            errors.append(f"{metadata_path.name}: missing feature file {feature_path.name}")
        if not metadata["lrc_lines"]:
            errors.append(f"{metadata_path.name}: no lyric lines")
        song_ids.append(str(metadata["song_id"]))

    if errors:
        raise SystemExit("Runtime database validation failed:\n" + "\n".join(errors))

    print(f"Runtime ready: {len(song_ids)} songs")
    print("Songs: " + " / ".join(song_ids))
    print(f"Optional one-click clips: {sum(1 for _ in (ROOT / 'data' / 'queries').rglob('*.wav'))}")


if __name__ == "__main__":
    main()
