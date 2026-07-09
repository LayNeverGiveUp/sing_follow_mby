from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Segment:
    song_id: str
    song_name: str
    line_id: str
    text: str
    start_ms: int
    end_ms: int
    features: List[float]
    reply_audio: str
    prompt_audio: Optional[str] = None
    lyrics_file: Optional[str] = None


@dataclass(frozen=True)
class Catalog:
    catalog_id: str
    reply_base_path: str
    prompt_base_path: str
    lyrics_base_path: str
    segments: List[Segment]


class CatalogStore:
    def __init__(self, catalog_dir: Path) -> None:
        self.catalog_dir = catalog_dir
        self._cache: Dict[str, Catalog] = {}

    def load(self, catalog_id: str) -> Catalog:
        if catalog_id in self._cache:
            return self._cache[catalog_id]

        path = self.catalog_dir / f"{catalog_id}.json"
        if not path.exists():
            raise ValueError(f"Unknown catalog_id: {catalog_id}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        segments: List[Segment] = []
        for song in payload.get("songs", []):
            for item in song.get("segments", []):
                segments.append(
                    Segment(
                        song_id=song["song_id"],
                        song_name=song["song_name"],
                        line_id=item["line_id"],
                        text=item["text"],
                        start_ms=int(item["start_ms"]),
                        end_ms=int(item["end_ms"]),
                        features=[float(value) for value in item["features"]],
                        reply_audio=item["reply_audio"],
                        prompt_audio=item.get("prompt_audio"),
                        lyrics_file=item.get("lyrics_file"),
                    )
                )

        catalog = Catalog(
            catalog_id=payload["catalog_id"],
            reply_base_path=payload.get("reply_base_path", "/static/replies"),
            prompt_base_path=payload.get("prompt_base_path", "/static/prompts"),
            lyrics_base_path=payload.get("lyrics_base_path", "data/lyrics"),
            segments=segments,
        )
        self._cache[catalog_id] = catalog
        return catalog
