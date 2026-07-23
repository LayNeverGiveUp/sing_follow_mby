from __future__ import annotations

import re
from pathlib import Path


_TIMESTAMP = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")


def parse_lrc(path: str | Path, duration: float) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LRC file does not exist: {path}")
    text = _read_text(path)
    entries: list[tuple[float, str]] = []
    for raw in text.splitlines():
        markers = list(_TIMESTAMP.finditer(raw))
        lyric = _TIMESTAMP.sub("", raw).strip()
        if not markers or not lyric:
            continue
        for marker in markers:
            fraction = marker.group(3) or "0"
            fraction_seconds = int(fraction) / (10 ** len(fraction))
            entries.append((int(marker.group(1)) * 60 + int(marker.group(2)) + fraction_seconds, lyric))
    entries.sort(key=lambda entry: entry[0])
    if not entries:
        raise ValueError(f"LRC has no timestamped lyric lines: {path}")
    result = []
    for index, (start_time, lyric) in enumerate(entries):
        end_time = entries[index + 1][0] if index + 1 < len(entries) else duration
        if end_time <= start_time:
            continue
        result.append({"index": len(result), "start_time": round(start_time, 3), "end_time": round(end_time, 3), "text": lyric})
    if not result:
        raise ValueError(f"LRC lines have invalid time ordering: {path}")
    return result


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="replace")
