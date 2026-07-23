from __future__ import annotations


def map_lyrics(lines: list[dict], matched_end_time: float, end_boundary_tolerance_seconds: float = 0.0) -> dict:
    """Map a DTW endpoint to the line being sung and its follow-up line.

    DTW endpoint quantization commonly lands a few frames past the next LRC
    timestamp when the query ends exactly at a line boundary.  In a follow-up
    experience that would skip a whole line, so the configurable tolerance
    assigns that tiny overshoot to the preceding line.
    """
    if not lines:
        raise ValueError("Database song has no LRC lines")
    effective_end_time = matched_end_time - max(0.0, end_boundary_tolerance_seconds)
    current = None
    for line in lines:
        if float(line["start_time"]) <= effective_end_time:
            current = line
        else:
            break
    if current is None:
        current = lines[0]
    next_index = int(current["index"]) + 1
    next_line = lines[next_index] if next_index < len(lines) else None
    return {
        "current_lyric_index": current["index"],
        "current_lyric_text": current["text"],
        "next_lyric_index": next_line["index"] if next_line else None,
        "next_lyric_text": next_line["text"] if next_line else None,
        "next_lyric_start_time": next_line["start_time"] if next_line else None,
    }
