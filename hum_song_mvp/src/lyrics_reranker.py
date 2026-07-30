from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from pypinyin import Style, lazy_pinyin


_CONTENT_RE = re.compile(r"[0-9a-z\u4e00-\u9fff]")


@dataclass(frozen=True)
class LyricPosition:
    line_index: int
    start_time: float
    end_time: float
    melody_cost: float
    source: str
    lyric_text: str
    outcome_key: str = ""
    song_id: str = ""


@dataclass(frozen=True)
class LyricCandidateScore:
    position: LyricPosition
    score: float
    character_score: float
    pinyin_score: float
    discriminative_score: float


@dataclass(frozen=True)
class LyricResolution:
    selected: LyricPosition | None
    normalized_text: str
    scores: list[LyricCandidateScore]
    margin: float | None
    reason: str | None


def build_lyric_window(lines: list[dict], start_time: float, end_time: float, padding_seconds: float = 0.0) -> str:
    """Return the lyric lines intersecting a melody candidate's time span."""
    window_start = max(0.0, float(start_time) - max(0.0, padding_seconds))
    window_end = float(end_time) + max(0.0, padding_seconds)
    selected: list[str] = []
    for index, line in enumerate(lines):
        line_start = float(line["start_time"])
        fallback_end = float(lines[index + 1]["start_time"]) if index + 1 < len(lines) else line_start + 30.0
        line_end = float(line.get("end_time", fallback_end))
        if line_start < window_end and line_end > window_start:
            text = str(line.get("text", "")).strip()
            if text:
                selected.append(text)
    return "".join(selected)


def normalize_lyrics(text: str, filler_characters: str = "啊呀啦哦噢嗯呜诶欸") -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    content = "".join(character for character in normalized if _CONTENT_RE.fullmatch(character))
    fillers = set(filler_characters)
    return "".join(character for character in content if character not in fillers)


def rerank_lyric_positions(
    transcript: str,
    positions: list[LyricPosition],
    settings: dict,
) -> LyricResolution:
    normalized_text = normalize_lyrics(transcript, str(settings.get("filler_characters", "啊呀啦哦噢嗯呜诶欸")))
    minimum_characters = int(settings.get("min_lexical_characters", 2))
    if len(normalized_text) < minimum_characters:
        return LyricResolution(None, normalized_text, [], None, "asr_no_lexical_content")
    if len(positions) < 2:
        return LyricResolution(positions[0] if positions else None, normalized_text, [], None, None)

    candidate_texts = [normalize_lyrics(position.lyric_text, "") for position in positions]
    shared_characters = set(candidate_texts[0])
    for candidate_text in candidate_texts[1:]:
        shared_characters.intersection_update(candidate_text)
    candidate_pinyin = [_to_pinyin(text) for text in candidate_texts]
    shared_pinyin = set(candidate_pinyin[0])
    for pinyin_tokens in candidate_pinyin[1:]:
        shared_pinyin.intersection_update(pinyin_tokens)

    query_pinyin = _to_pinyin(normalized_text)
    character_weight = float(settings.get("character_weight", 0.45))
    pinyin_weight = float(settings.get("pinyin_weight", 0.35))
    discriminative_weight = float(settings.get("discriminative_weight", 0.20))
    scores: list[LyricCandidateScore] = []
    for position, candidate_text, pinyin_tokens in zip(positions, candidate_texts, candidate_pinyin):
        character_score = _partial_similarity(list(normalized_text), list(candidate_text))
        pinyin_score = _partial_similarity(query_pinyin, pinyin_tokens)
        distinctive_characters = set(candidate_text) - shared_characters
        distinctive_pinyin = set(pinyin_tokens) - shared_pinyin
        character_coverage = _set_coverage(set(normalized_text), distinctive_characters)
        pinyin_coverage = _set_coverage(set(query_pinyin), distinctive_pinyin)
        discriminative_score = max(character_coverage, pinyin_coverage)
        score = (
            character_weight * character_score
            + pinyin_weight * pinyin_score
            + discriminative_weight * discriminative_score
        )
        scores.append(
            LyricCandidateScore(
                position=position,
                score=float(score),
                character_score=float(character_score),
                pinyin_score=float(pinyin_score),
                discriminative_score=float(discriminative_score),
            )
        )
    scores.sort(key=lambda item: item.score, reverse=True)
    best = scores[0]
    margin = best.score - scores[1].score
    if best.score < float(settings.get("min_lyrics_score", 0.45)):
        return LyricResolution(None, normalized_text, scores, margin, "asr_low_confidence")
    if margin < float(settings.get("min_lyrics_margin", 0.12)):
        return LyricResolution(None, normalized_text, scores, margin, "lyrics_margin_too_small")
    if best.discriminative_score < float(settings.get("min_discriminative_score", 0.20)):
        return LyricResolution(None, normalized_text, scores, margin, "lyrics_no_discriminative_evidence")
    return LyricResolution(best.position, normalized_text, scores, margin, None)


def _to_pinyin(text: str) -> list[str]:
    return [token for token in lazy_pinyin(text, style=Style.NORMAL, errors="ignore") if token]


def _set_coverage(query: set[str], distinctive: set[str]) -> float:
    if not distinctive:
        return 0.0
    return len(query & distinctive) / len(distinctive)


def _partial_similarity(query: list[str], candidate: list[str]) -> float:
    if not query or not candidate:
        return 0.0
    if len(query) > len(candidate):
        return _partial_similarity(candidate, query)
    minimum = max(1, len(query) - 2)
    maximum = min(len(candidate), len(query) + 2)
    best = 0.0
    for length in range(minimum, maximum + 1):
        for start in range(0, len(candidate) - length + 1):
            segment = candidate[start : start + length]
            distance = _edit_distance(query, segment)
            best = max(best, 1.0 - distance / max(len(query), len(segment)))
    return best


def _edit_distance(first: list[str], second: list[str]) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_item in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_item in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_item != second_item),
                )
            )
        previous = current
    return previous[-1]
