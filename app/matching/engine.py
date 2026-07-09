from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

from app.audio.features import normalize_features
from app.matching.catalog import Catalog, Segment
from app.matching.dtw import dtw_distance


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    song_id: Optional[str]
    song_name: Optional[str]
    confidence: float
    matched_line: Optional[str]
    line_id: Optional[str]
    handoff_type: str
    reply_audio: Optional[str]
    prompt_audio: Optional[str]
    lyrics_file: Optional[str]
    processing_ms: int


class RealtimeMatcher:
    def __init__(self, catalog: Catalog, confidence_threshold: float = 0.72, top_k: int = 5) -> None:
        self.catalog = catalog
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self._features: List[float] = []
        self._last_candidates: List[Segment] = []

    @property
    def feature_count(self) -> int:
        return len(self._features)

    def append_features(self, values: List[float]) -> None:
        if not values:
            return
        self._features.extend(float(value) for value in values)
        self._last_candidates = self._recall()

    def finalize(self) -> MatchResult:
        started = perf_counter()
        if len(self._features) < 3:
            return MatchResult(
                matched=False,
                song_id=None,
                song_name=None,
                confidence=0.0,
                matched_line=None,
                line_id=None,
                handoff_type="fallback",
                reply_audio=None,
                prompt_audio=None,
                lyrics_file=None,
                processing_ms=_elapsed_ms(started),
            )

        query = normalize_features(self._features)
        candidates = self.catalog.segments
        scored = []
        for segment in candidates:
            distance = dtw_distance(query, normalize_features(segment.features))
            scored.append((distance, segment))

        scored.sort(key=lambda item: item[0])
        best_distance, best_segment = scored[0]
        confidence = _distance_to_confidence(best_distance)
        matched = confidence >= self.confidence_threshold

        return MatchResult(
            matched=matched,
            song_id=best_segment.song_id if matched else None,
            song_name=best_segment.song_name if matched else None,
            confidence=round(confidence, 4),
            matched_line=best_segment.text if matched else None,
            line_id=best_segment.line_id if matched else None,
            handoff_type="next_line" if matched else "fallback",
            reply_audio=best_segment.reply_audio if matched else None,
            prompt_audio=best_segment.prompt_audio if matched else None,
            lyrics_file=best_segment.lyrics_file if matched else None,
            processing_ms=_elapsed_ms(started),
        )

    def _recall(self) -> List[Segment]:
        if not self._features:
            return self.catalog.segments[: self.top_k]

        query = normalize_features(self._features)
        scored = []
        for segment in self.catalog.segments:
            reference = normalize_features(segment.features)
            length = min(len(query), len(reference))
            if length == 0:
                score = float("inf")
            else:
                score = sum(abs(query[-length + idx] - reference[idx]) for idx in range(length)) / length
            scored.append((score, segment))

        scored.sort(key=lambda item: item[0])
        return [segment for _, segment in scored[: self.top_k]]


def _distance_to_confidence(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance / 12.0))


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)
