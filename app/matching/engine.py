from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from app.audio.features import normalize_features, stabilize_pitch_contour
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
    asr_transcript: Optional[str]
    recall_source: str
    candidate_count: int
    processing_ms: int


class RealtimeMatcher:
    def __init__(
        self,
        catalog: Catalog,
        confidence_threshold: float = 0.72,
        top_k: int = 5,
        text_recall_k: int = 16,
    ) -> None:
        self.catalog = catalog
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.text_recall_k = text_recall_k
        self._features: List[float] = []
        self._last_candidates: List[Segment] = []
        self._transcript: Optional[str] = None
        self._prepared_cache: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}

    @property
    def feature_count(self) -> int:
        return len(self._features)

    @property
    def transcript(self) -> Optional[str]:
        return self._transcript

    def append_features(self, values: List[float]) -> None:
        if not values:
            return
        self._features.extend(float(value) for value in values)
        self._last_candidates = self._recall()

    def set_transcript(self, text: str) -> None:
        normalized = text.strip()
        if normalized:
            self._transcript = normalized

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
                asr_transcript=self._transcript,
                recall_source="none",
                candidate_count=0,
                processing_ms=_elapsed_ms(started),
            )

        query_pitch, query_interval = _prepare_match_features(self._features)
        candidates, recall_source = self._recall_candidates()
        scored = []
        for segment in candidates:
            reference_pitch, reference_interval = self._prepare_segment_features(segment)
            distance = _combined_audio_distance(query_pitch, query_interval, reference_pitch, reference_interval)
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
            asr_transcript=self._transcript,
            recall_source=recall_source,
            candidate_count=len(candidates),
            processing_ms=_elapsed_ms(started),
        )

    def _recall(self) -> List[Segment]:
        if not self._features:
            return self.catalog.segments[: self.top_k]

        query_pitch, _ = _prepare_match_features(self._features)
        scored = []
        for segment in self.catalog.segments:
            reference, _ = self._prepare_segment_features(segment)
            length = min(len(query_pitch), len(reference))
            if length == 0:
                score = float("inf")
            else:
                score = sum(abs(query_pitch[-length + idx] - reference[idx]) for idx in range(length)) / length
            scored.append((score, segment))

        scored.sort(key=lambda item: item[0])
        return [segment for _, segment in scored[: self.top_k]]

    def _recall_candidates(self) -> Tuple[List[Segment], str]:
        if not self._transcript:
            return self.catalog.segments, "audio_full_catalog"

        text_scored = [
            (_text_similarity(self._transcript, segment.text), segment)
            for segment in self.catalog.segments
        ]
        text_scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [segment for score, segment in text_scored[: self.text_recall_k] if score > 0.0]
        if not candidates:
            return self.catalog.segments, "audio_full_catalog"
        return candidates, "asr_text_recall"

    def _prepare_segment_features(self, segment: Segment) -> Tuple[List[float], List[float]]:
        key = (segment.song_id, segment.line_id)
        if key not in self._prepared_cache:
            self._prepared_cache[key] = _prepare_match_features(segment.features)
        return self._prepared_cache[key]


def _distance_to_confidence(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance / 12.0))


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _prepare_match_features(values: List[float]) -> Tuple[List[float], List[float]]:
    pitch = stabilize_pitch_contour(values)
    return normalize_features(pitch), _interval_features(pitch)


def _interval_features(values: List[float]) -> List[float]:
    if len(values) < 2:
        return []
    return [max(-12.0, min(12.0, values[index] - values[index - 1])) for index in range(1, len(values))]


def _combined_audio_distance(
    query_pitch: List[float],
    query_interval: List[float],
    reference_pitch: List[float],
    reference_interval: List[float],
) -> float:
    pitch_distance = dtw_distance(query_pitch, reference_pitch)
    if not query_interval or not reference_interval:
        return pitch_distance
    interval_distance = dtw_distance(query_interval, reference_interval)
    return 0.65 * pitch_distance + 0.35 * interval_distance


def _text_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0

    left_chars = set(left_norm)
    right_chars = set(right_norm)
    char_overlap = len(left_chars & right_chars) / max(1, len(left_chars | right_chars))
    sequence = _lcs_length(left_norm, right_norm) / max(len(left_norm), len(right_norm))
    return 0.45 * char_overlap + 0.55 * sequence


def _normalize_text(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _lcs_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
