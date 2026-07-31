from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio_io import load_mono_audio, trim_outer_silence
from .config import load_config
from .confidence import decide
from .dtw_matcher import DtwResult, subsequence_dtw_nbest
from .lyrics_asr import get_lyrics_asr
from .lyrics_reranker import LyricPosition, rerank_lyric_positions
from .lyric_mapper import map_lyrics
from .phrase_matcher import (
    PhraseMatch,
    PreparedPhrase,
    match_prepared_lyric_phrases,
    prepare_lyric_phrases,
)
from .pitch_extractor import PitchFeatures, extract_features


@dataclass(frozen=True)
class Candidate:
    metadata: dict
    result: DtwResult


@dataclass(frozen=True)
class PhraseCandidate:
    metadata: dict
    match: PhraseMatch


@dataclass(frozen=True)
class SelectedPosition:
    metadata: dict
    start_time: float
    end_time: float
    lyrics: dict
    score: float
    margin: float | None
    route: str


@dataclass(frozen=True)
class DatabaseSong:
    metadata: dict
    reference: PitchFeatures
    phrases: tuple[PreparedPhrase, ...]


class PrefetchedLyricsAsr:
    """Expose a normal ASR adapter while its request runs beside melody matching."""

    def __init__(self, delegate, future) -> None:
        self.delegate = delegate
        self.future = future
        self.enabled = bool(getattr(delegate, "enabled", False))
        self.status = getattr(delegate, "status", "not_configured")
        self.elapsed_ms: float | None = None

    def transcribe(self, samples: np.ndarray, sample_rate: int):
        del samples, sample_rate
        transcript, elapsed_ms = self.future.result()
        self.elapsed_ms = elapsed_ms
        return transcript


_MATCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="melody-match")
_ASR_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lyrics-asr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize a hummed phrase against a local song database.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    payload = recognize(args.audio, args.database_dir, load_config(args.config))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def recognize(audio_path: Path, database_dir: Path, config: dict) -> dict:
    samples = load_mono_audio(audio_path, int(config["audio"]["sample_rate"]))
    return recognize_samples(samples, database_dir, config, get_lyrics_asr(config))


def recognize_samples(samples: np.ndarray, database_dir: Path, config: dict, lyrics_asr=None) -> dict:
    total_started = perf_counter()
    stage_started = perf_counter()
    trimmed, start_sample, _ = trim_outer_silence(samples, float(config["audio"]["trim_top_db"]))
    trim_ms = (perf_counter() - stage_started) * 1000
    lyrics_asr = _prefetch_lyrics_asr(trimmed, config, lyrics_asr)
    stage_started = perf_counter()
    query = extract_features(trimmed, config)
    feature_ms = (perf_counter() - stage_started) * 1000
    if str(config["matching"].get("algorithm", "frame_dtw")) == "hybrid_phrase":
        match_wall_started = perf_counter()
        frame_future = _MATCH_EXECUTOR.submit(_timed_frame_match, query, database_dir, config)
        phrase_future = _MATCH_EXECUTOR.submit(_timed_phrase_match, query, database_dir, config)
        candidates, frame_positions, candidate_ms, match_ms = frame_future.result()
        phrase_candidates, phrase_ms = phrase_future.result()
        match_wall_ms = (perf_counter() - match_wall_started) * 1000
        candidates.sort(key=lambda candidate: candidate.result.normalized_cost)
        payload = _recognize_hybrid(
            samples,
            trimmed,
            start_sample,
            query,
            candidates,
            frame_positions,
            phrase_candidates,
            config,
            {
                "trim_silence": round(trim_ms, 1),
                "f0_and_onset": round(feature_ms, 1),
                "database_load_and_dtw": round(match_ms, 1),
                "phrase_contour_matching": round(phrase_ms, 1),
                "melody_matching_wall": round(match_wall_ms, 1),
                "per_song_dtw": candidate_ms,
                "total_recognition": round((perf_counter() - total_started) * 1000, 1),
            },
            lyrics_asr,
        )
        payload["diagnostics"]["stage_ms"]["total_recognition"] = round(
            (perf_counter() - total_started) * 1000,
            1,
        )
        return payload
    candidate_ms: dict[str, float] = {}
    stage_started = perf_counter()
    candidates, frame_positions = _match_all(query, database_dir, config, candidate_ms)
    match_ms = (perf_counter() - stage_started) * 1000
    candidates.sort(key=lambda candidate: candidate.result.normalized_cost)
    best = candidates[0] if candidates else None
    second_cost = candidates[1].result.normalized_cost if len(candidates) > 1 else None
    decision = decide(query, best.result if best else None, second_cost, config)
    diagnostics = _diagnostics(
        samples, trimmed, query, candidates, config,
        {
            "trim_silence": round(trim_ms, 1),
            "f0_and_onset": round(feature_ms, 1),
            "database_load_and_dtw": round(match_ms, 1),
            "per_song_dtw": candidate_ms,
            "total_recognition": round((perf_counter() - total_started) * 1000, 1),
        },
    )
    if not decision.accepted or best is None:
        payload = _rejected_payload(decision, best, len(candidates))
        payload["diagnostics"] = diagnostics
        return payload
    hop = float(best.metadata["feature_hop_seconds"])
    start_time = best.result.start_frame * hop
    end_time = best.result.end_frame * hop
    lyrics = map_lyrics(
        best.metadata["lrc_lines"],
        end_time,
        float(config.get("lyric_mapping", {}).get("end_boundary_tolerance_seconds", 0.0)),
    )
    payload = {
        "accepted": True,
        "recognition_status": "accepted",
        "position_resolved": True,
        "song_id": best.metadata["song_id"],
        "matched_start_time": round(start_time, 3),
        "matched_end_time": round(end_time, 3),
        **lyrics,
        "score": round(decision.score, 4),
        "top2_margin": round(decision.margin, 4) if decision.margin is not None else None,
        "query_trim_start_time": round(start_sample / float(config["audio"]["sample_rate"]), 3),
    }
    payload["diagnostics"] = diagnostics
    return payload


def _diagnostics(
    samples: np.ndarray,
    trimmed: np.ndarray,
    query: PitchFeatures,
    candidates: list[Candidate],
    config: dict,
    stage_ms: dict,
) -> dict:
    sample_rate = float(config["audio"]["sample_rate"])
    finite_pitch = query.pitch[np.isfinite(query.pitch)]
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return {
        "input_duration_seconds": round(samples.size / sample_rate, 3),
        "trimmed_duration_seconds": round(trimmed.size / sample_rate, 3),
        "input_rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 1),
        "voiced_seconds": round(float(np.count_nonzero(query.voiced)) * float(config["pitch"]["hop_seconds"]), 3),
        "pitch_range_semitones": round(float(np.ptp(finite_pitch)) if finite_pitch.size else 0.0, 3),
        "stage_ms": stage_ms,
        "candidates": [
            {
                "song_id": candidate.metadata["song_id"],
                "normalized_cost": round(candidate.result.normalized_cost, 4),
                "raw_normalized_cost": round(candidate.result.raw_normalized_cost, 4),
                "paired_voiced_seconds": round(candidate.result.paired_voiced_seconds, 3),
                "query_voiced_coverage": round(candidate.result.query_voiced_coverage, 3),
            }
            for candidate in candidates
        ],
    }


def _match_all(
    query: PitchFeatures,
    database_dir: Path,
    config: dict,
    candidate_ms: dict[str, float] | None = None,
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    candidates = []
    positions: dict[str, list[Candidate]] = {}
    for song in _database_songs(database_dir, config):
        started = perf_counter()
        algorithm = str(config["matching"].get("algorithm", "frame_dtw"))
        if algorithm in {"frame_dtw", "hybrid_phrase"}:
            results = subsequence_dtw_nbest(query, song.reference, config)
        else:
            raise ValueError(f"Unsupported matching.algorithm: {algorithm}")
        if candidate_ms is not None:
            candidate_ms[song.metadata["song_id"]] = round((perf_counter() - started) * 1000, 1)
        song_positions = [Candidate(song.metadata, result) for result in results]
        if song_positions:
            candidates.append(song_positions[0])
            positions[song.metadata["song_id"]] = song_positions
    return candidates, positions


def _match_all_phrases(query: PitchFeatures, database_dir: Path, config: dict) -> list[PhraseCandidate]:
    candidates: list[PhraseCandidate] = []
    for song in _database_songs(database_dir, config):
        candidates.extend(
            PhraseCandidate(song.metadata, match)
            for match in match_prepared_lyric_phrases(query, song.phrases, config)
        )
    return sorted(candidates, key=lambda candidate: candidate.match.cost)


def _timed_frame_match(
    query: PitchFeatures,
    database_dir: Path,
    config: dict,
) -> tuple[list[Candidate], dict[str, list[Candidate]], dict[str, float], float]:
    candidate_ms: dict[str, float] = {}
    started = perf_counter()
    candidates, positions = _match_all(query, database_dir, config, candidate_ms)
    return candidates, positions, candidate_ms, (perf_counter() - started) * 1000


def _timed_phrase_match(
    query: PitchFeatures,
    database_dir: Path,
    config: dict,
) -> tuple[list[PhraseCandidate], float]:
    started = perf_counter()
    candidates = _match_all_phrases(query, database_dir, config)
    return candidates, (perf_counter() - started) * 1000


def _prefetch_lyrics_asr(samples: np.ndarray, config: dict, lyrics_asr):
    settings = config.get("lyrics_asr", {})
    if (
        lyrics_asr is None
        or not bool(getattr(lyrics_asr, "enabled", False))
        or not bool(settings.get("speculative_parallel", False))
        or str(config["matching"].get("algorithm", "frame_dtw")) != "hybrid_phrase"
        or samples.size == 0
    ):
        return lyrics_asr
    sample_rate = int(config["audio"]["sample_rate"])
    future = _ASR_EXECUTOR.submit(_timed_asr_transcribe, lyrics_asr, samples, sample_rate)
    return PrefetchedLyricsAsr(lyrics_asr, future)


def _timed_asr_transcribe(lyrics_asr, samples: np.ndarray, sample_rate: int):
    started = perf_counter()
    transcript = lyrics_asr.transcribe(samples, sample_rate)
    return transcript, (perf_counter() - started) * 1000


def _best_phrase_candidates_by_song(candidates: list[PhraseCandidate]) -> list[PhraseCandidate]:
    """Keep one melody winner per song before computing a song-level margin."""
    best_by_song: dict[str, PhraseCandidate] = {}
    for candidate in candidates:
        song_id = str(candidate.metadata["song_id"])
        current = best_by_song.get(song_id)
        if current is None or candidate.match.cost < current.match.cost:
            best_by_song[song_id] = candidate
    return sorted(best_by_song.values(), key=lambda candidate: candidate.match.cost)


def _recognize_hybrid(
    samples: np.ndarray,
    trimmed: np.ndarray,
    start_sample: int,
    query: PitchFeatures,
    frame_candidates: list[Candidate],
    frame_positions: dict[str, list[Candidate]],
    phrase_candidates: list[PhraseCandidate],
    config: dict,
    stage_ms: dict,
    lyrics_asr=None,
) -> dict:
    settings = config["phrase_matching"]
    hop = float(config["pitch"]["hop_seconds"])
    voiced_seconds = float(np.count_nonzero(query.voiced)) * hop
    finite_pitch = query.pitch[np.isfinite(query.pitch)]
    pitch_range = float(np.ptp(finite_pitch)) if finite_pitch.size else 0.0
    diagnostics = _diagnostics(samples, trimmed, query, frame_candidates, config, stage_ms)
    diagnostics["phrase_candidates"] = [
        {
            "song_id": candidate.metadata["song_id"],
            "line_index": candidate.match.line_index,
            "cost": round(candidate.match.cost, 4),
        }
        for candidate in phrase_candidates[:5]
    ]
    phrase_song_candidates = _best_phrase_candidates_by_song(phrase_candidates)
    diagnostics["phrase_song_candidates"] = [
        {
            "song_id": candidate.metadata["song_id"],
            "line_index": candidate.match.line_index,
            "cost": round(candidate.match.cost, 4),
        }
        for candidate in phrase_song_candidates[:5]
    ]
    if voiced_seconds < float(settings["min_query_voiced_seconds"]):
        return _hybrid_rejected("insufficient_voiced_audio", diagnostics, frame_candidates, phrase_candidates)
    if pitch_range < float(config["confidence"]["min_pitch_range_semitones"]):
        return _hybrid_rejected("insufficient_pitch_variation", diagnostics, frame_candidates, phrase_candidates)

    trusted_frames = [
        candidate
        for candidate in frame_candidates
        if candidate.result.paired_voiced_seconds >= float(settings["min_trusted_frame_voiced_seconds"])
        and candidate.result.query_voiced_coverage >= float(settings["min_trusted_frame_coverage"])
    ]
    trusted_frames.sort(key=lambda candidate: candidate.result.normalized_cost)
    frame = trusted_frames[0] if trusted_frames else None
    frame_margin = (
        trusted_frames[1].result.normalized_cost - frame.result.normalized_cost
        if frame is not None and len(trusted_frames) > 1
        else None
    )
    frame_strong = bool(
        frame is not None
        and frame.result.query_voiced_coverage >= float(settings["strong_frame_coverage"])
        and frame.result.normalized_cost <= float(settings.get("max_strong_frame_cost", 0.50))
    )
    phrase = phrase_song_candidates[0] if phrase_song_candidates else None
    phrase_margin = (
        phrase_song_candidates[1].match.cost - phrase.match.cost
        if phrase is not None and len(phrase_song_candidates) > 1
        else None
    )
    phrase_usable = phrase is not None and phrase.match.cost <= float(settings["max_cost"])
    phrase_confident = bool(
        phrase is not None
        and phrase.match.cost <= float(settings.get("max_standalone_cost", settings["max_cost"]))
        and (phrase_margin is None or phrase_margin >= float(settings["min_standalone_margin"]))
    )
    strong_frame_conflict = bool(
        frame_strong and phrase_usable and frame.metadata["song_id"] != phrase.metadata["song_id"]
    )
    diagnostics["hybrid_evidence"] = {
        "frame_song_id": frame.metadata["song_id"] if frame is not None else None,
        "frame_margin": round(frame_margin, 4) if frame_margin is not None else None,
        "frame_strong": frame_strong,
        "phrase_song_id": phrase.metadata["song_id"] if phrase is not None else None,
        "phrase_distinct_song_margin": round(phrase_margin, 4) if phrase_margin is not None else None,
        "strong_frame_conflict": strong_frame_conflict,
    }

    position_already_resolved = False
    if frame_strong and phrase_usable and frame.metadata["song_id"] == phrase.metadata["song_id"]:
        frame_end = frame.result.end_frame * float(frame.metadata["feature_hop_seconds"])
        selected = SelectedPosition(
            metadata=frame.metadata,
            start_time=frame.result.start_frame * float(frame.metadata["feature_hop_seconds"]),
            end_time=frame_end,
            lyrics=map_lyrics(
                frame.metadata["lrc_lines"],
                frame_end,
                float(config.get("lyric_mapping", {}).get("end_boundary_tolerance_seconds", 0.0)),
            ),
            score=max(
                0.0,
                min(1.0, 1.0 - frame.result.normalized_cost / float(config["confidence"]["score_cost_scale"])),
            ),
            margin=phrase_margin,
            route="strong_frame_phrase_agreement",
        )
    elif phrase_confident and not strong_frame_conflict:
        selected = SelectedPosition(
            metadata=phrase.metadata,
            start_time=phrase.match.start_frame * float(phrase.metadata["feature_hop_seconds"]),
            end_time=phrase.match.end_frame * float(phrase.metadata["feature_hop_seconds"]),
            lyrics=_lyrics_for_index(phrase.metadata["lrc_lines"], phrase.match.line_index),
            score=_phrase_score(phrase.match.cost, settings),
            margin=phrase_margin,
            route="standalone_phrase",
        )
    elif phrase_usable:
        selected, resolution_reason = _resolve_cross_song_position(
            trimmed,
            phrase_song_candidates,
            phrase_candidates,
            phrase_margin,
            config,
            lyrics_asr,
            diagnostics,
        )
        if selected is None:
            return _hybrid_rejected(
                resolution_reason or "phrase_margin_too_small",
                diagnostics,
                frame_candidates,
                phrase_candidates,
                best_song=phrase.metadata["song_id"],
                recognition_status="song_only",
            )
        position_already_resolved = True
    elif (
        frame_strong
        and (frame_margin is None or frame_margin >= float(settings.get("min_standalone_frame_margin", 0.02)))
    ):
        frame_end = frame.result.end_frame * float(frame.metadata["feature_hop_seconds"])
        selected = SelectedPosition(
            metadata=frame.metadata,
            start_time=frame.result.start_frame * float(frame.metadata["feature_hop_seconds"]),
            end_time=frame_end,
            lyrics=map_lyrics(
                frame.metadata["lrc_lines"],
                frame_end,
                float(config.get("lyric_mapping", {}).get("end_boundary_tolerance_seconds", 0.0)),
            ),
            score=max(
                0.0,
                min(1.0, 1.0 - frame.result.normalized_cost / float(config["confidence"]["score_cost_scale"])),
            ),
            margin=frame_margin,
            route="strong_frame",
        )
    else:
        return _hybrid_rejected("no_reliable_hybrid_candidate", diagnostics, frame_candidates, phrase_candidates)

    if not position_already_resolved:
        selected, resolution_reason = _resolve_repeated_position(
            trimmed,
            selected,
            frame_positions,
            phrase_candidates,
            config,
            lyrics_asr,
            diagnostics,
        )
        if selected is None:
            return _hybrid_rejected(
                resolution_reason or "ambiguous_repeated_melody",
                diagnostics,
                frame_candidates,
                phrase_candidates,
                best_song=phrase.metadata["song_id"] if phrase is not None else (frame.metadata["song_id"] if frame else None),
                recognition_status="song_only",
            )

    diagnostics["hybrid_route"] = selected.route
    return {
        "accepted": True,
        "recognition_status": "accepted",
        "position_resolved": True,
        "song_id": selected.metadata["song_id"],
        "matched_start_time": round(selected.start_time, 3),
        "matched_end_time": round(selected.end_time, 3),
        **selected.lyrics,
        "score": round(selected.score, 4),
        "top2_margin": round(selected.margin, 4) if selected.margin is not None else None,
        "query_trim_start_time": round(start_sample / float(config["audio"]["sample_rate"]), 3),
        "diagnostics": diagnostics,
    }


def _resolve_cross_song_position(
    samples: np.ndarray,
    song_candidates: list[PhraseCandidate],
    phrase_candidates: list[PhraseCandidate],
    song_margin: float | None,
    config: dict,
    lyrics_asr,
    diagnostics: dict,
) -> tuple[SelectedPosition | None, str | None]:
    """Use words only after melody has narrowed the search to a few songs."""
    settings = config["phrase_matching"]
    top_songs = song_candidates[: int(settings.get("cross_song_asr_top_k", 4))]
    if not top_songs:
        return None, "no_reliable_hybrid_candidate"

    metadata_by_song = {str(candidate.metadata["song_id"]): candidate.metadata for candidate in top_songs}
    if bool(settings.get("cross_song_expand_all_lines", False)):
        positions = _expand_positions_for_asr(
            [],
            metadata_by_song,
            phrase_candidates,
            config,
            source="phrase_dtw_cross_song_catalog",
        )
    else:
        positions = _shortlisted_cross_song_positions(top_songs, phrase_candidates, config)
    diagnostics["cross_song_position_candidates"] = [
        {
            "song_id": position.song_id,
            "line_index": position.line_index,
            "melody_cost": round(position.melody_cost, 4),
            "lyric_text": position.lyric_text,
        }
        for position in positions
    ]
    best = top_songs[0]
    selected = SelectedPosition(
        metadata=best.metadata,
        start_time=best.match.start_frame * float(best.metadata["feature_hop_seconds"]),
        end_time=best.match.end_frame * float(best.metadata["feature_hop_seconds"]),
        lyrics=_lyrics_for_index(best.metadata["lrc_lines"], best.match.line_index),
        score=_phrase_score(best.match.cost, settings),
        margin=song_margin,
        route="phrase_cross_song",
    )
    if len({position.song_id for position in positions}) < 2:
        return None, "phrase_margin_too_small"
    return _resolve_positions_with_lyrics(
        samples,
        selected,
        positions,
        metadata_by_song,
        config,
        lyrics_asr,
        diagnostics,
    )


def _resolve_repeated_position(
    samples: np.ndarray,
    selected: SelectedPosition,
    frame_positions: dict[str, list[Candidate]],
    phrase_candidates: list[PhraseCandidate],
    config: dict,
    lyrics_asr,
    diagnostics: dict,
) -> tuple[SelectedPosition | None, str | None]:
    settings = config.get("position_resolution", {})
    if not bool(settings.get("enabled", True)):
        return selected, None
    positions = _ambiguous_lyric_positions(selected, frame_positions, phrase_candidates, config)
    diagnostics["position_candidates"] = [
        {
            "song_id": position.song_id or selected.metadata["song_id"],
            "line_index": position.line_index,
            "start_time": round(position.start_time, 3),
            "end_time": round(position.end_time, 3),
            "melody_cost": round(position.melody_cost, 4),
            "source": position.source,
            "lyric_text": position.lyric_text,
        }
        for position in positions
    ]
    if len(positions) < 2:
        diagnostics["position_resolution"] = "melody_unique"
        return selected, None

    return _resolve_positions_with_lyrics(
        samples,
        selected,
        positions,
        {str(selected.metadata["song_id"]): selected.metadata},
        config,
        lyrics_asr,
        diagnostics,
    )


def _resolve_positions_with_lyrics(
    samples: np.ndarray,
    selected: SelectedPosition,
    positions: list[LyricPosition],
    metadata_by_song: dict[str, dict],
    config: dict,
    lyrics_asr,
    diagnostics: dict,
) -> tuple[SelectedPosition | None, str | None]:
    """Resolve melody-ambiguous positions without letting ASR search the full catalog."""

    asr_status = getattr(lyrics_asr, "status", "not_configured") if lyrics_asr is not None else "not_configured"
    diagnostics["position_resolution"] = "lyrics_required"
    diagnostics["lyrics_asr"] = {"triggered": False, "status": asr_status}
    if lyrics_asr is None or not bool(getattr(lyrics_asr, "enabled", False)):
        return None, "asr_unavailable_for_ambiguous_melody"

    started = perf_counter()
    try:
        transcript = lyrics_asr.transcribe(samples, int(config["audio"]["sample_rate"]))
    except Exception as exc:
        diagnostics["lyrics_asr"] = {
            "triggered": True,
            "status": "error",
            "error": type(exc).__name__,
            "detail": str(exc),
        }
        diagnostics.setdefault("stage_ms", {})["lyrics_asr"] = round((perf_counter() - started) * 1000, 1)
        return None, "asr_unavailable_for_ambiguous_melody"
    wait_ms = (perf_counter() - started) * 1000
    stage_ms = diagnostics.setdefault("stage_ms", {})
    stage_ms["lyrics_asr_wait"] = round(wait_ms, 1)
    stage_ms["lyrics_asr"] = round(
        float(getattr(lyrics_asr, "elapsed_ms", None) or wait_ms),
        1,
    )
    if transcript is None or not str(transcript.text).strip():
        diagnostics["lyrics_asr"] = {"triggered": True, "status": "no_text", "source": asr_status}
        return None, "asr_no_lexical_content"

    resolution = rerank_lyric_positions(str(transcript.text), positions, config.get("lyrics_asr", {}))
    diagnostics["lyrics_asr"] = {
        "triggered": True,
        "status": "resolved" if resolution.selected is not None else "unresolved",
        "source": transcript.source,
        "text": transcript.text,
        "normalized_text": resolution.normalized_text,
        "confidence": transcript.confidence,
        "margin": round(resolution.margin, 4) if resolution.margin is not None else None,
        "candidate_scores": [
            {
                "song_id": score.position.song_id or selected.metadata["song_id"],
                "line_index": score.position.line_index,
                "score": round(score.score, 4),
                "character_score": round(score.character_score, 4),
                "pinyin_score": round(score.pinyin_score, 4),
                "discriminative_score": round(score.discriminative_score, 4),
            }
            for score in resolution.scores
        ],
    }
    if resolution.selected is None:
        return None, resolution.reason or "lyrics_margin_too_small"
    position = resolution.selected
    song_id = position.song_id or str(selected.metadata["song_id"])
    metadata = metadata_by_song[song_id]
    diagnostics["position_resolution"] = "lyrics_rerank"
    return (
        SelectedPosition(
            metadata=metadata,
            start_time=position.start_time,
            end_time=position.end_time,
            lyrics=_lyrics_for_index(metadata["lrc_lines"], position.line_index),
            score=(
                _phrase_score(position.melody_cost, config["phrase_matching"])
                if position.source.startswith("phrase")
                else selected.score
            ),
            margin=selected.margin,
            route=f"{selected.route}_lyrics_rerank",
        ),
        None,
    )


def _ambiguous_lyric_positions(
    selected: SelectedPosition,
    frame_positions: dict[str, list[Candidate]],
    phrase_candidates: list[PhraseCandidate],
    config: dict,
) -> list[LyricPosition]:
    song_id = selected.metadata["song_id"]
    lines = selected.metadata["lrc_lines"]
    settings = config.get("position_resolution", {})
    padding = float(config.get("lyrics_asr", {}).get("lyric_window_padding_seconds", 0.0))
    positions: list[LyricPosition] = []
    if selected.route.startswith(("trusted_frame", "strong_frame")):
        phrase_settings = config["phrase_matching"]
        for candidate in frame_positions.get(song_id, []):
            if candidate.result.paired_voiced_seconds < float(phrase_settings["min_trusted_frame_voiced_seconds"]):
                continue
            if candidate.result.query_voiced_coverage < float(phrase_settings["min_trusted_frame_coverage"]):
                continue
            hop = float(candidate.metadata["feature_hop_seconds"])
            start_time = candidate.result.start_frame * hop
            end_time = candidate.result.end_frame * hop
            mapped = map_lyrics(
                lines,
                end_time,
                float(config.get("lyric_mapping", {}).get("end_boundary_tolerance_seconds", 0.0)),
            )
            line_index = int(mapped["current_lyric_index"])
            positions.append(
                LyricPosition(
                    line_index=line_index,
                    start_time=start_time,
                    end_time=end_time,
                    melody_cost=float(candidate.result.normalized_cost),
                    source="frame_dtw",
                    lyric_text=_lyric_text_for_index(lines, line_index),
                    outcome_key=_lyric_outcome_key(lines, line_index),
                    song_id=str(song_id),
                )
            )
        if bool(settings.get("require_phrase_support_for_frame_ambiguity", True)) and positions:
            phrase_costs = {
                int(candidate.match.line_index): float(candidate.match.cost)
                for candidate in phrase_candidates
                if candidate.metadata["song_id"] == song_id
            }
            supported_costs = [phrase_costs[position.line_index] for position in positions if position.line_index in phrase_costs]
            if supported_costs:
                phrase_limit = min(supported_costs) + float(settings.get("phrase_ambiguity_cost_delta", 0.10))
                positions = [
                    position
                    for position in positions
                    if position.line_index in phrase_costs and phrase_costs[position.line_index] <= phrase_limit
                ]
        delta = float(settings.get("frame_ambiguity_cost_delta", 0.12))
        ratio = float(settings.get("frame_ambiguity_cost_ratio", 1.60))
    else:
        for candidate in phrase_candidates:
            if candidate.metadata["song_id"] != song_id or candidate.match.cost > float(config["phrase_matching"]["max_cost"]):
                continue
            hop = float(candidate.metadata["feature_hop_seconds"])
            start_time = candidate.match.start_frame * hop
            end_time = candidate.match.end_frame * hop
            positions.append(
                LyricPosition(
                    line_index=int(candidate.match.line_index),
                    start_time=start_time,
                    end_time=end_time,
                    melody_cost=float(candidate.match.cost),
                    source="phrase_dtw",
                    lyric_text=_lyric_text_for_index(lines, int(candidate.match.line_index)),
                    outcome_key=_lyric_outcome_key(lines, int(candidate.match.line_index)),
                    song_id=str(song_id),
                )
            )
        delta = float(settings.get("phrase_ambiguity_cost_delta", 0.10))
        ratio = float(settings.get("phrase_ambiguity_cost_ratio", 1.30))

    by_line: dict[int, LyricPosition] = {}
    for position in positions:
        existing = by_line.get(position.line_index)
        if existing is None or position.melody_cost < existing.melody_cost:
            by_line[position.line_index] = position
    distinct = sorted(by_line.values(), key=lambda item: item.melody_cost)
    if len(distinct) < 2:
        return distinct
    best_cost = distinct[0].melody_cost
    cost_limit = min(best_cost + delta, best_cost * ratio if best_cost > 1e-8 else best_cost + delta)
    ambiguous = [position for position in distinct if position.melody_cost <= cost_limit]
    if len({position.outcome_key for position in ambiguous}) < 2:
        return ambiguous[:1]
    if bool(settings.get("expand_all_song_lines_for_asr", False)):
        return _expand_positions_for_asr(
            ambiguous,
            {str(song_id): selected.metadata},
            phrase_candidates,
            config,
            source="phrase_dtw_same_song_catalog",
        )
    return ambiguous


def _shortlisted_cross_song_positions(
    top_songs: list[PhraseCandidate],
    phrase_candidates: list[PhraseCandidate],
    config: dict,
) -> list[LyricPosition]:
    settings = config["phrase_matching"]
    positions: list[LyricPosition] = []
    per_song_limit = int(settings.get("cross_song_positions_per_song", 3))
    delta = float(settings.get("cross_song_position_cost_delta", 0.15))
    ratio = float(settings.get("cross_song_position_cost_ratio", 1.50))
    max_cost = float(settings["max_cost"])
    for song_candidate in top_songs:
        song_id = str(song_candidate.metadata["song_id"])
        lines = song_candidate.metadata["lrc_lines"]
        best_cost = float(song_candidate.match.cost)
        cost_limit = min(max_cost, best_cost + delta, best_cost * ratio if best_cost > 1e-8 else best_cost + delta)
        by_outcome: dict[str, LyricPosition] = {}
        for candidate in phrase_candidates:
            if str(candidate.metadata["song_id"]) != song_id or candidate.match.cost > cost_limit:
                continue
            hop = float(candidate.metadata["feature_hop_seconds"])
            line_index = int(candidate.match.line_index)
            outcome_key = _lyric_outcome_key(lines, line_index)
            position = LyricPosition(
                line_index=line_index,
                start_time=candidate.match.start_frame * hop,
                end_time=candidate.match.end_frame * hop,
                melody_cost=float(candidate.match.cost),
                source="phrase_dtw_cross_song",
                lyric_text=_lyric_text_for_index(lines, line_index),
                outcome_key=outcome_key,
                song_id=song_id,
            )
            existing = by_outcome.get(outcome_key)
            if existing is None or position.melody_cost < existing.melody_cost:
                by_outcome[outcome_key] = position
        positions.extend(sorted(by_outcome.values(), key=lambda item: item.melody_cost)[:per_song_limit])
    return sorted(positions, key=lambda item: item.melody_cost)


def _expand_positions_for_asr(
    positions: list[LyricPosition],
    metadata_by_song: dict[str, dict],
    phrase_candidates: list[PhraseCandidate],
    config: dict,
    source: str,
) -> list[LyricPosition]:
    """Add every lyric outcome inside melody-shortlisted songs for ASR reranking."""
    phrase_by_line: dict[tuple[str, int], PhraseCandidate] = {}
    for candidate in phrase_candidates:
        song_id = str(candidate.metadata["song_id"])
        if song_id not in metadata_by_song:
            continue
        key = (song_id, int(candidate.match.line_index))
        existing = phrase_by_line.get(key)
        if existing is None or candidate.match.cost < existing.match.cost:
            phrase_by_line[key] = candidate

    expanded = list(positions)
    fallback_cost = float(config["phrase_matching"]["max_cost"])
    for song_id, metadata in metadata_by_song.items():
        lines = metadata["lrc_lines"]
        for line in lines:
            line_index = int(line["index"])
            candidate = phrase_by_line.get((song_id, line_index))
            if candidate is not None:
                hop = float(candidate.metadata["feature_hop_seconds"])
                start_time = candidate.match.start_frame * hop
                end_time = candidate.match.end_frame * hop
                melody_cost = float(candidate.match.cost)
            else:
                start_time = float(line["start_time"])
                end_time = float(line.get("end_time", start_time))
                melody_cost = fallback_cost
            expanded.append(
                LyricPosition(
                    line_index=line_index,
                    start_time=start_time,
                    end_time=end_time,
                    melody_cost=melody_cost,
                    source=source,
                    lyric_text=str(line.get("text", "")).strip(),
                    outcome_key=_lyric_outcome_key(lines, line_index),
                    song_id=song_id,
                )
            )

    by_outcome: dict[tuple[str, str], LyricPosition] = {}
    for position in expanded:
        song_id = position.song_id or next(iter(metadata_by_song))
        outcome_key = position.outcome_key or _lyric_outcome_key(
            metadata_by_song[song_id]["lrc_lines"],
            position.line_index,
        )
        key = (song_id, outcome_key)
        existing = by_outcome.get(key)
        if existing is None or position.melody_cost < existing.melody_cost:
            by_outcome[key] = position
    return sorted(by_outcome.values(), key=lambda item: item.melody_cost)


def _lyric_text_for_index(lines: list[dict], line_index: int) -> str:
    line = next((item for item in lines if int(item["index"]) == line_index), None)
    return str(line.get("text", "")).strip() if line else ""


def _lyric_outcome_key(lines: list[dict], line_index: int) -> str:
    current = next((line for line in lines if int(line["index"]) == line_index), None)
    following = next((line for line in lines if int(line["index"]) == line_index + 1), None)
    return f"{current.get('text', '') if current else ''}\n{following.get('text', '') if following else ''}"


def _lyrics_for_index(lines: list[dict], index: int) -> dict:
    current = next((line for line in lines if int(line["index"]) == index), None)
    if current is None:
        raise ValueError(f"Unknown lyric index selected by phrase matcher: {index}")
    next_line = next((line for line in lines if int(line["index"]) == index + 1), None)
    return {
        "current_lyric_index": int(current["index"]),
        "current_lyric_text": current["text"],
        "next_lyric_index": int(next_line["index"]) if next_line else None,
        "next_lyric_text": next_line["text"] if next_line else None,
        "next_lyric_start_time": float(next_line["start_time"]) if next_line else None,
    }


def _phrase_score(cost: float, settings: dict) -> float:
    return max(0.0, min(1.0, 1.0 - cost / float(settings["max_cost"])))


def _hybrid_rejected(
    reason: str,
    diagnostics: dict,
    frame_candidates: list[Candidate],
    phrase_candidates: list[PhraseCandidate],
    best_song: str | None = None,
    recognition_status: str = "rejected",
) -> dict:
    if best_song is None:
        if phrase_candidates:
            best_song = phrase_candidates[0].metadata["song_id"]
        elif frame_candidates:
            best_song = frame_candidates[0].metadata["song_id"]
    return {
        "accepted": False,
        "recognition_status": recognition_status,
        "position_resolved": False,
        "song_id": None,
        "matched_start_time": None,
        "matched_end_time": None,
        "current_lyric_index": None,
        "current_lyric_text": None,
        "next_lyric_index": None,
        "next_lyric_text": None,
        "next_lyric_start_time": None,
        "score": 0.0,
        "top2_margin": None,
        "reason": reason,
        "best_candidate_song_id": best_song,
        "candidate_count": len(frame_candidates) + len(phrase_candidates),
        "diagnostics": diagnostics,
    }


def warm_recognition_database(database_dir: Path, config: dict) -> None:
    """Load immutable song features and phrase contours before the first request."""
    _database_songs(database_dir, config)


def _database_songs(database_dir: Path, config: dict) -> tuple[DatabaseSong, ...]:
    resolved = database_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {resolved}")
    metadata_paths = sorted(resolved.glob("*.json"))
    if not metadata_paths:
        raise ValueError(f"No song metadata JSON files found in {resolved}")
    signature_paths = sorted((*metadata_paths, *resolved.glob("*.npz")))
    signature = tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in signature_paths
    )
    return _load_database_songs(
        str(resolved),
        signature,
        int(config["phrase_matching"]["contour_points"]),
        float(config["pitch"]["hop_seconds"]),
    )


@lru_cache(maxsize=4)
def _load_database_songs(
    database_dir: str,
    signature: tuple[tuple[str, int, int], ...],
    contour_points: int,
    pitch_hop_seconds: float,
) -> tuple[DatabaseSong, ...]:
    # `signature` deliberately participates in the cache key so rebuilding any
    # JSON/NPZ file invalidates this snapshot without a process restart.
    del signature
    directory = Path(database_dir)
    phrase_config = {
        "pitch": {"hop_seconds": pitch_hop_seconds},
        "phrase_matching": {"contour_points": contour_points},
    }
    songs: list[DatabaseSong] = []
    for metadata_path in sorted(directory.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = {"song_id", "features_file", "feature_hop_seconds", "lrc_lines"}
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"Invalid database metadata {metadata_path}: missing {', '.join(sorted(missing))}")
        feature_path = directory / metadata["features_file"]
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file referenced by {metadata_path} does not exist: {feature_path}")
        reference = _load_features(feature_path)
        for values in vars(reference).values():
            if isinstance(values, np.ndarray):
                values.setflags(write=False)
        songs.append(
            DatabaseSong(
                metadata=metadata,
                reference=reference,
                phrases=prepare_lyric_phrases(reference, metadata["lrc_lines"], phrase_config),
            )
        )
    return tuple(songs)


def _load_features(path: Path) -> PitchFeatures:
    with np.load(path) as values:
        required = ("time", "pitch", "relative_pitch", "delta_pitch", "voiced", "confidence", "onset_strength")
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"Invalid feature file {path}: missing {', '.join(missing)}")
        return PitchFeatures(**{name: values[name] for name in required})


def _rejected_payload(decision, best: Candidate | None, candidate_count: int) -> dict:
    return {
        "accepted": False,
        "recognition_status": "rejected",
        "position_resolved": False,
        "song_id": None,
        "matched_start_time": None,
        "matched_end_time": None,
        "current_lyric_index": None,
        "current_lyric_text": None,
        "next_lyric_index": None,
        "next_lyric_text": None,
        "next_lyric_start_time": None,
        "score": round(decision.score, 4),
        "top2_margin": round(decision.margin, 4) if decision.margin is not None else None,
        "reason": decision.reason,
        "best_candidate_song_id": best.metadata["song_id"] if best else None,
        "candidate_count": candidate_count,
    }


if __name__ == "__main__":
    main()
