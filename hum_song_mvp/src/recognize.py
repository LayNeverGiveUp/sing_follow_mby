from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio_io import load_mono_audio, trim_outer_silence
from .config import load_config
from .confidence import decide
from .dtw_matcher import DtwResult, subsequence_dtw
from .lyric_mapper import map_lyrics
from .phrase_matcher import PhraseMatch, match_lyric_phrases
from .pitch_extractor import PitchFeatures, extract_features


@dataclass(frozen=True)
class Candidate:
    metadata: dict
    result: DtwResult


@dataclass(frozen=True)
class PhraseCandidate:
    metadata: dict
    match: PhraseMatch


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
    return recognize_samples(samples, database_dir, config)


def recognize_samples(samples: np.ndarray, database_dir: Path, config: dict) -> dict:
    total_started = perf_counter()
    stage_started = perf_counter()
    trimmed, start_sample, _ = trim_outer_silence(samples, float(config["audio"]["trim_top_db"]))
    trim_ms = (perf_counter() - stage_started) * 1000
    stage_started = perf_counter()
    query = extract_features(trimmed, config)
    feature_ms = (perf_counter() - stage_started) * 1000
    candidate_ms: dict[str, float] = {}
    stage_started = perf_counter()
    candidates = _match_all(query, database_dir, config, candidate_ms)
    match_ms = (perf_counter() - stage_started) * 1000
    candidates.sort(key=lambda candidate: candidate.result.normalized_cost)
    if str(config["matching"].get("algorithm", "frame_dtw")) == "hybrid_phrase":
        phrase_started = perf_counter()
        phrase_candidates = _match_all_phrases(query, database_dir, config)
        phrase_ms = (perf_counter() - phrase_started) * 1000
        return _recognize_hybrid(
            samples,
            trimmed,
            start_sample,
            query,
            candidates,
            phrase_candidates,
            config,
            {
                "trim_silence": round(trim_ms, 1),
                "f0_and_onset": round(feature_ms, 1),
                "database_load_and_dtw": round(match_ms, 1),
                "phrase_contour_matching": round(phrase_ms, 1),
                "per_song_dtw": candidate_ms,
                "total_recognition": round((perf_counter() - total_started) * 1000, 1),
            },
        )
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


def _match_all(query: PitchFeatures, database_dir: Path, config: dict, candidate_ms: dict[str, float] | None = None) -> list[Candidate]:
    if not database_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {database_dir}")
    metadata_paths = sorted(database_dir.glob("*.json"))
    if not metadata_paths:
        raise ValueError(f"No song metadata JSON files found in {database_dir}")
    candidates = []
    for metadata_path in metadata_paths:
        started = perf_counter()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = {"song_id", "features_file", "feature_hop_seconds", "lrc_lines"}
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"Invalid database metadata {metadata_path}: missing {', '.join(sorted(missing))}")
        feature_path = database_dir / metadata["features_file"]
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file referenced by {metadata_path} does not exist: {feature_path}")
        reference = _load_features(feature_path)
        algorithm = str(config["matching"].get("algorithm", "frame_dtw"))
        if algorithm in {"frame_dtw", "hybrid_phrase"}:
            result = subsequence_dtw(query, reference, config)
        else:
            raise ValueError(f"Unsupported matching.algorithm: {algorithm}")
        if candidate_ms is not None:
            candidate_ms[metadata["song_id"]] = round((perf_counter() - started) * 1000, 1)
        if result is not None:
            candidates.append(Candidate(metadata, result))
    return candidates


def _match_all_phrases(query: PitchFeatures, database_dir: Path, config: dict) -> list[PhraseCandidate]:
    candidates: list[PhraseCandidate] = []
    for metadata_path in sorted(database_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reference = _load_features(database_dir / metadata["features_file"])
        candidates.extend(
            PhraseCandidate(metadata, match)
            for match in match_lyric_phrases(query, reference, metadata["lrc_lines"], config)
        )
    return sorted(candidates, key=lambda candidate: candidate.match.cost)


def _recognize_hybrid(
    samples: np.ndarray,
    trimmed: np.ndarray,
    start_sample: int,
    query: PitchFeatures,
    frame_candidates: list[Candidate],
    phrase_candidates: list[PhraseCandidate],
    config: dict,
    stage_ms: dict,
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
    phrase = phrase_candidates[0] if phrase_candidates else None
    phrase_margin = (
        phrase_candidates[1].match.cost - phrase.match.cost
        if phrase is not None and len(phrase_candidates) > 1
        else None
    )
    phrase_usable = phrase is not None and phrase.match.cost <= float(settings["max_cost"])

    route: str
    if frame is not None:
        frame_end = frame.result.end_frame * float(frame.metadata["feature_hop_seconds"])
        frame_lyrics = map_lyrics(
            frame.metadata["lrc_lines"],
            frame_end,
            float(config.get("lyric_mapping", {}).get("end_boundary_tolerance_seconds", 0.0)),
        )
        if (
            phrase_usable
            and phrase.metadata["song_id"] == frame.metadata["song_id"]
            and frame.result.query_voiced_coverage < float(settings["strong_frame_coverage"])
        ):
            selected_metadata = phrase.metadata
            selected_start = phrase.match.start_frame * float(selected_metadata["feature_hop_seconds"])
            selected_end = phrase.match.end_frame * float(selected_metadata["feature_hop_seconds"])
            selected_lyrics = _lyrics_for_index(selected_metadata["lrc_lines"], phrase.match.line_index)
            score = _phrase_score(phrase.match.cost, settings)
            route = "frame_song_phrase_line"
        else:
            selected_metadata = frame.metadata
            selected_start = frame.result.start_frame * float(selected_metadata["feature_hop_seconds"])
            selected_end = frame_end
            selected_lyrics = frame_lyrics
            score = max(
                0.0,
                min(1.0, 1.0 - frame.result.normalized_cost / float(config["confidence"]["score_cost_scale"])),
            )
            route = "trusted_frame"
    elif (
        phrase_usable
        and phrase_margin is not None
        and phrase_margin >= float(settings["min_standalone_margin"])
    ):
        selected_metadata = phrase.metadata
        selected_start = phrase.match.start_frame * float(selected_metadata["feature_hop_seconds"])
        selected_end = phrase.match.end_frame * float(selected_metadata["feature_hop_seconds"])
        selected_lyrics = _lyrics_for_index(selected_metadata["lrc_lines"], phrase.match.line_index)
        score = _phrase_score(phrase.match.cost, settings)
        route = "standalone_phrase"
    else:
        reason = "phrase_margin_too_small" if phrase_usable else "no_reliable_hybrid_candidate"
        return _hybrid_rejected(reason, diagnostics, frame_candidates, phrase_candidates)

    diagnostics["hybrid_route"] = route
    return {
        "accepted": True,
        "song_id": selected_metadata["song_id"],
        "matched_start_time": round(selected_start, 3),
        "matched_end_time": round(selected_end, 3),
        **selected_lyrics,
        "score": round(score, 4),
        "top2_margin": round(phrase_margin, 4) if phrase_margin is not None else None,
        "query_trim_start_time": round(start_sample / float(config["audio"]["sample_rate"]), 3),
        "diagnostics": diagnostics,
    }


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
) -> dict:
    best_song = None
    if phrase_candidates:
        best_song = phrase_candidates[0].metadata["song_id"]
    elif frame_candidates:
        best_song = frame_candidates[0].metadata["song_id"]
    return {
        "accepted": False,
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
