from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtw_matcher import DtwResult
from .pitch_extractor import PitchFeatures


@dataclass(frozen=True)
class ConfidenceDecision:
    accepted: bool
    score: float
    margin: float | None
    reason: str | None


def decide(query: PitchFeatures, best: DtwResult | None, second_cost: float | None, config: dict) -> ConfidenceDecision:
    rules = config["confidence"]
    hop = float(config["pitch"]["hop_seconds"])
    voiced_seconds = float(np.count_nonzero(query.voiced)) * hop
    finite_pitch = query.pitch[np.isfinite(query.pitch)]
    pitch_range = float(np.ptp(finite_pitch)) if finite_pitch.size else 0.0
    if voiced_seconds < float(rules["min_voiced_seconds"]):
        margin = None if second_cost is None or best is None else second_cost - best.normalized_cost
        score = 0.0 if best is None else max(0.0, min(1.0, 1.0 - best.normalized_cost / float(rules["score_cost_scale"])))
        return ConfidenceDecision(False, score, margin, "insufficient_voiced_audio")
    if pitch_range < float(rules["min_pitch_range_semitones"]):
        return ConfidenceDecision(False, 0.0, None, "insufficient_pitch_variation")
    if best is None:
        return ConfidenceDecision(False, 0.0, None, "no_valid_dtw_path")
    if best.paired_voiced_seconds < float(rules["min_paired_voiced_seconds"]):
        return ConfidenceDecision(False, 0.0, None, "insufficient_paired_voiced_alignment")
    if best.query_voiced_coverage < float(rules["min_query_voiced_coverage"]):
        return ConfidenceDecision(False, 0.0, None, "insufficient_query_voiced_coverage")
    score = max(0.0, min(1.0, 1.0 - best.normalized_cost / float(rules["score_cost_scale"])))
    margin = None if second_cost is None else second_cost - best.normalized_cost
    if best.normalized_cost >= float(rules["absolute_cost_threshold"]):
        return ConfidenceDecision(False, score, margin, "cost_above_threshold")
    if margin is not None and margin <= float(rules["margin_threshold"]):
        return ConfidenceDecision(False, score, margin, "top2_margin_too_small")
    return ConfidenceDecision(True, score, margin, None)
