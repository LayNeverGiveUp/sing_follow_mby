from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from pypinyin import Style, lazy_pinyin
import soundfile as sf

from .lyrics_asr import AsrTranscript, AsrWord


_CONTENT_RE = re.compile(r"[0-9a-z\u4e00-\u9fff]")


@dataclass(frozen=True)
class LyricToken:
    text: str
    pinyin: str
    line_index: int
    position_in_line: int


@dataclass(frozen=True)
class TimedToken:
    text: str
    pinyin: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class TokenMatch:
    lyric_index: int
    asr_index: int
    exact: bool


def validate_song_alignment(
    lrc_lines: list[dict],
    transcript: AsrTranscript,
    duration: float,
    settings: dict,
) -> dict:
    """Compare timestamped LRC against word-timed singing ASR output."""
    lyric_tokens, line_lengths = _build_lyric_tokens(lrc_lines)
    asr_tokens = _build_asr_tokens(transcript.words)
    if not lyric_tokens:
        raise ValueError("LRC contains no alignable lyric characters")
    if not asr_tokens:
        raise ValueError("ASR result contains no word timestamps; alignment validation cannot continue")

    matches = _align_tokens(lyric_tokens, asr_tokens)
    matched_lyric_indices = {match.lyric_index for match in matches}
    exact_matches = sum(match.exact for match in matches)
    character_coverage = len(matched_lyric_indices) / len(lyric_tokens)
    exact_character_coverage = exact_matches / len(lyric_tokens)
    asr_character_coverage = len({match.asr_index for match in matches}) / len(asr_tokens)

    by_line: dict[int, list[TokenMatch]] = {}
    for match in matches:
        line_index = lyric_tokens[match.lyric_index].line_index
        by_line.setdefault(line_index, []).append(match)
    line_rows = _build_line_rows(lrc_lines, line_lengths, lyric_tokens, asr_tokens, by_line, settings)
    anchored = [row for row in line_rows if row["observed_start_time"] is not None]
    minimum_anchors = int(settings.get("min_anchor_lines", 6))
    if len(anchored) < minimum_anchors:
        return _insufficient_report(
            lrc_lines,
            transcript,
            line_rows,
            character_coverage,
            exact_character_coverage,
            asr_character_coverage,
            minimum_anchors,
        )

    lrc_times = np.asarray([row["lrc_start_time"] for row in anchored], dtype=np.float64)
    observed_times = np.asarray([row["observed_start_time"] for row in anchored], dtype=np.float64)
    weights = np.asarray([max(0.1, row["character_coverage"]) for row in anchored], dtype=np.float64)
    slope, intercept, inliers = _robust_linear_fit(lrc_times, observed_times, weights, settings)
    for anchor_index, row in enumerate(anchored):
        predicted = slope * row["lrc_start_time"] + intercept
        row["predicted_start_time"] = round(float(predicted), 3)
        row["residual_seconds"] = round(float(row["observed_start_time"] - predicted), 3)
        row["fit_inlier"] = bool(inliers[anchor_index])

    residuals = np.abs(np.asarray([row["residual_seconds"] for row in anchored], dtype=np.float64))
    median_error = float(np.median(residuals))
    p95_error = float(np.percentile(residuals, 95))
    max_error = float(np.max(residuals))
    line_coverage = len(anchored) / len(lrc_lines)
    discontinuities = _find_discontinuities(anchored, settings)
    unmatched_spans = _find_unmatched_spans(line_rows, settings)
    metrics = {
        "lrc_character_count": len(lyric_tokens),
        "asr_character_count": len(asr_tokens),
        "matched_character_count": len(matched_lyric_indices),
        "character_coverage": round(character_coverage, 4),
        "exact_character_coverage": round(exact_character_coverage, 4),
        "asr_character_coverage": round(asr_character_coverage, 4),
        "anchored_line_count": len(anchored),
        "lrc_line_count": len(lrc_lines),
        "line_coverage": round(line_coverage, 4),
        "median_absolute_error_seconds": round(median_error, 3),
        "p95_absolute_error_seconds": round(p95_error, 3),
        "max_absolute_error_seconds": round(max_error, 3),
        "discontinuity_count": len(discontinuities),
        "unmatched_span_count": len(unmatched_spans),
    }
    mapping = {
        "formula": "audio_time = slope * lrc_time + intercept_seconds",
        "slope": round(float(slope), 8),
        "intercept_seconds": round(float(intercept), 3),
        "drift_ratio_delta": round(float(slope - 1.0), 8),
        "drift_seconds_over_song": round(float((slope - 1.0) * duration), 3),
    }
    verdict, reasons = _decide_verdict(metrics, mapping, settings)
    can_auto_correct = _can_auto_correct(verdict, metrics, mapping, settings)
    return {
        "verdict": verdict,
        "reasons": reasons,
        "can_auto_correct": can_auto_correct,
        "duration_seconds": round(float(duration), 3),
        "asr": {
            "source": transcript.source,
            "text": transcript.text,
            "word_count": len(transcript.words),
        },
        "metrics": metrics,
        "time_mapping": mapping,
        "discontinuities": discontinuities,
        "unmatched_spans": unmatched_spans,
        "lines": line_rows,
    }


def corrected_lrc_text(lrc_lines: list[dict], report: dict) -> str:
    if not report.get("can_auto_correct"):
        raise ValueError("Alignment report is not safe for automatic LRC correction")
    mapping = report["time_mapping"]
    slope = float(mapping["slope"])
    intercept = float(mapping["intercept_seconds"])
    output = []
    for line in lrc_lines:
        corrected = max(0.0, slope * float(line["start_time"]) + intercept)
        output.append(f"{_format_lrc_timestamp(corrected)}{str(line['text']).strip()}")
    return "\n".join(output) + "\n"


def write_review_bundle(
    output_dir: Path,
    audio_samples: np.ndarray,
    sample_rate: int,
    lrc_lines: list[dict],
    report: dict,
    clip_count: int = 8,
) -> list[dict]:
    """Write small anchor clips and an HTML page for human verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "review_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in report.get("lines", []) if row.get("observed_start_time") is not None]
    selected = _select_review_rows(rows, clip_count)
    manifest: list[dict] = []
    duration = audio_samples.size / float(sample_rate)
    for row in selected:
        index = int(row["line_index"])
        start = max(0.0, float(row["observed_start_time"]) - 0.6)
        next_line = lrc_lines[index + 1] if index + 1 < len(lrc_lines) else None
        nominal_length = (
            float(next_line["start_time"]) - float(lrc_lines[index]["start_time"])
            if next_line is not None
            else 6.0
        )
        end = min(duration, start + min(12.0, max(3.0, nominal_length + 1.2)))
        filename = f"line_{index:03d}.wav"
        sf.write(clips_dir / filename, audio_samples[int(start * sample_rate) : int(end * sample_rate)], sample_rate)
        manifest.append(
            {
                "line_index": index,
                "text": row["text"],
                "lrc_start_time": row["lrc_start_time"],
                "observed_start_time": row["observed_start_time"],
                "residual_seconds": row.get("residual_seconds"),
                "character_coverage": row["character_coverage"],
                "audio": f"review_clips/{filename}",
            }
        )
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "review.html").write_text(_review_html(report, manifest), encoding="utf-8")
    return manifest


def transcript_to_dict(transcript: AsrTranscript) -> dict:
    return {
        "text": transcript.text,
        "source": transcript.source,
        "confidence": transcript.confidence,
        "words": [
            {
                "text": word.text,
                "start_time": word.start_time,
                "end_time": word.end_time,
                "confidence": word.confidence,
            }
            for word in transcript.words
        ],
    }


def transcript_from_dict(payload: dict) -> AsrTranscript:
    words = tuple(
        AsrWord(
            text=str(word["text"]),
            start_time=float(word["start_time"]),
            end_time=float(word["end_time"]),
            confidence=float(word["confidence"]) if word.get("confidence") is not None else None,
        )
        for word in payload.get("words", [])
    )
    return AsrTranscript(
        text=str(payload.get("text", "")),
        source=str(payload.get("source", "saved_transcript")),
        confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        words=words,
    )


def _build_lyric_tokens(lines: list[dict]) -> tuple[list[LyricToken], dict[int, int]]:
    tokens: list[LyricToken] = []
    lengths: dict[int, int] = {}
    for line in lines:
        index = int(line["index"])
        characters = _content_characters(str(line.get("text", "")))
        lengths[index] = len(characters)
        tokens.extend(
            LyricToken(character, _pinyin(character), index, position)
            for position, character in enumerate(characters)
        )
    return tokens, lengths


def _build_asr_tokens(words: tuple[AsrWord, ...]) -> list[TimedToken]:
    tokens: list[TimedToken] = []
    for word in words:
        characters = _content_characters(word.text)
        if not characters:
            continue
        duration = max(0.0, word.end_time - word.start_time)
        for position, character in enumerate(characters):
            start = word.start_time + duration * position / len(characters)
            end = word.start_time + duration * (position + 1) / len(characters)
            tokens.append(TimedToken(character, _pinyin(character), start, end))
    return tokens


def _align_tokens(lyrics: list[LyricToken], asr: list[TimedToken]) -> list[TokenMatch]:
    gap = -0.9
    previous = np.arange(len(asr) + 1, dtype=np.float32) * gap
    pointers = np.zeros((len(lyrics) + 1, len(asr) + 1), dtype=np.uint8)
    pointers[0, 1:] = 3
    pointers[1:, 0] = 2
    for lyric_index, lyric in enumerate(lyrics, start=1):
        current = np.empty(len(asr) + 1, dtype=np.float32)
        current[0] = lyric_index * gap
        for asr_index, asr_token in enumerate(asr, start=1):
            similarity = 2.5 if lyric.text == asr_token.text else (1.2 if lyric.pinyin == asr_token.pinyin else -1.6)
            diagonal = previous[asr_index - 1] + similarity
            up = previous[asr_index] + gap
            left = current[asr_index - 1] + gap
            if diagonal >= up and diagonal >= left:
                current[asr_index] = diagonal
                pointers[lyric_index, asr_index] = 1
            elif up >= left:
                current[asr_index] = up
                pointers[lyric_index, asr_index] = 2
            else:
                current[asr_index] = left
                pointers[lyric_index, asr_index] = 3
        previous = current

    lyric_index = len(lyrics)
    asr_index = len(asr)
    matches: list[TokenMatch] = []
    while lyric_index > 0 or asr_index > 0:
        direction = pointers[lyric_index, asr_index]
        if direction == 1:
            lyric_token = lyrics[lyric_index - 1]
            asr_token = asr[asr_index - 1]
            if lyric_token.text == asr_token.text or lyric_token.pinyin == asr_token.pinyin:
                matches.append(
                    TokenMatch(
                        lyric_index=lyric_index - 1,
                        asr_index=asr_index - 1,
                        exact=lyric_token.text == asr_token.text,
                    )
                )
            lyric_index -= 1
            asr_index -= 1
        elif direction == 2:
            lyric_index -= 1
        elif direction == 3:
            asr_index -= 1
        else:
            break
    matches.reverse()
    return matches


def _build_line_rows(lines, line_lengths, lyric_tokens, asr_tokens, by_line, settings) -> list[dict]:
    rows = []
    minimum_matches = int(settings.get("min_matched_characters_per_line", 2))
    minimum_coverage = float(settings.get("min_line_character_coverage_for_anchor", 0.20))
    token_offsets: dict[int, list[tuple[int, TimedToken, bool]]] = {}
    for line_index, matches in by_line.items():
        token_offsets[line_index] = [
            (
                lyric_tokens[match.lyric_index].position_in_line,
                asr_tokens[match.asr_index],
                match.exact,
            )
            for match in matches
        ]
    for line in lines:
        index = int(line["index"])
        length = line_lengths.get(index, 0)
        values = token_offsets.get(index, [])
        coverage = len({value[0] for value in values}) / length if length else 0.0
        observed = None
        if len(values) >= minimum_matches and coverage >= minimum_coverage:
            observed = _estimate_line_start(values, line, length)
        rows.append(
            {
                "line_index": index,
                "text": str(line.get("text", "")),
                "lrc_start_time": round(float(line["start_time"]), 3),
                "observed_start_time": round(float(observed), 3) if observed is not None else None,
                "predicted_start_time": None,
                "residual_seconds": None,
                "fit_inlier": None,
                "character_count": length,
                "matched_character_count": len({value[0] for value in values}),
                "exact_match_count": sum(value[2] for value in values),
                "character_coverage": round(coverage, 4),
            }
        )
    return rows


def _estimate_line_start(values: list[tuple[int, TimedToken, bool]], line: dict, line_length: int) -> float:
    values = sorted(values, key=lambda value: value[0])
    slopes = []
    for first, second in zip(values, values[1:]):
        position_delta = second[0] - first[0]
        time_delta = second[1].start_time - first[1].start_time
        if 0 < position_delta <= 6 and time_delta > 0:
            slopes.append(time_delta / position_delta)
    nominal_duration = max(0.5, float(line.get("end_time", line["start_time"])) - float(line["start_time"]))
    fallback = nominal_duration / max(1, line_length)
    step = float(np.median(slopes)) if slopes else fallback
    step = min(0.8, max(0.04, step))
    estimates = [token.start_time - position * step for position, token, _ in values]
    weights = [1.0 if exact else 0.7 for _, _, exact in values]
    return max(0.0, _weighted_median(np.asarray(estimates), np.asarray(weights)))


def _robust_linear_fit(x, y, weights, settings) -> tuple[float, float, np.ndarray]:
    inliers = np.ones(x.size, dtype=bool)
    minimum_residual = float(settings.get("fit_minimum_outlier_seconds", 0.75))
    for _ in range(5):
        if np.count_nonzero(inliers) < 2:
            break
        coefficients = np.polyfit(x[inliers], y[inliers], 1, w=np.sqrt(weights[inliers]))
        slope, intercept = float(coefficients[0]), float(coefficients[1])
        residuals = y - (slope * x + intercept)
        centered = residuals[inliers] - np.median(residuals[inliers])
        mad = float(np.median(np.abs(centered)))
        limit = max(minimum_residual, 3.5 * 1.4826 * mad)
        updated = np.abs(residuals) <= limit
        if np.array_equal(updated, inliers) or np.count_nonzero(updated) < 2:
            break
        inliers = updated
    coefficients = np.polyfit(x[inliers], y[inliers], 1, w=np.sqrt(weights[inliers]))
    return float(coefficients[0]), float(coefficients[1]), inliers


def _find_discontinuities(anchored: list[dict], settings: dict) -> list[dict]:
    threshold = float(settings.get("discontinuity_jump_seconds", 1.5))
    result = []
    for previous, current in zip(anchored, anchored[1:]):
        jump = abs(float(current["residual_seconds"]) - float(previous["residual_seconds"]))
        if jump >= threshold:
            result.append(
                {
                    "after_line_index": previous["line_index"],
                    "before_line_index": current["line_index"],
                    "residual_jump_seconds": round(jump, 3),
                }
            )
    return result


def _find_unmatched_spans(rows: list[dict], settings: dict) -> list[dict]:
    minimum_lines = int(settings.get("unmatched_span_min_lines", 3))
    spans = []
    start = None
    for position, row in enumerate([*rows, {"observed_start_time": 0.0}]):
        if row["observed_start_time"] is None and start is None:
            start = position
        elif row["observed_start_time"] is not None and start is not None:
            if position - start >= minimum_lines:
                spans.append(
                    {
                        "start_line_index": rows[start]["line_index"],
                        "end_line_index": rows[position - 1]["line_index"],
                        "line_count": position - start,
                    }
                )
            start = None
    return spans


def _decide_verdict(metrics: dict, mapping: dict, settings: dict) -> tuple[str, list[str]]:
    pass_failures = _threshold_failures(metrics, mapping, settings, "pass")
    if not pass_failures:
        return "pass", ["LRC text and timestamps are consistent with this audio version"]
    warning_failures = _threshold_failures(metrics, mapping, settings, "warning")
    if not warning_failures:
        return "warning", pass_failures
    return "fail", warning_failures


def _threshold_failures(metrics: dict, mapping: dict, settings: dict, level: str) -> list[str]:
    failures = []
    checks = (
        (metrics["character_coverage"] >= float(settings[f"min_character_coverage_{level}"]), "character coverage is too low"),
        (metrics["line_coverage"] >= float(settings[f"min_line_coverage_{level}"]), "too few lyric lines have reliable ASR anchors"),
        (metrics["median_absolute_error_seconds"] <= float(settings[f"max_median_error_seconds_{level}"]), "median timestamp error is too large"),
        (metrics["p95_absolute_error_seconds"] <= float(settings[f"max_p95_error_seconds_{level}"]), "timestamp errors vary too much across the song"),
        (abs(mapping["drift_ratio_delta"]) <= float(settings[f"max_drift_ratio_delta_{level}"]), "audio and LRC clocks drift too far apart"),
        (metrics["discontinuity_count"] <= int(settings[f"max_discontinuities_{level}"]), "timeline has abrupt jumps consistent with a different edit"),
        (metrics["unmatched_span_count"] <= int(settings[f"max_unmatched_spans_{level}"]), "long lyric sections are missing from ASR alignment"),
    )
    return [message for passed, message in checks if not passed]


def _can_auto_correct(verdict: str, metrics: dict, mapping: dict, settings: dict) -> bool:
    return bool(
        verdict in {"pass", "warning"}
        and metrics["character_coverage"] >= float(settings.get("min_character_coverage_auto_correct", 0.60))
        and metrics["line_coverage"] >= float(settings.get("min_line_coverage_auto_correct", 0.60))
        and metrics["discontinuity_count"] == 0
        and metrics["unmatched_span_count"] == 0
        and abs(mapping["drift_ratio_delta"]) <= float(settings.get("max_drift_ratio_delta_auto_correct", 0.03))
    )


def _insufficient_report(lines, transcript, line_rows, char_cov, exact_cov, asr_cov, minimum_anchors) -> dict:
    anchored = sum(row["observed_start_time"] is not None for row in line_rows)
    return {
        "verdict": "fail",
        "reasons": [f"only {anchored} lyric lines have ASR time anchors; at least {minimum_anchors} are required"],
        "can_auto_correct": False,
        "duration_seconds": None,
        "asr": {"source": transcript.source, "text": transcript.text, "word_count": len(transcript.words)},
        "metrics": {
            "character_coverage": round(char_cov, 4),
            "exact_character_coverage": round(exact_cov, 4),
            "asr_character_coverage": round(asr_cov, 4),
            "anchored_line_count": anchored,
            "lrc_line_count": len(lines),
            "line_coverage": round(anchored / len(lines), 4) if lines else 0.0,
        },
        "time_mapping": None,
        "discontinuities": [],
        "unmatched_spans": _find_unmatched_spans(line_rows, {"unmatched_span_min_lines": 3}),
        "lines": line_rows,
    }


def _select_review_rows(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    count = min(count, len(rows))
    indices = set(np.linspace(0, len(rows) - 1, count, dtype=int).tolist())
    worst = sorted(
        range(len(rows)),
        key=lambda index: abs(float(rows[index].get("residual_seconds") or 0.0)),
        reverse=True,
    )
    for index in worst[: max(1, count // 3)]:
        indices.add(index)
    selected = sorted((rows[index] for index in indices), key=lambda row: row["line_index"])
    return selected[:count]


def _review_html(report: dict, manifest: list[dict]) -> str:
    table_rows = []
    for item in manifest:
        table_rows.append(
            "<tr>"
            f"<td>{item['line_index']}</td>"
            f"<td>{html.escape(str(item['text']))}</td>"
            f"<td>{item['lrc_start_time']}</td>"
            f"<td>{item['observed_start_time']}</td>"
            f"<td>{item['residual_seconds']}</td>"
            f"<td><audio controls preload='none' src='{html.escape(item['audio'])}'></audio></td>"
            "</tr>"
        )
    verdict = html.escape(str(report.get("verdict", "unknown")))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>LRC 版本校验</title>
<style>body{{font-family:system-ui;margin:28px;max-width:1100px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}.verdict{{font-size:22px;font-weight:700}}</style>
</head><body><h1>LRC 与音频版本校验</h1><p class="verdict">结论：{verdict}</p>
<p>以下片段包含全曲均匀采样点和误差最大的锚点，请确认播放器开头是否与显示歌词一致。</p>
<table><thead><tr><th>行</th><th>歌词</th><th>LRC 时间</th><th>ASR 时间</th><th>拟合残差</th><th>试听</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""


def _content_characters(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [character for character in normalized if _CONTENT_RE.fullmatch(character)]


def _pinyin(character: str) -> str:
    values = lazy_pinyin(character, style=Style.NORMAL, errors="default")
    return values[0] if values else character


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = ordered_weights.sum() / 2.0
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")])


def _format_lrc_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"[{minutes:02d}:{remaining:05.2f}]"
