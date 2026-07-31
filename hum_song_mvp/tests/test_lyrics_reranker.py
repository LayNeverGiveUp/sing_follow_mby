import pytest

from src.lyrics_reranker import LyricPosition, build_lyric_window, normalize_lyrics, rerank_lyric_positions


def positions():
    return [
        LyricPosition(6, 30.0, 35.0, 0.05, "frame", "一杯敬朝阳一杯敬月光"),
        LyricPosition(18, 90.0, 95.0, 0.06, "frame", "一杯敬故乡一杯敬远方"),
    ]


def settings():
    return {
        "min_lexical_characters": 2,
        "character_weight": 0.45,
        "pinyin_weight": 0.35,
        "discriminative_weight": 0.20,
        "min_lyrics_score": 0.45,
        "min_lyrics_margin": 0.12,
        "min_discriminative_score": 0.20,
    }


def test_reranker_selects_different_lyrics_on_same_melody():
    result = rerank_lyric_positions("一杯敬故乡，一杯敬远方", positions(), settings())
    assert result.reason is None
    assert result.selected is not None
    assert result.selected.line_index == 18


def test_reranker_tolerates_homophone_errors():
    result = rerank_lyric_positions("一杯敬越光", positions(), settings())
    assert result.reason is None
    assert result.selected is not None
    assert result.selected.line_index == 6


def test_reranker_rejects_humming_without_lexical_content():
    result = rerank_lyric_positions("啦啦啦啊哦", positions(), settings())
    assert result.selected is None
    assert result.reason == "asr_no_lexical_content"


def test_reranker_rejects_words_shared_by_all_candidates():
    result = rerank_lyric_positions("一杯敬", positions(), settings())
    assert result.selected is None
    assert result.reason in {"lyrics_margin_too_small", "lyrics_no_discriminative_evidence"}


def test_reranker_uses_melody_to_break_identical_lyric_tie():
    repeated = [
        LyricPosition(0, 10.0, 12.0, 0.40, "phrase", "如果有一天我变得很有钱"),
        LyricPosition(4, 30.0, 32.0, 0.62, "phrase", "如果有一天我变得很有钱"),
    ]
    active_settings = settings()
    active_settings.update(
        {
            "melody_tiebreak_enabled": True,
            "melody_tiebreak_lexical_epsilon": 0.001,
            "melody_tiebreak_min_cost_margin": 0.10,
        }
    )

    result = rerank_lyric_positions("如果有一天我变得很有钱", repeated, active_settings)

    assert result.reason is None
    assert result.selected is not None
    assert result.selected.line_index == 0
    assert result.margin == pytest.approx(0.22)


def test_build_lyric_window_uses_intersecting_lines():
    lines = [
        {"index": 0, "start_time": 10.0, "end_time": 12.0, "text": "第一句"},
        {"index": 1, "start_time": 12.0, "end_time": 15.0, "text": "第二句"},
        {"index": 2, "start_time": 15.0, "end_time": 18.0, "text": "第三句"},
    ]
    assert build_lyric_window(lines, 11.5, 15.2) == "第一句第二句第三句"
    assert normalize_lyrics("啦啦啦，一杯敬月光！") == "一杯敬月光"
