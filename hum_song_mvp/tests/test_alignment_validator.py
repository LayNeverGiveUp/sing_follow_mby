from __future__ import annotations

from src.alignment_validator import corrected_lrc_text, validate_song_alignment
from src.config import load_config
from src.lyrics_asr import AsrTranscript, AsrWord


TEXTS = [
    "清晨走过安静街道",
    "晚风吹动远方树梢",
    "星光落在你的眼角",
    "我们听见岁月歌谣",
    "山川穿过漫长隧道",
    "海浪拥抱孤独小岛",
    "故事写进温柔怀抱",
    "明天依然值得寻找",
]


def lrc_lines():
    return [
        {
            "index": index,
            "start_time": 10.0 + index * 8.0,
            "end_time": 18.0 + index * 8.0,
            "text": text,
        }
        for index, text in enumerate(TEXTS)
    ]


def transcript(slope=1.0, intercept=0.0, step_after_line=None, step_seconds=0.0):
    words = []
    for line in lrc_lines():
        line_shift = step_seconds if step_after_line is not None and line["index"] >= step_after_line else 0.0
        start = slope * line["start_time"] + intercept + line_shift
        for position, character in enumerate(line["text"]):
            char_start = start + position * 0.28
            words.append(AsrWord(character, char_start, char_start + 0.22, 0.9))
    return AsrTranscript("".join(TEXTS), "synthetic", words=tuple(words))


def settings():
    return load_config()["alignment_validation"]


def test_global_offset_and_small_drift_pass_and_can_be_corrected():
    report = validate_song_alignment(lrc_lines(), transcript(1.002, 1.5), 80.0, settings())

    assert report["verdict"] == "pass"
    assert report["can_auto_correct"]
    assert report["metrics"]["character_coverage"] == 1.0
    assert abs(report["time_mapping"]["slope"] - 1.002) < 0.001
    assert abs(report["time_mapping"]["intercept_seconds"] - 1.5) < 0.1
    corrected = corrected_lrc_text(lrc_lines(), report)
    assert corrected.startswith("[00:11.52]清晨走过安静街道")


def test_mid_song_timeline_jump_is_not_silently_auto_corrected():
    report = validate_song_alignment(
        lrc_lines(),
        transcript(step_after_line=4, step_seconds=5.0),
        80.0,
        settings(),
    )

    assert report["verdict"] != "pass"
    assert not report["can_auto_correct"]
    assert report["metrics"]["discontinuity_count"] >= 1


def test_unrelated_asr_text_fails_with_insufficient_anchors():
    words = tuple(AsrWord(character, index * 0.2, index * 0.2 + 0.1) for index, character in enumerate("数字天气新闻测试内容完全不同"))
    report = validate_song_alignment(
        lrc_lines(),
        AsrTranscript("数字天气新闻测试内容完全不同", "synthetic", words=words),
        80.0,
        settings(),
    )

    assert report["verdict"] == "fail"
    assert not report["can_auto_correct"]
    assert report["metrics"]["anchored_line_count"] < settings()["min_anchor_lines"]
