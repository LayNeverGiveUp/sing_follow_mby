import numpy as np

from src.build_database import filter_lyric_lines, refine_line_end_times
from src.pitch_extractor import PitchFeatures


def test_filter_lyric_lines_removes_credits_and_out_of_range_timestamps():
    lines = [
        {"index": 0, "start_time": 1.0, "text": "第一句"},
        {"index": 1, "start_time": 3.0, "text": "鼓Drums：张三"},
        {"index": 2, "start_time": 4.0, "text": "第二句"},
        {"index": 3, "start_time": 12.0, "text": "人声编辑Vocal Editing：李四"},
    ]

    filtered = filter_lyric_lines(lines, "示例歌曲", duration=10.0)

    assert [line["text"] for line in filtered] == ["第一句", "第二句"]
    assert [line["index"] for line in filtered] == [0, 1]
    assert filtered[0]["end_time"] == 4.0
    assert filtered[1]["end_time"] == 10.0


def test_refine_line_end_times_removes_trailing_instrumental_gap():
    features = PitchFeatures(
        time=np.arange(20, dtype=np.float32) * 0.1,
        pitch=np.full(20, np.nan, dtype=np.float32),
        relative_pitch=np.full(20, np.nan, dtype=np.float32),
        delta_pitch=np.zeros(20, dtype=np.float32),
        voiced=np.array([False, True, True, True, True, False] + [False] * 6 + [True, True, True] + [False] * 5),
        confidence=np.zeros(20, dtype=np.float32),
        onset_strength=np.zeros(20, dtype=np.float32),
    )
    config = {"pitch": {"hop_seconds": 0.1}, "lyric_segmentation": {"min_voiced_run_seconds": 0.2, "max_intra_phrase_silence_seconds": 0.5, "vocal_end_padding_seconds": 0.2, "min_nominal_line_duration_to_trim_seconds": 1.0, "min_trailing_silence_to_trim_seconds": 0.5}}
    lines = [{"index": 0, "start_time": 0.0, "end_time": 1.5, "text": "一句"}]

    refined = refine_line_end_times(lines, features, config, duration=1.5)

    assert refined[0]["nominal_end_time"] == 1.5
    assert refined[0]["end_time"] == 0.7
    assert refined[0]["vocal_end_source"] == "f0_long_silence_trim"
