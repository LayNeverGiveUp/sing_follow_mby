import numpy as np

from src.config import load_config
from src.dtw_matcher import DtwResult
from src.lyrics_asr import AsrTranscript, DisabledLyricsAsr
from src.phrase_matcher import PhraseMatch
from src.pitch_extractor import PitchFeatures
from src.recognize import (
    Candidate,
    PhraseCandidate,
    SelectedPosition,
    _best_phrase_candidates_by_song,
    _recognize_hybrid,
    _resolve_cross_song_position,
    _resolve_repeated_position,
)


class FakeAsr:
    enabled = True
    status = "fake"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        return AsrTranscript(self.text, "fake", 0.9)


def metadata():
    return {
        "song_id": "test_song",
        "feature_hop_seconds": 0.025,
        "lrc_lines": [
            {"index": 0, "start_time": 10.0, "end_time": 12.0, "text": "一杯敬朝阳一杯敬月光"},
            {"index": 1, "start_time": 20.0, "end_time": 22.0, "text": "一杯敬故乡一杯敬远方"},
            {"index": 2, "start_time": 22.0, "end_time": 25.0, "text": "下一句"},
        ],
    }


def dtw_result(start_frame, end_frame, cost):
    return DtwResult(
        normalized_cost=cost,
        raw_normalized_cost=cost,
        start_frame=start_frame,
        end_frame=end_frame,
        path=[(0, start_frame), (1, end_frame)],
        speed_ratio=1.0,
        paired_voiced_seconds=2.0,
        query_voiced_coverage=1.0,
    )


def phrase_result(line_index, cost, start_frame=400, end_frame=480):
    return PhraseMatch(
        line_index=line_index,
        start_frame=start_frame,
        end_frame=end_frame,
        cost=cost,
        pitch_cost=cost,
        slope_cost=0.0,
        range_penalty=0.0,
        path_length=72,
        segment_coverage=1.0,
    )


def selected_position(meta):
    return SelectedPosition(
        metadata=meta,
        start_time=10.0,
        end_time=12.0,
        lyrics={
            "current_lyric_index": 0,
            "current_lyric_text": meta["lrc_lines"][0]["text"],
            "next_lyric_index": 1,
            "next_lyric_text": meta["lrc_lines"][1]["text"],
            "next_lyric_start_time": 20.0,
        },
        score=0.9,
        margin=0.01,
        route="trusted_frame",
    )


def ambiguous_frame_positions(meta):
    return {
        "test_song": [
            Candidate(meta, dtw_result(400, 480, 0.050)),
            Candidate(meta, dtw_result(800, 880, 0.055)),
        ]
    }


def resolution_config():
    config = load_config()
    config["position_resolution"]["require_phrase_support_for_frame_ambiguity"] = False
    return config


def test_real_lyrics_select_the_correct_repeated_position():
    meta = metadata()
    fake = FakeAsr("一杯敬故乡，一杯敬远方")
    resolved, reason = _resolve_repeated_position(
        np.ones(16000, dtype=np.float32) * 0.1,
        selected_position(meta),
        ambiguous_frame_positions(meta),
        [],
        resolution_config(),
        fake,
        {"stage_ms": {}},
    )

    assert reason is None
    assert resolved is not None
    assert resolved.lyrics["current_lyric_index"] == 1
    assert resolved.route == "trusted_frame_lyrics_rerank"
    assert fake.calls == 1


def test_humming_does_not_guess_a_repeated_position():
    meta = metadata()
    resolved, reason = _resolve_repeated_position(
        np.ones(16000, dtype=np.float32) * 0.1,
        selected_position(meta),
        ambiguous_frame_positions(meta),
        [],
        resolution_config(),
        FakeAsr("啦啦啦啊哦"),
        {"stage_ms": {}},
    )

    assert resolved is None
    assert reason == "asr_no_lexical_content"


def test_missing_asr_rejects_instead_of_using_melody_top1():
    meta = metadata()
    resolved, reason = _resolve_repeated_position(
        np.ones(16000, dtype=np.float32) * 0.1,
        selected_position(meta),
        ambiguous_frame_positions(meta),
        [],
        resolution_config(),
        DisabledLyricsAsr(),
        {"stage_ms": {}},
    )

    assert resolved is None
    assert reason == "asr_unavailable_for_ambiguous_melody"


def test_unique_position_does_not_call_asr():
    meta = metadata()
    fake = FakeAsr("不应该被调用")
    resolved, reason = _resolve_repeated_position(
        np.ones(16000, dtype=np.float32) * 0.1,
        selected_position(meta),
        {"test_song": [Candidate(meta, dtw_result(400, 480, 0.050))]},
        [],
        resolution_config(),
        fake,
        {"stage_ms": {}},
    )

    assert reason is None
    assert resolved is not None
    assert resolved.lyrics["current_lyric_index"] == 0
    assert fake.calls == 0


def test_phrase_song_margin_ignores_multiple_positions_in_the_same_song():
    first = metadata()
    second = {**metadata(), "song_id": "other_song"}
    candidates = [
        PhraseCandidate(first, phrase_result(0, 0.40)),
        PhraseCandidate(first, phrase_result(1, 0.41)),
        PhraseCandidate(second, phrase_result(0, 0.90)),
    ]

    grouped = _best_phrase_candidates_by_song(candidates)

    assert [candidate.metadata["song_id"] for candidate in grouped] == ["test_song", "other_song"]
    assert grouped[1].match.cost - grouped[0].match.cost == 0.5


def test_asr_reranks_only_melody_shortlisted_songs():
    wrong = {
        "song_id": "wrong_song",
        "feature_hop_seconds": 0.025,
        "lrc_lines": [
            {"index": 0, "start_time": 10.0, "end_time": 12.0, "text": "风吹过安静山岗"},
            {"index": 1, "start_time": 12.0, "end_time": 14.0, "text": "错误下一句"},
        ],
    }
    correct = {
        "song_id": "correct_song",
        "feature_hop_seconds": 0.025,
        "lrc_lines": [
            {"index": 0, "start_time": 20.0, "end_time": 23.0, "text": "如同昨夜天光乍破了远山的轮廓"},
            {"index": 1, "start_time": 23.0, "end_time": 25.0, "text": "正确下一句"},
        ],
    }
    candidates = [
        PhraseCandidate(wrong, phrase_result(0, 0.50)),
        PhraseCandidate(correct, phrase_result(0, 0.52, 800, 920)),
    ]
    diagnostics = {"stage_ms": {}}
    fake = FakeAsr("如同昨夜天光乍破了远山的轮廓")

    resolved, reason = _resolve_cross_song_position(
        np.ones(16000, dtype=np.float32) * 0.1,
        _best_phrase_candidates_by_song(candidates),
        candidates,
        0.02,
        load_config(),
        fake,
        diagnostics,
    )

    assert reason is None
    assert resolved is not None
    assert resolved.metadata["song_id"] == "correct_song"
    assert resolved.route == "phrase_cross_song_lyrics_rerank"
    assert resolved.margin == 0.02
    assert fake.calls == 1


def test_strong_frame_phrase_agreement_keeps_frame_line_position():
    first = metadata()
    second = {**metadata(), "song_id": "other_song"}
    frame = Candidate(first, dtw_result(400, 470, 0.05))
    phrase_candidates = [
        PhraseCandidate(first, phrase_result(1, 0.20, 800, 880)),
        PhraseCandidate(second, phrase_result(0, 0.25)),
    ]
    frame_count = 40
    pitch = np.linspace(60.0, 66.0, frame_count, dtype=np.float32)
    query = PitchFeatures(
        time=np.arange(frame_count, dtype=np.float32) * 0.025,
        pitch=pitch,
        relative_pitch=pitch - np.median(pitch),
        delta_pitch=np.gradient(pitch),
        voiced=np.ones(frame_count, dtype=bool),
        confidence=np.ones(frame_count, dtype=np.float32),
        onset_strength=np.zeros(frame_count, dtype=np.float32),
    )

    result = _recognize_hybrid(
        np.ones(32000, dtype=np.float32) * 0.1,
        np.ones(32000, dtype=np.float32) * 0.1,
        0,
        query,
        [frame],
        {"test_song": [frame]},
        phrase_candidates,
        load_config(),
        {},
        DisabledLyricsAsr(),
    )

    assert result["accepted"] is True
    assert result["current_lyric_index"] == 0
    assert result["diagnostics"]["hybrid_route"] == "strong_frame_phrase_agreement"
