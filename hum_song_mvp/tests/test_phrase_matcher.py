import numpy as np

from src.config import load_config
from src.phrase_matcher import contour_dtw, phrase_contour, segmental_contour_dtw


def test_phrase_contour_removes_global_transposition():
    low = phrase_contour(np.asarray([60.0, 60.0, 62.0, 64.0, 64.0]), 16)
    high = phrase_contour(np.asarray([67.0, 67.0, 69.0, 71.0, 71.0]), 16)
    assert low is not None
    assert high is not None
    assert np.allclose(low, high, atol=1e-5)


def test_phrase_contour_preserves_phrase_wide_shape():
    rising = phrase_contour(np.asarray([60.0, 61.0, 62.0, 64.0, 65.0]), 24)
    falling = phrase_contour(np.asarray([65.0, 64.0, 62.0, 61.0, 60.0]), 24)
    assert rising is not None
    assert falling is not None
    config = load_config()["phrase_matching"]
    same = contour_dtw(rising, rising, config)
    different = contour_dtw(rising, falling, config)
    assert same["cost"] < different["cost"]


def test_segmental_contour_can_ignore_mismatched_phrase_edges():
    config = dict(load_config()["phrase_matching"])
    config["segment_coverages"] = [1.0, 0.65]
    query = np.sin(np.linspace(-1.2, 1.2, 72)).astype(np.float32) * 4.0
    middle = np.sin(np.linspace(-1.2, 1.2, 48)).astype(np.float32) * 4.0
    reference = np.concatenate(
        (
            np.linspace(5.0, -5.0, 12, dtype=np.float32),
            middle,
            np.linspace(-5.0, 5.0, 12, dtype=np.float32),
        )
    )
    full_only = dict(config)
    full_only["segment_coverages"] = [1.0]
    full = segmental_contour_dtw(query, reference, full_only)
    segmental = segmental_contour_dtw(query, reference, config)
    assert segmental["segment_coverage"] < 1.0
    assert segmental["cost"] < full["cost"]
