from pathlib import Path

import numpy as np
import pytest

from fishnet.detection.evaluate import ImageDetections, evaluate
from fishnet.labels import Box, Sample


def _box(x, y, w, h, class_id, ignore=False):
    return Box(class_id, x, y, w, h, distance=10.0, visibility=0.8, ignore=ignore)


def _sample(boxes):
    return Sample("s", Path("s.png"), 640, 640, 0, -10.0, tuple(boxes))


def _dets(*rows):
    """rows: (x, y, w, h, score, class)"""
    a = np.array(rows, dtype=np.float64).reshape(-1, 6)
    return ImageDetections(a[:, :4], a[:, 4], a[:, 5].astype(int))


GT = [_box(100, 100, 40, 20, 1), _box(300, 300, 30, 30, 2)]


def test_perfect_detections_score_one():
    m = evaluate([_sample(GT)], [_dets((100, 100, 40, 20, 0.9, 1), (300, 300, 30, 30, 0.8, 2))])
    assert m["AP"] == pytest.approx(1.0) and m["AR"] == pytest.approx(1.0)
    assert m["AP_per_class"]["jack"] == pytest.approx(1.0)


def test_detection_on_ignore_region_is_not_a_false_positive_for_any_class():
    sample = _sample([*GT, _box(500, 500, 6, 6, 0, ignore=True)])
    # a high-scoring "jack" on the anchovy ignore speck must not cost precision
    dets = _dets((500, 500, 6, 6, 0.99, 1), (100, 100, 40, 20, 0.9, 1), (300, 300, 30, 30, 0.8, 2))
    assert evaluate([sample], [dets])["AP"] == pytest.approx(1.0)


def test_unmatched_high_score_detection_lowers_ap():
    dets = _dets((500, 500, 20, 20, 0.99, 1), (100, 100, 40, 20, 0.9, 1), (300, 300, 30, 30, 0.8, 2))
    assert evaluate([_sample(GT)], [dets])["AP_per_class"]["jack"] < 0.99


def test_missed_instance_lowers_recall():
    m = evaluate([_sample(GT)], [_dets((100, 100, 40, 20, 0.9, 1))])
    assert m["AR"] < 1.0


def test_no_detections_returns_zeros():
    m = evaluate([_sample(GT)], [_dets()])
    assert m["AP"] == 0.0 and m["AR"] == 0.0
