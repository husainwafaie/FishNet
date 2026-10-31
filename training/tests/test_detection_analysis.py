import numpy as np
import pytest

from fishnet.detection.analysis import box_iou, greedy_match, nearer_coverage
from fishnet.labels import Box


def _box(x, y, w, h, class_id=1, ignore=False, distance=10.0):
    return Box(class_id, x, y, w, h, distance=distance, visibility=0.8, ignore=ignore)


def test_box_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]])
    assert box_iou(a, b)[0] == pytest.approx([1.0, 50 / 150, 0.0])


def test_greedy_match_tp_fp_and_miss():
    gt = [_box(0, 0, 10, 10), _box(50, 50, 10, 10)]
    dets = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    m = greedy_match(gt, dets, np.array([0.9, 0.8]), np.array([1, 1]))
    assert m.gt_matched.tolist() == [True, False]
    assert m.det_status.tolist() == ["tp", "fp"]


def test_wrong_class_does_not_match():
    m = greedy_match([_box(0, 0, 10, 10, class_id=2)], np.array([[0, 0, 10, 10]], float),
                     np.array([0.9]), np.array([1]))
    assert m.det_status.tolist() == ["fp"] and not m.gt_matched.any()


def test_higher_score_claims_gt_first():
    gt = [_box(0, 0, 10, 10)]
    dets = np.array([[1, 0, 10, 10], [0, 0, 10, 10]], float)
    m = greedy_match(gt, dets, np.array([0.5, 0.9]), np.array([1, 1]))
    assert m.det_status.tolist() == ["fp", "tp"]


def test_detection_on_ignore_region_is_ignored_for_any_class():
    gt = [_box(0, 0, 4, 4, class_id=0, ignore=True)]
    m = greedy_match(gt, np.array([[0, 0, 4, 4]], float), np.array([0.9]), np.array([2]))
    assert m.det_status.tolist() == ["ignored"] and m.gt_matched.size == 0


def test_nearer_coverage_counts_only_closer_instances():
    back = _box(0, 0, 10, 10, distance=20.0)
    front = _box(0, 0, 10, 5, distance=5.0)  # covers the top half of `back`
    cov = nearer_coverage([back, front])
    assert cov == pytest.approx([0.5, 0.0])
