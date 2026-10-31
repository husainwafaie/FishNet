import math

import numpy as np
import pytest

from fishnet.detection.targets import STRIDE, draw_gaussian, encode
from fishnet.labels import Box


def _box(x, y, w, h, class_id=1, ignore=False):
    return Box(class_id, x, y, w, h, distance=10.0, visibility=0.8, ignore=ignore)


def test_single_box_peak_offset_and_size():
    t = encode([_box(101, 202, 40, 20, class_id=2)], 640, 640)
    cx, cy = (101 + 20) / STRIDE, (202 + 10) / STRIDE  # 30.25, 53.0
    ix, iy = int(cx), int(cy)
    assert t.heatmap.shape == (3, 160, 160)
    assert t.heatmap[2, iy, ix] == pytest.approx(1.0)
    assert t.heatmap[2].max() == pytest.approx(1.0) and t.heatmap[:2].max() == 0
    assert t.reg_mask.sum() == 1 and t.reg_mask[iy, ix] == 1
    assert t.offset[:, iy, ix] == pytest.approx([cx - ix, cy - iy])
    assert t.size[:, iy, ix] == pytest.approx([math.log(40), math.log(20)])


def test_gaussian_is_elongated_along_the_long_axis():
    hm = np.zeros((40, 40), np.float32)
    draw_gaussian(hm, 20.5, 20.5, sigma_x=3.0, sigma_y=1.0)
    assert hm[20, 23] > hm[23, 20]


def test_tiny_box_still_gets_a_peak():
    t = encode([_box(300, 300, 3, 2, class_id=0)], 640, 640)
    assert t.heatmap[0].max() == pytest.approx(1.0)


def test_ignore_box_masks_its_footprint_without_creating_positives():
    t = encode([_box(40, 40, 9, 7, class_id=0, ignore=True)], 640, 640)
    assert t.heatmap.max() == 0 and t.reg_mask.sum() == 0
    assert t.ignore[10:12, 10:13].all()
    assert t.ignore.sum() == t.ignore[10:12, 10:13].size


def test_edge_box_lands_in_last_cell():
    t = encode([_box(634, 634, 6, 6)], 640, 640)  # centre 637 px -> cell 159.25
    assert t.reg_mask[159, 159] == 1
    assert t.offset[:, 159, 159] == pytest.approx([0.25, 0.25])


def test_centre_collision_keeps_smaller_instance_regression():
    big, small = _box(96, 96, 32, 32), _box(108, 108, 8, 8)  # both centre in cell (28, 28)
    t = encode([small, big], 640, 640)
    assert t.reg_mask.sum() == 1
    assert t.size[:, 28, 28] == pytest.approx([math.log(8), math.log(8)])
