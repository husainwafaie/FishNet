import json

import numpy as np
import torch
from PIL import Image

from fishnet.detection.data import DetectionSet, hflip_box
from fishnet.labels import Box, load_samples


def test_hflip_box_mirrors_and_is_an_involution():
    b = Box(1, 10, 20, 30, 15, 5.0, 0.9, False)
    f = hflip_box(b, 640)
    assert (f.x, f.y, f.w, f.h) == (600, 20, 30, 15)
    assert hflip_box(f, 640) == b


def _write_sample(root, name, boxes, size=64):
    (root / "images").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, : size // 2] = 255  # left half white, so a flip is observable
    Image.fromarray(img).save(root / "images" / f"{name}.png")
    label = {"image": f"{name}.png", "width": size, "height": size, "seed": 0,
             "camera": {"position": [0, -10, 0]}, "boxes": boxes}
    (root / "labels" / f"{name}.json").write_text(json.dumps(label))


def test_items_have_stride_4_targets_and_uint8_frames(tmp_path):
    _write_sample(tmp_path, "a", [{"species": "jack", "classId": 1, "x": 8, "y": 8, "w": 16, "h": 8,
                                   "distance": 5, "visibility": 0.9}])
    ds = DetectionSet(load_samples(tmp_path), train=False)
    img, t = ds[0]
    assert img.dtype == torch.uint8 and img.shape == (3, 64, 64)
    assert t["heatmap"].shape == (3, 16, 16) and t["reg_mask"].sum() == 1
    assert t["heatmap"][1].max() == 1.0


def test_train_flip_moves_pixels_and_targets_together(tmp_path):
    _write_sample(tmp_path, "a", [{"species": "jack", "classId": 1, "x": 4, "y": 8, "w": 8, "h": 8,
                                   "distance": 5, "visibility": 0.9}])
    ds = DetectionSet(load_samples(tmp_path), train=True)
    seen = set()
    for seed in range(20):
        torch.manual_seed(seed)
        img, t = ds[0]
        flipped = bool(img[0, 0, -1] == 255)
        peak_x = int(t["reg_mask"].nonzero()[0, 1])
        assert (peak_x >= 8) == flipped  # object on the left half unless flipped
        seen.add(flipped)
    assert seen == {True, False}
