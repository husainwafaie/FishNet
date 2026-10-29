import json

import numpy as np
import pytest
import torch
from PIL import Image

from fishnet.augment import IMAGENET_MEAN, IMAGENET_STD, Preprocess
from fishnet.crops import CropSet, DeviceCrops, crop_window
from fishnet.labels import Box, load_samples
from fishnet.metrics import binned_accuracy, confusion_matrix, summarize


def _box(x, y, w, h, class_id=1, ignore=False):
    return Box(class_id, x, y, w, h, distance=10.0, visibility=0.8, ignore=ignore)


@pytest.mark.parametrize(
    "box",
    [
        _box(300, 300, 20, 10),  # interior
        _box(0, 0, 12, 8),  # top-left corner
        _box(620, 630, 20, 10),  # bottom-right corner
        _box(100, 0, 140, 60),  # large, touching top edge
        _box(320, 320, 2, 2),  # below min side
    ],
)
def test_crop_window_is_square_inside_image_and_contains_box(box):
    x0, y0, x1, y1 = crop_window(box, 640, 640)
    assert x1 - x0 == y1 - y0
    assert 0 <= x0 and 0 <= y0 and x1 <= 640 and y1 <= 640
    assert x0 <= box.x + 0.5 and y0 <= box.y + 0.5
    assert box.x + box.w <= x1 + 0.5 and box.y + box.h <= y1 + 0.5


def test_crop_dataset_skips_ignore_and_preserves_labels(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / "images" / "s0.png")
    boxes = [
        {"species": "grouper", "classId": 2, "x": 10, "y": 10, "w": 20, "h": 12,
         "distance": 5, "visibility": 0.9},
        {"species": "anchovy", "classId": 0, "x": 40, "y": 40, "w": 3, "h": 2,
         "distance": 30, "visibility": 0.4, "ignore": True},
    ]
    label = {"image": "s0.png", "width": 64, "height": 64, "seed": 1,
             "camera": {"position": [0, -10, 0], "target": [0, 0, 0], "fov": 60}, "boxes": boxes}
    (tmp_path / "labels" / "s0.json").write_text(json.dumps(label))

    crops = CropSet(load_samples(tmp_path), size=32)
    assert len(crops) == 1
    assert tuple(crops.images.shape) == (1, 3, 32, 32) and crops.images.dtype == torch.uint8
    assert crops.labels.tolist() == [2]


def test_device_crops_batches_cover_every_index_once():
    images = torch.arange(10, dtype=torch.uint8).view(10, 1, 1, 1).expand(10, 3, 2, 2).contiguous()
    dev = DeviceCrops(images, torch.arange(10))
    seen = torch.cat([y for _, y in dev.batches(batch_size=3, shuffle=True)])
    assert sorted(seen.tolist()) == list(range(10))


def test_preprocess_eval_is_deterministic_and_normalised():
    x = torch.full((4, 3, 16, 16), 128, dtype=torch.uint8)
    pre = Preprocess(train=False)
    out = pre(x)
    assert out.dtype == torch.float32 and torch.equal(out, pre(x))
    expected = (128 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-5)


def test_preprocess_train_keeps_shape_and_range():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (8, 3, 24, 24), dtype=torch.uint8)
    out = Preprocess(train=True)(x)
    assert out.shape == x.shape
    lo = (0 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    hi = (1 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert (out >= lo.view(1, 3, 1, 1) - 1e-5).all() and (out <= hi.view(1, 3, 1, 1) + 1e-5).all()


def test_label_class_id_mismatch_raises(tmp_path):
    (tmp_path / "labels").mkdir()
    label = {"image": "s.png", "width": 8, "height": 8, "seed": 0,
             "camera": {"position": [0, 0, 0]},
             "boxes": [{"species": "jack", "classId": 2, "x": 0, "y": 0, "w": 1, "h": 1,
                        "distance": 1, "visibility": 1}]}
    (tmp_path / "labels" / "s.json").write_text(json.dumps(label))
    with pytest.raises(ValueError):
        load_samples(tmp_path)


def test_summarize_balanced_accuracy_and_confusion():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[3, 1, 0], [0, 2, 0], [1, 0, 1]]
    s = summarize(y_true, y_pred, ["a", "b", "c"])
    assert s["accuracy"] == pytest.approx(6 / 8)
    assert s["balanced_accuracy"] == pytest.approx((0.75 + 1.0 + 0.5) / 3)


def test_binned_accuracy_half_open_bins():
    out = binned_accuracy([0.1, 0.3, 0.6, 0.9], [True, False, True, True], [0.0, 0.3, 1.0])
    assert [(b["n"], b["accuracy"]) for b in out] == [(1, 1.0), (3, pytest.approx(2 / 3))]
