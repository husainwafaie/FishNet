import torch

from fishnet.detection.decode import decode
from fishnet.detection.targets import encode
from fishnet.labels import Box


def _box(x, y, w, h, class_id):
    return Box(class_id, x, y, w, h, distance=10.0, visibility=0.8, ignore=False)


def _decode_targets(boxes, k=10):
    t = encode(boxes, 640, 640)
    as_batch = lambda a: torch.from_numpy(a)[None]  # noqa: E731
    return decode(as_batch(t.heatmap), as_batch(t.size), as_batch(t.offset), k=k)


def test_round_trip_recovers_boxes_and_classes():
    boxes = [_box(101, 202, 40, 20, 2), _box(400, 50, 12, 6, 0), _box(300, 500, 25, 30, 1)]
    det = _decode_targets(boxes)
    confident = det.scores[0] > 0.999
    assert int(confident.sum()) == len(boxes)
    got = sorted(zip(det.classes[0][confident].tolist(), det.boxes[0][confident].tolist()))
    want = sorted((b.class_id, [b.x, b.y, b.w, b.h]) for b in boxes)
    for (gc, gb), (wc, wb) in zip(got, want):
        assert gc == wc
        assert torch.allclose(torch.tensor(gb), torch.tensor(wb, dtype=torch.float32), atol=1e-3)


def test_adjacent_same_class_instances_both_survive_peak_extraction():
    boxes = [_box(100, 100, 8, 8, 0), _box(108, 100, 8, 8, 0)]  # centres two cells apart
    det = _decode_targets(boxes)
    assert int((det.scores[0] > 0.999).sum()) == 2


def test_empty_heatmap_yields_zero_scores():
    det = _decode_targets([], k=5)
    assert det.scores.shape == (1, 5) and float(det.scores.max()) == 0.0
