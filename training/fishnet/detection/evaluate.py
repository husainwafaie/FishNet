"""COCO-style detection metrics (pycocotools) with class-agnostic ignore regions.

Ignore boxes are registered as iscrowd=1 annotations under every category, so a
detection of any class landing on an ignore region is neither a true nor a false
positive. AP is read from COCOeval's precision tensor directly (single maxDets)
rather than via summarize(), which assumes maxDets=[1, 10, 100].
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..labels import CLASS_NAMES, Sample

MAX_DETS = 300
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0, 1e10),
    "<150": (0, 150),
    "150-400": (150, 400),
    "400-1500": (400, 1500),
    ">1500": (1500, 1e10),
}


@dataclass
class ImageDetections:
    boxes: np.ndarray  # (K, 4) x, y, w, h
    scores: np.ndarray  # (K,)
    classes: np.ndarray  # (K,) int class index


def _category_id(class_id: int) -> int:
    return class_id + 1


def ground_truth(samples: Sequence[Sample]) -> COCO:
    images, annotations = [], []
    for image_id, s in enumerate(samples):
        images.append({"id": image_id, "width": s.width, "height": s.height})
        for b in s.boxes:
            categories = range(len(CLASS_NAMES)) if b.ignore else [b.class_id]
            for c in categories:
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": _category_id(c),
                    "bbox": [b.x, b.y, b.w, b.h],
                    "area": b.w * b.h,
                    "iscrowd": int(b.ignore),
                })
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": _category_id(i), "name": n} for i, n in enumerate(CLASS_NAMES)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _mean_valid(a: np.ndarray) -> float:
    valid = a[a > -1]
    return float(valid.mean()) if valid.size else float("nan")


def evaluate(samples: Sequence[Sample], detections: Sequence[ImageDetections], score_floor: float = 0.0) -> dict:
    gt = ground_truth(samples)
    results = [
        {"image_id": image_id, "category_id": _category_id(int(c)), "bbox": [float(v) for v in box], "score": float(s)}
        for image_id, det in enumerate(detections)
        for box, s, c in zip(det.boxes, det.scores, det.classes)
        if s > score_floor
    ]
    names = list(AREA_RANGES)
    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0,
                "AP_per_class": dict.fromkeys(CLASS_NAMES, 0.0), "AP_by_area": dict.fromkeys(names, 0.0)}

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.maxDets = [MAX_DETS]
        ev.params.areaRng = [list(r) for r in AREA_RANGES.values()]
        ev.params.areaRngLbl = names
        ev.evaluate()
        ev.accumulate()

    precision = ev.eval["precision"]  # (T, R, K, A, M)
    recall = ev.eval["recall"]  # (T, K, A, M)
    iou = ev.params.iouThrs
    t50, t75 = int(np.argmin(abs(iou - 0.5))), int(np.argmin(abs(iou - 0.75)))
    return {
        "AP": _mean_valid(precision[:, :, :, 0, 0]),
        "AP50": _mean_valid(precision[t50, :, :, 0, 0]),
        "AP75": _mean_valid(precision[t75, :, :, 0, 0]),
        "AR": _mean_valid(recall[:, :, 0, 0]),
        "AP_per_class": {n: _mean_valid(precision[:, :, k, 0, 0]) for k, n in enumerate(CLASS_NAMES)},
        "AP_by_area": {n: _mean_valid(precision[:, :, :, a, 0]) for a, n in enumerate(names)},
    }
