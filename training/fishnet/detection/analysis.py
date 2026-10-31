"""Error-analysis primitives: IoU, greedy score-ordered matching, and an occlusion proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..labels import Box


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, 4), b: (M, 4) xywh -> (N, M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass
class Match:
    gt_matched: np.ndarray  # (G,) bool, per non-ignore GT
    det_status: np.ndarray  # (D,) "tp" | "fp" | "ignored"


def greedy_match(gt: Sequence[Box], boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                 iou_thr: float = 0.5) -> Match:
    """COCO-like: detections in descending score claim the best unmatched same-class GT; unmatched
    detections overlapping an ignore box (any class) are 'ignored' rather than false positives."""
    pos = [b for b in gt if not b.ignore]
    ign = [b for b in gt if b.ignore]
    gt_xywh = np.array([[b.x, b.y, b.w, b.h] for b in pos]).reshape(-1, 4)
    gt_cls = np.array([b.class_id for b in pos], dtype=int)
    ign_xywh = np.array([[b.x, b.y, b.w, b.h] for b in ign]).reshape(-1, 4)

    iou = box_iou(boxes, gt_xywh)
    matched = np.zeros(len(pos), bool)
    status = np.empty(len(boxes), dtype=object)
    for d in np.argsort(-scores, kind="stable"):
        cand = np.where((gt_cls == classes[d]) & ~matched & (iou[d] >= iou_thr))[0]
        if cand.size:
            matched[cand[np.argmax(iou[d, cand])]] = True
            status[d] = "tp"
            continue
        if len(ign) and (intersection_over_first(boxes[d], ign_xywh) >= iou_thr).any():
            status[d] = "ignored"
            continue
        status[d] = "fp"
    return Match(matched, status)


def intersection_over_first(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Crowd-style overlap: |box ∩ other| / |box| for each row of `others` (xywh)."""
    x0 = np.maximum(box[0], others[:, 0])
    y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[0] + box[2], others[:, 0] + others[:, 2])
    y1 = np.minimum(box[1] + box[3], others[:, 1] + others[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / max(box[2] * box[3], 1e-9)


def nearer_coverage(boxes: Sequence[Box]) -> np.ndarray:
    """For each non-ignore box, the fraction of its area covered by boxes of nearer non-ignore instances.

    An upper-bound occlusion proxy: fish bodies are thinner than their boxes.
    """
    pos = [b for b in boxes if not b.ignore]
    out = np.zeros(len(pos))
    for i, b in enumerate(pos):
        w, h = max(int(np.ceil(b.w)), 1), max(int(np.ceil(b.h)), 1)
        covered = np.zeros((h, w), bool)
        for o in pos:
            if o is b or o.distance >= b.distance:
                continue
            x0, y0 = max(b.x, o.x), max(b.y, o.y)
            x1, y1 = min(b.x + b.w, o.x + o.w), min(b.y + b.h, o.y + o.h)
            if x1 > x0 and y1 > y0:
                covered[int(y0 - b.y):int(np.ceil(y1 - b.y)), int(x0 - b.x):int(np.ceil(x1 - b.x))] = True
        out[i] = covered.mean()
    return out
