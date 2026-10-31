"""CenterNet target encoding at output stride 4.

Per image: a per-class centre heatmap with an elliptical Gaussian per instance
(TTFNet-style, sigma = alpha * size / 6), log-size and sub-cell offset regression
maps populated at centre cells, and an ignore mask zeroing the negative heatmap
loss inside COCO-style ignore boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..labels import NUM_CLASSES, Box

STRIDE = 4
GAUSSIAN_ALPHA = 0.54


@dataclass
class Targets:
    heatmap: np.ndarray  # (C, H, W) float32 in [0, 1]
    size: np.ndarray  # (2, H, W) log(w), log(h) in input pixels
    offset: np.ndarray  # (2, H, W) sub-cell centre offset in [0, 1)
    reg_mask: np.ndarray  # (H, W) 1 at instance centre cells
    ignore: np.ndarray  # (H, W) 1 where negative heatmap loss is suppressed


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, sigma_x: float, sigma_y: float) -> None:
    """Max-composite an elliptical Gaussian (peak 1 at the integer centre cell) into `heatmap` (H, W)."""
    h, w = heatmap.shape
    ix, iy = int(cx), int(cy)
    rx, ry = max(1, math.ceil(3 * sigma_x)), max(1, math.ceil(3 * sigma_y))
    x0, x1 = max(0, ix - rx), min(w, ix + rx + 1)
    y0, y1 = max(0, iy - ry), min(h, iy + ry + 1)
    xs = np.arange(x0, x1, dtype=np.float32) - ix
    ys = np.arange(y0, y1, dtype=np.float32) - iy
    g = np.exp(-(xs[None, :] ** 2) / (2 * sigma_x**2) - (ys[:, None] ** 2) / (2 * sigma_y**2))
    np.maximum(heatmap[y0:y1, x0:x1], g, out=heatmap[y0:y1, x0:x1])


def encode(boxes: Sequence[Box], img_w: int, img_h: int, stride: int = STRIDE) -> Targets:
    out_w, out_h = img_w // stride, img_h // stride
    heatmap = np.zeros((NUM_CLASSES, out_h, out_w), np.float32)
    size = np.zeros((2, out_h, out_w), np.float32)
    offset = np.zeros((2, out_h, out_w), np.float32)
    reg_mask = np.zeros((out_h, out_w), np.float32)
    ignore = np.zeros((out_h, out_w), np.float32)

    for b in boxes:
        if b.ignore:
            x0, y0 = int(b.x // stride), int(b.y // stride)
            x1, y1 = math.ceil((b.x + b.w) / stride), math.ceil((b.y + b.h) / stride)
            ignore[max(0, y0):min(out_h, y1), max(0, x0):min(out_w, x1)] = 1

    # Larger first so that, on a centre-cell collision, the smaller instance owns the regression targets
    for b in sorted((b for b in boxes if not b.ignore), key=lambda b: -b.area):
        cx = min((b.x + b.w / 2) / stride, out_w - 1e-3)
        cy = min((b.y + b.h / 2) / stride, out_h - 1e-3)
        ix, iy = int(cx), int(cy)
        draw_gaussian(heatmap[b.class_id], cx, cy,
                      max(GAUSSIAN_ALPHA * b.w / stride / 6, 1e-3),
                      max(GAUSSIAN_ALPHA * b.h / stride / 6, 1e-3))
        size[:, iy, ix] = (math.log(max(b.w, 1.0)), math.log(max(b.h, 1.0)))
        offset[:, iy, ix] = (cx - ix, cy - iy)
        reg_mask[iy, ix] = 1

    return Targets(heatmap, size, offset, reg_mask, ignore)
