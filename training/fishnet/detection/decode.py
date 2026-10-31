"""CenterNet decoding: 3x3 max-pool peak extraction + top-k, no NMS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .targets import STRIDE

MAX_LOG_SIZE = math.log(1024.0)


@dataclass
class Detections:
    boxes: torch.Tensor  # (B, K, 4) x, y, w, h in input pixels
    scores: torch.Tensor  # (B, K)
    classes: torch.Tensor  # (B, K) long


def _gather(maps: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """maps (B, D, H, W), idx (B, K) flat spatial indices -> (B, K, D)."""
    b, d = maps.shape[:2]
    return maps.reshape(b, d, -1).gather(2, idx[:, None, :].expand(-1, d, -1)).transpose(1, 2)


@torch.no_grad()
def decode(heatmap: torch.Tensor, size: torch.Tensor, offset: torch.Tensor, k: int = 200,
           stride: int = STRIDE) -> Detections:
    """heatmap: (B, C, H, W) probabilities; size/offset: (B, 2, H, W) raw regression outputs."""
    b, c, h, w = heatmap.shape
    peaks = heatmap * (F.max_pool2d(heatmap, 3, stride=1, padding=1) == heatmap)
    scores, flat = peaks.reshape(b, -1).topk(min(k, c * h * w))
    classes = flat // (h * w)
    pix = flat % (h * w)
    ys, xs = (pix // w).float(), (pix % w).float()

    off = _gather(offset, pix)
    wh = _gather(size, pix).clamp(max=MAX_LOG_SIZE).exp()
    cx = (xs + off[..., 0]) * stride
    cy = (ys + off[..., 1]) * stride
    boxes = torch.stack([cx - wh[..., 0] / 2, cy - wh[..., 1] / 2, wh[..., 0], wh[..., 1]], dim=-1)
    return Detections(boxes, scores, classes)
