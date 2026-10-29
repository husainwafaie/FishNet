"""Per-instance square crops for species classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .labels import NUM_CLASSES, Box, Sample

CONTEXT_PAD = 0.25  # fraction of the box's long side added on each side
MIN_CROP_SIDE = 16  # px; floor for tiny instances


def crop_window(
    box: Box,
    img_w: int,
    img_h: int,
    pad: float = CONTEXT_PAD,
    min_side: int = MIN_CROP_SIDE,
) -> tuple[int, int, int, int]:
    """Square window centred on `box`, expanded by `pad`, shifted to lie inside the image.

    Shifting (rather than zero-padding) keeps crops free of synthetic border
    artefacts; since side >= max(w, h) the box stays fully contained.
    """
    side = max(box.w, box.h) * (1 + 2 * pad)
    side = int(round(min(max(side, min_side), img_w, img_h)))
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    x0 = min(max(int(round(cx - side / 2)), 0), img_w - side)
    y0 = min(max(int(round(cy - side / 2)), 0), img_h - side)
    return x0, y0, x0 + side, y0 + side


@dataclass(frozen=True)
class CropMeta:
    key: str
    class_id: int
    distance: float
    visibility: float
    area: float


class CropSet:
    """Non-ignore instances from `samples`, decoded once into a uint8 NCHW tensor.

    Augmentation/normalisation is applied batch-wise on device (see augment.py).
    """

    def __init__(self, samples: Sequence[Sample], size: int, desc: str = "crops"):
        crops: list[np.ndarray] = []
        self.meta: list[CropMeta] = []
        for sample in tqdm(samples, desc=desc, leave=False):
            boxes = [b for b in sample.boxes if not b.ignore]
            if not boxes:
                continue
            with Image.open(sample.image_path) as im:
                im = im.convert("RGB")
                for b in boxes:
                    window = crop_window(b, im.width, im.height)
                    crop = im.crop(window).resize((size, size), Image.Resampling.BILINEAR)
                    crops.append(np.asarray(crop))
                    self.meta.append(CropMeta(sample.key, b.class_id, b.distance, b.visibility, b.area))
        if not crops:
            raise ValueError(f"{desc}: no non-ignore instances in {len(samples)} samples")

        self.images = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.tensor([m.class_id for m in self.meta], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def class_counts(self) -> torch.Tensor:
        return torch.bincount(self.labels, minlength=NUM_CLASSES)

    def to(self, device: torch.device) -> "DeviceCrops":
        return DeviceCrops(self.images.to(device), self.labels.to(device))


@dataclass
class DeviceCrops:
    images: torch.Tensor  # uint8 NCHW
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def batches(self, batch_size: int, shuffle: bool):
        n, device = len(self.labels), self.labels.device
        order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            yield self.images[idx], self.labels[idx]
