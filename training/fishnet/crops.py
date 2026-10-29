"""Per-instance square crops for species classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .labels import NUM_CLASSES, Box, Sample

CONTEXT_PAD = 0.25  # fraction of the box's long side added on each side
MIN_CROP_SIDE = 16  # px; floor for tiny instances

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(size: int, train: bool) -> v2.Compose:
    ops: list = []
    if train:
        # No hue/saturation jitter: colour is a class cue, and its depth-dependent
        # attenuation is the distribution shift under study.
        ops += [
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)


@dataclass(frozen=True)
class CropMeta:
    key: str
    class_id: int
    distance: float
    visibility: float
    area: float


class CropDataset(Dataset):
    """Non-ignore instances from `samples`, decoded once and held as uint8 NCHW."""

    def __init__(self, samples: Sequence[Sample], size: int, train: bool, desc: str = "crops"):
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
        self.transform = build_transform(size, train)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[i]), self.labels[i]

    def class_counts(self) -> torch.Tensor:
        return torch.bincount(self.labels, minlength=NUM_CLASSES)
