"""Full-frame detection data: frames decoded once to uint8 CHW; targets encoded per item after geometric aug."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from ..labels import Box, Sample
from .targets import encode


def hflip_box(b: Box, img_w: int) -> Box:
    return replace(b, x=img_w - (b.x + b.w))


class DetectionSet(Dataset):
    def __init__(self, samples: Sequence[Sample], train: bool, desc: str = "frames"):
        self.samples = list(samples)
        self.train = train
        frames = []
        for s in tqdm(self.samples, desc=desc, leave=False):
            with Image.open(s.image_path) as im:
                frames.append(np.asarray(im.convert("RGB")))
        self.images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img, sample = self.images[i], self.samples[i]
        _, h, w = img.shape
        boxes = sample.boxes
        if self.train and torch.rand(()) < 0.5:
            img = img.flip(-1)
            boxes = tuple(hflip_box(b, w) for b in boxes)
        t = encode(boxes, w, h)
        targets = {k: torch.from_numpy(getattr(t, k)) for k in ("heatmap", "size", "offset", "reg_mask", "ignore")}
        return img, targets
