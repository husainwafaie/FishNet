"""Batched on-device preprocessing: uint8 NCHW -> normalised float, with optional train-time augmentation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Preprocess:
    """Per-sample random hflip, resized crop (affine resample), brightness/contrast jitter.

    No hue/saturation jitter: colour is a class cue, and its depth-dependent
    attenuation is the distribution shift under study.
    """

    def __init__(
        self,
        train: bool,
        scale: tuple[float, float] = (0.75, 1.0),
        ratio: tuple[float, float] = (0.9, 1.1),
        brightness: float = 0.2,
        contrast: float = 0.2,
    ):
        self.train = train
        self.scale, self.ratio = scale, ratio
        self.brightness, self.contrast = brightness, contrast

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().div_(255)
        if self.train:
            x = self._augment(x)
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _rand(n: int, lo: float, hi: float, like: torch.Tensor) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(n, device=like.device)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        flip = self._rand(n, 0, 1, x) < 0.5
        x = torch.where(flip.view(n, 1, 1, 1), x.flip(-1), x)

        # RandomResizedCrop as an affine resample in normalised coords: crop of
        # relative size (w, h), centre offset so the window stays inside.
        area = self._rand(n, *self.scale, x)
        r = torch.exp(self._rand(n, math.log(self.ratio[0]), math.log(self.ratio[1]), x))
        w = torch.sqrt(area * r).clamp(max=1.0)
        h = torch.sqrt(area / r).clamp(max=1.0)
        cx = self._rand(n, -1, 1, x) * (1 - w)
        cy = self._rand(n, -1, 1, x) * (1 - h)
        zeros = torch.zeros_like(w)
        theta = torch.stack([torch.stack([w, zeros, cx], 1), torch.stack([zeros, h, cy], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        b = self._rand(n, 1 - self.brightness, 1 + self.brightness, x).view(n, 1, 1, 1)
        c = self._rand(n, 1 - self.contrast, 1 + self.contrast, x).view(n, 1, 1, 1)
        x = x * b
        gray_mean = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean(dim=(1, 2)).view(n, 1, 1, 1)
        x = (x - gray_mean) * c + gray_mean
        return x.clamp_(0, 1)
