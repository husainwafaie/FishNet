"""CenterNet: ImageNet ResNet-18 (frozen BN) + top-down GroupNorm neck to stride 4 + three conv heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

HEATMAP_PRIOR = 0.1
SIZE_PRIOR_PX = 16.0


def _conv_gn_relu(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.GroupNorm(32, c_out),
        nn.ReLU(inplace=True),
    )


def _head(c_in: int, c_out: int, bias: float, hidden: int = 64) -> nn.Sequential:
    head = nn.Sequential(nn.Conv2d(c_in, hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden, c_out, 1))
    nn.init.constant_(head[-1].bias, bias)
    return head


class CenterNet(nn.Module):
    """Backbone BN is frozen (FrozenBatchNorm2d): batch statistics at detection batch sizes are
    too noisy to update ImageNet stats; the neck uses GroupNorm, which is batch-size independent."""

    def __init__(self, num_classes: int, pretrained: bool = True, width: int = 128):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512))
        self.smooth = nn.ModuleList(_conv_gn_relu(width, width) for _ in range(3))
        self.heatmap = _head(width, num_classes, bias=-math.log((1 - HEATMAP_PRIOR) / HEATMAP_PRIOR))
        self.size = _head(width, 2, bias=math.log(SIZE_PRIOR_PX))
        self.offset = _head(width, 2, bias=0.5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c2 = self.layer1(self.stem(x))  # stride 4
        c3 = self.layer2(c2)  # 8
        c4 = self.layer3(c3)  # 16
        c5 = self.layer4(c4)  # 32
        p = self.lateral[3](c5)
        for lateral, smooth, c in zip(self.lateral[2::-1], self.smooth, (c4, c3, c2)):
            p = smooth(lateral(c) + F.interpolate(p, size=c.shape[-2:], mode="nearest"))
        return {"heatmap": self.heatmap(p), "size": self.size(p), "offset": self.offset(p)}
