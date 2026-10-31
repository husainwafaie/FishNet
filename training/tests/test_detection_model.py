import math

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from fishnet.detection.model import HEATMAP_PRIOR, CenterNet


def test_output_shapes_at_stride_4():
    model = CenterNet(num_classes=3, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 96))
    assert out["heatmap"].shape == (2, 3, 32, 24)
    assert out["size"].shape == (2, 2, 32, 24)
    assert out["offset"].shape == (2, 2, 32, 24)


def test_backbone_uses_frozen_bn_and_no_batchnorm_remains():
    model = CenterNet(num_classes=3, pretrained=False)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert any(isinstance(m, FrozenBatchNorm2d) for m in model.modules())


def test_heatmap_bias_encodes_prior():
    model = CenterNet(num_classes=3, pretrained=False)
    bias = model.heatmap[-1].bias
    assert torch.allclose(torch.sigmoid(bias), torch.full_like(bias, HEATMAP_PRIOR), atol=1e-6)
    assert torch.allclose(model.size[-1].bias, torch.full((2,), math.log(16.0)))
