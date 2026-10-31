"""CenterNet losses: penalty-reduced focal loss on centre heatmaps + masked L1 regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor) -> torch.Tensor:
    """logits/target: (B, C, H, W); ignore: (B, H, W) suppresses the negative term (all classes).

    Normalised by the number of positive cells (target == 1), as in CenterNet.
    """
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    p = log_p.exp()
    pos = target >= 1 - 1e-6
    keep_neg = (~pos) & (ignore[:, None] < 0.5)
    pos_loss = (log_p * (1 - p) ** FOCAL_ALPHA)[pos].sum()
    neg_loss = (log_1mp * p**FOCAL_ALPHA * (1 - target) ** FOCAL_BETA)[keep_neg].sum()
    return -(pos_loss + neg_loss) / pos.sum().clamp(min=1)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target: (B, 2, H, W); mask: (B, H, W). Mean L1 per positive cell (summed over channels)."""
    return ((pred - target).abs() * mask[:, None]).sum() / mask.sum().clamp(min=1)


@dataclass
class LossWeights:
    heatmap: float = 1.0
    size: float = 1.0
    offset: float = 1.0


def centernet_loss(out: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor],
                   w: LossWeights = LossWeights()) -> dict[str, torch.Tensor]:
    hm = heatmap_focal_loss(out["heatmap"], tgt["heatmap"], tgt["ignore"])
    size = masked_l1(out["size"], tgt["size"], tgt["reg_mask"])
    off = masked_l1(out["offset"], tgt["offset"], tgt["reg_mask"])
    return {"total": w.heatmap * hm + w.size * size + w.offset * off, "heatmap": hm, "size": size, "offset": off}
