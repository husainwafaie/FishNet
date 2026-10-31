import torch

from fishnet.detection.loss import heatmap_focal_loss, masked_l1

H = W = 16


def _target_with_peak(c=1, y=5, x=7):
    t = torch.zeros(1, 3, H, W)
    t[0, c, y, x] = 1.0
    return t


def test_confident_correct_prediction_has_near_zero_heatmap_loss():
    target = _target_with_peak()
    logits = torch.where(target >= 1, torch.tensor(12.0), torch.tensor(-12.0))
    assert heatmap_focal_loss(logits, target, torch.zeros(1, H, W)) < 1e-3


def test_ignore_mask_removes_false_positive_penalty():
    target = _target_with_peak()
    base = torch.where(target >= 1, torch.tensor(12.0), torch.tensor(-12.0))
    with_fp = base.clone()
    with_fp[0, 0, 12, 12] = 8.0  # confident detection where there is no labelled object
    ignore = torch.zeros(1, H, W)
    ignore[0, 11:14, 11:14] = 1

    clean = heatmap_focal_loss(base, target, ignore)
    fp_ignored = heatmap_focal_loss(with_fp, target, ignore)
    fp_counted = heatmap_focal_loss(with_fp, target, torch.zeros(1, H, W))
    assert torch.isclose(fp_ignored, clean, atol=1e-5)
    assert fp_counted > clean + 1.0


def test_ignore_mask_does_not_suppress_positive_loss():
    target = _target_with_peak(y=12, x=12)
    ignore = torch.zeros(1, H, W)
    ignore[0, 11:14, 11:14] = 1
    missed = torch.full((1, 3, H, W), -12.0)
    assert heatmap_focal_loss(missed, target, ignore) > 5.0


def test_masked_l1_only_counts_positive_cells():
    target = torch.zeros(1, 2, H, W)
    pred = torch.full((1, 2, H, W), 3.0)
    mask = torch.zeros(1, H, W)
    mask[0, 4, 4] = 1
    pred[0, :, 4, 4] = torch.tensor([0.5, -0.5])
    assert torch.isclose(masked_l1(pred, target, mask), torch.tensor(1.0))
