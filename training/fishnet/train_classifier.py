"""Train a species classifier on per-instance crops and evaluate it on the held-out test split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import DATASET_ROOT, RUNS_ROOT
from .augment import Preprocess
from .crops import CropSet, DeviceCrops
from .labels import CLASS_NAMES, NUM_CLASSES, load_samples
from .metrics import binned_accuracy, summarize
from .models import build_model, count_parameters
from .plots import plot_confusion, plot_history
from .splits import partition

DEFAULT_LR = {"cnn": 1e-3, "resnet18": 1e-3}
VISIBILITY_BINS = [0.0, 0.3, 0.5, 0.7, 1.01]
AREA_BINS = [0, 150, 400, 1500, float("inf")]  # instance box area, px^2 at capture resolution
MAX_EVALS = 25  # validation checkpoints per run, regardless of epoch count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    p.add_argument("--no-pretrained", action="store_true", help="random init for resnet18")
    p.add_argument("--data", type=Path, default=DATASET_ROOT)
    p.add_argument("--input-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--min-steps", type=int, default=0, help="raise epochs until at least this many optimizer steps")
    p.add_argument("--train-fraction", type=float, default=1.0, help="image-level subsample of the train split")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default: per-model)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--amp", action="store_true",
                   help="fp16 autocast; off by default (cuDNN fp16 kernels emit NaN on GTX 16-series)")
    p.add_argument("--no-precise-bn", action="store_true", help="evaluate with EMA running BN stats")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class EpochResult:
    loss: float
    y_true: np.ndarray
    y_pred: np.ndarray


def run_epoch(model, batches, preprocess, criterion, use_amp, optimizer=None, scheduler=None, scaler=None) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(training):
        for x, y in batches:
            x = preprocess(x)
            with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            preds.append(logits.argmax(1))
            targets.append(y)
    return EpochResult(total_loss / n, torch.cat(targets).cpu().numpy(), torch.cat(preds).cpu().numpy())


@torch.no_grad()
def recalibrate_bn(model: nn.Module, dev: DeviceCrops, preprocess, batch_size: int) -> None:
    """Precise BN: replace lagging EMA running stats with exact population stats.

    Runs in fp32; accumulating BN statistics under fp16 autocast overflows.
    """
    torch.optim.swa_utils.update_bn((preprocess(x) for x, _ in dev.batches(batch_size, shuffle=False)), model)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate_split(model, crops: CropSet, dev: DeviceCrops, batch_size, preprocess, criterion, use_amp) -> dict:
    res = run_epoch(model, dev.batches(batch_size, shuffle=False), preprocess, criterion, use_amp)
    correct = res.y_true == res.y_pred
    summary = summarize(res.y_true, res.y_pred, CLASS_NAMES)
    summary["loss"] = res.loss
    summary["accuracy_by_visibility"] = binned_accuracy([m.visibility for m in crops.meta], correct, VISIBILITY_BINS)
    summary["accuracy_by_area"] = binned_accuracy([m.area for m in crops.meta], correct, AREA_BINS)
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp
    torch.backends.cudnn.benchmark = True
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.model]

    variant = args.model + ("-scratch" if args.model == "resnet18" and args.no_pretrained else "")
    run_name = args.run_name or f"{variant}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_ROOT / "classifier" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data)
    parts = partition(samples, lambda s: s.key)
    if args.train_fraction < 1.0:
        k = max(1, round(args.train_fraction * len(parts["train"])))
        parts["train"] = sorted(random.Random(args.seed).sample(parts["train"], k), key=lambda s: s.key)
    crops = {split: CropSet(parts[split], args.input_size, desc=f"{split} crops") for split in ("train", "val", "test")}
    dev = {split: c.to(device) for split, c in crops.items()}
    train_pre, eval_pre = Preprocess(train=True), Preprocess(train=False)

    counts = crops["train"].class_counts().float()
    class_weights = None if args.no_class_weights else (counts.sum() / (NUM_CLASSES * counts.clamp(min=1)))
    train_criterion = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(device))
    eval_criterion = nn.CrossEntropyLoss()

    model = build_model(args.model, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(crops["train"]) / args.batch_size)
    epochs = max(args.epochs, math.ceil(args.min_steps / steps_per_epoch))
    scheduler = warmup_cosine(optimizer, int(args.warmup_epochs * steps_per_epoch), epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "variant": variant,
        "lr": lr,
        "epochs_effective": epochs,
        "total_steps": epochs * steps_per_epoch,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp": use_amp,
        "parameters": count_parameters(model),
        "split_images": {k: len(v) for k, v in parts.items()},
        "split_instances": {k: len(v) for k, v in crops.items()},
        "train_class_counts": dict(zip(CLASS_NAMES, counts.int().tolist())),
        "class_weights": None if class_weights is None else dict(zip(CLASS_NAMES, class_weights.tolist())),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{variant}: {config['parameters']:,} params | lr {lr:g} | {config['split_instances']} instances | "
          f"{config['device']} amp={use_amp}")

    history: list[dict] = []
    best_score, best_epoch = -1.0, -1
    t0 = time.time()
    eval_every = max(1, epochs // MAX_EVALS)
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, dev["train"].batches(args.batch_size, shuffle=True), train_pre,
                       train_criterion, use_amp, optimizer, scheduler, scaler)
        if epoch % eval_every and epoch != epochs:
            continue
        if not args.no_precise_bn:
            recalibrate_bn(model, dev["train"], eval_pre, 2 * args.batch_size)
        va = run_epoch(model, dev["val"].batches(args.batch_size, shuffle=False), eval_pre, eval_criterion, use_amp)
        va_summary = summarize(va.y_true, va.y_pred, CLASS_NAMES)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr.loss,
            "train_accuracy": float((tr.y_true == tr.y_pred).mean()),
            "val_loss": va.loss,
            "val_accuracy": va_summary["accuracy"],
            "val_balanced_accuracy": va_summary["balanced_accuracy"],
            "val_macro_f1": va_summary["macro_f1"],
        }
        history.append(row)
        improved = row["val_balanced_accuracy"] > best_score
        if improved:
            best_score, best_epoch = row["val_balanced_accuracy"], epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(f"ep {epoch:>3} | train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.3f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.3f} "
              f"bal {row['val_balanced_accuracy']:.3f}{' *' if improved else ''}")
    train_seconds = time.time() - t0

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    results = {
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
        **{
            split: evaluate_split(model, crops[split], dev[split], args.batch_size, eval_pre, eval_criterion, use_amp)
            for split in ("val", "test")
        },
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_history(history, run_dir / "curves.png")
    plot_confusion(results["test"]["confusion"], CLASS_NAMES, run_dir / "confusion_test.png", f"{variant} — test")

    te = results["test"]
    print(f"\nbest epoch {best_epoch} ({train_seconds:.0f}s) | test acc {te['accuracy']:.3f} "
          f"bal {te['balanced_accuracy']:.3f} macro-F1 {te['macro_f1']:.3f}")
    for name, pc in te["per_class"].items():
        print(f"  {name:<8} P {pc['precision']:.3f}  R {pc['recall']:.3f}  n={pc['support']}")
    print(f"run dir: {run_dir}")
    return {"config": config, "history": history, "results": results}


if __name__ == "__main__":
    main()
