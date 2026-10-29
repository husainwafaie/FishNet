"""Train a species classifier on per-instance crops and evaluate it on the held-out test split."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import DATASET_ROOT, RUNS_ROOT
from .crops import CropDataset
from .labels import CLASS_NAMES, NUM_CLASSES, load_samples
from .metrics import binned_accuracy, summarize
from .models import build_model, count_parameters
from .plots import plot_confusion, plot_history
from .splits import partition

DEFAULT_LR = {"cnn": 3e-3, "resnet18": 1e-3}
VISIBILITY_BINS = [0.0, 0.3, 0.5, 0.7, 1.01]
AREA_BINS = [0, 150, 400, 1500, float("inf")]  # instance box area, px^2 at capture resolution


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    p.add_argument("--no-pretrained", action="store_true", help="random init for resnet18")
    p.add_argument("--data", type=Path, default=DATASET_ROOT)
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default: per-model)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=0)
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


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scheduler=None, scaler=None) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
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
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    return EpochResult(total_loss / n, torch.cat(targets).numpy(), torch.cat(preds).numpy())


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate_split(model, ds: CropDataset, loader, criterion, device, use_amp) -> dict:
    res = run_epoch(model, loader, criterion, device, use_amp)
    correct = res.y_true == res.y_pred
    summary = summarize(res.y_true, res.y_pred, CLASS_NAMES)
    summary["loss"] = res.loss
    summary["accuracy_by_visibility"] = binned_accuracy([m.visibility for m in ds.meta], correct, VISIBILITY_BINS)
    summary["accuracy_by_area"] = binned_accuracy([m.area for m in ds.meta], correct, AREA_BINS)
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.model]

    variant = args.model + ("-scratch" if args.model == "resnet18" and args.no_pretrained else "")
    run_name = args.run_name or f"{variant}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_ROOT / "classifier" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data)
    parts = partition(samples, lambda s: s.key)
    datasets = {
        split: CropDataset(parts[split], args.input_size, train=(split == "train"), desc=f"{split} crops")
        for split in ("train", "val", "test")
    }
    pin = device.type == "cuda"
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=args.workers, pin_memory=pin, drop_last=False)
        for split, ds in datasets.items()
    }

    counts = datasets["train"].class_counts().float()
    class_weights = None if args.no_class_weights else (counts.sum() / (NUM_CLASSES * counts.clamp(min=1)))
    train_criterion = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(device))
    eval_criterion = nn.CrossEntropyLoss()

    model = build_model(args.model, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    scheduler = warmup_cosine(optimizer, int(args.warmup_epochs * steps_per_epoch), args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "variant": variant,
        "lr": lr,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp": use_amp,
        "parameters": count_parameters(model),
        "split_images": {k: len(v) for k, v in parts.items()},
        "split_instances": {k: len(v) for k, v in datasets.items()},
        "train_class_counts": dict(zip(CLASS_NAMES, counts.int().tolist())),
        "class_weights": None if class_weights is None else dict(zip(CLASS_NAMES, class_weights.tolist())),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{variant}: {config['parameters']:,} params | {config['split_instances']} instances | "
          f"{config['device']} amp={use_amp}")

    history: list[dict] = []
    best_score, best_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], train_criterion, device, use_amp, optimizer, scheduler, scaler)
        va = run_epoch(model, loaders["val"], eval_criterion, device, use_amp)
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
        "val": evaluate_split(model, datasets["val"], loaders["val"], eval_criterion, device, use_amp),
        "test": evaluate_split(model, datasets["test"], loaders["test"], eval_criterion, device, use_amp),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_history(history, run_dir / "curves.png")
    plot_confusion(results["test"]["confusion"], CLASS_NAMES, run_dir / "confusion_test.png",
                   f"{variant} — test")

    te = results["test"]
    print(f"\nbest epoch {best_epoch} ({train_seconds:.0f}s) | test acc {te['accuracy']:.3f} "
          f"bal {te['balanced_accuracy']:.3f} macro-F1 {te['macro_f1']:.3f}")
    for name, pc in te["per_class"].items():
        print(f"  {name:<8} P {pc['precision']:.3f}  R {pc['recall']:.3f}  n={pc['support']}")
    print(f"run dir: {run_dir}")
    return {"config": config, "history": history, "results": results}


if __name__ == "__main__":
    main()
