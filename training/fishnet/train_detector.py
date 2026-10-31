"""Train the CenterNet fish detector; evaluate COCO-style AP on val (model selection) and test."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import DATASET_ROOT, RUNS_ROOT
from .augment import Preprocess
from .detection.data import DetectionSet
from .detection.decode import decode
from .detection.evaluate import MAX_DETS, ImageDetections, evaluate
from .detection.loss import centernet_loss
from .detection.model import CenterNet
from .labels import NUM_CLASSES, load_samples
from .splits import partition
from .train_classifier import seed_everything, warmup_cosine

LOSS_KEYS = ("total", "heatmap", "size", "offset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATASET_ROOT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--clip-grad", type=float, default=35.0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--overfit", type=int, default=0,
                   help="sanity check: train and evaluate on the first N train frames, no augmentation")
    p.add_argument("--amp", action="store_true", help="fp16 autocast (unreliable on GTX 16-series)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, ds: DetectionSet, preprocess, device, batch_size: int, use_amp: bool) -> list[ImageDetections]:
    model.eval()
    out: list[ImageDetections] = []
    for i in range(0, len(ds), batch_size):
        x = preprocess(ds.images[i : i + batch_size].to(device))
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x)
        det = decode(pred["heatmap"].float().sigmoid(), pred["size"].float(), pred["offset"].float(), k=MAX_DETS)
        for b in range(x.shape[0]):
            out.append(ImageDetections(det.boxes[b].cpu().numpy(), det.scores[b].cpu().numpy(),
                                       det.classes[b].cpu().numpy()))
    return out


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp
    torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"centernet-r18-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_ROOT / "detector" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    parts = partition(load_samples(args.data), lambda s: s.key)
    if args.train_fraction < 1.0:
        k = max(1, round(args.train_fraction * len(parts["train"])))
        parts["train"] = sorted(random.Random(args.seed).sample(parts["train"], k), key=lambda s: s.key)
    if args.overfit:
        parts = {"train": parts["train"][: args.overfit], "val": parts["train"][: args.overfit],
                 "test": parts["train"][: args.overfit]}

    augment = not args.overfit
    sets = {
        "train": DetectionSet(parts["train"], train=augment, desc="train frames"),
        "val": DetectionSet(parts["val"], train=False, desc="val frames"),
        "test": DetectionSet(parts["test"], train=False, desc="test frames"),
    }
    loader = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", drop_last=len(sets["train"]) > args.batch_size)
    train_pre = Preprocess(train=augment, geometric=False)
    eval_pre = Preprocess(train=False)

    model = CenterNet(NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loader)
    scheduler = warmup_cosine(optimizer, max(1, int(args.warmup_epochs * steps_per_epoch)),
                              args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp": use_amp,
        "parameters": sum(p.numel() for p in params),
        "split_images": {k: len(v) for k, v in sets.items()},
        "steps_per_epoch": steps_per_epoch,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"centernet-r18: {config['parameters']:,} params | frames {config['split_images']} | "
          f"lr {args.lr:g} bs {args.batch_size} | {config['device']} amp={use_amp}", flush=True)

    history: list[dict] = []
    best_ap, best_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = dict.fromkeys(LOSS_KEYS, 0.0)
        n_batches = 0
        for imgs, tgts in loader:
            x = train_pre(imgs.to(device, non_blocking=True))
            tgts = {k: v.to(device, non_blocking=True) for k, v in tgts.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            losses = centernet_loss({k: v.float() for k, v in out.items()}, tgts)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for k in LOSS_KEYS:
                sums[k] += losses[k].item()
            n_batches += 1

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **{f"loss_{k}": v / n_batches for k, v in sums.items()}}
        msg = (f"ep {epoch:>3} | loss {row['loss_total']:.3f} (hm {row['loss_heatmap']:.3f} "
               f"size {row['loss_size']:.3f} off {row['loss_offset']:.3f})")
        if not math.isfinite(row["loss_total"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}: {row}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(sets["val"].samples, predict(model, sets["val"], eval_pre, device, args.batch_size, use_amp))
            row.update({f"val_{k}": m[k] for k in ("AP", "AP50", "AP75", "AR")})
            improved = m["AP"] > best_ap
            if improved:
                best_ap, best_epoch = m["AP"], epoch
                torch.save(model.state_dict(), run_dir / "best.pt")
            msg += f" | val AP {m['AP']:.3f} AP50 {m['AP50']:.3f} AR {m['AR']:.3f}{' *' if improved else ''}"
        history.append(row)
        print(msg, flush=True)
    train_seconds = time.time() - t0

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    results = {"best_epoch": best_epoch, "train_seconds": round(train_seconds, 1)}
    for split in ("val", "test"):
        results[split] = evaluate(sets[split].samples,
                                  predict(model, sets[split], eval_pre, device, args.batch_size, use_amp))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    te = results["test"]
    print(f"\nbest epoch {best_epoch} ({train_seconds:.0f}s) | test AP {te['AP']:.3f} AP50 {te['AP50']:.3f} "
          f"AP75 {te['AP75']:.3f} AR {te['AR']:.3f}")
    print("  per class: " + "  ".join(f"{n} {v:.3f}" for n, v in te["AP_per_class"].items()))
    print("  by area:   " + "  ".join(f"{n} {v:.3f}" for n, v in te["AP_by_area"].items()))
    print(f"run dir: {run_dir}")
    return {"config": config, "history": history, "results": results}


if __name__ == "__main__":
    main()
