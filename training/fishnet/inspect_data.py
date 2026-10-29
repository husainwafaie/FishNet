"""Dataset statistics and visual sanity checks: split/class counts, label overlays, crop grids."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from . import DATASET_ROOT, RUNS_ROOT
from .crops import CropDataset
from .labels import CLASS_NAMES, Sample, load_samples
from .plots import plot_crop_grid
from .splits import SPLITS, partition

OVERLAY_COLORS = {0: (127, 212, 255), 1: (255, 209, 102), 2: (255, 107, 107)}


def split_stats(samples: list[Sample]) -> dict:
    pos, ign = Counter(), Counter()
    for s in samples:
        for b in s.boxes:
            (ign if b.ignore else pos)[CLASS_NAMES[b.class_id]] += 1
    return {
        "images": len(samples),
        "images_without_positives": sum(all(b.ignore for b in s.boxes) for s in samples),
        "positives": {n: pos[n] for n in CLASS_NAMES},
        "ignore": {n: ign[n] for n in CLASS_NAMES},
    }


def draw_overlay(sample: Sample, path: Path) -> None:
    with Image.open(sample.image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for b in sample.boxes:
            color = (160, 160, 160) if b.ignore else OVERLAY_COLORS[b.class_id]
            draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=color, width=1)
        im.save(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATASET_ROOT)
    p.add_argument("--out", type=Path, default=RUNS_ROOT / "inspect")
    p.add_argument("--overlays", type=int, default=8)
    p.add_argument("--input-size", type=int, default=96)
    args = p.parse_args(argv)

    samples = load_samples(args.data)
    parts = partition(samples, lambda s: s.key)
    stats = {split: split_stats(parts[split]) for split in SPLITS}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))

    header = f"{'split':<6}{'images':>8}" + "".join(f"{n:>10}" for n in CLASS_NAMES) + f"{'ignore':>9}"
    print(header)
    for split, st in stats.items():
        row = f"{split:<6}{st['images']:>8}" + "".join(f"{st['positives'][n]:>10}" for n in CLASS_NAMES)
        print(row + f"{sum(st['ignore'].values()):>9}")

    overlay_dir = args.out / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    for s in samples[: args.overlays]:
        draw_overlay(s, overlay_dir / f"{s.key}.png")

    train = CropDataset(parts["train"], args.input_size, train=False, desc="train crops")
    plot_crop_grid(train.images, train.labels, CLASS_NAMES, args.out / "crops_train.png")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
