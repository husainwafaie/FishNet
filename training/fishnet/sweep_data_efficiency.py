"""Learning curve: test metrics vs. train-set size for scratch CNN, scratch ResNet-18, and ImageNet-pretrained ResNet-18.

Train subsets are image-level; val/test are always the full splits. Every run gets
at least --min-steps optimizer steps so small subsets are not under-optimised.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import RUNS_ROOT  # noqa: E402
from .train_classifier import main as train  # noqa: E402

VARIANTS: dict[str, list[str]] = {
    "cnn": ["--model", "cnn"],
    "resnet18-scratch": ["--model", "resnet18", "--no-pretrained"],
    "resnet18": ["--model", "resnet18"],
}
FRACTIONS = [0.02, 0.05, 0.1, 0.25, 1.0]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--min-steps", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=25)
    args = p.parse_args(argv)

    out_dir = RUNS_ROOT / "sweep_data_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in args.variants:
        for frac in args.fractions:
            for seed in args.seeds:
                name = f"{variant}-f{frac:g}-s{seed}"
                metrics_path = RUNS_ROOT / "classifier" / "sweep" / name / "metrics.json"
                config_path = metrics_path.with_name("config.json")
                if not metrics_path.exists():
                    train([*VARIANTS[variant], "--train-fraction", str(frac), "--seed", str(seed),
                           "--min-steps", str(args.min_steps), "--epochs", str(args.epochs),
                           "--run-name", f"sweep/{name}"])
                metrics = json.loads(metrics_path.read_text())
                config = json.loads(config_path.read_text())
                te = metrics["test"]
                rows.append({
                    "variant": variant,
                    "fraction": frac,
                    "seed": seed,
                    "train_images": config["split_images"]["train"],
                    "train_instances": config["split_instances"]["train"],
                    "test_accuracy": te["accuracy"],
                    "test_balanced_accuracy": te["balanced_accuracy"],
                    "test_macro_f1": te["macro_f1"],
                    "grouper_recall": te["per_class"]["grouper"]["recall"],
                    "train_seconds": metrics["train_seconds"],
                })
                print(f"[sweep] {name}: bal {te['balanced_accuracy']:.3f} (n_img={rows[-1]['train_images']})",
                      flush=True)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for variant in args.variants:
        pts = sorted((r["train_images"], r["test_balanced_accuracy"]) for r in rows if r["variant"] == variant)
        ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=variant)
    ax.set_xscale("log")
    ticks = sorted({r["train_images"] for r in rows})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set(xlabel="training images (log scale)", ylabel="test balanced accuracy",
           title="Data efficiency: scratch vs. ImageNet-pretrained")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=120)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
