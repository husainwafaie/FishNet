"""Matplotlib figures for runs and dataset inspection (Agg backend, file output only)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, [h["train_accuracy"] for h in history], label="train acc")
    ax_acc.plot(epochs, [h["val_accuracy"] for h in history], label="val acc")
    ax_acc.plot(epochs, [h["val_balanced_accuracy"] for h in history], label="val balanced acc")
    ax_acc.set(xlabel="epoch", ylim=(0, 1.02), title="Accuracy")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], path: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_crop_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    path: Path,
    per_class: int = 12,
    seed: int = 0,
) -> None:
    """Random uint8 NCHW crops, one row per class."""
    gen = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(per_class * 0.9, len(class_names) * 1.0))
    for c, name in enumerate(class_names):
        idx = torch.nonzero(labels == c).flatten()
        idx = idx[torch.randperm(len(idx), generator=gen)[:per_class]]
        for j in range(per_class):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(images[idx[j]].permute(1, 2, 0).numpy())
        axes[c, 0].set_title(name, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
