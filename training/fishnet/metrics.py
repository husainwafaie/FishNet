"""Classification metrics, including accuracy stratified by instance attributes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    idx = num_classes * np.asarray(y_true) + np.asarray(y_pred)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    precision = _safe_div(tp, cm.sum(axis=0).astype(float))
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    present = support > 0
    return {
        "n": int(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion": cm.tolist(),
    }


def binned_accuracy(values: np.ndarray, correct: np.ndarray, edges: Sequence[float]) -> list[dict]:
    """Accuracy within half-open bins [edges[i], edges[i+1])."""
    values, correct = np.asarray(values), np.asarray(correct, dtype=bool)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        out.append({"lo": lo, "hi": hi, "n": n, "accuracy": float(correct[mask].mean()) if n else None})
    return out
