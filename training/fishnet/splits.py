"""Deterministic image-level train/val/test assignment.

Splitting is per image (never per crop/box) to prevent leakage between
correlated instances from the same frame. Assignment hashes the sample key,
so it is stable as the dataset grows: existing samples never change split.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence, TypeVar

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

T = TypeVar("T")


def split_of(key: str, val_fraction: float = VAL_FRACTION, test_fraction: float = TEST_FRACTION) -> Split:
    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def partition(items: Sequence[T], key_fn, **fractions) -> dict[Split, list[T]]:
    out: dict[Split, list[T]] = {s: [] for s in SPLITS}
    for item in items:
        out[split_of(key_fn(item), **fractions)].append(item)
    return out
