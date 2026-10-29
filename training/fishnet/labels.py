"""Parsing of datagen label files (dataset/labels/*.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Index == classId emitted by datagen/src/scene/species.ts
CLASS_NAMES: tuple[str, ...] = ("anchovy", "jack", "grouper")
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float
    distance: float
    visibility: float
    ignore: bool

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class Sample:
    key: str
    image_path: Path
    width: int
    height: int
    seed: int
    camera_y: float
    boxes: tuple[Box, ...]


def _parse_box(raw: dict) -> Box:
    class_id = int(raw["classId"])
    if CLASS_NAMES[class_id] != raw["species"]:
        raise ValueError(f"classId {class_id} != species {raw['species']!r}")
    return Box(
        class_id=class_id,
        x=float(raw["x"]),
        y=float(raw["y"]),
        w=float(raw["w"]),
        h=float(raw["h"]),
        distance=float(raw["distance"]),
        visibility=float(raw["visibility"]),
        ignore=bool(raw.get("ignore", False)),
    )


def load_sample(label_path: Path, images_dir: Path) -> Sample:
    raw = json.loads(label_path.read_text())
    return Sample(
        key=label_path.stem,
        image_path=images_dir / raw["image"],
        width=int(raw["width"]),
        height=int(raw["height"]),
        seed=int(raw["seed"]),
        camera_y=float(raw["camera"]["position"][1]),
        boxes=tuple(_parse_box(b) for b in raw["boxes"]),
    )


def load_samples(root: Path) -> list[Sample]:
    """All samples under `root` (expects root/labels and root/images), key-sorted."""
    labels_dir, images_dir = root / "labels", root / "images"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"no labels directory at {labels_dir}")
    return [load_sample(p, images_dir) for p in sorted(labels_dir.glob("*.json"))]
