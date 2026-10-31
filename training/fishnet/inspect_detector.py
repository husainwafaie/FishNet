"""Error analysis for a trained detector run.

Picks the operating score threshold that maximises F1@IoU0.5 on val, then on the test
split reports recall stratified by class, box area, fog visibility and occlusion
(nearer-instance box coverage), classifies false positives (wrong class /
mislocalised / background), and renders overlays of the worst frames.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import DATASET_ROOT, RUNS_ROOT
from .augment import Preprocess
from .detection.analysis import box_iou, greedy_match, nearer_coverage
from .detection.data import DetectionSet
from .detection.evaluate import ImageDetections
from .detection.model import CenterNet
from .labels import CLASS_NAMES, NUM_CLASSES, Sample, load_samples
from .splits import partition
from .train_detector import predict

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
AREA_BINS = [0, 150, 400, 1500, float("inf")]
VISIBILITY_BINS = [0.0, 0.3, 0.5, 0.7, 1.01]
COVERAGE_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
COLORS = {0: (127, 212, 255), 1: (255, 209, 102), 2: (255, 107, 107)}
MISS, FALSE_POS = (255, 40, 40), (230, 60, 230)


def _keep(det: ImageDetections, thr: float) -> ImageDetections:
    m = det.scores >= thr
    return ImageDetections(det.boxes[m], det.scores[m], det.classes[m])


def f1_at(samples, dets, thr) -> float:
    tp = fp = fn = 0
    for s, d in zip(samples, dets):
        m = greedy_match(s.boxes, *_fields(_keep(d, thr)))
        tp += int((m.det_status == "tp").sum())
        fp += int((m.det_status == "fp").sum())
        fn += int((~m.gt_matched).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _fields(d: ImageDetections):
    return d.boxes, d.scores, d.classes


def _binned(values, hits, edges) -> list[dict]:
    values, hits = np.asarray(values), np.asarray(hits, bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (values >= lo) & (values < hi)
        rows.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "recall": float(hits[sel].mean()) if sel.any() else None})
    return rows


def is_truncated(b, img_w: int, img_h: int, eps: float = 0.5) -> bool:
    """Box clipped by the viewport (labels are clamped to the frame by the datagen labeler)."""
    return b.x <= eps or b.y <= eps or b.x + b.w >= img_w - eps or b.y + b.h >= img_h - eps


def _fp_kind(box, cls, sample: Sample) -> str:
    pos = [b for b in sample.boxes if not b.ignore]
    if not pos:
        return "background"
    gt = np.array([[b.x, b.y, b.w, b.h] for b in pos])
    iou = box_iou(box[None], gt)[0]
    gt_cls = np.array([b.class_id for b in pos])
    if (iou >= 0.5).any() and (gt_cls[iou >= 0.5] != cls).all():
        return "wrong_class"
    if (iou >= 0.1).any():
        return "mislocalised"  # includes duplicates on an already-matched instance
    return "background"


def draw(sample: Sample, det: ImageDetections, match, path: Path) -> None:
    pos = [b for b in sample.boxes if not b.ignore]
    with Image.open(sample.image_path) as im:
        im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    for b in sample.boxes:
        if b.ignore:
            d.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=(150, 150, 150))
    for b, hit in zip(pos, match.gt_matched):
        if not hit:
            d.rectangle([b.x - 2, b.y - 2, b.x + b.w + 2, b.y + b.h + 2], outline=MISS, width=2)
    for box, cls, status in zip(det.boxes, det.classes, match.det_status):
        if status == "ignored":
            continue
        x, y, w, h = box
        d.rectangle([x, y, x + w, y + h], outline=COLORS[int(cls)] if status == "tp" else FALSE_POS, width=1)
    im.save(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", help="run name under runs/detector/")
    p.add_argument("--data", type=Path, default=DATASET_ROOT)
    p.add_argument("--overlays", type=int, default=12)
    args = p.parse_args(argv)

    run_dir = RUNS_ROOT / "detector" / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNet(NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    parts = partition(load_samples(args.data), lambda s: s.key)
    pre = Preprocess(train=False)

    val = DetectionSet(parts["val"], train=False, desc="val frames")
    val_dets = predict(model, val, pre, device, 8, False)
    f1s = {float(t): f1_at(val.samples, val_dets, t) for t in THRESHOLDS}
    thr = max(f1s, key=f1s.get)

    test = DetectionSet(parts["test"], train=False, desc="test frames")
    test_dets = [_keep(d, thr) for d in predict(model, test, pre, device, 8, False)]

    rec = {"class": [], "area": [], "visibility": [], "coverage": [], "truncated": [], "hit": []}
    fp_kinds: Counter = Counter()
    det_counts: Counter = Counter()
    per_image = []
    for s, d in zip(test.samples, test_dets):
        m = greedy_match(s.boxes, *_fields(d))
        pos = [b for b in s.boxes if not b.ignore]
        cov = nearer_coverage(s.boxes)
        for b, c, hit in zip(pos, cov, m.gt_matched):
            rec["class"].append(b.class_id)
            rec["area"].append(b.area)
            rec["visibility"].append(b.visibility)
            rec["coverage"].append(c)
            rec["truncated"].append(is_truncated(b, s.width, s.height))
            rec["hit"].append(bool(hit))
        for box, cls, status in zip(d.boxes, d.classes, m.det_status):
            det_counts[status] += 1
            if status == "fp":
                fp_kinds[_fp_kind(box, int(cls), s)] += 1
        per_image.append((int((~m.gt_matched).sum()) + int((m.det_status == "fp").sum()), s, d, m))

    hits, cls, trunc = np.array(rec["hit"]), np.array(rec["class"]), np.array(rec["truncated"])
    tp, fp = det_counts["tp"], det_counts["fp"]
    report = {
        "threshold": thr,
        "val_f1_by_threshold": f1s,
        "test": {
            "precision": tp / max(tp + fp, 1),
            "recall": float(hits.mean()),
            "f1": 2 * tp / max(2 * tp + fp + int((~hits).sum()), 1),
            "recall_by_class": {n: float(hits[cls == i].mean()) for i, n in enumerate(CLASS_NAMES)},
            "recall_by_area": _binned(rec["area"], hits, AREA_BINS),
            "recall_by_visibility": _binned(rec["visibility"], hits, VISIBILITY_BINS),
            "recall_by_coverage": _binned(rec["coverage"], hits, COVERAGE_BINS),
            "recall_by_truncation": {
                "interior": {"n": int((~trunc).sum()), "recall": float(hits[~trunc].mean())},
                "frame_edge": {"n": int(trunc.sum()), "recall": float(hits[trunc].mean()) if trunc.any() else None},
            },
            "detections": dict(det_counts),
            "false_positive_kinds": dict(fp_kinds),
        },
    }
    out_dir = run_dir / "analysis"
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    for _, s, d, m in sorted(per_image, key=lambda r: -r[0])[: args.overlays]:
        draw(s, d, m, out_dir / "overlays" / f"{s.key}.png")

    t = report["test"]
    print(f"threshold {thr:.2f} (val F1 {f1s[thr]:.3f}) | test P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f}")
    print("recall by class:      " + "  ".join(f"{k} {v:.3f}" for k, v in t["recall_by_class"].items()))
    for key in ("area", "visibility", "coverage"):
        cells = "  ".join(f"[{r['lo']:g},{r['hi']:g}) {r['recall']:.3f} n={r['n']}" for r in t[f"recall_by_{key}"]
                          if r["recall"] is not None)
        print(f"recall by {key:<11} {cells}")
    tr = t["recall_by_truncation"]
    print(f"recall by truncation interior {tr['interior']['recall']:.3f} n={tr['interior']['n']}  "
          f"frame_edge {tr['frame_edge']['recall'] or 0:.3f} n={tr['frame_edge']['n']}")
    print(f"detections: {dict(det_counts)} | false positives: {dict(fp_kinds)}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
