# FishNet

[![CI](https://github.com/husainwafaie/FishNet/actions/workflows/ci.yml/badge.svg)](https://github.com/husainwafaie/FishNet/actions/workflows/ci.yml)

Synthetic-data fish detection — an end-to-end ML project. A procedural three.js
underwater scene generates unlimited, perfectly labeled training images; a PyTorch
model learns to detect and classify fish species in them; the trained model ships
back into the browser for real-time inference.

The premise: the expensive part of computer vision is labeled data. If you own the
renderer, labels are free — every fish's position, species, and screen-space
bounding box can be computed exactly, for any frame, with zero annotation. This is
the same synthetic-data technique used in robotics and autonomous driving, at
hobby scale.

**Status: Stage 2a (species classifier) complete; 2b (detector) next.**

## Roadmap

| Stage | What | Where |
|---|---|---|
| 1 ✅ | Synthetic scene + automatic pixel-accurate labels | `datagen/` |
| 2a ✅ | Per-instance species classifier; data-efficiency study | `training/` |
| 2b | Object detector (PyTorch) | `training/` |
| 3 | Evaluation: mAP by depth band and distance, failure analysis | `training/` |
| 4 | ONNX export, quantization, live in-browser inference | `datagen/` |

## Stage 1 — the data generator

```bash
cd datagen
npm install
npm run dev   # http://localhost:5175
```

An interactive underwater scene: three procedural fish species (anchovy, jack,
grouper) scattered in schools through a 90×50×90 water volume, under exponential
fog whose color darkens with depth. **Boxes: on** overlays the computed labels
live — if the boxes hug the fish, the dataset is trustworthy.

**Capture 1 / 100 / 1000** renders randomized 640×640 samples and writes them to
`dataset/images/*.png` + `dataset/labels/*.json` via a dev-server endpoint. Each
sample randomizes:

- the entire fish layout (seeded — the seed is recorded, so any sample can be
  exactly re-rendered)
- camera position, aim, and field of view (aim is biased toward fish so most
  frames have content; empty frames are kept occasionally as natural negatives)
- tail-wag phase

### Label format

```jsonc
{
  "image": "2026-10-29T07-21-28_00001.png",
  "width": 640, "height": 640,
  "seed": 622120360,                  // reproduces layout + camera exactly
  "camera": { "position": [...], "target": [...], "fov": 56.7 },
  "boxes": [
    { "species": "jack", "classId": 1,
      "x": 635, "y": 246, "w": 21, "h": 12,   // pixels, top-left origin
      "distance": 57.96,                       // world units from camera
      "visibility": 0.197 },                   // fog transmittance, 0–1
    { "species": "anchovy", "classId": 0, "ignore": true, ... }
  ]
}
```

### Design decisions that matter for training

- **Visibility is physical, not heuristic.** The labeler computes fog
  transmittance with the exact FogExp2 curve the renderer applies
  (`e^−(density·dist)²`). Fish below 12% transmittance are genuinely invisible in
  the pixels and are not labeled.
- **Ignore regions.** Fish that are visible but tiny (< 80 px²) become
  COCO-style `ignore: true` boxes. Training should mask these regions out of the
  loss — treating specks as background would teach the detector to miss small
  fish, and treating them as positives would add noise.
- **The intended difficulty is depth.** Near the surface the species are easy to
  separate by color. At depth, fog strips color and everything converges to
  murky silhouettes — so the depth/distance fields in every label enable
  evaluation sliced by domain shift (e.g. mAP above vs. below 30 m).
- **Honest caveat:** train and test imagery come from the same renderer, so
  there is no sim-to-real gap to bridge. The studied domain gap here is
  depth/turbidity, and the difficulty can be raised later (more species, tint
  overlap, turbidity randomization).

## Stage 2a — species classifier

```bash
conda env create -f training/environment.yml   # py3.12, torch cu128, editable install
conda activate fishnet
cd training
python -m fishnet.inspect_data                  # split stats, label overlays, crop grids
python -m fishnet.train_classifier --model cnn  # or --model resnet18 [--no-pretrained]
python -m fishnet.sweep_data_efficiency         # learning curves
pytest
```

Non-ignore instances are cropped (square, 25% context pad, shifted inside the
frame) and resized to 64×64. Splits are image-level and assigned by SHA-256 of the
sample key (70/15/15), so crops of one frame never straddle splits and assignments
are stable as the dataset grows.

Dataset: 1,001 frames → 8,876 / 1,859 / 1,694 train/val/test instances; class
imbalance ~10:4:1 (jack:anchovy:grouper), handled with inverse-frequency weighted
cross-entropy and model selection on val balanced accuracy.

**Training:** AdamW, linear warmup + cosine decay, batched on-device augmentation
(hflip, affine RandomResizedCrop, brightness/contrast; no hue/saturation jitter
since colour is a class cue), precise BN before each evaluation, fp32.

### Results (test split)

| Model | Params | Balanced acc | Macro-F1 |
|---|---|---|---|
| SmallCNN (scratch) | 1.17M | 0.995 | 0.995 |
| ResNet-18 (scratch) | 11.2M | 0.987 | 0.988 |
| ResNet-18 (ImageNet init) | 11.2M | 0.991 | 0.992 |

At full data the task saturates: accuracy is flat (≥0.99) across fog visibility
and box-area strata, so the full-data comparison is uninformative. The
data-efficiency sweep (each run ≥1,500 optimizer steps; single seed, so
sub-percent differences are within noise) separates the models:

![Learning curve](docs/figures/classifier_learning_curve.png)

- ImageNet initialisation dominates in the low-data regime: 0.971 balanced
  accuracy from 14 training frames (159 instances) vs. 0.836–0.843 from scratch.
- Above ~36 frames the scratch SmallCNN matches it; scratch ResNet-18 lags at
  every size, consistent with its 10× parameter count relative to the data.

### Training-stability findings

- **BN running-stat lag.** With EMA BN statistics, eval-mode val accuracy
  oscillated between epochs (e.g. 0.75 / 0.64 balanced) while train accuracy rose
  smoothly. Evaluating the same weights with batch statistics, or with running
  stats re-estimated on the training set, rose monotonically (0.97 → 0.99),
  isolating the cause to running-stat lag rather than augmentation shift.
  Precise BN (`swa_utils.update_bn`) before each eval removes the oscillation and
  brings val ≥ 0.99 forward from epoch 14 to epoch 3.
  ![Baseline curves with precise BN](docs/figures/cnn_baseline_curves.png)
- **fp16 on GTX 16-series.** Under fp16 autocast, cuDNN conv kernels produced NaN
  logits for some batch shapes (ResNet-18, n=31); GradScaler silently skipped
  those steps. BN statistic accumulation under autocast also overflowed. AMP is
  therefore opt-in (`--amp`).

## Architecture

```
datagen/
├── vite.config.ts        Dev server + /api/save endpoint (writes dataset/)
└── src/
    ├── lib/prng.ts       Seeded PRNG — every layout is a pure function of seed
    ├── scene/
    │   ├── environment.ts  Depth-graded fog/lighting, dune seafloor
    │   ├── species.ts      The class list: sizes, colors, depth bands
    │   ├── fishGeometry.ts Procedural fish (head at +Z — labeler relies on it)
    │   └── FishField.ts    InstancedMesh per species; CPU-side transforms
    └── capture/
        ├── labeler.ts      Projects instance bboxes → pixel-space labels
        ├── overlay.ts      Live label visualization (the proof of correctness)
        └── recorder.ts     Randomized batch capture → POST /api/save

training/
├── environment.yml       conda env (py3.12, torch cu128)
├── fishnet/
│   ├── labels.py         Label JSON → typed Sample/Box; class list
│   ├── splits.py         Hash-based image-level train/val/test
│   ├── crops.py          Instance crops, cached uint8 NCHW, on-device batching
│   ├── augment.py        Batched on-device augmentation + normalisation
│   ├── models.py         SmallCNN, ResNet-18
│   ├── metrics.py        Confusion, per-class P/R/F1, stratified accuracy
│   ├── train_classifier.py
│   ├── sweep_data_efficiency.py
│   └── inspect_data.py
└── tests/
```
