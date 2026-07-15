# fishnet

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

**Status: Stage 1 (data generation) complete.**

## Roadmap

| Stage | What | Where |
|---|---|---|
| 1 ✅ | Synthetic scene + automatic pixel-accurate labels | `datagen/` |
| 2 | Species classifier, then object detector (PyTorch) | `training/` |
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

**Capture 1 / Capture 100** renders randomized samples and writes them to
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
  "image": "2026-07-14T06-59-50_00000.png",
  "width": 730, "height": 738,
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
```
