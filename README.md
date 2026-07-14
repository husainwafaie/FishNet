# fishnet

Synthetic-data fish detection — an end-to-end ML project: a procedural underwater
scene generates perfectly labeled training images, a PyTorch model learns to detect
fish species in them, and the trained model runs live in the browser.

**Status: Stage 1 — synthetic data generator.** See roadmap below.

## Roadmap

1. **Data generation** (`datagen/`) — three.js underwater scene with procedural fish
   species; renders frames with automatic pixel-accurate bounding-box labels
2. **Training** (`training/`) — species classifier, then object detector, in PyTorch
3. **Evaluation** — mAP split by depth band and distance; failure-mode analysis
4. **Deployment** — ONNX export, quantization, real-time in-browser inference

## Development

```bash
cd datagen
npm install
npm run dev
```
