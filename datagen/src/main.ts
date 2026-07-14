import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createEnvironment } from './scene/environment';
import { FishField } from './scene/FishField';
import { computeLabels } from './capture/labeler';
import { Overlay } from './capture/overlay';
import { captureBatch } from './capture/recorder';
import './style.css';

const app = document.getElementById('app')!;

// preserveDrawingBuffer: capture reads the canvas back after render.
// pixelRatio stays 1 so label pixel coordinates match canvas pixels 1:1.
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 500);
camera.position.set(28, -12, 28);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -18, 0);
controls.enableDamping = true;

const env = createEnvironment(scene);
const field = new FishField(scene);
field.layout(1);

function relayout(): void {
  field.layout((Math.random() * 0xffffffff) >>> 0);
}
document.getElementById('btn-relayout')!.addEventListener('click', relayout);

const overlay = new Overlay(document.getElementById('overlay') as HTMLCanvasElement);
let showBoxes = true;
const overlayBtn = document.getElementById('btn-overlay') as HTMLButtonElement;

function toggleOverlay(): void {
  showBoxes = !showBoxes;
  overlayBtn.textContent = `Boxes: ${showBoxes ? 'on' : 'off'}`;
  if (!showBoxes) overlay.clear();
}
overlayBtn.addEventListener('click', toggleOverlay);

window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') relayout();
  if (e.key === 'b' || e.key === 'B') toggleOverlay();
});

// ── Capture ──────────────────────────────────────────────────────────────
const statusEl = document.getElementById('status')!;
const captureButtons = [
  document.getElementById('btn-capture-1') as HTMLButtonElement,
  document.getElementById('btn-capture-100') as HTMLButtonElement,
];
let capturing = false;

async function runCapture(count: number): Promise<void> {
  if (capturing) return;
  capturing = true; // pauses the interactive loop — capture owns the camera
  captureButtons.forEach((b) => (b.disabled = true));
  overlay.clear();
  try {
    await captureBatch(count, { renderer, scene, camera, field, env }, (done, total) => {
      statusEl.textContent = `capturing ${done}/${total}…`;
    });
    statusEl.textContent = `saved ${count} sample${count === 1 ? '' : 's'} to dataset/`;
  } catch (err) {
    statusEl.textContent = String(err);
  } finally {
    // Hand the camera back to the user in a sane state
    camera.fov = 60;
    camera.updateProjectionMatrix();
    capturing = false;
    captureButtons.forEach((b) => (b.disabled = false));
  }
}
captureButtons[0].addEventListener('click', () => void runCapture(1));
captureButtons[1].addEventListener('click', () => void runCapture(100));

function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  overlay.resize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

const timer = new THREE.Timer();
renderer.setAnimationLoop(() => {
  if (capturing) return;
  timer.update();
  controls.update();
  env.update(camera.position.y);
  field.update(timer.getElapsed());
  renderer.render(scene, camera);
  if (showBoxes) {
    overlay.draw(computeLabels(camera, field, window.innerWidth, window.innerHeight));
  }
});
