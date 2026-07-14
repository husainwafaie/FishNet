import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createEnvironment } from './scene/environment';
import { FishField } from './scene/FishField';
import { computeLabels } from './capture/labeler';
import { Overlay } from './capture/overlay';
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

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  controls.update();
  env.update(camera.position.y);
  field.update(clock.getElapsedTime());
  renderer.render(scene, camera);
  if (showBoxes) {
    overlay.draw(computeLabels(camera, field, window.innerWidth, window.innerHeight));
  }
});
