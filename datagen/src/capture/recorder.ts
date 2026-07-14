import * as THREE from 'three';
import { mulberry32 } from '../lib/prng';
import { computeLabels, type FishBox } from './labeler';
import { FishField, VOLUME_XZ } from '../scene/FishField';
import type { Environment } from '../scene/environment';

/**
 * Batch capture: every sample randomizes the fish layout, camera pose, and
 * tail-wag phase, renders one frame, computes ground-truth labels, and posts
 * the PNG + JSON pair to the dev server's /api/save endpoint.
 *
 * Each sample records its seed, so any image in the dataset can be exactly
 * re-rendered later (layout and camera derive from the seed alone).
 */

export interface CaptureContext {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  field: FishField;
  env: Environment;
}

export interface SampleLabel {
  image: string;
  width: number;
  height: number;
  seed: number;
  camera: {
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
  };
  boxes: FishBox[];
}

/** Frames with no visible fish get resampled this many times before being
 * kept as a natural negative example. */
const MAX_EMPTY_RESAMPLES = 4;

function randomizeCamera(
  camera: THREE.PerspectiveCamera,
  field: FishField,
  rand: () => number,
): [number, number, number] {
  camera.position.set(
    (rand() * 2 - 1) * (VOLUME_XZ + 2),
    -3 - rand() * 41, // y ∈ [-44, -3] — below surface, above dune crests
    (rand() * 2 - 1) * (VOLUME_XZ + 2),
  );
  camera.fov = 45 + rand() * 25;
  camera.updateProjectionMatrix();

  // Bias toward content: usually aim near a random fish, sometimes anywhere
  const target = new THREE.Vector3(
    (rand() * 2 - 1) * VOLUME_XZ,
    -5 - rand() * 40,
    (rand() * 2 - 1) * VOLUME_XZ,
  );
  if (rand() < 0.7) {
    const groups = field.groups.filter((g) => g.fish.length > 0);
    if (groups.length > 0) {
      const group = groups[Math.floor(rand() * groups.length)];
      const fish = group.fish[Math.floor(rand() * group.fish.length)];
      target.setFromMatrixPosition(fish.matrix);
      target.x += (rand() * 2 - 1) * 6;
      target.y += (rand() * 2 - 1) * 4;
      target.z += (rand() * 2 - 1) * 6;
    }
  }
  camera.lookAt(target);
  return [target.x, target.y, target.z];
}

async function canvasToBase64Png(canvas: HTMLCanvasElement): Promise<string> {
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png');
  });
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
  return dataUrl.slice(dataUrl.indexOf(',') + 1);
}

export async function captureBatch(
  count: number,
  ctx: CaptureContext,
  onProgress: (done: number, total: number) => void,
): Promise<void> {
  const { renderer, scene, camera, field, env } = ctx;
  const canvas = renderer.domElement;
  const runId = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

  for (let i = 0; i < count; i++) {
    let seed = 0;
    let boxes: FishBox[] = [];
    let target: [number, number, number] = [0, 0, 0];

    for (let attempt = 0; attempt <= MAX_EMPTY_RESAMPLES; attempt++) {
      seed = (Math.random() * 0xffffffff) >>> 0;
      const rand = mulberry32(seed ^ 0x9e3779b9);
      field.layout(seed);
      target = randomizeCamera(camera, field, rand);
      field.update(rand() * 100); // random tail-wag snapshot
      env.update(camera.position.y);
      renderer.render(scene, camera);
      boxes = computeLabels(camera, field, canvas.width, canvas.height);
      if (boxes.length > 0) break;
    }

    const name = `${runId}_${String(i).padStart(5, '0')}`;
    const label: SampleLabel = {
      image: `${name}.png`,
      width: canvas.width,
      height: canvas.height,
      seed,
      camera: {
        position: [camera.position.x, camera.position.y, camera.position.z],
        target,
        fov: camera.fov,
      },
      boxes,
    };

    const imageBase64 = await canvasToBase64Png(canvas);
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name, imageBase64, label }),
    });
    if (!res.ok) throw new Error(`save failed (${res.status}): ${await res.text()}`);

    onProgress(i + 1, count);
  }
}
