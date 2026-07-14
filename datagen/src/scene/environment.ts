import * as THREE from 'three';
import { mulberry32 } from '../lib/prng';

/** Seafloor rest height; dunes rise a few units above this. */
export const FLOOR_Y = -50;

/**
 * FogExp2 density. Exported because the labeler derives per-fish visibility
 * from the exact same curve the renderer uses: transmittance
 * = exp(-(density·dist)²).
 */
export const FOG_DENSITY = 0.022;

const SURFACE_COLOR = new THREE.Color('#1d5f7d');
const DEEP_COLOR = new THREE.Color('#04101a');

export interface Environment {
  /** Re-grade fog colour and light intensity for the camera's depth. */
  update(cameraY: number): void;
}

/**
 * Underwater atmosphere: exponential fog whose colour darkens with camera
 * depth (the poor man's colour absorption — reds/greens die first because the
 * deep colour is nearly pure blue-black), plus lights that dim as you sink.
 * This is what makes species colour an unreliable cue for the detector at
 * depth, which is the intended difficulty of the dataset.
 */
export function createEnvironment(scene: THREE.Scene): Environment {
  const fog = new THREE.FogExp2(SURFACE_COLOR.clone(), FOG_DENSITY);
  scene.fog = fog;
  // Same Color instance: the background greys out with depth automatically
  scene.background = fog.color;

  const hemi = new THREE.HemisphereLight('#9fd8ef', '#0a1a24', 0.9);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight('#cfeaf5', 1.6);
  sun.position.set(20, 60, 10);
  scene.add(sun);

  scene.add(createSeafloor());

  return {
    update(cameraY: number): void {
      const t = THREE.MathUtils.clamp(-cameraY / -FLOOR_Y, 0, 1);
      fog.color.lerpColors(SURFACE_COLOR, DEEP_COLOR, t);
      sun.intensity = 1.6 * (1 - 0.8 * t);
      hemi.intensity = 0.9 * (1 - 0.6 * t);
    },
  };
}

function createSeafloor(): THREE.Mesh {
  const geo = new THREE.PlaneGeometry(400, 400, 96, 96);
  geo.rotateX(-Math.PI / 2);

  // Low-frequency dunes from three fixed-phase sines — deterministic, so the
  // floor never changes between layouts and fish depth bands can trust it
  const rand = mulberry32(0x5eaf100e);
  const p1 = rand() * Math.PI * 2;
  const p2 = rand() * Math.PI * 2;
  const p3 = rand() * Math.PI * 2;
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const h =
      Math.sin(x * 0.05 + p1) * 1.2 +
      Math.sin(z * 0.08 + p2) * 0.9 +
      Math.sin((x + z) * 0.03 + p3) * 1.5;
    pos.setY(i, FLOOR_Y + h);
  }
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({ color: '#33413f', roughness: 1 });
  return new THREE.Mesh(geo, mat);
}
