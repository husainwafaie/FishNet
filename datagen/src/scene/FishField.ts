import * as THREE from 'three';
import { mulberry32, rangeFrom } from '../lib/prng';
import { createFishGeometry } from './fishGeometry';
import { SPECIES, type SpeciesConfig } from './species';
import { FLOOR_Y } from './environment';

/**
 * FishField — one InstancedMesh per species (one draw call each), with the
 * entire layout a pure function of a seed. `layout(seed)` scatters schools
 * of fish through the water volume; the per-instance transforms are kept on
 * the CPU so the labeler can project exact bounding boxes for every fish.
 *
 * Tail wag runs in the vertex shader (patched via onBeforeCompile so fish
 * keep scene lighting and fog). The wag displaces vertices laterally by at
 * most WAG_AMP·scale, which is inside the bounding-box slack the tail-fin
 * quad already provides, so labels stay valid at any wag phase.
 */

const MAX_PER_SPECIES = 512;
/** School centres land in [-VOLUME_XZ, VOLUME_XZ]² horizontally. */
export const VOLUME_XZ = 40;
/** Fish never spawn above this (they'd poke through the "surface"). */
const CEILING_Y = -1.5;

const WAG_AMP = 0.07;

const FISH_VERT_DECLS = /* glsl */ `
  uniform float uTime;
  uniform float uTailFreq;
  uniform float uWagAmp;
  attribute float aPhase;
`;

// Local z runs tail (−0.78) → head (+0.5); weight ramps toward the tail and
// the position.z term makes the sine a travelling body wave
const FISH_VERT_WAG = /* glsl */ `
  float tailWeight = 1.0 - smoothstep(-0.5, 0.35, position.z);
  transformed.x += sin(uTime * uTailFreq + aPhase + position.z * 4.0)
                   * uWagAmp * tailWeight;
`;

export interface FishInstance {
  /** World transform — cloned per layout, consumed by the labeler. */
  matrix: THREE.Matrix4;
  /** Body length (uniform scale). */
  scale: number;
}

export interface SpeciesGroup {
  cfg: SpeciesConfig;
  mesh: THREE.InstancedMesh;
  /** Local-space bounds of the fish geometry (shared by all instances). */
  boundsLocal: THREE.Box3;
  fish: FishInstance[];
}

export class FishField {
  readonly groups: SpeciesGroup[] = [];
  private readonly uniformsPerSpecies: Array<Record<string, THREE.IUniform> | null>;

  constructor(scene: THREE.Scene) {
    this.uniformsPerSpecies = SPECIES.map(() => null);

    SPECIES.forEach((cfg, si) => {
      const geometry = createFishGeometry(cfg.bodyAspect);
      geometry.computeBoundingBox();

      // Fixed per-instance wag phases (never relaid out — no need)
      const phaseRand = mulberry32(0xf15b0e5 + si * 131);
      const phases = new Float32Array(MAX_PER_SPECIES);
      for (let i = 0; i < MAX_PER_SPECIES; i++) phases[i] = phaseRand() * Math.PI * 2;
      geometry.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phases, 1));

      const material = new THREE.MeshStandardMaterial({
        color: cfg.color,
        roughness: 0.4,
        metalness: 0.45,
        side: THREE.DoubleSide, // tail fin is a flat quad
      });
      material.onBeforeCompile = (shader) => {
        shader.uniforms.uTime = { value: 0 };
        shader.uniforms.uTailFreq = { value: cfg.tailFreq };
        shader.uniforms.uWagAmp = { value: WAG_AMP };
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', `#include <common>\n${FISH_VERT_DECLS}`)
          .replace('#include <begin_vertex>', `#include <begin_vertex>\n${FISH_VERT_WAG}`);
        this.uniformsPerSpecies[si] = shader.uniforms;
      };
      material.customProgramCacheKey = () => `fish-${cfg.name}`;

      const mesh = new THREE.InstancedMesh(geometry, material, MAX_PER_SPECIES);
      mesh.count = 0;
      // Instances sit far from the geometry's origin-centred bounding sphere
      mesh.frustumCulled = false;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      scene.add(mesh);

      this.groups.push({
        cfg,
        mesh,
        boundsLocal: geometry.boundingBox!.clone(),
        fish: [],
      });
    });
  }

  /** Deterministically rebuild the whole fish population from a seed. */
  layout(seed: number): void {
    const rand = mulberry32(seed);
    const range = rangeFrom(rand);
    const dummy = new THREE.Object3D();
    const base = new THREE.Color();
    const tinted = new THREE.Color();

    for (const group of this.groups) {
      const { cfg, mesh } = group;
      group.fish.length = 0;
      base.set(cfg.color);

      const schoolCount = Math.round(range(cfg.schoolCount));
      let i = 0;
      for (let s = 0; s < schoolCount; s++) {
        const cx = (rand() * 2 - 1) * VOLUME_XZ;
        const cz = (rand() * 2 - 1) * VOLUME_XZ;
        const cy = range(cfg.yBand);
        const radius = range(cfg.schoolRadius);
        // Schools swim coherently: shared heading, per-fish jitter
        const heading = rand() * Math.PI * 2;
        const fishCount = Math.round(range(cfg.fishPerSchool));

        for (let f = 0; f < fishCount && i < MAX_PER_SPECIES; f++, i++) {
          const theta = rand() * Math.PI * 2;
          const rr = Math.sqrt(rand()) * radius;
          const px = cx + Math.cos(theta) * rr;
          const pz = cz + Math.sin(theta) * rr;
          const py = THREE.MathUtils.clamp(
            cy + (rand() - 0.5) * radius * 0.7,
            FLOOR_Y + 3,
            CEILING_Y,
          );

          dummy.position.set(px, py, pz);
          dummy.rotation.set((rand() - 0.5) * 0.3, heading + (rand() - 0.5) * 0.7, 0, 'YXZ');
          const scale = range(cfg.bodyLength);
          dummy.scale.setScalar(scale);
          dummy.updateMatrix();

          mesh.setMatrixAt(i, dummy.matrix);
          const jitter = 1 + (rand() - 0.5) * 2 * cfg.tintJitter;
          tinted.copy(base).multiplyScalar(jitter);
          mesh.setColorAt(i, tinted);
          group.fish.push({ matrix: dummy.matrix.clone(), scale });
        }
      }

      mesh.count = i;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  /** Advance the GPU tail-wag clock (uniforms exist once shaders compile). */
  update(time: number): void {
    for (const uniforms of this.uniformsPerSpecies) {
      if (uniforms) uniforms.uTime.value = time;
    }
  }
}
