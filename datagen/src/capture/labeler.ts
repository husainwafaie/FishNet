import * as THREE from 'three';
import type { FishField } from '../scene/FishField';
import { FOG_DENSITY } from '../scene/environment';

/**
 * Ground-truth extraction: for every fish instance, transform the 8 corners
 * of its local-space bounding box by its instance matrix and project them
 * through the camera; the screen-space min/max rectangle is the label.
 *
 * Because layouts and camera poses are the only inputs, these labels are
 * pixel-accurate by construction — no annotation, no heuristic detection.
 */

export interface FishBox {
  species: string;
  classId: number;
  /** Pixel-space box, top-left origin, clamped to the viewport. */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Camera-to-fish distance, world units. */
  distance: number;
  /** Fog transmittance at that distance (0 = fully fogged, 1 = clear). */
  visibility: number;
  /**
   * COCO-style ignore region: the fish is visible but too small to be a
   * trustworthy positive (a speck). Training should mask these regions out
   * of the loss rather than treat them as background — otherwise visible
   * fish the labeler skipped would teach the detector to miss small fish.
   */
  ignore?: true;
}

/** Below this transmittance a fish is invisible in the murk — not labeled. */
const MIN_VISIBILITY = 0.12;
/** Boxes at least this big (after viewport clamping) are real positives. */
const MIN_AREA_PX = 80;
/** Boxes between this and MIN_AREA_PX become ignore regions; below, drop. */
const IGNORE_AREA_PX = 6;
/** Require this fraction of the box to survive viewport clamping. */
const MIN_INSIDE_FRACTION = 0.25;

const corner = new THREE.Vector3();
const viewPos = new THREE.Vector3();
const centre = new THREE.Vector3();

export function computeLabels(
  camera: THREE.PerspectiveCamera,
  field: FishField,
  width: number,
  height: number,
): FishBox[] {
  camera.updateMatrixWorld();
  const boxes: FishBox[] = [];

  for (const group of field.groups) {
    const bb = group.boundsLocal;
    for (const fish of group.fish) {
      centre.setFromMatrixPosition(fish.matrix);
      const distance = camera.position.distanceTo(centre);

      // Same curve the renderer uses: FogExp2 factor = 1 − e^(−(d·dist)²)
      const visibility = Math.exp(-((FOG_DENSITY * distance) ** 2));
      if (visibility < MIN_VISIBILITY) continue;

      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      let behindCamera = false;

      for (let c = 0; c < 8; c++) {
        corner.set(
          c & 1 ? bb.max.x : bb.min.x,
          c & 2 ? bb.max.y : bb.min.y,
          c & 4 ? bb.max.z : bb.min.z,
        );
        corner.applyMatrix4(fish.matrix);

        // Projection is undefined for points at/behind the eye plane; a fish
        // straddling it would need polygon clipping — skip those rare cases
        viewPos.copy(corner).applyMatrix4(camera.matrixWorldInverse);
        if (viewPos.z > -camera.near) {
          behindCamera = true;
          break;
        }

        corner.project(camera);
        const px = (corner.x * 0.5 + 0.5) * width;
        const py = (0.5 - corner.y * 0.5) * height;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }
      if (behindCamera) continue;

      const fullArea = (maxX - minX) * (maxY - minY);
      const x0 = THREE.MathUtils.clamp(minX, 0, width);
      const y0 = THREE.MathUtils.clamp(minY, 0, height);
      const x1 = THREE.MathUtils.clamp(maxX, 0, width);
      const y1 = THREE.MathUtils.clamp(maxY, 0, height);
      const w = x1 - x0;
      const h = y1 - y0;
      const area = w * h;
      if (area < IGNORE_AREA_PX) continue;
      if (fullArea > 0 && area / fullArea < MIN_INSIDE_FRACTION) continue;

      boxes.push({
        ...(area < MIN_AREA_PX ? { ignore: true as const } : {}),
        species: group.cfg.name,
        classId: group.cfg.classId,
        x: Math.round(x0),
        y: Math.round(y0),
        w: Math.round(w),
        h: Math.round(h),
        distance: Math.round(distance * 100) / 100,
        visibility: Math.round(visibility * 1000) / 1000,
      });
    }
  }

  return boxes;
}
