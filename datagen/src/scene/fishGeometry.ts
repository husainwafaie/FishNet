import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Procedural fish: an ellipsoid body plus a flat tail-fin quad.
 *
 * Convention the whole pipeline relies on: unit body length along Z with the
 * HEAD at +Z — instance orientation aims fish with a yaw/pitch rotation, and
 * the labeler takes this geometry's bounding box as the local-space extents
 * it projects to screen space.
 */
export function createFishGeometry(bodyAspect: number): THREE.BufferGeometry {
  const body = new THREE.SphereGeometry(0.5, 20, 12);
  // Slim the sphere into a fusiform body: full unit length in Z, height =
  // aspect, width a bit narrower than height
  body.scale(bodyAspect * 0.6, bodyAspect, 1);

  const tail = new THREE.PlaneGeometry(0.32, bodyAspect * 1.3);
  tail.rotateY(Math.PI / 2); // into the Y/Z plane
  tail.translate(0, 0, -0.62); // trailing the body

  const geo = mergeGeometries([body.toNonIndexed(), tail.toNonIndexed()]);
  body.dispose();
  tail.dispose();
  return geo;
}
