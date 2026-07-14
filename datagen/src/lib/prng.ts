/**
 * Mulberry32 — tiny seedable PRNG. Every scene layout is a pure function of
 * its seed, so any dataset sample can be exactly reproduced from the seed
 * recorded in its label file.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform sample in [min, max). */
export function rangeFrom(rand: () => number): (r: readonly [number, number]) => number {
  return ([min, max]) => min + rand() * (max - min);
}
