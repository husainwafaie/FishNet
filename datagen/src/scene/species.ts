/**
 * The detector's class list. Sizes/colours are chosen so the species are
 * easy to tell apart near the surface but converge to murky silhouettes at
 * depth (fog strips colour), keeping the learning problem honest.
 */
export interface SpeciesConfig {
  name: string;
  /** Stable class id written into label files. */
  classId: number;
  color: string;
  /** Per-fish lightness jitter (0–1) so instances aren't clones. */
  tintJitter: number;
  /** Body length range, world units. */
  bodyLength: readonly [number, number];
  /** Body height at unit length. */
  bodyAspect: number;
  /** Schools per layout (min/max, inclusive-ish — rounded). */
  schoolCount: readonly [number, number];
  fishPerSchool: readonly [number, number];
  /** Scatter envelope around the school centre. */
  schoolRadius: readonly [number, number];
  /** Depth band for school centres (world y, min/max). */
  yBand: readonly [number, number];
  /** Tail undulation frequency (Hz) for the vertex-shader wag. */
  tailFreq: number;
  /** Overlay/debug colour for this species' boxes. */
  overlayColor: string;
}

export const SPECIES: readonly SpeciesConfig[] = [
  {
    name: 'anchovy',
    classId: 0,
    color: '#b9cdd9',
    tintJitter: 0.12,
    bodyLength: [0.35, 0.6],
    bodyAspect: 0.24,
    schoolCount: [3, 6],
    fishPerSchool: [20, 45],
    schoolRadius: [2, 4],
    yBand: [-16, -4],
    tailFreq: 7,
    overlayColor: '#7fd4ff',
  },
  {
    name: 'jack',
    classId: 1,
    color: '#4b7d99',
    tintJitter: 0.08,
    bodyLength: [1.1, 1.7],
    bodyAspect: 0.3,
    schoolCount: [2, 4],
    fishPerSchool: [8, 16],
    schoolRadius: [3, 6],
    yBand: [-32, -12],
    tailFreq: 3.5,
    overlayColor: '#ffd166',
  },
  {
    // Solitary bottom-dweller: schoolCount is effectively the fish count
    name: 'grouper',
    classId: 2,
    color: '#4d5449',
    tintJitter: 0.1,
    bodyLength: [2.2, 3.2],
    bodyAspect: 0.38,
    schoolCount: [2, 6],
    fishPerSchool: [1, 1],
    schoolRadius: [0, 0],
    yBand: [-45, -34],
    tailFreq: 1.4,
    overlayColor: '#ff6b6b',
  },
];
