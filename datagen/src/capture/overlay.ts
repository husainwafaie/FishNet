import type { FishBox } from './labeler';
import { SPECIES } from '../scene/species';

const colorByClass = new Map(SPECIES.map((s) => [s.classId, s.overlayColor]));

/**
 * Debug overlay: draws the computed labels on a 2D canvas stacked above the
 * WebGL canvas. If these boxes hug the fish, the dataset labels are right —
 * this is the visual proof the pipeline is trustworthy.
 */
export class Overlay {
  private readonly ctx: CanvasRenderingContext2D;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(boxes: FishBox[]): void {
    const { ctx } = this;
    this.clear();
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.textBaseline = 'bottom';
    ctx.lineWidth = 1.5;

    for (const box of boxes) {
      const color = colorByClass.get(box.classId) ?? '#ffffff';
      ctx.strokeStyle = color;
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = color;
      ctx.fillText(
        `${box.species} ${(box.visibility * 100).toFixed(0)}%`,
        box.x + 2,
        box.y - 2,
      );
    }
  }
}
