// Unit test for the FRONTEND_SPEC §4.1 coordinate contract and the MQTT.md direction
// convention. Run: node web/tools/test-coords.mjs
import assert from 'node:assert/strict';
import {
  computeFit, normToCanvas, canvasToNorm, clamp01, side, forwardNormal,
  isSelfIntersecting, pointInPolygon, bboxToRect,
} from '../dist/js/coords.js';

let n = 0;
const ok = (name) => { n += 1; console.log('  ok', name); };

// --- fit geometry -----------------------------------------------------------
{
  // 4:3 source in a 16:9 canvas -> horizontal letterbox, height-bound scale.
  const fit = computeFit(1400, 750, 1280, 960);
  assert.equal(fit.scale, 750 / 960);
  assert.equal(fit.dispH, 750);
  assert.ok(Math.abs(fit.dispW - 1000) < 1e-9);
  assert.ok(Math.abs(fit.offsetX - 200) < 1e-9);
  assert.equal(fit.offsetY, 0);
  ok('contain fit: 4:3 in 16:9 letterboxes horizontally with scale=min()');
}
{
  // 16:9 source in a tall canvas -> vertical letterbox.
  const fit = computeFit(800, 800, 1920, 1080);
  assert.equal(fit.scale, 800 / 1920);
  assert.ok(Math.abs(fit.dispH - 450) < 1e-9);
  assert.ok(Math.abs(fit.offsetY - 175) < 1e-9);
  assert.equal(fit.offsetX, 0);
  ok('contain fit: 16:9 in a square canvas letterboxes vertically');
}

// --- round trip -------------------------------------------------------------
for (const [cw, ch, fw, fh] of [[1400, 750, 1280, 960], [800, 800, 1920, 1080], [640, 360, 640, 360], [1000, 400, 2560, 1440]]) {
  const fit = computeFit(cw, ch, fw, fh);
  for (const p of [[0, 0], [1, 1], [0.5, 0.5], [0.137, 0.913], [0.999, 0.001]]) {
    const back = canvasToNorm(normToCanvas(p, fit), fit);
    assert.ok(Math.abs(back[0] - p[0]) < 1e-9, `x ${p} -> ${back}`);
    assert.ok(Math.abs(back[1] - p[1]) < 1e-9, `y ${p} -> ${back}`);
  }
}
ok('norm -> canvas -> norm round-trips exactly on 4 canvas/frame combinations');

// Corners land on the letterbox box edges, not the canvas edges.
{
  const fit = computeFit(1400, 750, 1280, 960);
  assert.deepEqual(normToCanvas([0, 0], fit).map(Math.round), [200, 0]);
  assert.deepEqual(normToCanvas([1, 1], fit).map(Math.round), [1200, 750]);
  ok('frame corners map to the letterbox rect, offset applied');
}

// --- clamping ---------------------------------------------------------------
{
  const fit = computeFit(1400, 750, 1280, 960);
  // Pointer dragged into the left letterbox bar and past the bottom edge.
  assert.deepEqual(canvasToNorm([-500, 2000], fit), [0, 1]);
  assert.deepEqual(canvasToNorm([9999, -20], fit), [1, 0]);
  assert.equal(clamp01(-0.2), 0);
  assert.equal(clamp01(1.7), 1);
  ok('canvas -> norm clamps out-of-frame drags into [0,1]');
}

// --- direction convention (contracts/MQTT.md) -------------------------------
{
  // side(p) = sign((end-start) x (p-start)); image coords, y down.
  const s = [0, 0];
  const e = [1, 0];              // arrow pointing right
  assert.equal(side(s, e, [0.5, 1]), 1);   // below the arrow => side > 0
  assert.equal(side(s, e, [0.5, -1]), -1); // above the arrow => side < 0
  // forward := centroid crosses from side>0 to side<0, i.e. travels "upward" here.
  const nf = forwardNormal(s, e);
  assert.ok(Math.abs(nf[0] - 0) < 1e-9);
  assert.ok(Math.abs(nf[1] + 1) < 1e-9, 'forward normal must point at -y for a rightward arrow');
  // Consistency check on an arbitrary segment: stepping along the forward normal must
  // move a point from the positive side to the negative side.
  for (const [a, b] of [[[0.1, 0.2], [0.8, 0.7]], [[0.9, 0.1], [0.2, 0.6]], [[0.3, 0.9], [0.35, 0.1]]]) {
    const nn = forwardNormal(a, b);
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const from = [mid[0] - nn[0] * 0.05, mid[1] - nn[1] * 0.05];
    const to = [mid[0] + nn[0] * 0.05, mid[1] + nn[1] * 0.05];
    assert.equal(side(a, b, from), 1);
    assert.equal(side(a, b, to), -1);
  }
  ok('forward arrow points from side>0 to side<0 for every tested segment');
}

// --- polygon helpers --------------------------------------------------------
{
  assert.equal(isSelfIntersecting([[0, 0], [1, 0], [1, 1], [0, 1]]), false);
  assert.equal(isSelfIntersecting([[0, 0], [1, 1], [1, 0], [0, 1]]), true); // bowtie
  assert.equal(isSelfIntersecting([[0, 0], [1, 0], [0.5, 1]]), false);
  ok('self-intersection detector: accepts convex/triangle, rejects bowtie');
}
{
  const sq = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]];
  assert.equal(pointInPolygon([0.5, 0.5], sq), true);
  assert.equal(pointInPolygon([0.1, 0.5], sq), false);
  ok('point-in-polygon ray casting');
}
{
  const fit = computeFit(1400, 750, 1280, 960);
  const r = bboxToRect([0.5, 0.5, 0.2, 0.4], fit);
  assert.ok(Math.abs(r.w - 0.2 * 1000) < 1e-9);
  assert.ok(Math.abs(r.h - 0.4 * 750) < 1e-9);
  assert.ok(Math.abs(r.x - (200 + 0.4 * 1000)) < 1e-9);
  ok('bbox [cx,cy,w,h] -> canvas rect honours the letterbox offset');
}

console.log(n + ' assertions groups passed');
