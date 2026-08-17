// frame_norm <-> canvas transforms. Normative source: FRONTEND_SPEC §4.1.
//
//   scale   = min(canvasW/frameW, canvasH/frameH)
//   offsetX = (canvasW - frameW*scale)/2      offsetY likewise
//   norm -> canvas: px = nx * frameW * scale + offsetX
//   canvas -> norm: nx = (px - offsetX) / (frameW * scale), clamped to [0,1]
//
// Every stored coordinate is frame_norm ([0,1], origin top-left, x right, y down,
// relative to the ORIGINAL frame w/h reported by the detector) — contracts/MQTT.md.

export function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

export function computeFit(canvasW, canvasH, frameW, frameH) {
  const fw = frameW > 0 ? frameW : 1;
  const fh = frameH > 0 ? frameH : 1;
  const scale = Math.min(canvasW / fw, canvasH / fh);
  const dispW = fw * scale;
  const dispH = fh * scale;
  return {
    scale,
    offsetX: (canvasW - dispW) / 2,
    offsetY: (canvasH - dispH) / 2,
    dispW,
    dispH,
    frameW: fw,
    frameH: fh,
  };
}

export function normToCanvas(n, fit) {
  return [
    n[0] * fit.frameW * fit.scale + fit.offsetX,
    n[1] * fit.frameH * fit.scale + fit.offsetY,
  ];
}

// Always clamped: a drag that leaves the letterbox box must still store a legal
// frame_norm value (schema rejects anything outside [0,1]).
export function canvasToNorm(p, fit) {
  return [
    clamp01((p[0] - fit.offsetX) / (fit.frameW * fit.scale)),
    clamp01((p[1] - fit.offsetY) / (fit.frameH * fit.scale)),
  ];
}

// Pointer position in the coordinate system of an SVG element's own box.
export function eventToCanvas(ev, el) {
  const r = el.getBoundingClientRect();
  return [ev.clientX - r.left, ev.clientY - r.top];
}

// ---- geometry ------------------------------------------------------------

// side(p) = sign((end-start) x (p-start)), contracts/MQTT.md `direction`.
export function side(start, end, p) {
  const c = (end[0] - start[0]) * (p[1] - start[1]) - (end[1] - start[1]) * (p[0] - start[0]);
  return c > 0 ? 1 : c < 0 ? -1 : 0;
}

// Unit normal pointing the way a `forward` crossing travels.
// forward := centroid moves from side > 0 to side < 0 (MQTT.md), and with the image
// convention (y down) the side>0 half-plane is the one reached by rotating
// start->end by +90 deg, so the forward normal is that vector rotated by -90 deg:
//   d = end - start  =>  n_forward = (d.y, -d.x) / |d|
export function forwardNormal(start, end) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const len = Math.hypot(dx, dy) || 1;
  return [dy / len, -dx / len];
}

export function midpoint(a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }

export function distToSegment(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  let tt = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2;
  tt = Math.max(0, Math.min(1, tt));
  return Math.hypot(p[0] - (a[0] + tt * dx), p[1] - (a[1] + tt * dy));
}

export function pointInPolygon(p, pts) {
  // Ray casting, mirrors the hub's rules/geometry.py port.
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    const hit = ((yi > p[1]) !== (yj > p[1]))
      && (p[0] < ((xj - xi) * (p[1] - yi)) / ((yj - yi) || Number.EPSILON) + xi);
    if (hit) inside = !inside;
  }
  return inside;
}

function properIntersect(a, b, c, d) {
  const s1 = side(a, b, c);
  const s2 = side(a, b, d);
  const s3 = side(c, d, a);
  const s4 = side(c, d, b);
  return s1 !== 0 && s2 !== 0 && s3 !== 0 && s4 !== 0 && s1 !== s2 && s3 !== s4;
}

// Self-intersection test for a closed polygon (FRONTEND_SPEC §4.2 save validation).
export function isSelfIntersecting(pts) {
  const n = pts.length;
  if (n < 4) return false;
  for (let i = 0; i < n; i++) {
    const a1 = pts[i], a2 = pts[(i + 1) % n];
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || (i + 1) % n === j) continue; // adjacent edges share a vertex
      const b1 = pts[j], b2 = pts[(j + 1) % n];
      if (properIntersect(a1, a2, b1, b2)) return true;
    }
  }
  return false;
}

export function polygonCentroid(pts) {
  let x = 0, y = 0;
  pts.forEach((p) => { x += p[0]; y += p[1]; });
  return [x / (pts.length || 1), y / (pts.length || 1)];
}

// bbox is [cx, cy, w, h] normalized (contracts/MQTT.md) -> canvas rect.
export function bboxToRect(bbox, fit) {
  const [cx, cy, w, h] = bbox;
  const tl = normToCanvas([cx - w / 2, cy - h / 2], fit);
  return { x: tl[0], y: tl[1], w: w * fit.frameW * fit.scale, h: h * fit.frameH * fit.scale };
}
