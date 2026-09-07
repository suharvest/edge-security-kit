// Video wall geometry and tile state — pure functions, no DOM, no fetch.
// FRONTEND_SPEC §9. Tested by web/tools/test-wall.mjs.
//
// The page component below this file does three things badly if this file gets
// them wrong, and all three are invisible in a one-camera demo: a grid that
// wastes half the screen at 6 tiles, an overlay drawn against the wrong frame
// size so every box is offset, and a tile that shows a live picture for a
// camera the hub says is offline.

// Layouts the operator can pick, in the order the selector shows them. `cols`
// is fixed and rows follow from the count, so a 6 stays 3x2 rather than
// becoming 3x2 on one screen and 2x3 on another.
export const LAYOUTS = [
  { tiles: 1, cols: 1 },
  { tiles: 2, cols: 2 },
  { tiles: 4, cols: 2 },
  { tiles: 6, cols: 3 },
  { tiles: 9, cols: 3 },
];

export const LAYOUT_TILES = LAYOUTS.map((l) => l.tiles);

// The smallest layout that shows every stream, so plugging in a fifth camera
// does not silently hide it behind a pager. Above nine, nine — a wall nobody
// can read is not a better default than one that says "9 of 12".
export function autoLayout(streamCount) {
  const found = LAYOUTS.find((l) => l.tiles >= streamCount);
  return found ? found.tiles : LAYOUTS[LAYOUTS.length - 1].tiles;
}

export function layoutFor(tiles) {
  return LAYOUTS.find((l) => l.tiles === tiles) || LAYOUTS[2];
}

// CSS grid for a layout. Rows are derived, not fixed: 6 tiles is 3x2, and 5
// streams in a 6-tile layout still gets two rows rather than one row of five.
export function gridStyle(tiles) {
  const { cols } = layoutFor(tiles);
  const rows = Math.ceil(tiles / cols);
  return `grid-template-columns:repeat(${cols},1fr);grid-template-rows:repeat(${rows},1fr)`;
}

// Flatten devices into wall tiles. Order is stable (device, then stream) so a
// status push does not reshuffle the wall under the operator's eyes.
export function tilesFor(devices, tiles) {
  const out = [];
  (devices || []).forEach((device) => {
    const streams = Array.isArray(device.streams) ? device.streams : [];
    streams.forEach((stream) => {
      out.push({
        key: device.device_id + '/' + stream.stream_id,
        device_id: device.device_id,
        stream_id: stream.stream_id,
        name: stream.name || stream.stream_id,
        // A stream is live only when the device is up AND the stream is
        // running. A device that is online with a stream stuck in
        // `reconnecting` must not show a live picture: the last frame it
        // served would keep painting a scene that is minutes old.
        online: !!device.online && stream.state === 'running',
        state: device.online ? (stream.state || 'stopped') : 'offline',
        fps: Number(stream.fps || 0),
        decode: stream.decode || null,
        conf_threshold:
          typeof stream.conf_threshold === 'number' ? stream.conf_threshold : null,
        live_url: stream.live_url || null,
        preview_url: stream.preview_url || null,
        // The overlay is drawn in the picture's own box, not the tile's. A tile
        // is 16:9 and a source may be 4:3; with the picture letterboxed inside
        // and the overlay stretched across the whole tile, every box would sit
        // off the person by the pad. `frame` comes from the status message, so
        // a stream that does not report it gets no aspect and the two stay
        // locked together by filling the tile instead.
        aspect: stream.frame && stream.frame.w && stream.frame.h
          ? stream.frame.w + '/' + stream.frame.h
          : null,
      });
    });
  });
  return { visible: out.slice(0, tiles), total: out.length };
}

// One line the operator reads before looking at any picture (the overview rule):
// how many streams there are, how many are live, and whether anything is on a
// fallback decode path.
export function wallSummary(devices) {
  const { visible, total } = tilesFor(devices, Infinity);
  const live = visible.filter((t) => t.online).length;
  return {
    streams: total,
    live,
    down: total - live,
    fps: Math.round(visible.reduce((sum, t) => sum + t.fps, 0) * 10) / 10,
    sw_decode: visible.filter((t) => t.decode === 'sw').length,
  };
}

// Index `GET /api/live` by tile key so a tile looks up its own boxes in O(1)
// rather than scanning the array once per tile per tick.
export function indexLive(payload) {
  const rows = (payload && payload.streams) || [];
  const map = new Map();
  rows.forEach((row) => {
    map.set(row.device_id + '/' + row.stream_id, row);
  });
  return map;
}

// Detections older than this are not drawn. A stalled stream keeping its last
// boxes on screen is the failure that makes an operator trust a frozen tile.
export const OVERLAY_STALE_MS = 4000;

export function overlayBoxes(entry, nowMs) {
  if (!entry || !entry.payload) return [];
  const age = nowMs - (entry.received_ms || 0);
  if (age > OVERLAY_STALE_MS) return [];
  const frame = entry.payload.frame || {};
  return (entry.payload.detections || []).map((det) => {
    const [cx, cy, w, h] = det.bbox;
    return {
      track_id: det.track_id,
      score: det.score,
      // Percentages of the tile, so the box scales with the picture without
      // the component knowing the rendered pixel size. The published bbox is
      // already frame_norm (contracts/MQTT.md), so frame.w/h are needed only
      // for the label, never for the geometry -- a tile that guessed 16:9 for
      // a 4:3 source would offset every box.
      left: (cx - w / 2) * 100,
      top: (cy - h / 2) * 100,
      width: w * 100,
      height: h * 100,
    };
  }).filter((b) => b.width > 0 && b.height > 0)
    .map((b) => Object.assign(b, { frame }));
}

// Rule shapes for a stream, in the same percentage space as the boxes, so the
// wall shows what the alert will be judged against rather than only the person.
export function overlayShapes(rulesBody) {
  if (!rulesBody) return { zones: [], lines: [] };
  const features = rulesBody.features || {};
  const zones = (features.zone_detection === false ? [] : rulesBody.zones || []).map((z) => ({
    id: z.id,
    name: z.name,
    dwell: z.dwell_seconds || 0,
    points: (z.points || []).map(([x, y]) => [x * 100, y * 100]),
  })).filter((z) => z.points.length >= 3);
  const lines = (features.line_crossing === false ? [] : rulesBody.lines || []).map((l) => ({
    id: l.id,
    name: l.name,
    direction: l.direction || 'any',
    x1: l.start[0] * 100,
    y1: l.start[1] * 100,
    x2: l.end[0] * 100,
    y2: l.end[1] * 100,
  }));
  return { zones, lines };
}

// The confidence slider's step and bounds. 0.05 because the difference between
// 0.35 and 0.36 is not something anyone can see in the scene, and a slider that
// pretends otherwise invites fiddling instead of a decision.
export const CONF_MIN = 0.05;
export const CONF_MAX = 0.95;
export const CONF_STEP = 0.05;

export function clampConf(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  const stepped = Math.round(n / CONF_STEP) * CONF_STEP;
  return Math.min(CONF_MAX, Math.max(CONF_MIN, Math.round(stepped * 100) / 100));
}

// An RTSP URL usually carries the camera password. It must never reach a
// screenshot, a projector or a support ticket, so the console shows the host
// and path and hides the credentials.
export function maskSource(source) {
  const raw = String(source || '');
  const at = raw.indexOf('@');
  const scheme = raw.indexOf('://');
  if (at < 0 || scheme < 0 || at < scheme) return raw;
  const user = raw.slice(scheme + 3, at).split(':')[0];
  return raw.slice(0, scheme + 3) + user + ':***@' + raw.slice(at + 1);
}

// Accept what a camera actually gives you: rtsp is the common case, but a file
// or an http MJPEG endpoint is a legitimate source and refusing them would send
// the operator back to the config file this dialog exists to replace.
export function validateSource(source) {
  const raw = String(source || '').trim();
  if (!raw) return 'wall.errSourceEmpty';
  if (!/^(rtsp|rtsps|http|https|file):\/\//i.test(raw)) return 'wall.errSourceScheme';
  return null;
}

export function validateStreamId(streamId, existing) {
  const raw = String(streamId || '').trim();
  if (!raw) return 'wall.errIdEmpty';
  // Topic segment, so the two MQTT wildcards and the separator are out.
  if (/[+#/\s]/.test(raw)) return 'wall.errIdChars';
  if ((existing || []).includes(raw)) return 'wall.errIdTaken';
  return null;
}

// Suggest the next free cam-N so the common case needs no typing.
export function suggestStreamId(existing) {
  const taken = new Set(existing || []);
  for (let i = 0; i < 100; i += 1) {
    const candidate = 'cam-' + i;
    if (!taken.has(candidate)) return candidate;
  }
  return 'cam-' + Date.now();
}
