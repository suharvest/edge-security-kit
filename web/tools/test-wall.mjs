// Unit tests for the video-wall geometry and tile state (FRONTEND_SPEC §9).
// Run: node web/tools/test-wall.mjs
//
// Everything here is a bug that a one-camera demo cannot show: a grid that
// wastes the screen at 6 tiles, an overlay drawn from a stale payload, a live
// picture for a stream the hub says is not running, or a camera password in a
// screenshot.
import assert from 'node:assert/strict';
import {
  LAYOUT_TILES, autoLayout, layoutFor, gridStyle, tilesFor, wallSummary,
  indexLive, overlayBoxes, overlayShapes, OVERLAY_STALE_MS,
  clampConf, CONF_MIN, CONF_MAX, maskSource, validateSource, validateStreamId,
  suggestStreamId,
} from '../dist/js/wall.js';

let n = 0;
const ok = (name) => { n += 1; console.log('  ok', name); };

// --- layout ------------------------------------------------------------------
{
  assert.deepEqual(LAYOUT_TILES, [1, 2, 4, 6, 9]);
  ok('the five layouts are 1/2/4/6/9');
}
{
  assert.equal(autoLayout(0), 1);
  assert.equal(autoLayout(1), 1);
  assert.equal(autoLayout(3), 4);
  assert.equal(autoLayout(5), 6);
  assert.equal(autoLayout(9), 9);
  // A tenth camera must not vanish silently -- the wall caps and says "9 of 10".
  assert.equal(autoLayout(12), 9);
  ok('auto layout is the smallest that fits, capped at 9');
}
{
  assert.equal(gridStyle(6), 'grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(2,1fr)');
  assert.equal(gridStyle(4), 'grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(2,1fr)');
  assert.equal(gridStyle(9), 'grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr)');
  assert.equal(gridStyle(2), 'grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(1,1fr)');
  ok('6 is 3x2 and 9 is 3x3, rows derived from columns');
}
{
  assert.equal(layoutFor(7).tiles, 4, 'an unknown tile count falls back to 4 rather than throwing');
  ok('an unknown layout falls back instead of breaking the page');
}

// --- tiles -------------------------------------------------------------------
const DEVICES = [
  {
    device_id: 'jetson-01',
    online: true,
    streams: [
      { stream_id: 'cam-0', state: 'running', fps: 5, decode: 'hw', conf_threshold: 0.35,
        live_url: 'http://d/live/cam-0', preview_url: 'http://d/preview/cam-0.jpg' },
      { stream_id: 'cam-1', state: 'reconnecting', fps: 0, decode: 'hw', name: 'North gate' },
    ],
  },
  {
    device_id: 'rk3588-02',
    online: false,
    streams: [{ stream_id: 'cam-0', state: 'running', fps: 5, decode: 'sw' }],
  },
];
{
  const { visible, total } = tilesFor(DEVICES, 9);
  assert.equal(total, 3);
  assert.deepEqual(visible.map((t) => t.key),
    ['jetson-01/cam-0', 'jetson-01/cam-1', 'rk3588-02/cam-0']);
  ok('tile order is device then stream, stable across pushes');
}
{
  const { visible } = tilesFor(DEVICES, 9);
  assert.equal(visible[0].online, true);
  // Device up, stream reconnecting: not live. Painting its last frame would
  // show a scene that is minutes old with nothing saying so.
  assert.equal(visible[1].online, false);
  assert.equal(visible[1].state, 'reconnecting');
  // Device down: not live regardless of what the stream entry still claims.
  assert.equal(visible[2].online, false);
  assert.equal(visible[2].state, 'offline');
  ok('a tile is live only when the device is up AND the stream is running');
}
{
  const { visible } = tilesFor(DEVICES, 2);
  assert.equal(visible.length, 2);
  assert.equal(visible[1].name, 'North gate', 'a named stream shows its name');
  assert.equal(tilesFor(DEVICES, 9).visible[0].name, 'cam-0', 'an unnamed one falls back to its id');
  ok('the visible set is capped by the layout and names fall back to the id');
}
{
  assert.deepEqual(tilesFor(undefined, 4), { visible: [], total: 0 });
  assert.deepEqual(tilesFor([{ device_id: 'x', online: true }], 4), { visible: [], total: 0 });
  ok('a device with no streams array does not crash the wall');
}
{
  const s = wallSummary(DEVICES);
  assert.deepEqual(s, { streams: 3, live: 1, down: 2, fps: 10, sw_decode: 1 });
  ok('the overview line counts streams, live, down, fps and sw decode');
}

// --- overlay -----------------------------------------------------------------
const NOW = 1_757_203_500_000;
const entry = (ageMs, detections) => ({
  device_id: 'jetson-01',
  stream_id: 'cam-0',
  received_ms: NOW - ageMs,
  payload: { frame: { w: 1280, h: 720 }, detections },
});
{
  const map = indexLive({ streams: [entry(0, []), { ...entry(0, []), stream_id: 'cam-1' }] });
  assert.equal(map.size, 2);
  assert.ok(map.has('jetson-01/cam-1'));
  assert.deepEqual(overlayBoxes(map.get('nope'), NOW), []);
  ok('live payloads index by tile key and a missing tile yields no boxes');
}
{
  const boxes = overlayBoxes(entry(500, [
    { track_id: 3, class: 'person', score: 0.91, bbox: [0.5, 0.5, 0.1, 0.4] },
  ]), NOW);
  assert.equal(boxes.length, 1);
  assert.equal(boxes[0].left, 45);
  assert.equal(boxes[0].top, 30);
  assert.equal(boxes[0].width, 10);
  assert.equal(boxes[0].height, 40);
  assert.equal(boxes[0].track_id, 3);
  ok('a frame_norm bbox becomes centre-to-corner percentages of the tile');
}
{
  // The published bbox is already normalized against the ORIGINAL frame, so a
  // 4:3 source and a 16:9 source produce the same percentages. A tile that
  // reintroduced frame.w/h into the geometry would offset every box on one of
  // them (contracts/MQTT.md coordinate_space).
  const wide = overlayBoxes(entry(0, [{ track_id: 1, class: 'person', score: 0.9, bbox: [0.25, 0.5, 0.1, 0.2] }]), NOW);
  const tall = overlayBoxes({
    ...entry(0, [{ track_id: 1, class: 'person', score: 0.9, bbox: [0.25, 0.5, 0.1, 0.2] }]),
    payload: { frame: { w: 1280, h: 960 }, detections: [{ track_id: 1, class: 'person', score: 0.9, bbox: [0.25, 0.5, 0.1, 0.2] }] },
  }, NOW);
  assert.equal(wide[0].left, tall[0].left);
  assert.equal(wide[0].height, tall[0].height);
  ok('overlay geometry does not depend on the source aspect ratio');
}
{
  const stale = overlayBoxes(entry(OVERLAY_STALE_MS + 1, [
    { track_id: 1, class: 'person', score: 0.9, bbox: [0.5, 0.5, 0.1, 0.4] },
  ]), NOW);
  assert.deepEqual(stale, [], 'a stalled stream must not keep its last boxes on screen');
  const fresh = overlayBoxes(entry(OVERLAY_STALE_MS - 1, [
    { track_id: 1, class: 'person', score: 0.9, bbox: [0.5, 0.5, 0.1, 0.4] },
  ]), NOW);
  assert.equal(fresh.length, 1);
  ok('boxes older than the stale window are dropped, not drawn');
}
{
  const zeroWidth = overlayBoxes(entry(0, [
    { track_id: 1, class: 'person', score: 0.9, bbox: [0.5, 0.5, 0, 0.4] },
  ]), NOW);
  assert.deepEqual(zeroWidth, []);
  ok('a fully clipped box is dropped rather than drawn as a line');
}

// --- rule shapes -------------------------------------------------------------
const RULES = {
  zones: [{ id: 'bay', name: 'bay', points: [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8]], dwell_seconds: 10 }],
  lines: [{ id: 'gate', name: 'gate', start: [0.5, 0.1], end: [0.5, 0.9], direction: 'forward' }],
  features: { zone_detection: true, loitering: true, line_crossing: true },
};
{
  const { zones, lines } = overlayShapes(RULES);
  assert.deepEqual(zones[0].points, [[60, 30], [90, 30], [90, 80]]);
  assert.deepEqual([lines[0].x1, lines[0].y1, lines[0].x2, lines[0].y2], [50, 10, 50, 90]);
  assert.equal(lines[0].direction, 'forward');
  ok('zones and lines land in the same percentage space as the boxes');
}
{
  const off = overlayShapes({ ...RULES, features: { zone_detection: false, line_crossing: false } });
  assert.deepEqual(off, { zones: [], lines: [] });
  ok('a disabled feature draws nothing -- the wall shows what will actually fire');
}
{
  assert.deepEqual(overlayShapes(null), { zones: [], lines: [] });
  const degenerate = overlayShapes({ zones: [{ id: 'z', name: 'z', points: [[0.1, 0.1], [0.2, 0.2]] }], lines: [] });
  assert.deepEqual(degenerate.zones, [], 'a two-point zone is not a polygon');
  ok('missing or degenerate rule bodies draw nothing instead of throwing');
}

// --- confidence slider -------------------------------------------------------
{
  assert.equal(clampConf(0.42), 0.4);
  assert.equal(clampConf(0.43), 0.45);
  assert.equal(clampConf(0), CONF_MIN);
  assert.equal(clampConf(-3), CONF_MIN);
  assert.equal(clampConf(2), CONF_MAX);
  assert.equal(clampConf(1), CONF_MAX);
  assert.equal(clampConf('0.6'), 0.6);
  assert.equal(clampConf('abc'), null);
  assert.equal(clampConf(undefined), null);
  ok('confidence snaps to the 0.05 step and clamps inside 0.05-0.95');
}
{
  // Floating point: 0.35 must not come back as 0.35000000000000003 and reach
  // the API as a value the audit row then reports verbatim.
  for (let v = CONF_MIN; v <= CONF_MAX + 1e-9; v += 0.05) {
    const c = clampConf(v);
    assert.equal(c, Math.round(c * 100) / 100);
    assert.equal(String(c).length <= 4, true, `ugly value ${c}`);
  }
  ok('every step lands on a clean two-decimal value');
}

// --- add-camera dialog -------------------------------------------------------
{
  assert.equal(maskSource('rtsp://admin:hunter2@192.168.1.64:554/Streaming/Channels/101'),
    'rtsp://admin:***@192.168.1.64:554/Streaming/Channels/101');
  assert.equal(maskSource('rtsp://192.168.1.64:554/live'), 'rtsp://192.168.1.64:554/live');
  assert.equal(maskSource(''), '');
  assert.equal(maskSource(null), '');
  ok('a camera password never reaches the screen');
}
{
  assert.equal(validateSource('rtsp://cam/live'), null);
  assert.equal(validateSource('  http://cam/mjpeg  '), null, 'an MJPEG endpoint is a legitimate source');
  assert.equal(validateSource('file:///clips/a.mp4'), null);
  assert.equal(validateSource(''), 'wall.errSourceEmpty');
  assert.equal(validateSource('192.168.1.64'), 'wall.errSourceScheme');
  ok('the source must carry a scheme the detector can open');
}
{
  assert.equal(validateStreamId('cam-2', ['cam-0']), null);
  assert.equal(validateStreamId('', []), 'wall.errIdEmpty');
  assert.equal(validateStreamId('cam 2', []), 'wall.errIdChars');
  assert.equal(validateStreamId('cam/2', []), 'wall.errIdChars', 'a slash would split the topic');
  assert.equal(validateStreamId('cam+2', []), 'wall.errIdChars', 'a wildcard would match every stream');
  assert.equal(validateStreamId('cam-0', ['cam-0']), 'wall.errIdTaken');
  ok('a stream id that would break the topic or collide is caught before the request');
}
{
  assert.equal(suggestStreamId([]), 'cam-0');
  assert.equal(suggestStreamId(['cam-0', 'cam-1']), 'cam-2');
  assert.equal(suggestStreamId(['cam-1']), 'cam-0');
  ok('the dialog suggests the first free cam-N');
}

console.log(`\n${n} wall assertions passed`);
