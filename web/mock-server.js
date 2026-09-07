#!/usr/bin/env node
/*
 * Mock hub for frontend development. Implements the subset of HUB_SPEC §4 the UI needs
 * plus the §5 WS push, with cookie session auth (§7). No npm dependencies: the
 * WebSocket handshake and frame codec are inline.
 *
 *   node web/mock-server.js [--port 8090] [--must-change] [--quiet]
 *
 * Scenario controls (no auth, dev only):
 *   POST/GET /mock/burst?n=3            fabricate N new alerts now
 *   POST/GET /mock/offline?device=ID    flip a device offline (LWT) and push device.status
 *   POST/GET /mock/online?device=ID     flip it back
 *   POST/GET /mock/decode?device=ID&stream=ID&decode=sw   flip decode health
 *   POST/GET /mock/reject?on=1          make PUT /api/rules answer 400 with per-item issues
 *   POST/GET /mock/fail?on=1            make PUT /api/rules answer 503 (network-class failure)
 *   POST/GET /mock/state                dump counters
 */
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const url = require('url');

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf('--' + name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const PORT = Number(argVal('port', process.env.PORT || 8090));
const MUST_CHANGE = args.includes('--must-change');
const QUIET = args.includes('--quiet');
const ROOT = path.join(__dirname, 'dist');
const MOCKDIR = path.join(__dirname, 'mock');
const AUTO_ALERT_MS = Number(argVal('interval', 15000));

const log = (...a) => { if (!QUIET) console.log(new Date().toISOString().slice(11, 19), ...a); };

// ---------------------------------------------------------------- fixture state

const USER = { username: 'operator', password: 'operator', must_change: MUST_CHANGE };
const sessions = new Set();

const EVENTS = ['zone_enter', 'loitering', 'line_cross'];
const RULES = {
  'jetson-01': { 'cam-01': ['restricted_area', 'gate_line'], 'cam-02': ['dock_zone'] },
  'rk3588-02': { 'cam-01': ['perimeter'] },
  'recamera-07': { 'cam-01': ['door_line'] },
};

const devices = [
  {
    device_id: 'jetson-01',
    name: 'Gate A · Jetson Orin',
    online: true,
    mode: 'hub',
    versions: { app: '0.2.0', model: 'yolov8n@fb38b3933007' },
    last_seen_ms: Date.now(),
    // contracts/mqtt-detection.schema.json: `streams` is an ARRAY of stream_status
    // objects, each carrying its own stream_id. The hub passes it through as-is.
    streams: [
      {
        stream_id: 'cam-01', name: 'Gate A',
        conf_threshold: 0.35,
        state: 'running', decode: 'hw', fps: 14.8, fallback_active: false,
        frame: { w: 1920, h: 1080 },
        preview_url: '/mock/preview-1920x1080.jpg',
        live_url: '/mock/live/cam-01',
      },
      // 4:3 source on a 16:9 canvas: proves the object-fit: contain math (§4.1).
      {
        stream_id: 'cam-02', name: 'Loading bay',
        conf_threshold: 0.45,
        state: 'running', decode: 'hw', fps: 13.2, fallback_active: false,
        frame: { w: 1280, h: 960 },
        preview_url: '/mock/preview-1280x960.jpg',
        live_url: '/mock/live/cam-02',
      },
    ],
  },
  {
    device_id: 'rk3588-02',
    name: 'Dock · RK3588',
    online: true,
    mode: 'hub',
    versions: { app: '0.2.0', model: 'yolov8n@fb38b3933007' },
    last_seen_ms: Date.now(),
    streams: [
      // No preview_url -> grey canvas fallback + /api/live overlay.
      {
        stream_id: 'cam-01', name: 'Dock ramp',
        conf_threshold: 0.5,
        state: 'running', decode: 'sw', fps: 6.2, fallback_active: true,
        frame: { w: 2560, h: 1440 },
        live_url: '/mock/live/rk-cam-01',
      },
    ],
  },
  {
    device_id: 'recamera-07',
    name: 'Side door · reCamera',
    online: false,
    mode: 'single_box',
    versions: { app: '0.1.9', model: 'yolo11n@0c31a7d19f42' },
    last_seen_ms: Date.now() - 52 * 60 * 1000,
    streams: [
      { stream_id: 'cam-01', name: 'Side door', conf_threshold: 0.4, state: 'stopped', decode: 'hw', fps: 0, frame: { w: 1920, h: 1080 } },
    ],
  },
];

const rules = {
  'jetson-01': {
    'cam-01': {
      rev: 4,
      updated_ms: Date.now() - 3600e3,
      body: {
        zones: [{ id: 'z-seed1', name: 'restricted_area', points: [[0.12, 0.45], [0.44, 0.36], [0.58, 0.72], [0.16, 0.86]], dwell_s: 10 }],
        lines: [{ id: 'l-seed1', name: 'gate_line', start: [0.62, 0.22], end: [0.9, 0.68], direction: 'forward' }],
        features: {},
        cooldown: 30,
      },
    },
    'cam-02': {
      rev: 1, updated_ms: Date.now() - 7200e3,
      body: { zones: [{ id: 'z-seed2', name: 'dock_zone', points: [[0.2, 0.2], [0.8, 0.25], [0.75, 0.8], [0.25, 0.75]], dwell_s: 15 }], lines: [], features: {}, cooldown: 30 },
    },
  },
  'rk3588-02': {
    'cam-01': { rev: 2, updated_ms: Date.now() - 600e3, body: { zones: [], lines: [{ id: 'l-seed2', name: 'perimeter', start: [0.1, 0.8], end: [0.9, 0.6], direction: 'any' }], features: {}, cooldown: 45 } },
  },
};

const hubConfig = { mqtt_host: 'mosquitto', mqtt_port: 1883, retention_days: 30, fp_warn_rate: 0.4, version: '0.2.0-mock' };

let alertSeq = 0;
const alerts = [];

// `streams` is an array; look one up by its own stream_id (never by index).
function streamOf(dev, sid) {
  return (dev && dev.streams || []).find((x) => String(x.stream_id) === String(sid)) || null;
}
function streamIdsOf(dev) { return (dev && dev.streams || []).map((x) => String(x.stream_id)); }

// The hub answers /live and /live/{d}/{s} with {received_ms, payload}, where
// payload is the detector's own sensecraft.detection/1 message. The mock once
// answered a flat {objects: [...]} instead; the rule editor was written against
// that and its reference boxes therefore never drew against a real hub. Both
// mock routes now build the payload here, so there is one shape to get wrong.
let liveFrameId = 0;
function liveEntry(dev, stream) {
  // Deterministic motion rather than random boxes: a screenshot or a GIF then
  // shows people walking rather than boxes teleporting, and two consecutive
  // frames are comparable.
  const t = Date.now() / 1000;
  const seed = (String(dev.device_id) + String(stream.stream_id))
    .split('').reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 997, 7);
  const n = 1 + (seed % 3);
  const detections = [];
  for (let i = 0; i < n; i++) {
    const phase = t * 0.11 + (seed + i * 37) / 40;
    const cx = 0.2 + 0.6 * (0.5 + 0.5 * Math.sin(phase));
    const cy = 0.42 + 0.22 * Math.sin(phase * 0.7 + i);
    detections.push({
      track_id: i + 1,
      class: 'person',
      score: Number((0.62 + 0.3 * Math.abs(Math.sin(phase * 1.7))).toFixed(2)),
      bbox: [Number(cx.toFixed(4)), Number(cy.toFixed(4)), 0.09, 0.26],
    });
  }
  liveFrameId += 1;
  return {
    device_id: dev.device_id,
    stream_id: String(stream.stream_id),
    received_ms: Date.now(),
    payload: {
      schema: 'sensecraft.detection/1',
      timestamp: Date.now(),
      session_id: 'mock-1',
      frame_id: liveFrameId,
      device_id: dev.device_id,
      stream_id: String(stream.stream_id),
      coordinate_space: 'frame_norm',
      frame: stream.frame || { w: 1280, h: 720 },
      inference_time_ms: 4.2,
      detections,
      health: { fps: stream.fps || 0, decode: stream.decode || 'hw',
                backend: 'mock', fallback_active: !!stream.fallback_active },
    },
  };
}

// Audit trail, written by the control endpoints below.
const auditRows = [];
let auditSeq = 0;
function audit(actor, action, device_id, stream_id, outcome, detail) {
  auditSeq += 1;
  auditRows.unshift({ id: auditSeq, ts_ms: Date.now(), actor, action,
                      device_id, stream_id, outcome, detail });
  return auditRows[0];
}
// Scenario knobs: /mock/control?mode=timeout|refuse|ok
let controlMode = 'ok';

function ruleNamesFor(d, s) { return (RULES[d] && RULES[d][s]) || ['rule']; }

function makeAlert(opts = {}) {
  const dev = opts.device_id || 'jetson-01';
  const stream = opts.stream_id || (dev === 'jetson-01' ? (Math.random() < 0.6 ? 'cam-01' : 'cam-02') : 'cam-01');
  const names = ruleNamesFor(dev, stream);
  const event_type = opts.event_type || EVENTS[Math.floor(Math.random() * EVENTS.length)];
  const rule_name = opts.rule_name
    || (event_type === 'line_cross' ? names.find((n) => n.includes('line')) || names[0] : names[0]);
  const id = ++alertSeq;
  const ts = opts.ts || Date.now();
  const a = {
    id,
    event_id: dev + '-' + stream + '-1755400000-' + id,
    ts_ms: ts,
    received_ms: ts,
    device_id: dev,
    stream_id: stream,
    event_type,
    rule_name,
    track_id: 1 + Math.floor(Math.random() * 40),
    score: Number((0.55 + Math.random() * 0.44).toFixed(2)),
    bbox: [Number((0.2 + Math.random() * 0.6).toFixed(3)), Number((0.25 + Math.random() * 0.5).toFixed(3)), 0.11, 0.28],
    direction: event_type === 'line_cross' ? (Math.random() < 0.5 ? 'forward' : 'backward') : null,
    dwell_s: event_type === 'loitering' ? Number((10 + Math.random() * 30).toFixed(1)) : null,
    state: opts.state || 'new',
    acted_by: opts.acted_by || null,
    acted_at: opts.acted_at || null,
    snapshot_state: opts.snapshot_state || 'pending',
    snapshot_url: null,
    simulated: !!opts.simulated,
    meta: opts.simulated ? { simulated: true } : {},
  };
  if (a.snapshot_state === 'received') a.snapshot_url = '/api/alerts/' + id + '/snapshot.jpg';
  alerts.push(a);
  return a;
}

// Seed history: a day's worth, mixed states, one rule deliberately noisy so the
// >40% false-positive suggestion shows up (§3.4).
(function seed() {
  const now = Date.now();
  for (let i = 22; i >= 1; i--) {
    const ts = now - i * 17 * 60 * 1000;
    const noisy = i % 3 === 0;
    const a = makeAlert({
      ts,
      device_id: noisy ? 'rk3588-02' : (i % 4 === 0 ? 'jetson-01' : 'jetson-01'),
      stream_id: noisy ? 'cam-01' : undefined,
      state: i > 6 ? (noisy ? 'dismissed' : (i % 2 ? 'acked' : 'dismissed')) : 'new',
      snapshot_state: 'received',
    });
    if (a.state !== 'new') { a.acted_by = 'operator'; a.acted_at = ts + 60000; }
  }
})();

// ---------------------------------------------------------------- websockets

const clients = new Set();

function wsAccept(key) {
  return crypto.createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
}

function wsFrame(text) {
  const payload = Buffer.from(text, 'utf8');
  const len = payload.length;
  let head;
  if (len < 126) {
    head = Buffer.alloc(2);
    head[1] = len;
  } else if (len < 65536) {
    head = Buffer.alloc(4);
    head[1] = 126;
    head.writeUInt16BE(len, 2);
  } else {
    head = Buffer.alloc(10);
    head[1] = 127;
    head.writeUInt32BE(0, 2);
    head.writeUInt32BE(len, 6);
  }
  head[0] = 0x81; // FIN + text
  return Buffer.concat([head, payload]);
}

function push(obj) {
  const buf = wsFrame(JSON.stringify(obj));
  for (const sock of Array.from(clients)) {
    if (sock.destroyed) { clients.delete(sock); continue; }
    try { sock.write(buf); } catch (e) { clients.delete(sock); }
  }
}

function handleUpgrade(req, socket) {
  const u = url.parse(req.url, true);
  if (u.pathname !== '/ws') { socket.destroy(); return; }
  if (!sessionOf(req)) {
    socket.write('HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n');
    socket.destroy();
    log('WS handshake rejected (no session)');
    return;
  }
  const key = req.headers['sec-websocket-key'];
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\n'
    + 'Upgrade: websocket\r\n'
    + 'Connection: Upgrade\r\n'
    + 'Sec-WebSocket-Accept: ' + wsAccept(key) + '\r\n\r\n',
  );
  socket.setNoDelay(true);
  clients.add(socket);
  log('WS client connected, total', clients.size);
  let buf = Buffer.alloc(0);
  socket.on('data', (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    // Minimal client-frame drain: the browser only sends {"type":"ping"} and close.
    while (buf.length >= 2) {
      const opcode = buf[0] & 0x0f;
      const masked = (buf[1] & 0x80) !== 0;
      let len = buf[1] & 0x7f;
      let off = 2;
      if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); off = 4; }
      else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); off = 10; }
      const total = off + (masked ? 4 : 0) + len;
      if (buf.length < total) return;
      buf = buf.slice(total);
      if (opcode === 0x8) { clients.delete(socket); socket.destroy(); return; }
    }
  });
  socket.on('close', () => { clients.delete(socket); log('WS client gone, total', clients.size); });
  socket.on('error', () => { clients.delete(socket); });
}

// ---------------------------------------------------------------- http helpers

// --any-session accepts any previously issued cookie value, so restarting the mock does
// not force a re-login — needed to exercise the WS reconnect/after_id path.
const ANY_SESSION = args.includes('--any-session');

function sessionOf(req) {
  const raw = req.headers.cookie || '';
  const m = /(?:^|;\s*)sid=([^;]+)/.exec(raw);
  if (!m) return null;
  if (ANY_SESSION) { sessions.add(m[1]); return m[1]; }
  return sessions.has(m[1]) ? m[1] : null;
}

function send(res, code, obj, extraHeaders) {
  const body = Buffer.from(JSON.stringify(obj === undefined ? {} : obj), 'utf8');
  res.writeHead(code, Object.assign({
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': body.length,
    'Cache-Control': 'no-store',
  }, extraHeaders || {}));
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = '';
    req.on('data', (c) => { data += c; });
    req.on('end', () => {
      if (!data) { resolve({}); return; }
      try { resolve(JSON.parse(data)); } catch (e) { resolve({ __raw: data }); }
    });
  });
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.mp3': 'audio/mpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function serveFile(res, file, fallbackType) {
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('not found'); return; }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file).toLowerCase()] || fallbackType || 'application/octet-stream',
      'Content-Length': data.length,
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  });
}

const SPA_ROUTES = ['/', '/login', '/wall', '/devices', '/rules', '/debug'];

// ---------------------------------------------------------------- scenario knobs

const knobs = { rejectRules: false, failRules: false };

function csvEscape(v) {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function filterAlerts(q) {
  let rows = alerts.slice();
  if (q.after_id) {
    const after = Number(q.after_id);
    rows = rows.filter((a) => a.id > after).sort((a, b) => a.id - b.id);
  } else {
    rows.sort((a, b) => b.ts_ms - a.ts_ms);
  }
  if (q.state) rows = rows.filter((a) => a.state === q.state);
  if (q.device_id) rows = rows.filter((a) => a.device_id === q.device_id);
  if (q.stream_id) rows = rows.filter((a) => a.stream_id === q.stream_id);
  if (q.event_type) rows = rows.filter((a) => a.event_type === q.event_type);
  if (q.rule_name) rows = rows.filter((a) => a.rule_name === q.rule_name);
  if (q.date_from) rows = rows.filter((a) => a.ts_ms >= Number(q.date_from));
  if (q.date_to) rows = rows.filter((a) => a.ts_ms <= Number(q.date_to));
  const offset = Number(q.offset || 0);
  const limit = Number(q.limit || 50);
  return rows.slice(offset, offset + limit);
}

function transition(id, target) {
  const a = alerts.find((x) => x.id === Number(id));
  if (!a) return { code: 404, body: { error: 'no such alert' } };
  if (a.state === target) return { code: 409, body: { error: 'already ' + target } };
  a.state = target;
  a.acted_by = USER.username;
  a.acted_at = Date.now();
  push({ type: 'alert.update', alert: { id: a.id, state: a.state, acted_by: a.acted_by, acted_at: a.acted_at } });
  return { code: 200, body: a };
}

function validateRuleBody(body) {
  const issues = [];
  (body.zones || []).forEach((z) => {
    if (!z.points || z.points.length < 3) issues.push({ target_type: 'zone', target_id: z.id, name: z.name, message: "zone '" + z.name + "' has fewer than 3 vertices" });
  });
  (body.lines || []).forEach((l) => {
    const d = Math.hypot(l.end[0] - l.start[0], l.end[1] - l.start[1]);
    if (d < 1e-4) issues.push({ target_type: 'line', target_id: l.id, name: l.name, message: "line '" + l.name + "' has coincident endpoints" });
  });
  const all = (body.zones || []).concat(body.lines || []);
  all.forEach((x) => {
    const pts = x.points || [x.start, x.end];
    pts.forEach((p) => {
      if (p[0] < 0 || p[0] > 1 || p[1] < 0 || p[1] > 1) {
        issues.push({ target_type: x.points ? 'zone' : 'line', target_id: x.id, name: x.name, message: "'" + x.name + "' has a coordinate outside [0,1]" });
      }
    });
  });
  return issues;
}

// ---------------------------------------------------------------- api routing

async function api(req, res, u) {
  const p = u.pathname.replace(/^\/api/, '');
  const q = u.query;
  const method = req.method;

  if (p === '/health') {
    return send(res, 200, { ok: true, mqtt_connected: true, schema_rejects: 0, version: hubConfig.version });
  }

  if (p === '/auth/login' && method === 'POST') {
    const b = await readBody(req);
    if (b.username !== USER.username || b.password !== USER.password) {
      return send(res, 401, { error: 'bad credentials' });
    }
    const sid = crypto.randomBytes(16).toString('hex');
    sessions.add(sid);
    log('login ok ->', sid.slice(0, 8));
    return send(res, 200, { username: USER.username, must_change: USER.must_change },
      { 'Set-Cookie': 'sid=' + sid + '; Path=/; HttpOnly; SameSite=Strict' });
  }

  const sid = sessionOf(req);
  if (!sid) return send(res, 401, { error: 'unauthorized' });

  // GET /auth/session: the frontend asks the hub who it is talking to instead of
  // remembering a username locally (HUB_SPEC §4).
  if (p === '/auth/session' && method === 'GET') {
    return send(res, 200, { username: USER.username, must_change: USER.must_change });
  }

  if (p === '/auth/logout' && method === 'POST') { sessions.delete(sid); return send(res, 200, { ok: true }); }

  if (p === '/auth/password' && method === 'POST') {
    const b = await readBody(req);
    if (b.old_password !== USER.password) return send(res, 400, { error: 'wrong current password' });
    USER.password = b.new_password;
    USER.must_change = false;
    return send(res, 200, { ok: true });
  }

  if (p === '/alerts' && method === 'GET') return send(res, 200, { alerts: filterAlerts(q), total: alerts.length });

  if (p === '/alerts/export.csv' && method === 'GET') {
    const rows = filterAlerts(Object.assign({}, q, { limit: 100000 }));
    const cols = ['id', 'ts_ms', 'device_id', 'stream_id', 'event_type', 'rule_name', 'track_id', 'score', 'state', 'acted_by', 'acted_at'];
    const lines = [cols.join(',')].concat(rows.map((a) => cols.map((c) => csvEscape(a[c])).join(',')));
    const body = Buffer.concat([Buffer.from('﻿', 'utf8'), Buffer.from(lines.join('\r\n'), 'utf8')]);
    res.writeHead(200, {
      'Content-Type': 'text/csv; charset=utf-8',
      'Content-Disposition': 'attachment; filename="alerts.csv"',
      'Content-Length': body.length,
    });
    return res.end(body);
  }

  let m = /^\/alerts\/(\d+)\/snapshot\.jpg$/.exec(p);
  if (m) {
    const a = alerts.find((x) => x.id === Number(m[1]));
    if (!a || a.snapshot_state === 'pending' || a.snapshot_state === 'none') {
      return send(res, 404, { error: 'no snapshot' });
    }
    return serveFile(res, path.join(MOCKDIR, 'snapshot.jpg'), 'image/jpeg');
  }

  m = /^\/alerts\/(\d+)$/.exec(p);
  if (m && method === 'GET') {
    const a = alerts.find((x) => x.id === Number(m[1]));
    return a ? send(res, 200, a) : send(res, 404, { error: 'no such alert' });
  }

  m = /^\/alerts\/(\d+)\/(ack|dismiss)$/.exec(p);
  if (m && method === 'POST') {
    const r = transition(m[1], m[2] === 'ack' ? 'acked' : 'dismissed');
    return send(res, r.code, r.body);
  }

  m = /^\/alerts\/(ack|dismiss)$/.exec(p);
  if (m && method === 'POST') {
    const b = await readBody(req);
    const ids = Array.isArray(b.ids) ? b.ids : [];
    if (ids.length > 500) return send(res, 400, { error: 'batch limit is 500' });
    const results = ids.map((id) => {
      const r = transition(id, m[1] === 'ack' ? 'acked' : 'dismissed');
      return { id: Number(id), status: r.code, alert: r.code === 200 ? r.body : undefined };
    });
    log('batch', m[1], ids.length, '->', results.filter((r) => r.status === 200).length, 'ok');
    return send(res, 200, { results });
  }

  if (p === '/devices' && method === 'GET') return send(res, 200, { devices });

  // Hub single-frame preview proxy (HUB_SPEC §4). The mock serves the same
  // fixture JPEG the device would, so the rule canvas exercises the real path.
  m = /^\/devices\/([^/]+)\/streams\/([^/]+)\/preview\.jpg$/.exec(p);
  if (m && method === 'GET') {
    const dev = devices.find((d) => d.device_id === decodeURIComponent(m[1]));
    if (!dev) return send(res, 404, { error: 'device not found' });
    const stream = streamOf(dev, decodeURIComponent(m[2]));
    if (!stream) return send(res, 404, { error: 'stream not found' });
    if (!stream.preview_url) return send(res, 404, { error: 'stream reports no preview_url' });
    const file = path.join(MOCKDIR, stream.preview_url.replace('/mock/', ''));
    if (!file.startsWith(MOCKDIR) || !fs.existsSync(file)) {
      return send(res, 502, { error: 'device unreachable (mock)', preview_url: stream.preview_url });
    }
    return serveFile(res, file);
  }

  m = /^\/devices\/([^/]+)\/config$/.exec(p);
  if (m) {
    const id = decodeURIComponent(m[1]);
    const dev = devices.find((d) => d.device_id === id);
    if (!dev) return send(res, 404, { error: 'no such device' });
    if (method === 'GET') {
      const streams = {};
      streamIdsOf(dev).forEach((s) => {
        streams[s] = {
          camera: { rtsp_url: 'rtsp://192.168.3.90:554/' + s, frame: (streamOf(dev, s) || {}).frame },
          rules: ((rules[id] || {})[s] || { body: { zones: [], lines: [], features: {}, cooldown: 30 } }).body,
        };
      });
      return send(res, 200, { device_id: id, exported_ms: Date.now(), streams });
    }
    if (method === 'PUT') {
      const b = await readBody(req);
      if (!b || !b.streams) return send(res, 400, { error: 'missing streams' });
      Object.keys(b.streams).forEach((s) => {
        rules[id] = rules[id] || {};
        const cur = rules[id][s] || { rev: 0 };
        rules[id][s] = { rev: cur.rev + 1, updated_ms: Date.now(), body: b.streams[s].rules || { zones: [], lines: [], features: {}, cooldown: 30 } };
      });
      log('device config restored for', id);
      return send(res, 200, { ok: true, rev: Math.max.apply(null, Object.keys(rules[id]).map((s) => rules[id][s].rev)), persisted_ms: Date.now() });
    }
  }

  if (p === '/rules' && method === 'GET') return send(res, 200, rules);

  m = /^\/rules\/([^/]+)\/([^/]+)$/.exec(p);
  if (m) {
    const d = decodeURIComponent(m[1]);
    const s = decodeURIComponent(m[2]);
    if (method === 'GET') {
      const r = (rules[d] || {})[s];
      if (!r) return send(res, 404, { error: 'no rules for stream' });
      return send(res, 200, { device_id: d, stream_id: s, rev: r.rev, updated_ms: r.updated_ms, body: r.body });
    }
    if (method === 'PUT') {
      const b = await readBody(req);
      if (knobs.failRules) return send(res, 503, { error: 'storage unavailable' });
      const issues = knobs.rejectRules
        ? [{ target_type: 'zone', target_id: (b.zones && b.zones[0] && b.zones[0].id) || null, name: (b.zones && b.zones[0] && b.zones[0].name) || 'zone_1', message: "zone is self-intersecting" }]
        : validateRuleBody(b);
      if (issues.length) return send(res, 400, { error: 'validation failed', issues });
      rules[d] = rules[d] || {};
      const cur = rules[d][s] || { rev: 0 };
      rules[d][s] = { rev: cur.rev + 1, updated_ms: Date.now(), body: b };
      log('rules PUT', d + '/' + s, '-> rev', rules[d][s].rev,
        '(zones', (b.zones || []).length, 'lines', (b.lines || []).length + ')');
      return send(res, 200, { ok: true, rev: rules[d][s].rev, persisted_ms: Date.now() });
    }
  }

  m = /^\/rules\/([^/]+)\/([^/]+)\/simulate$/.exec(p);
  if (m && method === 'POST') {
    const d = decodeURIComponent(m[1]);
    const s = decodeURIComponent(m[2]);
    const b = await readBody(req);
    const r = (rules[d] || {})[s];
    if (!r) return send(res, 404, { error: 'no rules for stream' });
    const target = (r.body.zones || []).concat(r.body.lines || []).find((x) => x.id === b.rule_id);
    if (!target) return send(res, 404, { error: 'no such rule_id: ' + b.rule_id });
    const isLine = !!target.start;
    const a = makeAlert({
      device_id: d,
      stream_id: s,
      event_type: isLine ? 'line_cross' : 'zone_enter',
      rule_name: target.name,
      simulated: true,
      snapshot_state: 'received',
    });
    push({ type: 'alert.new', alert: a });
    log('simulate ->', target.name, 'alert', a.id);
    return send(res, 200, { ok: true, alert_id: a.id });
  }

  if (p === '/live' && method === 'GET') {
    const streams = [];
    devices.forEach((dev) => {
      if (!dev.online) return;
      (dev.streams || []).forEach((st) => {
        if (st.state !== 'running') return;
        streams.push(liveEntry(dev, st));
      });
    });
    return send(res, 200, { streams, count: streams.length, now_ms: Date.now() });
  }

  m = /^\/live\/([^/]+)\/([^/]+)$/.exec(p);
  if (m && method === 'GET') {
    const d = decodeURIComponent(m[1]);
    const sid = decodeURIComponent(m[2]);
    const dev = devices.find((x) => x.device_id === d);
    const stream = streamOf(dev, sid);
    if (!stream) return send(res, 404, { error: 'no such stream' });
    return send(res, 200, liveEntry(dev, stream));
  }

  // ---- runtime control (contracts/MQTT.md "Control downlink") --------------
  // The three outcomes are what the console has to render differently, so the
  // mock can produce all three: /mock/control?mode=timeout|refuse|ok.
  function controlOutcome(action, device_id, stream_id, params, applied) {
    if (controlMode === 'timeout') {
      audit('operator', action, device_id, stream_id, 'timeout', { params });
      return send(res, 504, { error: device_id + ' did not answer ' + action + ' within 8s', params });
    }
    if (controlMode === 'refuse') {
      audit('operator', action, device_id, stream_id, 'rejected', { params });
      return send(res, 409, { error: 'device refused: single-stream runtime', params, applied: {} });
    }
    audit('operator', action, device_id, stream_id, 'ok', { params, applied });
    return send(res, action === 'add_stream' ? 201 : 200, { ok: true, applied });
  }

  m = /^\/devices\/([^/]+)\/streams\/([^/]+)\/conf$/.exec(p);
  if (m && method === 'PUT') {
    const d = decodeURIComponent(m[1]);
    const sid = decodeURIComponent(m[2]);
    const b = await readBody(req);
    const v = b && b.conf_threshold;
    if (typeof v !== 'number' || v < 0 || v > 1) return send(res, 400, { error: 'conf_threshold must be between 0 and 1' });
    const dev = devices.find((x) => x.device_id === d);
    const stream = streamOf(dev, sid);
    if (!stream) return send(res, 404, { error: 'no such stream' });
    const applied = { stream_id: sid, conf_threshold: v, persisted: true };
    if (controlMode === 'ok') {
      stream.conf_threshold = v;
      push({ type: 'device.status', device: dev });
    }
    return controlOutcome('set_conf_threshold', d, sid, { stream_id: sid, conf_threshold: v }, applied);
  }

  m = /^\/devices\/([^/]+)\/streams$/.exec(p);
  if (m && method === 'POST') {
    const d = decodeURIComponent(m[1]);
    const dev = devices.find((x) => x.device_id === d);
    if (!dev) return send(res, 404, { error: 'device not found' });
    const b = await readBody(req) || {};
    if (!b.stream_id) return send(res, 400, { error: 'stream_id is required' });
    if (!b.source) return send(res, 400, { error: 'source is required' });
    if (streamIdsOf(dev).includes(String(b.stream_id))) {
      return send(res, 409, { error: 'stream ' + b.stream_id + ' already exists on ' + d });
    }
    const applied = { stream_id: b.stream_id, source: b.source, state: 'running', persisted: true };
    if (controlMode === 'ok') {
      dev.streams.push({
        stream_id: String(b.stream_id), name: b.name || '',
        state: 'running', decode: 'hw', fps: 12.0, fallback_active: false,
        conf_threshold: 0.35, frame: { w: 1920, h: 1080 },
        preview_url: '/mock/preview-1920x1080.jpg',
        live_url: '/mock/live/' + String(b.stream_id),
      });
      push({ type: 'device.status', device: dev });
    }
    return controlOutcome('add_stream', d, String(b.stream_id), b, applied);
  }

  m = /^\/devices\/([^/]+)\/streams\/([^/]+)$/.exec(p);
  if (m && method === 'DELETE') {
    const d = decodeURIComponent(m[1]);
    const sid = decodeURIComponent(m[2]);
    const dev = devices.find((x) => x.device_id === d);
    if (!dev || !streamOf(dev, sid)) return send(res, 404, { error: 'no such stream' });
    if (controlMode === 'ok') {
      dev.streams = dev.streams.filter((x) => String(x.stream_id) !== sid);
      push({ type: 'device.status', device: dev });
    }
    return controlOutcome('remove_stream', d, sid, { stream_id: sid }, { stream_id: sid, removed: true, persisted: true });
  }

  if (p === '/audit' && method === 'GET') {
    const limit = Number(u.query.limit || 100);
    return send(res, 200, { audit: auditRows.slice(0, limit), count: Math.min(auditRows.length, limit) });
  }

  if (p === '/config') {
    if (method === 'GET') return send(res, 200, hubConfig);
    if (method === 'PUT') {
      const b = await readBody(req);
      Object.assign(hubConfig, b);
      return send(res, 200, { ok: true, restart_required: [] });
    }
  }

  return send(res, 404, { error: 'no such endpoint: ' + method + ' ' + p });
}

function mock(req, res, u) {
  const q = u.query;
  const p = u.pathname;
  const m = /^\/mock\/live\/([^/]+)$/.exec(p);
  if (m) {
    // The detector's own /live page is a refreshing still (platforms/generic
    // preview.py). The mock serves the same shape so the wall exercises the
    // iframe path rather than only the hub's single-frame proxy.
    const sid = decodeURIComponent(m[1]);
    const jpeg = sid === 'cam-02' ? '/mock/preview-1280x960.jpg' : '/mock/preview-1920x1080.jpg';
    const body = '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;height:100%;background:#06080b}'
      + 'img{width:100%;height:100%;object-fit:fill;display:block}</style>'
      + '<img id="f" alt="' + sid + '"><script>var i=document.getElementById("f");'
      + 'function t(){i.src="' + jpeg + '?t="+Date.now();}t();setInterval(t,1000);<\/script>';
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    return res.end(body);
  }
  if (p === '/mock/control') {
    controlMode = ['ok', 'refuse', 'timeout'].includes(q.mode) ? q.mode : 'ok';
    return send(res, 200, { control_mode: controlMode });
  }
  if (p === '/mock/burst') {
    const n = Number(q.n || 1);
    const made = [];
    for (let i = 0; i < n; i++) {
      const a = makeAlert({ device_id: q.device || undefined, event_type: q.event_type || undefined, snapshot_state: 'pending' });
      push({ type: 'alert.new', alert: a });
      made.push(a.id);
      // Late snapshot -> alert.update, exercising §3 snapshot_state transitions.
      setTimeout(() => {
        a.snapshot_state = 'received';
        a.snapshot_url = '/api/alerts/' + a.id + '/snapshot.jpg';
        push({ type: 'alert.update', alert: { id: a.id, snapshot_state: 'received', snapshot_url: a.snapshot_url } });
      }, 2500);
    }
    log('burst', made.join(','));
    return send(res, 200, { created: made });
  }
  if (p === '/mock/offline' || p === '/mock/online') {
    const dev = devices.find((d) => d.device_id === q.device);
    if (!dev) return send(res, 404, { error: 'no such device' });
    dev.online = p === '/mock/online';
    dev.last_seen_ms = Date.now();
    if (!dev.online) dev.streams.forEach((st) => { st.fps = 0; st.state = 'stopped'; });
    push({ type: 'device.status', device: { device_id: dev.device_id, online: dev.online, last_seen_ms: dev.last_seen_ms, streams: dev.streams } });
    log('device', dev.device_id, dev.online ? 'online' : 'offline');
    return send(res, 200, { ok: true, online: dev.online });
  }
  if (p === '/mock/decode') {
    const dev = devices.find((d) => d.device_id === q.device);
    const target = streamOf(dev, q.stream);
    if (!target) return send(res, 404, { error: 'no such stream' });
    target.decode = q.decode === 'sw' ? 'sw' : 'hw';
    target.fallback_active = q.decode === 'sw';
    push({ type: 'device.status', device: { device_id: dev.device_id, online: dev.online, streams: dev.streams } });
    return send(res, 200, { ok: true });
  }
  if (p === '/mock/reject') { knobs.rejectRules = q.on !== '0'; return send(res, 200, knobs); }
  if (p === '/mock/fail') { knobs.failRules = q.on !== '0'; return send(res, 200, knobs); }
  if (p === '/mock/state') {
    return send(res, 200, {
      alerts: alerts.length,
      by_state: alerts.reduce((m2, a) => { m2[a.state] = (m2[a.state] || 0) + 1; return m2; }, {}),
      ws_clients: clients.size,
      sessions: sessions.size,
      knobs,
    });
  }
  const file = path.join(MOCKDIR, p.replace('/mock/', ''));
  if (file.startsWith(MOCKDIR) && fs.existsSync(file)) return serveFile(res, file);
  return send(res, 404, { error: 'no such mock route' });
}

const server = http.createServer(async (req, res) => {
  const u = url.parse(req.url, true);
  try {
    if (u.pathname.startsWith('/api/')) return await api(req, res, u);
    if (u.pathname.startsWith('/mock/')) return mock(req, res, u);
    if (SPA_ROUTES.includes(u.pathname)) return serveFile(res, path.join(ROOT, 'index.html'));
    const file = path.join(ROOT, u.pathname);
    if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end('no'); }
    return serveFile(res, file);
  } catch (e) {
    console.error('server error', e);
    if (!res.headersSent) send(res, 500, { error: String(e && e.message) });
  }
});

server.on('upgrade', handleUpgrade);

// Background traffic: a fresh alert every AUTO_ALERT_MS, snapshot landing 2.5 s later.
if (AUTO_ALERT_MS > 0) {
  setInterval(() => {
    if (!clients.size) return;
    const a = makeAlert({ snapshot_state: 'pending' });
    push({ type: 'alert.new', alert: a });
    setTimeout(() => {
      a.snapshot_state = 'received';
      a.snapshot_url = '/api/alerts/' + a.id + '/snapshot.jpg';
      push({ type: 'alert.update', alert: { id: a.id, snapshot_state: 'received', snapshot_url: a.snapshot_url } });
    }, 2500);
    log('auto alert', a.id, a.event_type, a.rule_name);
  }, AUTO_ALERT_MS);
}

// Heartbeat fps jitter so the devices page shows live numbers.
setInterval(() => {
  let changed = false;
  devices.forEach((d) => {
    if (!d.online) return;
    d.streams.forEach((st) => {
      if (!st.fps) return;
      const base = st.decode === 'sw' ? 6.2 : 14.2;
      st.fps = Number((base + (Math.random() - 0.5) * 1.6).toFixed(1));
      changed = true;
    });
    d.last_seen_ms = Date.now();
  });
  if (changed && clients.size) {
    devices.forEach((d) => push({ type: 'device.status', device: { device_id: d.device_id, online: d.online, last_seen_ms: d.last_seen_ms, streams: d.streams } }));
  }
}, 30000);

server.listen(PORT, () => {
  console.log('mock hub listening on http://127.0.0.1:' + PORT);
  console.log('  login: ' + USER.username + ' / ' + USER.password + (MUST_CHANGE ? '  (must_change=1)' : ''));
  console.log('  static root: ' + ROOT);
  console.log('  seeded alerts: ' + alerts.length + ', devices: ' + devices.length);
});
