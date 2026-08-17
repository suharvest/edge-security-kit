#!/usr/bin/env node
/* Structural gate: the mock hub's device payload must have the same `streams`
   shape as the contract fixture the real hub passes through.

   Background: contracts/mqtt-detection.schema.json declares status.streams as an
   ARRAY of stream_status objects, and hub/edge_hub/device_registry.py stores it
   verbatim. The mock server once modelled it as an object keyed by stream id, the
   frontend was written against the mock, and every Object.keys() over it produced
   array indices ("0", "1") as stream ids — visible as a stream named 0 and as
   rules saved to a stream that does not exist.

   This script checks three things:
     1. the fixture's streams is an array whose items carry stream_id (the truth);
     2. the mock's GET /api/devices agrees — array, same required fields;
     3. no frontend source treats device.streams as an object map.

   Run: node web/tools/check-streams-shape.mjs
*/
import { spawn } from 'node:child_process';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..');
const REPO = path.join(WEB, '..');
const FIXTURE = path.join(REPO, 'contracts', 'fixtures', 'status-generic.json');
const PORT = Number(process.env.PORT || 8791);

const failures = [];
const fail = (msg) => { failures.push(msg); console.error('FAIL ' + msg); };
const pass = (msg) => console.log('PASS ' + msg);

// ---- 1. the contract fixture -------------------------------------------------

const fixture = JSON.parse(readFileSync(FIXTURE, 'utf8'));
if (!Array.isArray(fixture.streams)) {
  fail('fixture ' + path.basename(FIXTURE) + ': streams is ' + typeof fixture.streams + ', expected an array');
} else if (!fixture.streams.length) {
  fail('fixture ' + path.basename(FIXTURE) + ': streams is empty, nothing to compare against');
} else {
  pass('fixture streams is a non-empty array (' + fixture.streams.length + ' item(s))');
}
const REQUIRED = ['stream_id', 'state'];
const fixtureKeys = Object.keys(fixture.streams[0] || {});
REQUIRED.forEach((k) => {
  if (!fixtureKeys.includes(k)) fail('fixture stream item is missing required field ' + k);
});

// ---- 3. static scan of the frontend -----------------------------------------
// Any `Object.keys(<something>.streams` / `.streams[<expr>]` outside the helper
// module is the exact bug this gate exists to prevent.

function jsFiles(dir, out = []) {
  readdirSync(dir).forEach((f) => {
    const full = path.join(dir, f);
    if (statSync(full).isDirectory()) { if (f !== 'vendor') jsFiles(full, out); }
    else if (f.endsWith('.js')) out.push(full);
  });
  return out;
}

const BAD = [
  /Object\.keys\(\s*[A-Za-z_$][\w$.]*\.streams/,
  /Object\.keys\(\s*\(?[A-Za-z_$][\w$.]*\.streams\s*\|\|/,
  /\.streams\s*\|\|\s*\{\}\s*\)\s*\[/,
];
let scanned = 0;
jsFiles(path.join(WEB, 'dist', 'js')).forEach((file) => {
  if (file.endsWith(path.join('js', 'util.js'))) return; // the helper itself
  scanned += 1;
  readFileSync(file, 'utf8').split('\n').forEach((line, i) => {
    // The device-config export really is an object keyed by stream id; skip it.
    if (line.includes('cfg.streams') || line.includes('streamsObj')) return;
    if (BAD.some((re) => re.test(line))) {
      fail(path.relative(REPO, file) + ':' + (i + 1) + ' treats device.streams as an object map: ' + line.trim());
    }
  });
});
pass('scanned ' + scanned + ' frontend source file(s) for object-map access to device.streams');

// ---- 2. the mock hub ---------------------------------------------------------

const child = spawn(process.execPath, [path.join(WEB, 'mock-server.js'), '--port', String(PORT), '--quiet', '--interval', '0'], {
  stdio: ['ignore', 'pipe', 'inherit'],
});
let childOut = '';
child.stdout.on('data', (b) => { childOut += b.toString(); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitUp() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch('http://127.0.0.1:' + PORT + '/api/health');
      if (r.ok) return true;
    } catch (e) { /* not listening yet */ }
    await sleep(100);
  }
  return false;
}

let code = 0;
try {
  if (!await waitUp()) throw new Error('mock server did not come up on :' + PORT + '\n' + childOut);

  const login = await fetch('http://127.0.0.1:' + PORT + '/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'operator', password: 'operator' }),
  });
  if (!login.ok) throw new Error('mock login failed: HTTP ' + login.status);
  const cookie = (login.headers.get('set-cookie') || '').split(';')[0];

  const res = await fetch('http://127.0.0.1:' + PORT + '/api/devices', { headers: { cookie } });
  if (!res.ok) throw new Error('GET /api/devices failed: HTTP ' + res.status);
  const body = await res.json();
  const devices = Array.isArray(body) ? body : body.devices;
  if (!Array.isArray(devices) || !devices.length) throw new Error('/api/devices returned no devices');
  pass('mock /api/devices returned ' + devices.length + ' device(s)');

  devices.forEach((d) => {
    const where = 'mock device ' + d.device_id;
    if (!Array.isArray(d.streams)) {
      fail(where + ': streams is ' + (d.streams === null ? 'null' : typeof d.streams) + ', expected an array (contract shape)');
      return;
    }
    d.streams.forEach((s, i) => {
      REQUIRED.forEach((k) => {
        if (s[k] === undefined) fail(where + ' streams[' + i + ']: missing required field ' + k);
      });
      if (typeof s.stream_id !== 'string' || !s.stream_id) {
        fail(where + ' streams[' + i + ']: stream_id must be a non-empty string, got ' + JSON.stringify(s.stream_id));
      }
      // The exact symptom of the object/array mix-up: indices used as ids.
      if (String(s.stream_id) === String(i)) {
        fail(where + ' streams[' + i + ']: stream_id equals the array index — this is what an object-map bug looks like');
      }
    });
    // §4 device rows carry versions:{app,model}, not a flat version.
    if (d.versions === undefined && d.version !== undefined) {
      fail(where + ': reports a flat `version` and no `versions` object (contract §status_message)');
    }
  });
  pass('mock device streams match the contract shape (array of stream_status)');
} catch (e) {
  fail(String((e && e.message) || e));
} finally {
  child.kill('SIGTERM');
}

console.log('---');
if (failures.length) {
  console.error(failures.length + ' failure(s)');
  code = 1;
} else {
  console.log('streams shape consistent: contract fixture == mock hub == frontend access pattern');
}
process.exit(code);
