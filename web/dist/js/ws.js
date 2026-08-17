// WS client for FRONTEND_SPEC §7 / HUB_SPEC §5.
// Same origin, same port — the URL is derived from location, never a hardcoded port.
// Session cookie rides the handshake; a rejected handshake means the session is gone.
import { api } from './api.js';
import { getState, commit, upsertAlert, upsertDevice, toast } from './store.js';
import { t } from './i18n.js';

const PING_MS = 30000;
const BACKOFF = [1000, 2000, 4000, 8000, 16000, 30000];

let sock = null;
let tries = 0;
let pingTimer = null;
let retryTimer = null;
let stopped = false;
let hooks = { onNewAlert: () => {}, onUnauthorized: () => {} };

export function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return proto + '//' + location.host + '/ws';
}

export function startWs(h) {
  hooks = Object.assign(hooks, h || {});
  stopped = false;
  open();
}

export function stopWs() {
  stopped = true;
  clearTimeout(retryTimer);
  clearInterval(pingTimer);
  if (sock) { try { sock.close(); } catch (e) {} }
  sock = null;
  commit({ conn: 'offline' });
}

function setConn(c) {
  if (getState().conn !== c) commit({ conn: c });
}

function open() {
  if (stopped) return;
  let s;
  try {
    s = new WebSocket(wsUrl());
  } catch (e) {
    scheduleRetry();
    return;
  }
  sock = s;
  s.onopen = () => {
    tries = 0;
    setConn('connected');
    clearInterval(pingTimer);
    pingTimer = setInterval(() => {
      if (s.readyState === 1) s.send(JSON.stringify({ type: 'ping' }));
    }, PING_MS);
    catchUp();
  };
  s.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handle(msg);
  };
  s.onclose = (ev) => {
    clearInterval(pingTimer);
    if (stopped) return;
    // 1008/4401-style policy closes and an immediate close on first connect are the
    // observable symptom of a dead session; probe REST to find out which it is.
    scheduleRetry();
    probeSession();
  };
  s.onerror = () => { /* onclose always follows */ };
}

function scheduleRetry() {
  setConn(tries === 0 ? 'reconnecting' : (tries > 2 ? 'offline' : 'reconnecting'));
  const wait = BACKOFF[Math.min(tries, BACKOFF.length - 1)];
  tries += 1;
  clearTimeout(retryTimer);
  retryTimer = setTimeout(open, wait);
}

let probing = false;
async function probeSession() {
  if (probing) return;
  probing = true;
  try {
    await api.alerts({ limit: 1 });
  } catch (e) {
    if (e.status === 401) hooks.onUnauthorized();
  } finally {
    probing = false;
  }
}

function handle(msg) {
  if (!msg || !msg.type) return;
  if (msg.type === 'alert.new' && msg.alert) {
    const kind = upsertAlert(msg.alert);
    if (kind === 'new') {
      const st = getState();
      commit({ newSinceScroll: st.newSinceScroll + 1 });
      hooks.onNewAlert(msg.alert);
    }
    return;
  }
  if (msg.type === 'alert.update' && msg.alert) {
    // Only merge if we already track the row; an update for an unseen row is
    // pulled in whole so the list does not end up with a partial record.
    const st = getState();
    if (st.alerts.some((a) => a.id === msg.alert.id)) upsertAlert(msg.alert);
    else api.alert(msg.alert.id).then((full) => full && upsertAlert(full)).catch(() => {});
    return;
  }
  if (msg.type === 'device.status' && msg.device) {
    upsertDevice(msg.device);
    return;
  }
}

// Reconnect catch-up, FRONTEND_SPEC §7:
//  - alert.new gaps come from GET /alerts?after_id=<last seen>, id ascending.
//  - alert.update gaps (late snapshots, remote reclassification) are NOT covered by
//    after_id, so re-GET every rendered alert that still lacks a snapshot.
async function catchUp() {
  const st = getState();
  if (!st.session.user) return;
  if (st.lastAlertId > 0) {
    try {
      const res = await api.alerts({ after_id: st.lastAlertId, limit: 500 });
      const rows = Array.isArray(res) ? res : (res && res.alerts) || [];
      rows.forEach((a) => {
        if (upsertAlert(a) === 'new') {
          commit({ newSinceScroll: getState().newSinceScroll + 1 });
          hooks.onNewAlert(a);
        }
      });
      if (rows.length) toast(t('conn.connected'), 'ok', 2500);
    } catch (e) { /* the next reconnect retries */ }
  }
  const stale = getState().alerts.filter(
    (a) => a.snapshot_state === 'pending' || (!a.snapshot_url && a.snapshot_state !== 'none'),
  ).slice(0, 50);
  for (const a of stale) {
    try { const full = await api.alert(a.id); if (full) upsertAlert(full); } catch (e) {}
  }
}
