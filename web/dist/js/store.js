// Minimal observable store + hooks. One global state object; components subscribe
// and re-render on any commit. Alert list state lives here so WS pushes, REST loads
// and the UndoBar all mutate one place (FRONTEND_SPEC §1 rationale).
import { useState, useEffect } from '../vendor/preact-htm.js';
import { onLangChange, getLang } from './i18n.js';

const state = {
  route: { path: '/', query: {} },
  // Filled from GET /api/auth/session at boot (HUB_SPEC §4). Nothing about the
  // signed-in identity is kept in localStorage: the cookie is the session, and a
  // remembered username only ever disagreed with it.
  session: { user: null, checked: false, mustChange: false },
  lang: getLang(),

  conn: 'offline', // connected | reconnecting | offline
  alerts: [],       // newest first
  alertsLoading: false,
  alertsError: null,
  lastAlertId: 0,
  newSinceScroll: 0,

  devices: [],
  devicesError: null,

  hubConfig: null,

  undo: null,   // { kind:'ack'|'dismiss', ids:[], deadline, prev:Map, timer }
  toasts: [],   // { id, kind:'info'|'error'|'ok', text }
};

const subs = new Set();
export function getState() { return state; }
export function commit(patch) {
  if (patch) Object.assign(state, patch);
  subs.forEach((fn) => fn());
}
export function subscribe(fn) { subs.add(fn); return () => subs.delete(fn); }

// Re-render everything on language switch.
onLangChange((l) => commit({ lang: l }));

export function useStore() {
  const [, force] = useState(0);
  useEffect(() => subscribe(() => force((n) => n + 1)), []);
  return state;
}

let toastSeq = 1;
export function toast(text, kind = 'info', ms = 4000) {
  const id = toastSeq++;
  state.toasts = state.toasts.concat([{ id, kind, text }]);
  commit();
  setTimeout(() => {
    state.toasts = state.toasts.filter((x) => x.id !== id);
    commit();
  }, ms);
}
export function dropToast(id) {
  state.toasts = state.toasts.filter((x) => x.id !== id);
  commit();
}

// ---- alert list helpers -------------------------------------------------

export function setAlerts(rows) {
  state.alerts = rows.slice().sort(byIdDesc);
  state.lastAlertId = state.alerts.reduce((m, a) => Math.max(m, a.id), 0);
  commit();
}

export function byIdDesc(a, b) { return (b.id || 0) - (a.id || 0); }

// Insert or replace a pushed alert. Returns 'new' | 'update' | 'dup'.
export function upsertAlert(alert) {
  const i = state.alerts.findIndex((a) => a.id === alert.id);
  if (i >= 0) {
    state.alerts = state.alerts.slice();
    state.alerts[i] = Object.assign({}, state.alerts[i], alert);
    commit();
    return 'update';
  }
  state.alerts = [alert].concat(state.alerts).sort(byIdDesc);
  if (alert.id > state.lastAlertId) state.lastAlertId = alert.id;
  commit();
  return 'new';
}

export function patchAlerts(ids, patch) {
  const set = new Set(ids);
  state.alerts = state.alerts.map((a) => (set.has(a.id) ? Object.assign({}, a, patch) : a));
  commit();
}

export function restoreAlerts(prev) {
  state.alerts = state.alerts.map((a) => (prev.has(a.id) ? Object.assign({}, a, prev.get(a.id)) : a));
  commit();
}

export function upsertDevice(dev) {
  const i = state.devices.findIndex((d) => d.device_id === dev.device_id);
  if (i >= 0) {
    state.devices = state.devices.slice();
    state.devices[i] = Object.assign({}, state.devices[i], dev);
  } else {
    state.devices = state.devices.concat([dev]);
  }
  commit();
}
