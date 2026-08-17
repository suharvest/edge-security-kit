// Deferred disposition with a 5 s undo window (FRONTEND_SPEC §8).
// The UI applies the new state optimistically and the REST call is held for 5 s;
// "Undo" simply drops the queued call. Single-alert disposition uses the same path so
// the cost of a mistake is identical either way.
import { api } from './api.js';
import { getState, commit, patchAlerts, restoreAlerts, toast } from './store.js';
import { t } from './i18n.js';
import { errMsg } from './util.js';

export const UNDO_MS = 5000;

function actor() { return getState().session.user || 'operator'; }

function targetState(kind) { return kind === 'ack' ? 'acked' : 'dismissed'; }

// Commit the queued REST call. Returns after the server has been told.
async function flush(entry) {
  const { kind, ids } = entry;
  clearInterval(entry.ticker);
  clearTimeout(entry.timer);
  if (getState().undo && getState().undo.token === entry.token) commit({ undo: null });
  try {
    let skipped = [];
    if (ids.length === 1) {
      try {
        const row = kind === 'ack' ? await api.ack(ids[0]) : await api.dismiss(ids[0]);
        if (row && row.id) patchAlerts([row.id], row);
      } catch (e) {
        if (e.status === 409 || e.status === 404) skipped = ids.slice();
        else throw e;
      }
    } else {
      const res = kind === 'ack' ? await api.ackBatch(ids) : await api.dismissBatch(ids);
      const list = Array.isArray(res) ? res : (res && (res.results || res.items)) || [];
      list.forEach((r) => {
        const st = r.status || r.code;
        if (st && Number(st) !== 200 && Number(st) !== 0) skipped.push(r.id);
        else if (r.alert) patchAlerts([r.alert.id], r.alert);
      });
    }
    if (skipped.length) {
      toast(t('alerts.skipped', { n: skipped.length }), 'error', 6000);
      // Pull the authoritative rows so the list stops showing our optimistic guess.
      for (const id of skipped) {
        try { const full = await api.alert(id); if (full) patchAlerts([id], full); } catch (e) {}
      }
    }
  } catch (e) {
    restoreAlerts(entry.prev);
    toast(t('undo.failed', { msg: errMsg(e) }), 'error', 7000);
  }
}

export function stage(kind, ids) {
  if (!ids || !ids.length) return;
  const st = getState();
  // A new disposition while one is queued: commit the queued one first, no silent drop.
  if (st.undo) flush(st.undo);

  const prev = new Map();
  st.alerts.forEach((a) => {
    if (ids.includes(a.id)) {
      prev.set(a.id, { state: a.state, acted_by: a.acted_by || null, acted_at: a.acted_at || null, _pending: undefined });
    }
  });
  patchAlerts(ids, { state: targetState(kind), acted_by: actor(), acted_at: Date.now(), _pending: kind });

  const entry = {
    token: Math.random().toString(36).slice(2),
    kind,
    ids: ids.slice(),
    prev,
    deadline: Date.now() + UNDO_MS,
    timer: null,
    ticker: null,
  };
  entry.timer = setTimeout(() => flush(entry), UNDO_MS);
  entry.ticker = setInterval(() => commit(), 250); // drives the countdown label
  commit({ undo: entry });
}

export function undoNow() {
  const entry = getState().undo;
  if (!entry) return;
  clearTimeout(entry.timer);
  clearInterval(entry.ticker);
  restoreAlerts(entry.prev);
  patchAlerts(entry.ids, { _pending: undefined });
  commit({ undo: null });
}

export function flushNow() {
  const entry = getState().undo;
  if (entry) flush(entry);
}

export function secondsLeft() {
  const entry = getState().undo;
  if (!entry) return 0;
  return Math.max(0, Math.ceil((entry.deadline - Date.now()) / 1000));
}

// Anything still queued must reach the hub before the tab goes away.
window.addEventListener('pagehide', () => { const e = getState().undo; if (e) flush(e); });
