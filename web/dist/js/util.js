import { t, getLang } from './i18n.js';

export function pad2(n) { return n < 10 ? '0' + n : String(n); }

export function absTime(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate())
    + ' ' + pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
}

export function hhmm(ms) {
  const d = new Date(ms || Date.now());
  return pad2(d.getHours()) + ':' + pad2(d.getMinutes());
}

export function clockTime(ms) {
  const d = new Date(ms || Date.now());
  return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
}

const REL = {
  zh: { now: '刚刚', s: '{n} 秒前', m: '{n} 分钟前', h: '{n} 小时前', d: '{n} 天前' },
  en: { now: 'just now', s: '{n}s ago', m: '{n}m ago', h: '{n}h ago', d: '{n}d ago' },
};

export function relTime(ms, nowMs) {
  if (!ms) return '—';
  const lang = REL[getLang()] ? getLang() : 'en';
  const tab = REL[lang];
  const diff = Math.max(0, Math.floor(((nowMs || Date.now()) - ms) / 1000));
  if (diff < 5) return tab.now;
  if (diff < 60) return tab.s.replace('{n}', diff);
  if (diff < 3600) return tab.m.replace('{n}', Math.floor(diff / 60));
  if (diff < 86400) return tab.h.replace('{n}', Math.floor(diff / 3600));
  return tab.d.replace('{n}', Math.floor(diff / 86400));
}

export function eventLabel(type) {
  const key = 'evt.' + type;
  const s = t(key);
  return s === key ? type : s;
}

// The hub names the device timestamp ts_ms on every alert payload — REST rows and
// WS alert.new/alert.update alike (HUB_SPEC §6 column name). There is no second
// spelling to fall back to; accepting two names hides a hub that sends neither.
export function alertTs(a) { return a.ts_ms || 0; }

export function dateInputValue(ms) {
  const d = new Date(ms);
  return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
}

// Filter time range -> {date_from, date_to} epoch-ms pair used by /api/alerts.
export function rangeBounds(range, fromStr, toStr) {
  const now = Date.now();
  if (range === '24h') return { date_from: now - 86400e3, date_to: null };
  if (range === '7d') return { date_from: now - 7 * 86400e3, date_to: null };
  if (range === 'custom') {
    const from = fromStr ? new Date(fromStr + 'T00:00:00').getTime() : null;
    const to = toStr ? new Date(toStr + 'T23:59:59.999').getTime() : null;
    return { date_from: from, date_to: to };
  }
  // today
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return { date_from: d.getTime(), date_to: null };
}

export function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function slug(s) {
  return String(s || '').replace(/[^A-Za-z0-9._-]+/g, '_');
}

export function uid(prefix) {
  return prefix + '-' + Math.random().toString(36).slice(2, 8) + Date.now().toString(36).slice(-4);
}

export function errMsg(e) {
  if (!e) return t('common.error');
  if (e.network) return t('conn.offline');
  return e.message || t('common.error');
}

// ---- device streams ----------------------------------------------------------
// `status.streams` is an ARRAY of stream_status objects (contracts/mqtt-detection
// .schema.json §stream_status), and the hub passes it through verbatim
// (device_registry.py). Treating it as an object keyed by stream id yields array
// indices ("0", "1", ...) as stream ids, which silently writes rules to a stream
// named "0". Every consumer goes through these helpers.
export function streamList(device) {
  const s = (device && device.streams) || [];
  if (Array.isArray(s)) return s.filter((x) => x && x.stream_id != null);
  // A legacy object map degrades instead of breaking: synthesize stream_id from the key.
  return Object.keys(s).map((k) => Object.assign({ stream_id: k }, s[k] || {}));
}

export function streamIds(device) {
  return streamList(device).map((s) => String(s.stream_id));
}

export function findStream(device, streamId) {
  if (streamId === null || streamId === undefined || streamId === '') return null;
  return streamList(device).find((s) => String(s.stream_id) === String(streamId)) || null;
}
