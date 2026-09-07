// REST client for HUB_SPEC §4. Same-origin, cookie session (§7): every request
// sends credentials; any 401 hands control to the 401 handler (router -> /login).
const BASE = '/api';

let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

export class ApiError extends Error {
  constructor(status, body) {
    super((body && body.error) || ('HTTP ' + status));
    this.status = status;
    this.body = body || null;
    this.network = false;
  }
}
export class NetworkError extends Error {
  constructor(cause) { super(String((cause && cause.message) || cause)); this.network = true; this.status = 0; }
}

async function req(method, path, body, opts = {}) {
  let res;
  const init = { method, credentials: 'same-origin', headers: {} };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  try {
    res = await fetch(BASE + path, init);
  } catch (e) {
    throw new NetworkError(e);
  }
  if (res.status === 401 && !opts.allow401) {
    onUnauthorized();
    throw new ApiError(401, { error: 'unauthorized' });
  }
  let payload = null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    try { payload = await res.json(); } catch (e) { payload = null; }
  } else if (opts.text) {
    payload = await res.text();
  }
  if (!res.ok) throw new ApiError(res.status, payload);
  return payload;
}

export function qs(params) {
  const u = new URLSearchParams();
  Object.keys(params || {}).forEach((k) => {
    const v = params[k];
    if (v === undefined || v === null || v === '' || v === 'all') return;
    u.set(k, String(v));
  });
  const s = u.toString();
  return s ? '?' + s : '';
}

export const api = {
  health: () => req('GET', '/health', undefined, { allow401: true }),

  login: (username, password) => req('POST', '/auth/login', { username, password }, { allow401: true }),
  // Authoritative source for "who am I" and must_change: the session cookie the hub
  // issued, not a username the browser wrote down for itself.
  session: () => req('GET', '/auth/session', undefined, { allow401: true }),
  logout: () => req('POST', '/auth/logout'),
  changePassword: (old_password, new_password) =>
    req('POST', '/auth/password', { old_password, new_password }, { allow401: true }),

  alerts: (params) => req('GET', '/alerts' + qs(params)),
  ack: (id) => req('POST', '/alerts/' + id + '/ack'),
  dismiss: (id) => req('POST', '/alerts/' + id + '/dismiss'),
  ackBatch: (ids) => req('POST', '/alerts/ack', { ids }),
  dismissBatch: (ids) => req('POST', '/alerts/dismiss', { ids }),
  alert: (id) => req('GET', '/alerts/' + id),
  exportUrl: (params) => BASE + '/alerts/export.csv' + qs(params),
  snapshotUrl: (id) => BASE + '/alerts/' + id + '/snapshot.jpg',

  devices: () => req('GET', '/devices'),
  deviceConfig: (id) => req('GET', '/devices/' + encodeURIComponent(id) + '/config'),
  putDeviceConfig: (id, body) => req('PUT', '/devices/' + encodeURIComponent(id) + '/config', body),
  // Hub-side single-frame proxy. The device's own preview_url is a device-local
  // address the browser usually cannot reach; this one is same-origin and carries
  // the session cookie. HUB_SPEC §4.
  streamPreviewUrl: (id, sid) =>
    BASE + '/devices/' + encodeURIComponent(id) + '/streams/' + encodeURIComponent(sid) + '/preview.jpg',

  // Runtime control (contracts/MQTT.md "Control downlink"). All three answer
  // 200 only when the detector confirmed the change is live; 409 is a refusal
  // with the device's own reason, 504 means the ack never arrived and the
  // caller must not redraw as if it had succeeded.
  setConfThreshold: (d, s, conf_threshold) =>
    req('PUT', '/devices/' + encodeURIComponent(d) + '/streams/' + encodeURIComponent(s) + '/conf',
      { conf_threshold }),
  addStream: (d, body) => req('POST', '/devices/' + encodeURIComponent(d) + '/streams', body),
  removeStream: (d, s) =>
    req('DELETE', '/devices/' + encodeURIComponent(d) + '/streams/' + encodeURIComponent(s)),
  audit: (params) => req('GET', '/audit' + qs(params)),

  rules: () => req('GET', '/rules'),
  streamRules: (d, s) => req('GET', '/rules/' + encodeURIComponent(d) + '/' + encodeURIComponent(s)),
  putStreamRules: (d, s, body) => req('PUT', '/rules/' + encodeURIComponent(d) + '/' + encodeURIComponent(s), body),
  simulate: (d, s, rule_id) =>
    req('POST', '/rules/' + encodeURIComponent(d) + '/' + encodeURIComponent(s) + '/simulate', { rule_id }),

  live: (d, s) => req('GET', '/live/' + encodeURIComponent(d) + '/' + encodeURIComponent(s)),
  // Batch form for the video wall: one request covers every tile, so a
  // screenful of overlay boxes comes from the same instant.
  liveAll: () => req('GET', '/live'),

  config: () => req('GET', '/config'),
  putConfig: (body) => req('PUT', '/config', body),
};

// Pull a 400 validation body into a per-target issue list.
// HUB_SPEC §4 only guarantees {"error": "..."}; a richer {issues:[...]} body is used
// when present so the sidebar can flag the offending entry (FRONTEND_SPEC §4.5).
export function validationIssues(err) {
  const b = err && err.body;
  if (!b) return [];
  if (Array.isArray(b.issues)) {
    return b.issues.map((it) => ({
      kind: it.kind || it.target_type || null,
      id: it.id || it.target_id || null,
      name: it.name || null,
      message: it.message || it.error || String(it),
    }));
  }
  if (b.error) return [{ kind: null, id: null, name: null, message: b.error }];
  return [];
}
