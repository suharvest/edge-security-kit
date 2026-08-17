// History-API router for the paths in FRONTEND_SPEC §2 (/login, /, /devices, /rules).
// The hub must serve index.html for those paths (SPA fallback). If the app is opened
// from a host that does not do that, the router transparently degrades to hash routing
// so deep links still work.
import { getState, commit } from './store.js';

export const ROUTES = ['/', '/login', '/devices', '/rules', '/debug'];

let hashMode = false;

function parseQuery(search) {
  const out = {};
  new URLSearchParams(search || '').forEach((v, k) => { out[k] = v; });
  return out;
}

function readLocation() {
  if (location.hash.startsWith('#/')) {
    hashMode = true;
    const raw = location.hash.slice(1);
    const qi = raw.indexOf('?');
    const path = qi < 0 ? raw : raw.slice(0, qi);
    return { path: ROUTES.includes(path) ? path : '/', query: parseQuery(qi < 0 ? '' : raw.slice(qi + 1)) };
  }
  if (ROUTES.includes(location.pathname)) {
    return { path: location.pathname, query: parseQuery(location.search) };
  }
  hashMode = true;
  return { path: '/', query: parseQuery(location.search) };
}

function buildUrl(path, query) {
  const u = new URLSearchParams();
  Object.keys(query || {}).forEach((k) => {
    const v = query[k];
    if (v === undefined || v === null || v === '') return;
    u.set(k, String(v));
  });
  const q = u.toString();
  if (hashMode) return location.pathname + '#' + path + (q ? '?' + q : '');
  return path + (q ? '?' + q : '');
}

export function href(path, query) { return buildUrl(path, query); }

export function navigate(path, query, replace) {
  const url = buildUrl(path, query);
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  commit({ route: { path, query: query || {} } });
  if (!replace) window.scrollTo(0, 0);
}

// Filter changes rewrite the URL in place — bookmarkable without flooding history.
export function setQuery(query) {
  const path = getState().route.path;
  history.replaceState(null, '', buildUrl(path, query));
  commit({ route: { path, query: query || {} } });
}

export function linkProps(path, query) {
  return {
    href: buildUrl(path, query),
    onClick: (e) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      e.preventDefault();
      navigate(path, query);
    },
  };
}

export function startRouter() {
  commit({ route: readLocation() });
  window.addEventListener('popstate', () => commit({ route: readLocation() }));
  window.addEventListener('hashchange', () => commit({ route: readLocation() }));
}
