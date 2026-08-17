// App entry: session bootstrap, routing, WS wiring, sound/notification side effects.
import { html, render, useEffect } from '../vendor/preact-htm.js';
import { t } from './i18n.js';
import { api, setUnauthorizedHandler } from './api.js';
import { useStore, commit, getState, toast } from './store.js';
import { startRouter, navigate } from './router.js';
import { TopBar } from './components/topbar.js';
import { Toasts } from './components/ui.js';
import { AlertsPage } from './pages/alerts.js';
import { DevicesPage } from './pages/devices.js';
import { RulesPage } from './pages/rules.js';
import { LoginPage } from './pages/login.js';
import { DebugPage } from './pages/debug.js';
import { startWs, stopWs } from './ws.js';
import { installUnlockHandler, playAlert } from './sound.js';
import { notifyAlert } from './notify.js';
import { eventLabel } from './util.js';

function toLogin() {
  const st = getState();
  stopWs();
  commit({ session: { user: null, checked: true, mustChange: false } });
  if (st.route.path !== '/login') navigate('/login');
}

setUnauthorizedHandler(toLogin);

function onNewAlert(alert) {
  playAlert();
  notifyAlert(alert, eventLabel(alert.event_type) + ' · ' + alert.rule_name
    + ' · ' + alert.device_id + '/' + alert.stream_id);
}

let wsStarted = false;
function ensureWs() {
  if (wsStarted) return;
  wsStarted = true;
  startWs({ onNewAlert, onUnauthorized: toLogin });
}

function goLogin() {
  commit({ session: { user: null, checked: true, mustChange: false } });
  if (getState().route.path !== '/login') navigate('/login', getState().route.query, true);
}

async function bootstrap() {
  startRouter();
  installUnlockHandler();

  // Identity comes from the hub, not from the browser: GET /api/auth/session
  // resolves the session cookie into a username plus must_change (HUB_SPEC §4).
  // A remembered localStorage username could outlive the cookie, which rendered
  // the workbench for a signed-out visitor until the first 401 came back.
  let session;
  try {
    session = await api.session();
  } catch (e) {
    if (e.status !== 401) toast(t('conn.offline'), 'error');
    goLogin();
    return;
  }
  if (!session || !session.username) { goLogin(); return; }

  commit({
    session: { user: session.username, checked: true, mustChange: !!session.must_change },
  });
  if (session.must_change) {
    // §7: the forced rotation must survive a reload, so the initial password
    // cannot be left in place by refreshing past the change form.
    navigate('/login', {}, true);
    return;
  }
  try {
    const res = await api.devices();
    commit({ devices: Array.isArray(res) ? res : (res && res.devices) || [] });
  } catch (e) {
    if (e.status !== 401) toast(t('conn.offline'), 'error');
  }
  ensureWs();
}

function App() {
  const st = useStore();
  const path = st.route.path;

  useEffect(() => {
    if (st.session.user && st.session.checked) ensureWs();
  }, [st.session.user, st.session.checked]);

  if (path === '/login' || !st.session.user) {
    return html`<div class="app"><${LoginPage} /><${Toasts} /></div>`;
  }
  const page = path === '/devices' ? html`<${DevicesPage} />`
    : path === '/rules' ? html`<${RulesPage} />`
      : path === '/debug' ? html`<${DebugPage} />`
        : html`<${AlertsPage} />`;
  return html`
    <div class="app">
      <${TopBar} />
      <main class="main">${page}</main>
      <${Toasts} />
    </div>`;
}

render(html`<${App} />`, document.getElementById('root'));
bootstrap();
