// Browser notifications, FRONTEND_SPEC §3.1: never call requestPermission() on load.
// A banner explains the benefit; only the explicit "Enable" click asks the browser.
// A denial is remembered so the banner stops nagging; the devices page keeps a
// re-onboarding entry point.
import { commit, toast } from './store.js';
import { t } from './i18n.js';

const KEY = 'notify.asked';

export function supported() { return typeof Notification !== 'undefined'; }
export function permission() { return supported() ? Notification.permission : 'unsupported'; }
export function asked() { return localStorage.getItem(KEY) === '1'; }

// Show the explainer only when we could still gain something by asking.
export function shouldPrompt() {
  return supported() && permission() === 'default' && !asked();
}

export function dismissPrompt() {
  localStorage.setItem(KEY, '1');
  commit();
}

export async function requestNow() {
  if (!supported()) return 'unsupported';
  localStorage.setItem(KEY, '1');
  let p = 'default';
  try { p = await Notification.requestPermission(); } catch (e) { p = permission(); }
  commit();
  if (p === 'granted') toast(t('notify.granted'), 'ok');
  else if (p === 'denied') toast(t('notify.denied'), 'error', 6000);
  return p;
}

// Re-onboarding from the devices page: if the browser already hard-denied us,
// requestPermission() resolves 'denied' without a prompt, so say so explicitly.
export async function reonboard() {
  if (permission() === 'denied') { toast(t('notify.denied'), 'error', 6000); return 'denied'; }
  localStorage.removeItem(KEY);
  return requestNow();
}

export function notifyAlert(alert, body) {
  if (permission() !== 'granted') return;
  if (!document.hidden) return; // only useful when the page is in the background
  try {
    const n = new Notification(t('notify.title'), { body, tag: 'alert-' + alert.id });
    n.onclick = () => { window.focus(); n.close(); };
  } catch (e) { /* Safari throws for non-SW notifications in some versions */ }
}
