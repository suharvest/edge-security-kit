import { html } from '../../vendor/preact-htm.js';
import { t, getLang, setLang } from '../i18n.js';
import { useStore, commit, toast } from '../store.js';
import { Icon } from '../icons.js';
import { api } from '../api.js';
import { navigate, linkProps } from '../router.js';
import { isSoundOn, setSoundOn, isUnlocked } from '../sound.js';
import { stopWs } from '../ws.js';
import { streamList } from '../util.js';

function ConnBadge() {
  const st = useStore();
  const map = {
    connected: { cls: 'ok', text: t('conn.connected') },
    reconnecting: { cls: 'warn', text: t('conn.reconnecting') },
    offline: { cls: 'bad', text: t('conn.offline') },
  };
  const m = map[st.conn] || map.offline;
  return html`<span class=${'conn conn-' + m.cls} title=${m.text}><i class="dot"></i>${m.text}</span>`;
}

// Streams reported on software decode become a badge count on the Devices tab
// (FRONTEND_SPEC §5).
function swDecodeCount(devices) {
  let n = 0;
  (devices || []).forEach((d) => {
    streamList(d).forEach((s) => { if (s.decode === 'sw') n += 1; });
  });
  return n;
}

export function TopBar() {
  const st = useStore();
  const lang = getLang();
  const soundOn = isSoundOn();
  const unlocked = isUnlocked();
  const swCount = swDecodeCount(st.devices);
  const pending = st.alerts.filter((a) => a.state === 'new').length;

  const tab = (path, label, badge) => html`
    <a ...${linkProps(path)} class=${'tab' + (st.route.path === path ? ' tab-on' : '')}>
      ${label}${badge ? html`<span class="tab-badge">${badge}</span>` : null}
    </a>`;

  const logout = async () => {
    try { await api.logout(); } catch (e) { /* dropping the session locally is enough */ }
    stopWs();
    commit({ session: { user: null, checked: true, mustChange: false }, alerts: [], devices: [] });
    navigate('/login');
  };

  return html`
    <header class="topbar">
      <div class="brand">${Icon.shield({ size: 18 })}<span>${t('app.title')}</span></div>
      <nav class="tabs">
        ${tab('/', t('nav.alerts'), pending)}
        ${tab('/devices', t('nav.devices'), swCount)}
        ${tab('/rules', t('nav.rules'), 0)}
      </nav>
      <div class="topbar-right">
        <${ConnBadge} />
        <button class=${'icon-btn' + (soundOn ? ' on' : '')}
                title=${soundOn ? t('sound.on') : t('sound.off')}
                aria-label=${soundOn ? t('sound.on') : t('sound.off')}
                onClick=${() => { setSoundOn(!soundOn); toast(!soundOn ? t('sound.on') : t('sound.off'), 'info', 2000); }}>
          ${soundOn ? Icon.bell({ size: 18 }) : Icon.bellOff({ size: 18 })}
          ${soundOn && !unlocked ? html`<i class="warn-dot"></i>` : null}
        </button>
        <div class="lang">
          <button class=${lang === 'zh' ? 'on' : ''} onClick=${() => setLang('zh')}>中文</button>
          <span>/</span>
          <button class=${lang === 'en' ? 'on' : ''} onClick=${() => setLang('en')}>EN</button>
        </div>
        <span class="user" title=${t('nav.user')}>${st.session.user || '—'}</span>
        <button class="link-btn" onClick=${logout}>${t('nav.logout')}</button>
      </div>
    </header>`;
}
