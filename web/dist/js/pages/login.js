// Login + forced initial password change (FRONTEND_SPEC §2, HUB_SPEC §7).
import { html, useState, useRef, useEffect } from '../../vendor/preact-htm.js';
import { t, getLang, setLang } from '../i18n.js';
import { api } from '../api.js';
import { commit, getState, toast } from '../store.js';
import { Field } from '../components/ui.js';
import { Icon } from '../icons.js';
import { navigate } from '../router.js';

export function LoginPage() {
  const [u, setU] = useState('');
  const [p, setP] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [mustChange, setMustChange] = useState(getState().session.mustChange);
  const [oldPw, setOldPw] = useState('');
  const [n1, setN1] = useState('');
  const [n2, setN2] = useState('');
  const first = useRef(null);
  const lang = getLang();

  useEffect(() => { if (first.current) first.current.focus(); }, [mustChange]);

  const submit = async (e) => {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      const res = await api.login(u, p);
      // The hub echoes the account name it authenticated; nothing is persisted
      // client-side, the cookie is the whole session (HUB_SPEC §7).
      const user = (res && res.username) || u;
      const needs = !!(res && res.must_change);
      commit({ session: { user, checked: true, mustChange: needs } });
      if (needs) {
        setMustChange(true);
        setOldPw(p);
      } else {
        navigate('/');
      }
    } catch (e2) {
      setErr(e2.network ? t('login.network') : (e2.status === 401 ? t('login.failed') : (e2.message || t('common.error'))));
    } finally {
      setBusy(false);
    }
  };

  const submitChange = async (e) => {
    e.preventDefault();
    setErr(null);
    if (n1.length < 8) { setErr(t('login.tooShort')); return; }
    if (n1 !== n2) { setErr(t('login.mismatch')); return; }
    setBusy(true);
    try {
      await api.changePassword(oldPw, n1);
      const user = getState().session.user || u;
      commit({ session: { user, checked: true, mustChange: false } });
      toast(t('login.changed'), 'ok');
      navigate('/');
    } catch (e2) {
      setErr(e2.network ? t('login.network') : (e2.message || t('common.error')));
    } finally {
      setBusy(false);
    }
  };

  return html`
    <div class="login-wrap">
      <form class="login-card" onSubmit=${mustChange ? submitChange : submit}>
        <div class="login-head">
          ${Icon.shield({ size: 22 })}
          <h1>${t('app.title')}</h1>
          <div class="lang">
            <button type="button" class=${lang === 'zh' ? 'on' : ''} onClick=${() => setLang('zh')}>中文</button>
            <span>/</span>
            <button type="button" class=${lang === 'en' ? 'on' : ''} onClick=${() => setLang('en')}>EN</button>
          </div>
        </div>

        ${!mustChange ? html`
          <h2>${t('login.heading')}</h2>
          <${Field} label=${t('login.username')}>
            <input ref=${first} value=${u} autocomplete="username" onInput=${(e) => setU(e.target.value)} />
          <//>
          <${Field} label=${t('login.password')} hint=${t('login.hint')}>
            <input type="password" value=${p} autocomplete="current-password" onInput=${(e) => setP(e.target.value)} />
          <//>
        ` : html`
          <h2>${t('login.mustChange')}</h2>
          <${Field} label=${t('login.oldPassword')}>
            <input type="password" value=${oldPw} autocomplete="current-password" onInput=${(e) => setOldPw(e.target.value)} />
          <//>
          <${Field} label=${t('login.newPassword')}>
            <input ref=${first} type="password" value=${n1} autocomplete="new-password" onInput=${(e) => setN1(e.target.value)} />
          <//>
          <${Field} label=${t('login.newPassword2')}>
            <input type="password" value=${n2} autocomplete="new-password" onInput=${(e) => setN2(e.target.value)} />
          <//>
        `}

        ${err ? html`<p class="err-text">${err}</p>` : null}
        <button class="btn btn-primary btn-block" type="submit" disabled=${busy}>
          ${busy ? t('common.loading') : (mustChange ? t('login.changeSubmit') : t('login.submit'))}
        </button>
      </form>
    </div>`;
}
