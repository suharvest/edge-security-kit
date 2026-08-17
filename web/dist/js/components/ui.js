import { html, useState, useEffect, useRef } from '../../vendor/preact-htm.js';
import { t } from '../i18n.js';
import { useStore, dropToast } from '../store.js';
import { Icon } from '../icons.js';

export function Modal({ title, onClose, children, footer, wide }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return html`
    <div class="modal-back" onClick=${(e) => { if (e.target.classList.contains('modal-back')) onClose(); }}>
      <div class=${'modal' + (wide ? ' modal-wide' : '')} role="dialog" aria-modal="true">
        <header>
          <h2>${title}</h2>
          <button class="icon-btn" onClick=${onClose} title=${t('common.close')} aria-label=${t('common.close')}>
            ${Icon.x({ size: 18 })}
          </button>
        </header>
        <div class="modal-body">${children}</div>
        ${footer ? html`<footer>${footer}</footer>` : null}
      </div>
    </div>`;
}

export function Toasts() {
  const st = useStore();
  if (!st.toasts.length) return null;
  return html`
    <div class="toasts">
      ${st.toasts.map((x) => html`
        <div class=${'toast toast-' + x.kind} key=${x.id} onClick=${() => dropToast(x.id)}>
          ${x.kind === 'error' ? Icon.warn({ size: 14 }) : x.kind === 'ok' ? Icon.check({ size: 14 }) : null}
          <span>${x.text}</span>
        </div>`)}
    </div>`;
}

export function Banner({ kind = 'info', children, onDismiss, action }) {
  return html`
    <div class=${'banner banner-' + kind}>
      <div class="banner-text">${children}</div>
      <div class="banner-actions">
        ${action || null}
        ${onDismiss ? html`<button class="link-btn" onClick=${onDismiss}>${t('common.dismissTip')}</button>` : null}
      </div>
    </div>`;
}

export function Field({ label, hint, children, error }) {
  return html`
    <label class=${'field' + (error ? ' field-err' : '')}>
      <span class="field-label">${label}</span>
      ${children}
      ${error ? html`<span class="field-msg">${error}</span>` : hint ? html`<span class="field-hint">${hint}</span>` : null}
    </label>`;
}

export function Select({ value, onChange, options, ariaLabel }) {
  return html`
    <select value=${value} aria-label=${ariaLabel || ''} onChange=${(e) => onChange(e.target.value)}>
      ${options.map((o) => html`<option value=${o.value} key=${o.value}>${o.label}</option>`)}
    </select>`;
}

export function Confirm({ title, body, confirmLabel, danger, onConfirm, onClose }) {
  return html`
    <${Modal} title=${title} onClose=${onClose} footer=${html`
      <button class="btn" onClick=${onClose}>${t('common.cancel')}</button>
      <button class=${danger ? 'btn btn-danger' : 'btn btn-primary'} onClick=${() => { onConfirm(); onClose(); }}>
        ${confirmLabel || t('common.confirm')}
      </button>`}>
      <p>${body}</p>
    <//>`;
}

// Danger confirmation that requires typing a word (FRONTEND_SPEC §8).
export function TypedConfirm({ title, body, word, confirmLabel, onConfirm, onClose }) {
  const [typed, setTyped] = useState('');
  const ok = typed.trim() === word;
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.focus(); }, []);
  return html`
    <${Modal} title=${title} onClose=${onClose} footer=${html`
      <button class="btn" onClick=${onClose}>${t('common.cancel')}</button>
      <button class="btn btn-danger" disabled=${!ok} onClick=${() => { onConfirm(); onClose(); }}>
        ${confirmLabel || t('common.confirm')}
      </button>`}>
      <p>${body}</p>
      <input ref=${ref} value=${typed} onInput=${(e) => setTyped(e.target.value)} placeholder=${word} />
      ${typed && !ok ? html`<p class="err-text">${t('danger.confirmMismatch')}</p>` : null}
    <//>`;
}

export function Spinner({ label }) {
  return html`<div class="spinner"><span class="dot"></span><span>${label || t('common.loading')}</span></div>`;
}

export function Empty({ text }) {
  return html`<div class="empty">${text || t('common.empty')}</div>`;
}
