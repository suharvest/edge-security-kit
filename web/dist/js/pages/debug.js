// Device-local debug page (FRONTEND_SPEC §6) — placeholder in the hub build.
// The real page is hosted by the detector container on :8080/debug and reuses the
// §4 canvas components from this same bundle.
import { html } from '../../vendor/preact-htm.js';
import { t } from '../i18n.js';
import { Banner } from '../components/ui.js';

export function DebugPage() {
  return html`
    <div class="pad">
      <${Banner} kind="warn">${t('debug.banner')}<//>
      <h1>${t('debug.title')}</h1>
      <p class="dim">${t('debug.placeholder')}</p>
    </div>`;
}
