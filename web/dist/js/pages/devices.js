// Devices page — FRONTEND_SPEC §5 (P2). LWT online state, decode badge, per-stream FPS,
// version, config backup/restore, password change, notification re-onboarding.
import { html, useState, useEffect, useMemo } from '../../vendor/preact-htm.js';
import { t } from '../i18n.js';
import { api } from '../api.js';
import { useStore, commit, toast } from '../store.js';
import { Icon } from '../icons.js';
import { Modal, Field, Empty, Spinner, Confirm } from '../components/ui.js';
import { absTime, relTime, downloadBlob, slug, errMsg } from '../util.js';
import { linkProps } from '../router.js';
import { permission, reonboard } from '../notify.js';

const FPS_WARN = 8; // §5 default threshold

function DecodeBadge({ decode }) {
  if (!decode) return html`<span class="badge">—</span>`;
  const sw = decode === 'sw';
  return html`
    <span class=${'badge ' + (sw ? 'badge-warn' : 'badge-ok')}
          title=${sw ? t('devices.decodeSw') : t('devices.decodeHw')}>
      ${decode}${sw ? ' ⚠' : ''}
    </span>`;
}

function StreamCells({ streams }) {
  const ids = Object.keys(streams || {});
  if (!ids.length) return html`<span class="dim">${t('devices.noStreams')}</span>`;
  return html`
    <div class="stream-cells">
      ${ids.map((id) => {
        const s = streams[id] || {};
        const low = typeof s.fps === 'number' && s.fps < FPS_WARN;
        return html`
          <div class="stream-cell" key=${id}>
            <span class="mono sid">${id}</span>
            <${DecodeBadge} decode=${s.decode} />
            <span class=${'fps' + (low ? ' fps-low' : '')} title=${low ? t('devices.fpsLow', { th: FPS_WARN }) : ''}>
              ${s.fps != null ? Number(s.fps).toFixed(1) : '—'}
            </span>
            ${s.fallback_active ? html`<span class="badge badge-warn">${t('devices.fallback')}</span>` : null}
          </div>`;
      })}
    </div>`;
}

function summarize(cfg) {
  let zones = 0;
  let lines = 0;
  const walk = (v) => {
    if (!v || typeof v !== 'object') return;
    if (Array.isArray(v)) { v.forEach(walk); return; }
    if (Array.isArray(v.zones)) zones += v.zones.length;
    if (Array.isArray(v.lines)) lines += v.lines.length;
    Object.keys(v).forEach((k) => { if (k !== 'zones' && k !== 'lines') walk(v[k]); });
  };
  walk(cfg);
  const streamsObj = (cfg && (cfg.streams || cfg.rules)) || {};
  const streams = Array.isArray(streamsObj) ? streamsObj.length : Object.keys(streamsObj).length;
  return { streams, zones, lines };
}

function RestoreDialog({ deviceId, current, onClose, onDone }) {
  const [file, setFile] = useState(null);
  const [parsed, setParsed] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  const pick = async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    setFile(f || null);
    setParsed(null);
    setErr(null);
    if (!f) return;
    try {
      const txt = await f.text();
      const obj = JSON.parse(txt);
      if (!obj || typeof obj !== 'object') throw new Error('bad');
      setParsed(obj);
    } catch (e) {
      setErr(t('devices.restoreBad'));
    }
  };

  const a = summarize(current || {});
  const b = parsed ? summarize(parsed) : null;

  const apply = async () => {
    setBusy(true);
    try {
      const res = await api.putDeviceConfig(deviceId, parsed);
      toast(t('devices.restoreDone', { rev: (res && res.rev) != null ? res.rev : '?' }), 'ok', 6000);
      onDone();
      onClose();
    } catch (e) {
      setErr(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return html`
    <${Modal} title=${t('devices.restoreTitle') + ' · ' + deviceId} onClose=${onClose} footer=${html`
      <button class="btn" onClick=${onClose}>${t('common.cancel')}</button>
      <button class="btn btn-danger" disabled=${!parsed || busy} onClick=${apply}>${t('devices.restoreApply')}</button>`}>
      <${Field} label=${t('devices.restorePick')}>
        <input type="file" accept="application/json,.json" onChange=${pick} />
      <//>
      ${err ? html`<p class="err-text">${err}</p>` : null}
      ${b ? html`
        <div class="diff">
          <h4>${t('devices.restoreDiff')}</h4>
          <ul>
            <li>${t('devices.restoreStreams', { from: a.streams, to: b.streams })}</li>
            <li>${t('devices.restoreZones', { from: a.zones, to: b.zones })}</li>
            <li>${t('devices.restoreLines', { from: a.lines, to: b.lines })}</li>
          </ul>
          <p class="mono small break">${file ? file.name : ''}</p>
        </div>` : null}
    <//>`;
}

function PasswordDialog({ onClose }) {
  const [oldPw, setOld] = useState('');
  const [n1, setN1] = useState('');
  const [n2, setN2] = useState('');
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setErr(null);
    if (n1.length < 8) { setErr(t('login.tooShort')); return; }
    if (n1 !== n2) { setErr(t('login.mismatch')); return; }
    setBusy(true);
    try {
      await api.changePassword(oldPw, n1);
      toast(t('login.changed'), 'ok');
      onClose();
    } catch (e) {
      setErr(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  return html`
    <${Modal} title=${t('devices.changePassword')} onClose=${onClose} footer=${html`
      <button class="btn" onClick=${onClose}>${t('common.cancel')}</button>
      <button class="btn btn-primary" disabled=${busy} onClick=${submit}>${t('common.save')}</button>`}>
      <${Field} label=${t('login.oldPassword')}><input type="password" value=${oldPw} onInput=${(e) => setOld(e.target.value)} /><//>
      <${Field} label=${t('login.newPassword')}><input type="password" value=${n1} onInput=${(e) => setN1(e.target.value)} /><//>
      <${Field} label=${t('login.newPassword2')}><input type="password" value=${n2} onInput=${(e) => setN2(e.target.value)} /><//>
      ${err ? html`<p class="err-text">${err}</p>` : null}
    <//>`;
}

function ConfigPanel({ device, hubConfig }) {
  const [cfg, setCfg] = useState(null);
  const [loading, setLoading] = useState(true);
  const [restoring, setRestoring] = useState(false);

  const load = () => {
    setLoading(true);
    api.deviceConfig(device.device_id)
      .then((res) => { setCfg(res); setLoading(false); })
      .catch((e) => { setLoading(false); if (e.status !== 401) toast(errMsg(e), 'error'); });
  };
  useEffect(load, [device.device_id]);

  const backup = () => {
    if (!cfg) return;
    downloadBlob(
      'config-' + slug(device.device_id) + '-' + new Date().toISOString().slice(0, 10) + '.json',
      new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' }),
    );
  };

  const s = cfg ? summarize(cfg) : null;
  return html`
    <div class="cfgpanel">
      ${loading ? html`<${Spinner} />` : null}
      <div class="cfg-cols">
        <div>
          <h4>${Icon.download({ size: 14 })} ${t('devices.backup')}</h4>
          <p class="dim small">${t('devices.backupHint')}</p>
          <button class="btn btn-sm" disabled=${!cfg} onClick=${backup}>${t('common.download')}</button>
          ${s ? html`<p class="dim small mono">streams ${s.streams} · zones ${s.zones} · lines ${s.lines}</p>` : null}
        </div>
        <div>
          <h4>${Icon.upload({ size: 14 })} ${t('devices.restore')}</h4>
          <p class="dim small">${t('devices.restorePick')}</p>
          <button class="btn btn-sm" onClick=${() => setRestoring(true)}>${t('common.upload')}</button>
        </div>
        <div>
          <h4>${Icon.gear({ size: 14 })} ${t('devices.mqtt')}</h4>
          <p class="dim small mono">
            ${t('devices.broker')}: ${(hubConfig && (hubConfig.mqtt_host || hubConfig.broker)) || '—'}
            ${hubConfig && hubConfig.mqtt_port ? ':' + hubConfig.mqtt_port : ''}
          </p>
          <p class="dim small mono">${t('devices.mode')}: ${device.mode || 'hub'}</p>
        </div>
      </div>
      ${restoring ? html`
        <${RestoreDialog} deviceId=${device.device_id} current=${cfg}
          onClose=${() => setRestoring(false)} onDone=${load} />` : null}
    </div>`;
}

export function DevicesPage() {
  const st = useStore();
  const [open, setOpen] = useState(null);
  const [loading, setLoading] = useState(!st.devices.length);
  const [pwd, setPwd] = useState(false);

  const load = () => {
    api.devices().then((res) => {
      const list = Array.isArray(res) ? res : (res && res.devices) || [];
      commit({ devices: list, devicesError: null });
      setLoading(false);
    }).catch((e) => {
      setLoading(false);
      if (e.status !== 401) commit({ devicesError: t('devices.loadFailed', { msg: errMsg(e) }) });
    });
  };
  useEffect(() => {
    load();
    api.config().then((c) => commit({ hubConfig: c })).catch(() => {});
  }, []);

  const swCount = useMemo(() => {
    let n = 0;
    st.devices.forEach((d) => Object.keys(d.streams || {}).forEach((k) => { if ((d.streams[k] || {}).decode === 'sw') n += 1; }));
    return n;
  }, [st.devices]);

  const localUrl = (d) => {
    const streams = d.streams || {};
    const k = Object.keys(streams).find((x) => streams[x] && streams[x].live_url);
    return k ? streams[k].live_url : null;
  };

  return html`
    <div class="devices-page">
      <div class="page-head">
        <h1>${t('devices.title')}</h1>
        <div class="page-head-right">
          ${swCount ? html`<span class="badge badge-warn">${t('devices.swBadgeCount', { n: swCount })}</span>` : null}
          ${permission() !== 'granted' ? html`
            <button class="btn btn-sm" onClick=${reonboard}>${Icon.bell({ size: 14 })} ${t('notify.reenable')}</button>` : null}
          <button class="btn btn-sm" onClick=${() => setPwd(true)}>${t('devices.changePassword')}</button>
          <button class="btn btn-sm" onClick=${load}>${Icon.refresh({ size: 14 })}</button>
        </div>
      </div>

      ${st.devicesError ? html`<div class="err-text pad">${st.devicesError}</div>` : null}
      ${loading && !st.devices.length ? html`<${Spinner} />` : null}
      ${!loading && !st.devices.length ? html`<${Empty} />` : null}

      ${st.devices.length ? html`
        <table class="dev-table">
          <thead>
            <tr>
              <th>${t('devices.device')}</th>
              <th>${t('devices.status')}</th>
              <th>${t('devices.streams')} · ${t('devices.decode')} · ${t('devices.fps')}</th>
              <th>${t('devices.version')}</th>
              <th>${t('devices.actions')}</th>
            </tr>
          </thead>
          <tbody>
            ${st.devices.map((d) => {
              const url = localUrl(d);
              const isOpen = open === d.device_id;
              return html`
                <${Fragmentish} key=${d.device_id}>
                  <tr class=${d.online ? '' : 'row-offline'}>
                    <td>
                      <b class="mono">${d.device_id}</b>
                      ${d.name ? html`<div class="dim small">${d.name}</div>` : null}
                    </td>
                    <td>
                      <span class=${'dotstate ' + (d.online ? 'on' : 'off')}>
                        <i></i>${d.online ? t('devices.online') : t('devices.offline')}
                      </span>
                      ${!d.online && d.last_seen_ms ? html`
                        <div class="dim small" title=${absTime(d.last_seen_ms)}>
                          ${t('devices.lastSeen', { t: relTime(d.last_seen_ms) })}
                        </div>` : null}
                    </td>
                    <td><${StreamCells} streams=${d.streams} /></td>
                    <td class="mono">${d.version || (d.info && d.info.version) || '—'}</td>
                    <td class="dev-actions">
                      <button class="btn btn-sm" onClick=${() => setOpen(isOpen ? null : d.device_id)}>
                        ${Icon.gear({ size: 14 })} ${t('devices.config')}
                      </button>
                      <a class=${'btn btn-sm' + (url ? '' : ' btn-disabled')} href=${url || '#'} target="_blank" rel="noopener"
                         title=${url ? url : t('devices.localMissing')}
                         onClick=${(e) => { if (!url) e.preventDefault(); }}>
                        ${Icon.ext({ size: 14 })} ${t('devices.localPage')}
                      </a>
                      <a class="btn btn-sm" ...${linkProps('/rules', { device_id: d.device_id, stream_id: Object.keys(d.streams || {})[0] || '' })}>
                        ${t('nav.rules')}
                      </a>
                    </td>
                  </tr>
                  ${isOpen ? html`
                    <tr class="cfg-row"><td colspan="5"><${ConfigPanel} device=${d} hubConfig=${st.hubConfig} /></td></tr>` : null}
                <//>`;
            })}
          </tbody>
        </table>` : null}

      ${pwd ? html`<${PasswordDialog} onClose=${() => setPwd(false)} />` : null}
    </div>`;
}

// A table row plus its expansion row must be siblings; htm needs a fragment host.
function Fragmentish(props) { return props.children; }
