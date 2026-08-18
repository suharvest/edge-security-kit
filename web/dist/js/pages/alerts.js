// Alert workbench — FRONTEND_SPEC §3 (+ §8 undo, §7 WS).
import { html, useState, useEffect, useMemo, useRef } from '../../vendor/preact-htm.js';
import { t } from '../i18n.js';
import { api } from '../api.js';
import { useStore, commit, setAlerts, getState, toast } from '../store.js';
import { setQuery } from '../router.js';
import { Icon, eventIcon } from '../icons.js';
import { Modal, Field, Select, Empty, Spinner, Banner } from '../components/ui.js';
import { stage, undoNow, secondsLeft, UNDO_MS } from '../undo.js';
import { absTime, relTime, eventLabel, alertTs, rangeBounds, dateInputValue, errMsg, streamIds, findStream } from '../util.js';
import { shouldPrompt, dismissPrompt, requestNow } from '../notify.js';
import { isSoundOn, isUnlocked } from '../sound.js';

const STATES = ['all', 'new', 'acked', 'dismissed'];
const EVENTS = ['all', 'zone_enter', 'loitering', 'line_cross'];
const RANGES = ['today', '24h', '7d', 'custom'];
const FP_WARN = 0.4;      // §3.4 threshold suggestion
const BATCH_MAX = 500;    // §3.3
const PAGE_LIMIT = 200;   // §3.1 initial pull

// ---- filter state <-> URL query ----------------------------------------------

function filterFromQuery(q) {
  return {
    state: STATES.includes(q.state) ? q.state : 'new',
    device_id: q.device_id || '',
    stream_id: q.stream_id || '',
    rule_name: q.rule_name || '',
    event_type: EVENTS.includes(q.event_type) ? q.event_type : 'all',
    range: RANGES.includes(q.range) ? q.range : 'today',
    from: q.from || dateInputValue(Date.now() - 7 * 86400e3),
    to: q.to || dateInputValue(Date.now()),
  };
}

function queryFromFilter(f) {
  const q = {};
  if (f.state !== 'new') q.state = f.state;
  if (f.device_id) q.device_id = f.device_id;
  if (f.stream_id) q.stream_id = f.stream_id;
  if (f.rule_name) q.rule_name = f.rule_name;
  if (f.event_type !== 'all') q.event_type = f.event_type;
  if (f.range !== 'today') q.range = f.range;
  if (f.range === 'custom') { q.from = f.from; q.to = f.to; }
  return q;
}

function restParams(f) {
  const b = rangeBounds(f.range, f.from, f.to);
  const p = { limit: PAGE_LIMIT };
  if (f.state !== 'all') p.state = f.state;
  if (f.device_id) p.device_id = f.device_id;
  if (f.stream_id) p.stream_id = f.stream_id;
  if (f.event_type !== 'all') p.event_type = f.event_type;
  // rule_name is a HUB_SPEC §4 filter parameter on /alerts and /alerts/export.csv
  // alike, so the list and the CSV taken from it cover the same rows. It used to be
  // applied in the browser instead, which left the export unfiltered.
  if (f.rule_name) p.rule_name = f.rule_name;
  if (b.date_from) p.date_from = b.date_from;
  if (b.date_to) p.date_to = b.date_to;
  return p;
}

// Gate for WS-pushed rows (§7): the REST pull is already filtered server-side, but
// alert.new arrives regardless of the active filter and must not slip into a filtered
// list. Every filter dimension the server understands is mirrored here for that reason.
function matches(a, f) {
  if (f.state !== 'all' && a.state !== f.state) return false;
  if (f.device_id && a.device_id !== f.device_id) return false;
  if (f.stream_id && a.stream_id !== f.stream_id) return false;
  if (f.event_type !== 'all' && a.event_type !== f.event_type) return false;
  if (f.rule_name && a.rule_name !== f.rule_name) return false;
  const b = rangeBounds(f.range, f.from, f.to);
  const ts = alertTs(a);
  if (b.date_from && ts < b.date_from) return false;
  if (b.date_to && ts > b.date_to) return false;
  return true;
}

// ---- stats -------------------------------------------------------------------

function computeStats(rows) {
  const s = { total: rows.length, pending: 0, acked: 0, dismissed: 0, byRule: {} };
  rows.forEach((a) => {
    if (a.state === 'new') s.pending += 1;
    else if (a.state === 'acked') s.acked += 1;
    else if (a.state === 'dismissed') s.dismissed += 1;
    const r = (s.byRule[a.rule_name] = s.byRule[a.rule_name] || { total: 0, dismissed: 0 });
    r.total += 1;
    if (a.state === 'dismissed') r.dismissed += 1;
  });
  const decided = s.acked + s.dismissed;
  s.fpRate = decided ? s.dismissed / decided : 0;
  Object.keys(s.byRule).forEach((k) => {
    const r = s.byRule[k];
    r.rate = r.total ? r.dismissed / r.total : 0;
  });
  return s;
}

function pct(x) { return Math.round(x * 100); }

// ---- components --------------------------------------------------------------

function FilterPanel({ f, set, devices, rules, onExport }) {
  const streamOptions = useMemo(() => {
    const out = new Set();
    devices.forEach((d) => {
      if (f.device_id && d.device_id !== f.device_id) return;
      streamIds(d).forEach((s) => out.add(s));
    });
    return Array.from(out);
  }, [devices, f.device_id]);

  const opt = (v, label) => ({ value: v, label });
  return html`
    <aside class="filters">
      <h2>${Icon.filter({ size: 14 })} ${t('alerts.filters')}</h2>

      <${Field} label=${t('alerts.state')}>
        <${Select} value=${f.state} onChange=${(v) => set({ state: v })}
          options=${STATES.map((s) => opt(s, s === 'all' ? t('common.all') : t('alerts.state.' + s)))} />
      <//>

      <${Field} label=${t('alerts.device')}>
        <${Select} value=${f.device_id} onChange=${(v) => set({ device_id: v, stream_id: '' })}
          options=${[opt('', t('common.all'))].concat(devices.map((d) => opt(d.device_id, d.name || d.device_id)))} />
      <//>

      <${Field} label=${t('alerts.stream')}>
        <${Select} value=${f.stream_id} onChange=${(v) => set({ stream_id: v })}
          options=${[opt('', t('common.all'))].concat(streamOptions.map((s) => opt(s, s)))} />
      <//>

      <${Field} label=${t('alerts.rule')}>
        <${Select} value=${f.rule_name} onChange=${(v) => set({ rule_name: v })}
          options=${[opt('', t('common.all'))].concat(rules.map((r) => opt(r, r)))} />
      <//>

      <${Field} label=${t('alerts.eventType')}>
        <${Select} value=${f.event_type} onChange=${(v) => set({ event_type: v })}
          options=${EVENTS.map((e) => opt(e, e === 'all' ? t('common.all') : eventLabel(e)))} />
      <//>

      <${Field} label=${t('alerts.range')}>
        <${Select} value=${f.range} onChange=${(v) => set({ range: v })}
          options=${RANGES.map((r) => opt(r, t('alerts.range.' + r)))} />
      <//>
      ${f.range === 'custom' ? html`
        <div class="row2">
          <${Field} label=${t('alerts.range.from')}>
            <input type="date" value=${f.from} onInput=${(e) => set({ from: e.target.value })} />
          <//>
          <${Field} label=${t('alerts.range.to')}>
            <input type="date" value=${f.to} onInput=${(e) => set({ to: e.target.value })} />
          <//>
        </div>` : null}

      <button class="btn btn-block" onClick=${onExport}>${Icon.download({ size: 14 })} ${t('alerts.export')}</button>
    </aside>`;
}

function ShiftStats({ stats }) {
  const hot = Object.keys(stats.byRule).filter((k) => stats.byRule[k].rate > FP_WARN && stats.byRule[k].total >= 4);
  return html`
    <div class="stats">
      <div class="stat"><b>${stats.total}</b><span>${t('stats.total')}</span></div>
      <div class="stat stat-new"><b>${stats.pending}</b><span>${t('stats.pending')}</span></div>
      <div class="stat"><b>${stats.acked}</b><span>${t('stats.acked')}</span></div>
      <div class="stat"><b>${stats.dismissed}</b><span>${t('stats.fp')}</span></div>
      <div class=${'stat' + (stats.fpRate > FP_WARN ? ' stat-warn' : '')}>
        <b>${pct(stats.fpRate)}%</b><span>${t('stats.fpRate')}</span>
      </div>
      ${hot.length ? html`
        <div class="stat-hint">
          ${Icon.warn({ size: 14 })}
          <span>${t('stats.byRule')}: ${hot.map((k) => html`
            <b class="rule-hot" title=${t('stats.fpWarn', { rate: pct(stats.byRule[k].rate) })}>
              ${k} ${pct(stats.byRule[k].rate)}%
            </b>`)}
          </span>
        </div>` : null}
    </div>`;
}

function StateBadge({ a, onReclassify }) {
  const other = a.state === 'acked' ? 'dismiss' : 'ack';
  return html`
    <div class=${'state-badge sb-' + a.state}>
      <span>${t('alerts.state.' + a.state)}</span>
      ${a.acted_by ? html`<i>${t('alerts.actedBy', { who: a.acted_by })}</i>` : null}
      ${a.acted_at ? html`<i>${absTime(a.acted_at)}</i>` : null}
      <button class="reclassify" onClick=${() => onReclassify(other)}>${Icon.refresh({ size: 12 })} ${t('alerts.reclassify')}</button>
    </div>`;
}

function Snapshot({ a, onOpen, thumb }) {
  const [failed, setFailed] = useState(false);
  const url = a.snapshot_url || api.snapshotUrl(a.id);
  const noImage = failed || a.snapshot_state === 'pending' || a.snapshot_state === 'timeout' || a.snapshot_state === 'none';
  if (noImage) {
    const label = a.snapshot_state === 'pending' ? t('alerts.snapshotPending')
      : a.snapshot_state === 'timeout' ? t('alerts.snapshotTimeout') : t('alerts.snapshotNone');
    const pending = !failed && a.snapshot_state === 'pending';
    return html`<div class=${'snap snap-empty' + (pending ? ' snap-pending' : '') + (thumb ? '' : ' snap-big')}
                     title=${label}><span>${label}</span></div>`;
  }
  return html`
    <img class=${'snap' + (thumb ? '' : ' snap-big')} src=${url} alt="snapshot" loading="lazy"
         onError=${() => setFailed(true)} onClick=${onOpen} />`;
}

function AlertCard({ a, flash, batch, checked, onCheck, onOpen, ruleStats, devices }) {
  const dev = devices.find((d) => d.device_id === a.device_id);
  const liveUrl = (findStream(dev, a.stream_id) || {}).live_url || a.live_url || null;
  const rs = ruleStats[a.rule_name];
  const hotRule = rs && rs.rate > FP_WARN && rs.total >= 4;
  const disp = (kind) => stage(kind, [a.id]);

  return html`
    <article class=${'card card-' + a.state + (flash ? ' card-flash' : '') + (a._pending ? ' card-pending' : '')}>
      ${batch ? html`
        <label class="card-check">
          <input type="checkbox" checked=${checked} onChange=${(e) => onCheck(a.id, e.target.checked)} />
        </label>` : null}
      <${Snapshot} a=${a} thumb=${true} onOpen=${() => onOpen(a)} />
      <div class="card-mid">
        <div class="card-line1">
          <span class=${'evt evt-' + a.event_type}>${eventIcon(a.event_type)} ${eventLabel(a.event_type)}</span>
          <b class="rule">${a.rule_name}</b>
          ${hotRule ? html`<span class="rule-hot" title=${t('stats.fpWarn', { rate: pct(rs.rate) })}>${Icon.warn({ size: 12 })} ${pct(rs.rate)}%</span>` : null}
          ${a.simulated || (a.meta && a.meta.simulated) ? html`<span class="tag tag-sim">${t('alerts.simulated')}</span>` : null}
        </div>
        <div class="card-line2">
          <span class="mono">${(dev && dev.name) || a.device_id}/${a.stream_id}</span>
          <span class="sep">·</span>
          <span title=${absTime(alertTs(a))}>${relTime(alertTs(a))}</span>
          <span class="dim">${absTime(alertTs(a))}</span>
        </div>
        <div class="card-line3 dim">
          <span>${t('alerts.track')} ${a.track_id}</span>
          ${a.score != null ? html`<span>${t('alerts.score')} ${Number(a.score).toFixed(2)}</span>` : null}
          ${a.dwell_s != null ? html`<span>${t('alerts.dwell')} ${Number(a.dwell_s).toFixed(1)}s</span>` : null}
          ${a.direction ? html`<span>${t('alerts.direction')} ${t('rules.dir.' + a.direction)}</span>` : null}
          <span class="mono">#${a.id}</span>
        </div>
      </div>
      <div class="card-right">
        ${a.state === 'new' ? html`
          <div class="disp-row">
            <button class="btn btn-ok" onClick=${() => disp('ack')}>${Icon.check({ size: 14 })} ${t('alerts.ack')}</button>
            <button class="btn btn-bad" onClick=${() => disp('dismiss')}>${Icon.x({ size: 14 })} ${t('alerts.dismiss')}</button>
          </div>
        ` : html`<${StateBadge} a=${a} onReclassify=${(k) => disp(k)} />`}
        <a class=${'btn btn-ghost' + (liveUrl ? '' : ' btn-disabled')}
           href=${liveUrl || '#'} target="_blank" rel="noopener"
           title=${liveUrl ? a.stream_id : t('alerts.liveMissing')}
           onClick=${(e) => { if (!liveUrl) e.preventDefault(); }}>
          ${Icon.ext({ size: 14 })} ${t('alerts.live')}
        </a>
      </div>
    </article>`;
}

function DetailModal({ a, onClose, devices }) {
  const dev = devices.find((d) => d.device_id === a.device_id);
  const liveUrl = (findStream(dev, a.stream_id) || {}).live_url || null;
  return html`
    <${Modal} wide=${true} title=${t('alerts.detail') + ' #' + a.id} onClose=${onClose} footer=${html`
      ${a.state === 'new' ? html`
        <button class="btn btn-ok" onClick=${() => { stage('ack', [a.id]); onClose(); }}>${t('alerts.ack')}</button>
        <button class="btn btn-bad" onClick=${() => { stage('dismiss', [a.id]); onClose(); }}>${t('alerts.dismiss')}</button>
      ` : null}
      ${liveUrl ? html`<a class="btn btn-ghost" href=${liveUrl} target="_blank" rel="noopener">${t('alerts.live')}</a>` : null}
      <button class="btn" onClick=${onClose}>${t('common.close')}</button>`}>
      <${Snapshot} a=${a} thumb=${false} onOpen=${() => {}} />
      <dl class="kv">
        <dt>${t('alerts.eventType')}</dt><dd>${eventLabel(a.event_type)}</dd>
        <dt>${t('alerts.rule')}</dt><dd>${a.rule_name}</dd>
        <dt>${t('alerts.device')}</dt><dd class="mono">${a.device_id}/${a.stream_id}</dd>
        <dt>${t('alerts.state')}</dt><dd>${t('alerts.state.' + a.state)}${a.acted_by ? ' · ' + a.acted_by : ''}</dd>
        <dt>ts</dt><dd class="mono">${absTime(alertTs(a))}</dd>
        <dt>${t('alerts.track')}</dt><dd class="mono">${a.track_id}</dd>
        ${a.score != null ? html`<dt>${t('alerts.score')}</dt><dd class="mono">${Number(a.score).toFixed(3)}</dd>` : null}
        ${a.direction ? html`<dt>${t('alerts.direction')}</dt><dd>${t('rules.dir.' + a.direction)}</dd>` : null}
        ${a.dwell_s != null ? html`<dt>${t('alerts.dwell')}</dt><dd class="mono">${a.dwell_s}s</dd>` : null}
        ${a.bbox ? html`<dt>bbox</dt><dd class="mono">${JSON.stringify(a.bbox)}</dd>` : null}
        <dt>event_id</dt><dd class="mono">${a.event_id || '—'}</dd>
      </dl>
    <//>`;
}

function ExportDialog({ f, onClose }) {
  const [range, setRange] = useState(f.range);
  const [from, setFrom] = useState(f.from);
  const [to, setTo] = useState(f.to);
  const params = restParams(Object.assign({}, f, { range, from, to }));
  delete params.limit;
  const url = api.exportUrl(params);
  return html`
    <${Modal} title=${t('export.title')} onClose=${onClose} footer=${html`
      <button class="btn" onClick=${onClose}>${t('common.cancel')}</button>
      <a class="btn btn-primary" href=${url} onClick=${() => setTimeout(onClose, 300)}>
        ${Icon.download({ size: 14 })} ${t('export.go')}
      </a>`}>
      <p class="dim">${t('export.inherit')}</p>
      <${Field} label=${t('alerts.range')}>
        <${Select} value=${range} onChange=${setRange}
          options=${RANGES.map((r) => ({ value: r, label: t('alerts.range.' + r) }))} />
      <//>
      ${range === 'custom' ? html`
        <div class="row2">
          <${Field} label=${t('alerts.range.from')}><input type="date" value=${from} onInput=${(e) => setFrom(e.target.value)} /><//>
          <${Field} label=${t('alerts.range.to')}><input type="date" value=${to} onInput=${(e) => setTo(e.target.value)} /><//>
        </div>` : null}
      <p class="dim small">${t('export.columns')}</p>
      <p class="mono small break">${url}</p>
    <//>`;
}

function UndoBar() {
  const st = useStore();
  const u = st.undo;
  if (!u) return null;
  const left = secondsLeft();
  const label = u.kind === 'ack' ? t('undo.acked', { n: u.ids.length }) : t('undo.dismissed', { n: u.ids.length });
  const frac = Math.max(0, Math.min(1, (u.deadline - Date.now()) / UNDO_MS));
  return html`
    <div class="undobar">
      <span>${label}</span>
      <button class="btn btn-ghost" onClick=${undoNow}>${Icon.undo({ size: 14 })} ${t('undo.undo')}</button>
      <span class="undo-count">${t('undo.seconds', { s: left })}</span>
      <i class="undo-progress" style=${'transform:scaleX(' + frac + ')'}></i>
    </div>`;
}

// ---- page --------------------------------------------------------------------

export function AlertsPage() {
  const st = useStore();
  const f = useMemo(() => filterFromQuery(st.route.query), [st.route.query]);
  const [batch, setBatch] = useState(false);
  const [sel, setSel] = useState({});
  const [detail, setDetail] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [flash, setFlash] = useState({});
  const listRef = useRef(null);
  const seen = useRef(null);
  const filterKey = JSON.stringify(f);

  const set = (patch) => setQuery(queryFromFilter(Object.assign({}, f, patch)));

  // Initial + on-filter-change REST pull (§3.1).
  useEffect(() => {
    let dead = false;
    seen.current = null;
    commit({ alertsLoading: true, alertsError: null });
    api.alerts(restParams(f)).then((res) => {
      if (dead) return;
      const rows = Array.isArray(res) ? res : (res && (res.alerts || res.items)) || [];
      setAlerts(rows);
      commit({ alertsLoading: false, newSinceScroll: 0 });
    }).catch((e) => {
      if (dead) return;
      if (e.status === 401) return;
      commit({ alertsLoading: false, alertsError: t('alerts.loadFailed', { msg: errMsg(e) }) });
    });
    return () => { dead = true; };
  }, [filterKey]);

  const visible = useMemo(() => st.alerts.filter((a) => matches(a, f)), [st.alerts, filterKey]);
  const stats = useMemo(() => computeStats(visible), [visible]);

  // 2 s highlight for rows that appeared after the initial load (§3.1).
  useEffect(() => {
    const ids = visible.map((a) => a.id);
    if (seen.current === null) { seen.current = new Set(ids); return; }
    const fresh = ids.filter((id) => !seen.current.has(id));
    if (!fresh.length) return;
    fresh.forEach((id) => seen.current.add(id));
    setFlash((prev) => { const n = Object.assign({}, prev); fresh.forEach((id) => { n[id] = 1; }); return n; });
    setTimeout(() => setFlash((prev) => {
      const n = Object.assign({}, prev);
      fresh.forEach((id) => { delete n[id]; });
      return n;
    }), 2000);
  }, [visible]);

  const devices = st.devices;
  const rules = useMemo(() => {
    const s = new Set(st.alerts.map((a) => a.rule_name).filter(Boolean));
    if (f.rule_name) s.add(f.rule_name);
    return Array.from(s).sort();
  }, [st.alerts, f.rule_name]);

  const onScroll = () => {
    const el = listRef.current;
    if (el && el.scrollTop <= 40 && getState().newSinceScroll) commit({ newSinceScroll: 0 });
  };
  const backToTop = () => {
    if (listRef.current) listRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    commit({ newSinceScroll: 0 });
  };

  const selIds = Object.keys(sel).filter((k) => sel[k]).map(Number).filter((id) => visible.some((a) => a.id === id));
  const doBatch = (kind) => {
    if (selIds.length > BATCH_MAX) { toast(t('alerts.batchLimit'), 'error', 6000); return; }
    stage(kind, selIds);
    setSel({});
  };
  const toggleAll = (on) => {
    if (!on) { setSel({}); return; }
    const next = {};
    visible.slice(0, BATCH_MAX).forEach((a) => { next[a.id] = true; });
    if (visible.length > BATCH_MAX) toast(t('alerts.batchLimit'), 'error', 6000);
    setSel(next);
  };

  const showSoundHint = isSoundOn() && !isUnlocked();
  const showNotifyHint = shouldPrompt();

  return html`
    <div class="wb">
      <${FilterPanel} f=${f} set=${set} devices=${devices} rules=${rules} onExport=${() => setExporting(true)} />
      <section class="wb-main">
        ${st.conn === 'offline' ? html`<${Banner} kind="bad">${t('conn.banner')}<//>` : null}
        ${showSoundHint ? html`<${Banner} kind="info">${Icon.bell({ size: 14 })} ${t('sound.unlock')}<//>` : null}
        ${showNotifyHint ? html`
          <${Banner} kind="info" onDismiss=${dismissPrompt} action=${html`
            <button class="btn btn-primary btn-sm" onClick=${requestNow}>${t('common.enable')}</button>`}>
            ${t('notify.prompt')}
          <//>` : null}

        <div class="wb-head">
          <${ShiftStats} stats=${stats} />
          <label class="batch-toggle">
            <input type="checkbox" checked=${batch} onChange=${(e) => { setBatch(e.target.checked); setSel({}); }} />
            <span>${t('alerts.batchMode')}</span>
          </label>
        </div>

        ${batch ? html`
          <div class="batchbar">
            <label>
              <input type="checkbox"
                     checked=${selIds.length > 0 && selIds.length >= Math.min(visible.length, BATCH_MAX)}
                     onChange=${(e) => toggleAll(e.target.checked)} />
              <span>${t('alerts.selectAll')}</span>
            </label>
            <span class="dim">${t('alerts.selected', { n: selIds.length })}</span>
            <button class="btn btn-ok btn-sm" disabled=${!selIds.length} onClick=${() => doBatch('ack')}>${t('alerts.batchAck')}</button>
            <button class="btn btn-bad btn-sm" disabled=${!selIds.length} onClick=${() => doBatch('dismiss')}>${t('alerts.batchDismiss')}</button>
          </div>` : null}

        ${st.newSinceScroll > 0 ? html`
          <button class="newbar" onClick=${backToTop}>${Icon.up({ size: 14 })} ${t('alerts.newBanner', { n: st.newSinceScroll })}</button>` : null}

        <div class="alert-list" ref=${listRef} onScroll=${onScroll}>
          ${st.alertsError ? html`<div class="err-text pad">${st.alertsError}</div>` : null}
          ${st.alertsLoading && !visible.length ? html`<${Spinner} />` : null}
          ${!st.alertsLoading && !visible.length && !st.alertsError ? html`<${Empty} text=${t('alerts.emptyList')} />` : null}
          ${visible.map((a) => html`
            <${AlertCard} key=${a.id} a=${a} flash=${!!flash[a.id]} batch=${batch}
              checked=${!!sel[a.id]} onCheck=${(id, on) => setSel((p) => Object.assign({}, p, { [id]: on }))}
              onOpen=${setDetail} ruleStats=${stats.byRule} devices=${devices} />`)}
        </div>
      </section>
      <${UndoBar} />
      ${detail ? html`<${DetailModal} a=${st.alerts.find((x) => x.id === detail.id) || detail}
                        devices=${devices} onClose=${() => setDetail(null)} />` : null}
      ${exporting ? html`<${ExportDialog} f=${f} onClose=${() => setExporting(false)} />` : null}
    </div>`;
}
