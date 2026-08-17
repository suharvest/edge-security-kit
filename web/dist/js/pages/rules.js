// Rule editor — FRONTEND_SPEC §4. Zones (migrated interaction), lines (new tool with
// forward/backward/any direction), dwell input, explicit save with typed failure
// feedback, simulate. All geometry is stored as frame_norm (§4.1 / contracts/MQTT.md).
import { html, useState, useEffect, useMemo, useRef } from '../../vendor/preact-htm.js';
import { t } from '../i18n.js';
import { api, validationIssues } from '../api.js';
import { useStore, commit, toast } from '../store.js';
import { setQuery } from '../router.js';
import { Icon } from '../icons.js';
import { Field, Select, Confirm, Spinner } from '../components/ui.js';
import { clockTime, uid, errMsg, streamIds, findStream } from '../util.js';
import {
  computeFit, normToCanvas, canvasToNorm, eventToCanvas, forwardNormal, midpoint,
  distToSegment, pointInPolygon, isSelfIntersecting, polygonCentroid, bboxToRect, clamp01,
} from '../coords.js';

const DIRECTIONS = ['forward', 'backward', 'any'];
const DEFAULT_DWELL = 10;      // §4.2
const DEFAULT_COOLDOWN = 30;
const HIT_PX = 9;
const FALLBACK_FRAME = { w: 1920, h: 1080 };

function emptyBody() {
  return { zones: [], lines: [], features: {}, cooldown: DEFAULT_COOLDOWN };
}

function normalizeBody(raw) {
  const b = raw && typeof raw === 'object' ? raw : {};
  const body = {
    zones: (b.zones || []).map((z, i) => ({
      id: z.id || uid('z'),
      name: z.name || ('zone_' + (i + 1)),
      points: (z.points || []).map((p) => [clamp01(Number(p[0])), clamp01(Number(p[1]))]),
      dwell_s: z.dwell_s != null ? Number(z.dwell_s) : DEFAULT_DWELL,
    })),
    lines: (b.lines || []).map((l, i) => ({
      id: l.id || uid('l'),
      name: l.name || ('line_' + (i + 1)),
      start: [clamp01(Number((l.start || [0, 0])[0])), clamp01(Number((l.start || [0, 0])[1]))],
      end: [clamp01(Number((l.end || [1, 1])[0])), clamp01(Number((l.end || [1, 1])[1]))],
      direction: DIRECTIONS.includes(l.direction) ? l.direction : 'any',
    })),
    features: b.features || {},
    cooldown: b.cooldown != null ? Number(b.cooldown) : DEFAULT_COOLDOWN,
  };
  return body;
}

// Local pre-flight validation, same rules the hub rejects with 400 (§4.2/§4.5).
function localIssues(body) {
  const out = [];
  const names = {};
  const seen = (kind, id, name) => {
    if (!name) return;
    if (names[name]) out.push({ kind, id, message: t('rules.err.dupName', { name }) });
    names[name] = true;
  };
  body.zones.forEach((z) => {
    seen('zone', z.id, z.name);
    if (z.points.length < 3) out.push({ kind: 'zone', id: z.id, message: t('rules.err.tooFewPoints', { name: z.name }) });
    else if (isSelfIntersecting(z.points)) out.push({ kind: 'zone', id: z.id, message: t('rules.err.selfIntersect', { name: z.name }) });
  });
  body.lines.forEach((l) => {
    seen('line', l.id, l.name);
    if (Math.hypot(l.end[0] - l.start[0], l.end[1] - l.start[1]) < 1e-4) {
      out.push({ kind: 'line', id: l.id, message: t('rules.err.degenerateLine', { name: l.name }) });
    }
  });
  return out;
}

// ---- canvas -----------------------------------------------------------------

function ArrowGlyph({ a, b, direction }) {
  const m = midpoint(a, b);
  const n = forwardNormal(a, b);
  const L = 32;
  const head = 9;
  const draw = (sign, key) => {
    const tip = [m[0] + n[0] * L * sign, m[1] + n[1] * L * sign];
    const base = [m[0] + n[0] * 8 * sign, m[1] + n[1] * 8 * sign];
    const perp = [-n[1], n[0]];
    const p1 = [tip[0] - n[0] * head * sign + perp[0] * head * 0.6, tip[1] - n[1] * head * sign + perp[1] * head * 0.6];
    const p2 = [tip[0] - n[0] * head * sign - perp[0] * head * 0.6, tip[1] - n[1] * head * sign - perp[1] * head * 0.6];
    return html`
      <g key=${key} class="arrow">
        <line x1=${base[0]} y1=${base[1]} x2=${tip[0]} y2=${tip[1]} />
        <polygon points=${tip[0] + ',' + tip[1] + ' ' + p1[0] + ',' + p1[1] + ' ' + p2[0] + ',' + p2[1]} />
      </g>`;
  };
  if (direction === 'any') return html`<g>${draw(1, 'f')}${draw(-1, 'b')}</g>`;
  return draw(direction === 'forward' ? 1 : -1, 'one');
}

function RuleCanvas({ body, setBody, frame, previewSources, live, showLive, tool, setTool, sel, setSel, issueIds }) {
  const wrapRef = useRef(null);
  const svgRef = useRef(null);
  const [size, setSize] = useState({ w: 640, h: 360 });
  const [draftZone, setDraftZone] = useState([]);
  const [draftLine, setDraftLine] = useState(null);
  const [hoverNorm, setHoverNorm] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const drag = useRef(null);
  const [pendingDelete, setPendingDelete] = useState(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null;
    if (ro) ro.observe(el);
    window.addEventListener('resize', measure);
    return () => { if (ro) ro.disconnect(); window.removeEventListener('resize', measure); };
  }, []);

  // §2.3 backdrop sourcing. Candidates are tried in order and the first that
  // decodes wins: the hub's same-origin single-frame proxy first (it works from
  // anywhere the UI works), the device's own preview_url only as an optimisation
  // for a browser that happens to sit on the device network. All candidates
  // failing degrades to the grey canvas, which is still fully drawable.
  const sources = (previewSources || []).filter(Boolean);
  const sourceKey = sources.join('|');
  useEffect(() => {
    setPreviewSrc(null);
    if (!sources.length) return;
    let dead = false;
    let index = 0;
    const img = new Image();
    const attempt = () => {
      if (dead || index >= sources.length) return;
      img.src = sources[index];
    };
    img.onload = () => { if (!dead) setPreviewSrc(sources[index]); };
    img.onerror = () => { index += 1; attempt(); };
    attempt();
    return () => { dead = true; img.onload = null; img.onerror = null; };
  }, [sourceKey]);
  const previewOk = !!previewSrc;

  const fit = useMemo(() => computeFit(size.w, size.h, frame.w, frame.h), [size, frame.w, frame.h]);
  const toC = (n) => normToCanvas(n, fit);

  const commitBody = (mut) => {
    const next = JSON.parse(JSON.stringify(body));
    mut(next);
    setBody(next);
  };

  const hitTest = (p) => {
    // vertices first, then endpoints, then bodies — smallest target wins.
    for (const z of body.zones) {
      for (let i = 0; i < z.points.length; i++) {
        const c = toC(z.points[i]);
        if (Math.hypot(c[0] - p[0], c[1] - p[1]) <= HIT_PX) return { kind: 'zone', id: z.id, part: i };
      }
    }
    for (const l of body.lines) {
      const s = toC(l.start), e = toC(l.end);
      if (Math.hypot(s[0] - p[0], s[1] - p[1]) <= HIT_PX) return { kind: 'line', id: l.id, part: 'start' };
      if (Math.hypot(e[0] - p[0], e[1] - p[1]) <= HIT_PX) return { kind: 'line', id: l.id, part: 'end' };
    }
    for (const l of body.lines) {
      if (distToSegment(p, toC(l.start), toC(l.end)) <= HIT_PX) return { kind: 'line', id: l.id, part: 'body' };
    }
    for (const z of body.zones) {
      if (z.points.length >= 3 && pointInPolygon(p, z.points.map(toC))) return { kind: 'zone', id: z.id, part: 'body' };
    }
    return null;
  };

  const finalizeZone = (pts) => {
    if (pts.length < 3) { setDraftZone([]); return; }
    const id = uid('z');
    commitBody((b) => {
      b.zones.push({ id, name: 'zone_' + (b.zones.length + 1), points: pts, dwell_s: DEFAULT_DWELL });
    });
    setDraftZone([]);
    setSel({ kind: 'zone', id });
    setTool('select');
  };

  const onPointerDown = (ev) => {
    if (ev.button !== 0) return;
    const p = eventToCanvas(ev, svgRef.current);
    if (tool === 'zone') {
      setDraftZone(draftZone.concat([canvasToNorm(p, fit)]));
      return;
    }
    if (tool === 'line') {
      const n = canvasToNorm(p, fit);
      if (!draftLine) { setDraftLine(n); return; }
      const id = uid('l');
      const startN = draftLine;
      commitBody((b) => {
        b.lines.push({ id, name: 'line_' + (b.lines.length + 1), start: startN, end: n, direction: 'any' });
      });
      setDraftLine(null);
      setSel({ kind: 'line', id });
      setTool('select');
      return;
    }
    const hit = hitTest(p);
    if (tool === 'delete') {
      if (hit) setPendingDelete(hit);
      return;
    }
    if (!hit) { setSel(null); return; }
    setSel({ kind: hit.kind, id: hit.id });
    const shape = hit.kind === 'zone'
      ? body.zones.find((z) => z.id === hit.id)
      : body.lines.find((l) => l.id === hit.id);
    drag.current = {
      hit,
      origin: canvasToNorm(p, fit),
      snapshot: JSON.parse(JSON.stringify(shape)),
    };
    if (svgRef.current.setPointerCapture) svgRef.current.setPointerCapture(ev.pointerId);
  };

  const onPointerMove = (ev) => {
    const p = eventToCanvas(ev, svgRef.current);
    const n = canvasToNorm(p, fit);
    setHoverNorm(n);
    const d = drag.current;
    if (!d) return;
    const dx = n[0] - d.origin[0];
    const dy = n[1] - d.origin[1];
    commitBody((b) => {
      if (d.hit.kind === 'zone') {
        const z = b.zones.find((x) => x.id === d.hit.id);
        if (!z) return;
        if (d.hit.part === 'body') {
          z.points = d.snapshot.points.map((q) => [clamp01(q[0] + dx), clamp01(q[1] + dy)]);
        } else {
          z.points = d.snapshot.points.slice();
          z.points[d.hit.part] = n;
        }
      } else {
        const l = b.lines.find((x) => x.id === d.hit.id);
        if (!l) return;
        if (d.hit.part === 'body') {
          l.start = [clamp01(d.snapshot.start[0] + dx), clamp01(d.snapshot.start[1] + dy)];
          l.end = [clamp01(d.snapshot.end[0] + dx), clamp01(d.snapshot.end[1] + dy)];
        } else {
          l[d.hit.part] = n;
        }
      }
    });
  };

  const onPointerUp = () => { drag.current = null; };

  const onDblClick = () => { if (tool === 'zone' && draftZone.length >= 3) finalizeZone(draftZone); };

  useEffect(() => {
    const onKey = (e) => {
      if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      if (e.key === 'Enter' && tool === 'zone' && draftZone.length >= 3) { finalizeZone(draftZone); }
      else if (e.key === 'Escape') { setDraftZone([]); setDraftLine(null); setSel(null); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && sel) { setPendingDelete({ kind: sel.kind, id: sel.id }); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tool, draftZone, sel, body]);

  const doDelete = (target) => {
    commitBody((b) => {
      if (target.kind === 'zone') b.zones = b.zones.filter((z) => z.id !== target.id);
      else b.lines = b.lines.filter((l) => l.id !== target.id);
    });
    setSel(null);
    setTool('select');
  };

  const delName = (() => {
    if (!pendingDelete) return '';
    const s = pendingDelete.kind === 'zone'
      ? body.zones.find((z) => z.id === pendingDelete.id)
      : body.lines.find((l) => l.id === pendingDelete.id);
    return (s && s.name) || pendingDelete.id;
  })();

  const toolBtn = (id, label, icon) => html`
    <button class=${'tool' + (tool === id ? ' tool-on' : '')} onClick=${() => { setTool(id); setDraftZone([]); setDraftLine(null); }}>
      ${icon} <span>${label}</span>
    </button>`;

  const hint = tool === 'zone' ? t('rules.tool.zoneHint')
    : tool === 'line' ? t('rules.tool.lineHint')
      : tool === 'delete' ? t('rules.tool.deleteHint') : '';

  const liveBoxes = showLive && live && Array.isArray(live.objects) ? live.objects : [];

  return html`
    <div class="canvas-col">
      <div class="toolbar">
        ${toolBtn('select', t('rules.tool.select'), Icon.cursor({ size: 14 }))}
        ${toolBtn('zone', t('rules.tool.zone'), Icon.zone({ size: 14 }))}
        ${toolBtn('line', t('rules.tool.line'), Icon.line({ size: 14 }))}
        ${toolBtn('delete', t('rules.tool.delete'), Icon.trash({ size: 14 }))}
        <span class="tool-hint">${hint}</span>
        <span class="frame-badge mono">${t('rules.canvas.frame', { w: frame.w, h: frame.h })}</span>
      </div>

      <div class=${'canvas-wrap tool-' + tool} ref=${wrapRef}>
        <svg ref=${svgRef} class="canvas" width=${size.w} height=${size.h}
             onPointerDown=${onPointerDown} onPointerMove=${onPointerMove}
             onPointerUp=${onPointerUp} onPointerLeave=${() => { onPointerUp(); setHoverNorm(null); }}
             onDblClick=${onDblClick}>
          <rect x="0" y="0" width=${size.w} height=${size.h} class="canvas-bg" />
          ${previewOk
            ? html`<image href=${previewSrc} x=${fit.offsetX} y=${fit.offsetY} width=${fit.dispW} height=${fit.dispH}
                          preserveAspectRatio="none" />`
            : html`<rect x=${fit.offsetX} y=${fit.offsetY} width=${fit.dispW} height=${fit.dispH} class="canvas-frame" />`}

          ${liveBoxes.map((o, i) => {
            const r = bboxToRect(o.bbox, fit);
            return html`<g key=${'lb' + i} class="livebox">
              <rect x=${r.x} y=${r.y} width=${r.w} height=${r.h} />
              <text x=${r.x + 2} y=${Math.max(10, r.y - 3)}>${(o.label || 'obj') + (o.track_id ? ' #' + o.track_id : '')}</text>
            </g>`;
          })}

          ${body.zones.map((z) => {
            const pts = z.points.map(toC);
            const on = sel && sel.kind === 'zone' && sel.id === z.id;
            const bad = issueIds.has(z.id);
            const c = polygonCentroid(pts);
            return html`
              <g key=${z.id} class=${'zone' + (on ? ' sel' : '') + (bad ? ' bad' : '')}>
                <polygon points=${pts.map((p) => p[0] + ',' + p[1]).join(' ')} />
                ${pts.length ? html`<text x=${c[0]} y=${c[1]}>${z.name}</text>` : null}
                ${on ? pts.map((p, i) => html`<circle key=${'v' + i} class="vtx" cx=${p[0]} cy=${p[1]} r="5" />`) : null}
              </g>`;
          })}

          ${body.lines.map((l) => {
            const a = toC(l.start), b = toC(l.end);
            const on = sel && sel.kind === 'line' && sel.id === l.id;
            const bad = issueIds.has(l.id);
            const m = midpoint(a, b);
            // Keep the label clear of the direction arrow: opposite side of the forward
            // normal, or pushed along the segment when both arrows are drawn.
            const nf = forwardNormal(a, b);
            const lab = l.direction === 'any'
              ? [m[0] - nf[1] * 42, m[1] + nf[0] * 42]
              : [m[0] - nf[0] * 26, m[1] - nf[1] * 26];
            return html`
              <g key=${l.id} class=${'line' + (on ? ' sel' : '') + (bad ? ' bad' : '')}>
                <line x1=${a[0]} y1=${a[1]} x2=${b[0]} y2=${b[1]} />
                <${ArrowGlyph} a=${a} b=${b} direction=${l.direction} />
                <text x=${lab[0]} y=${lab[1]} text-anchor="middle">${l.name}</text>
                ${on ? html`<g><circle class="vtx" cx=${a[0]} cy=${a[1]} r="5" /><circle class="vtx" cx=${b[0]} cy=${b[1]} r="5" /></g>` : null}
              </g>`;
          })}

          ${draftZone.length ? html`
            <g class="draft">
              <polyline points=${draftZone.map(toC).map((p) => p[0] + ',' + p[1]).join(' ')
                + (hoverNorm ? ' ' + toC(hoverNorm).join(',') : '')} />
              ${draftZone.map(toC).map((p, i) => html`<circle key=${'d' + i} cx=${p[0]} cy=${p[1]} r="4" />`)}
            </g>` : null}

          ${draftLine && hoverNorm ? html`
            <g class="draft">
              <line x1=${toC(draftLine)[0]} y1=${toC(draftLine)[1]} x2=${toC(hoverNorm)[0]} y2=${toC(hoverNorm)[1]} />
              <circle cx=${toC(draftLine)[0]} cy=${toC(draftLine)[1]} r="4" />
            </g>` : null}
        </svg>

        ${!previewOk ? html`
          <div class="canvas-note">
            ${sources.length ? t('rules.canvas.previewFail') : t('rules.canvas.noPreview', { w: frame.w, h: frame.h })}
          </div>` : null}
        ${hoverNorm ? html`<div class="canvas-coord mono">${hoverNorm[0].toFixed(3)}, ${hoverNorm[1].toFixed(3)}</div>` : null}
      </div>

      ${pendingDelete ? html`
        <${Confirm} danger=${true} title=${t('common.delete')}
          body=${t('rules.deleteConfirm', { name: delName })}
          confirmLabel=${t('common.delete')}
          onConfirm=${() => doDelete(pendingDelete)}
          onClose=${() => setPendingDelete(null)} />` : null}
    </div>`;
}

// ---- sidebar ----------------------------------------------------------------

function DirectionToggle({ value, onChange }) {
  const next = DIRECTIONS[(DIRECTIONS.indexOf(value) + 1) % DIRECTIONS.length];
  return html`
    <button class=${'dirtoggle dir-' + value} title=${t('rules.dir.' + value + 'Tip')} onClick=${() => onChange(next)}>
      ${Icon.arrows({ size: 12 })} ${t('rules.dir.' + value)}
    </button>`;
}

function RuleSidebar({ body, setBody, sel, setSel, issues, onSimulate, simBusy }) {
  const issueFor = (id) => issues.filter((i) => i.id === id);
  const rename = (kind, id, name) => {
    const next = JSON.parse(JSON.stringify(body));
    const arr = kind === 'zone' ? next.zones : next.lines;
    const item = arr.find((x) => x.id === id);
    if (item) item.name = name;
    setBody(next);
  };
  const patchLine = (id, patch) => {
    const next = JSON.parse(JSON.stringify(body));
    const l = next.lines.find((x) => x.id === id);
    if (l) Object.assign(l, patch);
    setBody(next);
  };
  const patchZone = (id, patch) => {
    const next = JSON.parse(JSON.stringify(body));
    const z = next.zones.find((x) => x.id === id);
    if (z) Object.assign(z, patch);
    setBody(next);
  };
  const del = (kind, id) => {
    const next = JSON.parse(JSON.stringify(body));
    if (kind === 'zone') next.zones = next.zones.filter((z) => z.id !== id);
    else next.lines = next.lines.filter((l) => l.id !== id);
    setBody(next);
    setSel(null);
  };

  const row = (kind, item, extra) => {
    const on = sel && sel.id === item.id;
    const errs = issueFor(item.id);
    return html`
      <li class=${'rule-row' + (on ? ' on' : '') + (errs.length ? ' bad' : '')} key=${item.id}
          onClick=${() => setSel({ kind, id: item.id })}>
        <div class="rule-row-head">
          ${kind === 'zone' ? Icon.zone({ size: 14 }) : Icon.line({ size: 14 })}
          <input class="name-input" value=${item.name}
                 onClick=${(e) => e.stopPropagation()}
                 onInput=${(e) => rename(kind, item.id, e.target.value)} />
          <button class="icon-btn" title=${t('common.delete')}
                  onClick=${(e) => { e.stopPropagation(); del(kind, item.id); }}>${Icon.trash({ size: 14 })}</button>
        </div>
        <div class="rule-row-body">${extra}</div>
        ${errs.map((e, i) => html`<div class="rule-row-err" key=${i}>${Icon.warn({ size: 12 })} ${e.message}</div>`)}
      </li>`;
  };

  return html`
    <aside class="rule-side">
      <h3>${t('rules.zones')} <span class="count">${body.zones.length}</span></h3>
      ${body.zones.length ? html`
        <ul class="rule-list">
          ${body.zones.map((z) => row('zone', z, html`
            <label class="inline" onClick=${(e) => e.stopPropagation()}>
              <span>${t('rules.dwell')}</span>
              <input class="num" type="number" min="0" step="1" value=${z.dwell_s}
                     onInput=${(e) => patchZone(z.id, { dwell_s: Number(e.target.value) })} />
              <span>${t('rules.dwellUnit')}</span>
              <i class="hint" title=${t('rules.dwellHint')}>?</i>
            </label>
            <span class="dim mono">${z.points.length} pts</span>`))}
        </ul>` : html`<p class="dim">${t('rules.noZones')}</p>`}

      <h3>${t('rules.lines')} <span class="count">${body.lines.length}</span></h3>
      ${body.lines.length ? html`
        <ul class="rule-list">
          ${body.lines.map((l) => row('line', l, html`
            <div class="inline" onClick=${(e) => e.stopPropagation()}>
              <span>${t('rules.direction')}</span>
              <${DirectionToggle} value=${l.direction} onChange=${(d) => patchLine(l.id, { direction: d })} />
            </div>`))}
        </ul>` : html`<p class="dim">${t('rules.noLines')}</p>`}

      <h3>${t('rules.cooldown')}</h3>
      <label class="inline">
        <input class="num" type="number" min="0" step="1" value=${body.cooldown}
               onInput=${(e) => { const n = JSON.parse(JSON.stringify(body)); n.cooldown = Number(e.target.value); setBody(n); }} />
        <span>${t('rules.dwellUnit')}</span>
        <i class="hint" title=${t('rules.cooldownHint')}>?</i>
      </label>

      <div class="side-foot">
        <button class="btn btn-block" disabled=${!sel || simBusy} onClick=${onSimulate}
                title=${sel ? t('rules.simulateHint') : t('rules.simulatePick')}>
          ${Icon.play({ size: 14 })} ${t('rules.simulate')}
        </button>
        <p class="dim small">${sel ? t('rules.simulateHint') : t('rules.simulatePick')}</p>
      </div>
    </aside>`;
}

// ---- page --------------------------------------------------------------------

export function RulesPage() {
  const st = useStore();
  const q = st.route.query;
  const [devices, setDevices] = useState(st.devices);
  const [body, setBody] = useState(emptyBody());
  const [saved, setSaved] = useState(null);       // last persisted body (JSON string)
  const [rev, setRev] = useState(null);
  const [status, setStatus] = useState('clean');  // clean|dirty|saving|saved|netfail|validfail
  const [savedAt, setSavedAt] = useState(null);
  const [issues, setIssues] = useState([]);
  const [tool, setTool] = useState('select');
  const [sel, setSel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [live, setLive] = useState(null);
  const [showLive, setShowLive] = useState(true);
  const [simBusy, setSimBusy] = useState(false);
  const [askRevert, setAskRevert] = useState(false);

  const deviceId = q.device_id || '';
  const streamId = q.stream_id || '';

  useEffect(() => {
    api.devices().then((res) => {
      const list = Array.isArray(res) ? res : (res && res.devices) || [];
      setDevices(list);
      commit({ devices: list });
    }).catch((e) => { if (e.status !== 401) toast(t('devices.loadFailed', { msg: errMsg(e) }), 'error'); });
  }, []);

  const dev = devices.find((d) => d.device_id === deviceId) || null;
  const sids = dev ? streamIds(dev) : [];
  const stream = findStream(dev, streamId);

  const frame = useMemo(() => {
    const f = (stream && stream.frame) || (live && live.frame) || null;
    if (f && f.w && f.h) return { w: Number(f.w), h: Number(f.h) };
    return FALLBACK_FRAME;
  }, [stream, live]);

  // Hub proxy first, the device-local URL as an opportunistic upgrade (§2.3).
  const previewSources = useMemo(() => {
    if (!deviceId || !streamId) return [];
    return [api.streamPreviewUrl(deviceId, streamId), (stream && stream.preview_url) || null].filter(Boolean);
  }, [deviceId, streamId, stream && stream.preview_url]);

  // Load rules for the selected stream.
  useEffect(() => {
    if (!deviceId || !streamId) { setBody(emptyBody()); setSaved(null); setStatus('clean'); return; }
    let dead = false;
    setLoading(true);
    setIssues([]);
    setSel(null);
    api.streamRules(deviceId, streamId).then((res) => {
      if (dead) return;
      const raw = (res && (res.body || res.rules)) || res || {};
      const nb = normalizeBody(raw);
      setBody(nb);
      setSaved(JSON.stringify(nb));
      setRev(res && res.rev != null ? res.rev : null);
      setSavedAt(res && res.updated_ms ? res.updated_ms : null);
      setStatus('clean');
      setLoading(false);
    }).catch((e) => {
      if (dead) return;
      setLoading(false);
      if (e.status === 404) { const nb = emptyBody(); setBody(nb); setSaved(JSON.stringify(nb)); setStatus('clean'); return; }
      if (e.status !== 401) toast(t('rules.loadFailed', { msg: errMsg(e) }), 'error');
    });
    api.live(deviceId, streamId).then((res) => { if (!dead) setLive(res); }).catch(() => { if (!dead) setLive(null); });
    return () => { dead = true; };
  }, [deviceId, streamId]);

  const dirty = saved !== null && JSON.stringify(body) !== saved;

  useEffect(() => {
    if (dirty && (status === 'clean' || status === 'saved')) setStatus('dirty');
  }, [dirty]);

  // §4.5 beforeunload guard.
  useEffect(() => {
    const h = (e) => { if (dirty) { e.preventDefault(); e.returnValue = t('rules.leaveConfirm'); return e.returnValue; } };
    window.addEventListener('beforeunload', h);
    return () => window.removeEventListener('beforeunload', h);
  }, [dirty]);

  const issueIds = useMemo(() => new Set(issues.map((i) => i.id).filter(Boolean)), [issues]);

  const pick = (patch) => setQuery(Object.assign({}, q, patch));

  const save = async () => {
    const local = localIssues(body);
    if (local.length) { setIssues(local); setStatus('validfail'); return; }
    setIssues([]);
    setStatus('saving');
    try {
      const res = await api.putStreamRules(deviceId, streamId, {
        zones: body.zones, lines: body.lines, features: body.features, cooldown: body.cooldown,
      });
      setSaved(JSON.stringify(body));
      setRev(res && res.rev != null ? res.rev : (rev == null ? 1 : rev + 1));
      setSavedAt(res && res.persisted_ms ? res.persisted_ms : Date.now());
      setStatus('saved');
    } catch (e) {
      if (e.status === 401) return;
      if (e.network || (e.status >= 500)) { setStatus('netfail'); return; }
      if (e.status === 400) {
        const list = validationIssues(e);
        // Map hub-reported target names back to local ids so the sidebar can flag them.
        const mapped = list.map((i) => {
          if (i.id) return i;
          const byName = body.zones.concat(body.lines).find((x) => i.name && x.name === i.name);
          return Object.assign({}, i, { id: byName ? byName.id : null });
        });
        setIssues(mapped);
        setStatus('validfail');
        return;
      }
      setStatus('netfail');
    }
  };

  const revert = () => {
    if (saved === null) return;
    setBody(JSON.parse(saved));
    setIssues([]);
    setSel(null);
    setStatus('clean');
  };

  const simulate = async () => {
    if (!sel) return;
    setSimBusy(true);
    try {
      await api.simulate(deviceId, streamId, sel.id);
      toast(t('rules.simulateOk'), 'ok', 5000);
    } catch (e) {
      if (e.status !== 401) toast(t('rules.simulateFail', { msg: errMsg(e) }), 'error', 6000);
    } finally {
      setSimBusy(false);
    }
  };

  const statusNode = (() => {
    if (status === 'saving') return html`<span class="save-status saving">${t('rules.status.saving')}</span>`;
    if (status === 'netfail') return html`<span class="save-status bad">${t('rules.status.netFail')}</span>`;
    if (status === 'validfail') return html`<span class="save-status bad">${t('rules.status.validFail')}</span>`;
    if (dirty) return html`<span class="save-status dirty">${t('rules.status.dirty')}</span>`;
    if (status === 'saved') {
      return html`<span class="save-status ok">${t('rules.status.saved', { time: clockTime(savedAt), rev: rev == null ? '?' : rev })}</span>`;
    }
    return html`<span class="save-status">${rev != null ? t('rules.status.saved', { time: savedAt ? clockTime(savedAt) : '—', rev }) : t('rules.status.clean')}</span>`;
  })();

  const ready = !!(deviceId && streamId);

  return html`
    <div class="rules-page">
      <div class="picker">
        <${Field} label=${t('rules.pickDevice')}>
          <${Select} value=${deviceId} onChange=${(v) => pick({ device_id: v, stream_id: '' })}
            options=${[{ value: '', label: '—' }].concat(devices.map((d) => ({ value: d.device_id, label: (d.name || d.device_id) + (d.online ? '' : ' (' + t('devices.offline') + ')') })))} />
        <//>
        <${Field} label=${t('rules.pickStream')}>
          <${Select} value=${streamId} onChange=${(v) => pick({ stream_id: v })}
            options=${[{ value: '', label: '—' }].concat(sids.map((s) => ({ value: s, label: s })))} />
        <//>
        <label class="inline live-toggle">
          <input type="checkbox" checked=${showLive} onChange=${(e) => setShowLive(e.target.checked)} />
          <span>${t('rules.canvas.overlay')}</span>
        </label>
        <div class="picker-right">
          ${statusNode}
          <button class="btn" disabled=${!dirty} onClick=${() => setAskRevert(true)}>${Icon.refresh({ size: 14 })} ${t('rules.revert')}</button>
          <button class=${'btn ' + (dirty ? 'btn-primary' : '')} disabled=${!ready || status === 'saving'} onClick=${save}>
            ${t('rules.save')}
          </button>
        </div>
      </div>

      ${!ready ? html`<div class="pad dim">${t('rules.pickHint')}</div>`
        : loading ? html`<${Spinner} />`
          : html`
            <div class="rules-body">
              <${RuleCanvas} body=${body} setBody=${setBody} frame=${frame} previewSources=${previewSources}
                live=${live} showLive=${showLive} tool=${tool} setTool=${setTool}
                sel=${sel} setSel=${setSel} issueIds=${issueIds} />
              <${RuleSidebar} body=${body} setBody=${setBody} sel=${sel} setSel=${setSel}
                issues=${issues} onSimulate=${simulate} simBusy=${simBusy} />
            </div>`}

      ${issues.length ? html`
        <div class="issue-strip">
          ${Icon.warn({ size: 14 })}
          <ul>${issues.map((i, k) => html`<li key=${k}>${i.message}</li>`)}</ul>
        </div>` : null}

      ${askRevert ? html`
        <${Confirm} danger=${true} title=${t('rules.revert')} body=${t('rules.revertConfirm')}
          confirmLabel=${t('rules.revert')} onConfirm=${revert} onClose=${() => setAskRevert(false)} />` : null}
    </div>`;
}
