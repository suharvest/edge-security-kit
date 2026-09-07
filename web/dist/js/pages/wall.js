// Video wall (FRONTEND_SPEC §9): an adaptive grid of live tiles with the
// detector's boxes and the stream's rule shapes drawn over each one.
//
// No video passes through the hub. Each tile embeds the device's own live_url
// (go2rtc WebRTC where a site runs one, the detector's refreshing still where
// it does not) and the browser fetches it directly; the hub supplies only the
// overlay JSON and, for a tile whose device is unreachable from the browser,
// the cached single-frame proxy. HUB_SPEC §11 has the reasoning.
import { html, useState, useEffect, useRef, useCallback } from '../../vendor/preact-htm.js';
import { t } from '../i18n.js';
import { useStore, commit, toast } from '../store.js';
import { api } from '../api.js';
import { Icon } from '../icons.js';
import { Modal, Field, Empty } from '../components/ui.js';
import {
  LAYOUT_TILES, autoLayout, gridStyle, tilesFor, wallSummary, indexLive,
  overlayBoxes, overlayShapes, clampConf, CONF_MIN, CONF_MAX, CONF_STEP,
  maskSource, validateSource, validateStreamId, suggestStreamId,
} from '../wall.js';

//: Overlay poll rate. Detectors publish at 5 fps and the boxes are a position
//: indicator, not the picture; 500 ms is one request per second per wall
//: regardless of how many tiles are on it.
const OVERLAY_INTERVAL_MS = 500;
const LAYOUT_KEY = 'wall.tiles';

function readLayout() {
  const stored = Number(localStorage.getItem(LAYOUT_KEY));
  return LAYOUT_TILES.includes(stored) ? stored : 0; // 0 = auto
}

// ---------------------------------------------------------------- tile ------

function Tile({ tile, live, rules, now, onConf, onRemove, big }) {
  const boxes = overlayBoxes(live, now);
  const { zones, lines } = overlayShapes(rules);
  const [failed, setFailed] = useState(false);

  // An offline tile falls back to the hub's cached still: the last thing the
  // camera actually saw beats a black rectangle, as long as it is labelled.
  const src = tile.online && tile.live_url && !failed
    ? tile.live_url
    : api.streamPreviewUrl(tile.device_id, tile.stream_id);

  return html`
    <div class=${'tile' + (tile.online ? '' : ' tile-down')} key=${tile.key}>
      <div class="tile-frame">
       <div class="tile-media" style=${tile.aspect ? `aspect-ratio:${tile.aspect}` : ''}>
        ${tile.online && tile.live_url && !failed
          ? html`<iframe class="tile-live" src=${src} title=${tile.name}
                         loading="lazy" onError=${() => setFailed(true)}></iframe>`
          : html`<img class="tile-still" src=${src} alt=${tile.name}
                      onError=${(e) => { e.target.style.visibility = 'hidden'; }} />`}
        <svg class="tile-overlay" viewBox="0 0 100 100" preserveAspectRatio="none"
             aria-hidden="true">
          ${zones.map((z) => html`
            <polygon key=${z.id} class="ov-zone"
                     points=${z.points.map((p) => p.join(',')).join(' ')} />`)}
          ${lines.map((l) => html`
            <line key=${l.id} class="ov-line" x1=${l.x1} y1=${l.y1} x2=${l.x2} y2=${l.y2} />`)}
          ${boxes.map((b) => html`
            <rect key=${b.track_id} class="ov-box"
                  x=${b.left} y=${b.top} width=${b.width} height=${b.height} />`)}
        </svg>
        <div class="tile-tags">
          ${boxes.map((b) => html`
            <span class="ov-tag" key=${b.track_id}
                  style=${`left:${b.left}%;top:${b.top}%`}>
              #${b.track_id} · ${b.score.toFixed(2)}
            </span>`)}
        </div>
       </div>
        ${!tile.online ? html`
          <div class="tile-veil">
            <span>${Icon.warn({ size: 16 })} ${t('wall.state.' + tile.state)}</span>
            <small>${t('wall.lastSnapshot')}</small>
          </div>` : null}
      </div>
      <footer class="tile-bar">
        <span class="tile-name" title=${tile.device_id + ' / ' + tile.stream_id}>
          <i class=${'dot ' + (tile.online ? 'dot-ok' : 'dot-bad')}></i>${tile.name}
        </span>
        <span class="tile-meta">
          ${tile.fps.toFixed(1)} fps${tile.decode === 'sw' ? ' · ' + t('wall.swDecode') : ''}
        </span>
        ${big && tile.conf_threshold !== null ? html`
          <${ConfSlider} tile=${tile} onConf=${onConf} />` : null}
        ${onRemove ? html`
          <button class="icon-btn" title=${t('wall.removeCamera')}
                  aria-label=${t('wall.removeCamera')}
                  onClick=${() => onRemove(tile)}>${Icon.trash({ size: 14 })}</button>` : null}
      </footer>
    </div>`;
}

// ------------------------------------------------------- confidence slider ---

export function ConfSlider({ tile, onConf }) {
  // Local while dragging, device value otherwise: showing the dragged value
  // after a failed save would leave the operator believing the site changed.
  const [dragging, setDragging] = useState(null);
  const [busy, setBusy] = useState(false);
  const value = dragging !== null ? dragging : tile.conf_threshold;

  const commitValue = async (raw) => {
    const next = clampConf(raw);
    setDragging(null);
    if (next === null || next === tile.conf_threshold) return;
    setBusy(true);
    try {
      await onConf(tile, next);
    } finally {
      setBusy(false);
    }
  };

  return html`
    <label class=${'conf' + (busy ? ' conf-busy' : '')}
           title=${t('wall.confHint')}>
      <span class="conf-label">${t('wall.conf')}</span>
      <input type="range" min=${CONF_MIN} max=${CONF_MAX} step=${CONF_STEP}
             value=${value} disabled=${busy}
             aria-label=${t('wall.confAria', { name: tile.name })}
             onInput=${(e) => setDragging(Number(e.target.value))}
             onChange=${(e) => commitValue(e.target.value)} />
      <output class="conf-value">${Number(value).toFixed(2)}</output>
    </label>`;
}

// ------------------------------------------------------ add-camera dialog ----

export function AddCameraDialog({ devices, onClose, onAdded }) {
  const targets = (devices || []).filter((d) => d.online);
  const [deviceId, setDeviceId] = useState(targets.length ? targets[0].device_id : '');
  const device = (devices || []).find((d) => d.device_id === deviceId);
  const existing = ((device && device.streams) || []).map((s) => s.stream_id);
  const [streamId, setStreamId] = useState(suggestStreamId(existing));
  const [source, setSource] = useState('');
  const [name, setName] = useState('');
  const [transport, setTransport] = useState('tcp');
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState(null);

  const idError = validateStreamId(streamId, existing);
  const sourceError = validateSource(source);
  const ready = !idError && !sourceError && deviceId && !busy;

  const submit = async () => {
    setBusy(true);
    setFailure(null);
    try {
      const body = { stream_id: streamId.trim(), source: source.trim(),
                     rtsp_transport: transport };
      if (name.trim()) body.name = name.trim();
      const res = await api.addStream(deviceId, body);
      toast(t('wall.addOk', { name: name.trim() || streamId }), 'ok');
      onAdded(res);
      onClose();
    } catch (e) {
      // The three outcomes read differently on purpose. A 504 says nothing
      // about the device, so the dialog stays open rather than claiming either
      // result (HUB_SPEC §4).
      setFailure(e.status === 504 ? t('wall.addTimeout')
        : e.status === 409 ? t('wall.addRefused', { reason: e.message })
          : e.message);
    } finally {
      setBusy(false);
    }
  };

  return html`
    <${Modal} title=${t('wall.addCamera')} onClose=${onClose} footer=${html`
      <button class="btn" onClick=${onClose}>${t('common.cancel')}</button>
      <button class="btn btn-primary" disabled=${!ready} onClick=${submit}>
        ${busy ? t('wall.adding') : t('wall.addConfirm')}
      </button>`}>
      ${!targets.length ? html`<p class="err-text">${t('wall.addNoDevice')}</p>` : null}
      <${Field} label=${t('wall.addDevice')} hint=${t('wall.addDeviceHint')}>
        <select value=${deviceId} onChange=${(e) => {
          setDeviceId(e.target.value);
          const next = (devices || []).find((d) => d.device_id === e.target.value);
          setStreamId(suggestStreamId(((next && next.streams) || []).map((s) => s.stream_id)));
        }}>
          ${targets.map((d) => html`
            <option value=${d.device_id} key=${d.device_id}>${d.device_id}</option>`)}
        </select>
      <//>
      <${Field} label=${t('wall.addSource')} hint=${t('wall.addSourceHint')}
                error=${source && sourceError ? t(sourceError) : null}>
        <input value=${source} placeholder="rtsp://admin:••••@192.168.1.64:554/Streaming/Channels/101"
               onInput=${(e) => setSource(e.target.value)} />
      <//>
      ${source && !sourceError ? html`
        <p class="hint-line">${t('wall.addMasked')} <code>${maskSource(source)}</code></p>` : null}
      <${Field} label=${t('wall.addName')} hint=${t('wall.addNameHint')}>
        <input value=${name} placeholder=${t('wall.addNamePlaceholder')}
               onInput=${(e) => setName(e.target.value)} />
      <//>
      <${Field} label=${t('wall.addStreamId')} hint=${t('wall.addStreamIdHint')}
                error=${idError ? t(idError) : null}>
        <input value=${streamId} onInput=${(e) => setStreamId(e.target.value)} />
      <//>
      <${Field} label=${t('wall.addTransport')} hint=${t('wall.addTransportHint')}>
        <select value=${transport} onChange=${(e) => setTransport(e.target.value)}>
          <option value="tcp">TCP</option>
          <option value="udp">UDP</option>
        </select>
      <//>
      ${failure ? html`<p class="err-text">${failure}</p>` : null}
    <//>`;
}

// ---------------------------------------------------------------- page ------

export function WallPage() {
  const st = useStore();
  const [chosen, setChosen] = useState(readLayout());
  const [live, setLive] = useState(new Map());
  const [rules, setRules] = useState({});
  const [now, setNow] = useState(Date.now());
  const [adding, setAdding] = useState(false);
  const [full, setFull] = useState(false);
  const rootRef = useRef(null);

  const devices = st.devices || [];
  const summary = wallSummary(devices);
  const tiles = chosen || autoLayout(summary.streams);
  const { visible, total } = tilesFor(devices, tiles);

  // Overlay poll. One request per tick for the whole wall, so the cost is flat
  // in the number of tiles (HUB_SPEC §4, GET /live).
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const payload = await api.liveAll();
        if (!alive) return;
        setLive(indexLive(payload));
        setNow(payload.now_ms || Date.now());
      } catch (e) { /* a dropped poll is the next tick's problem */ }
    };
    tick();
    const timer = setInterval(tick, OVERLAY_INTERVAL_MS);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  // Rule shapes change only when someone saves them, so they are fetched once
  // rather than on the overlay tick.
  useEffect(() => {
    api.rules().then((res) => setRules(res.rules || res || {})).catch(() => {});
  }, [st.devices.length]);

  // Fullscreen for an HDMI screen: F toggles, Esc leaves. The listener is on
  // window rather than the container so the key works before anything is
  // focused -- an operator walking up to a wall-mounted screen has not clicked.
  const toggleFull = useCallback(() => {
    const node = rootRef.current;
    if (!document.fullscreenElement) {
      setFull(true);
      if (node && node.requestFullscreen) node.requestFullscreen().catch(() => {});
    } else {
      setFull(false);
      if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target && e.target.tagName) || '';
      // Never steal the key from a field the operator is typing in.
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFull(); }
      if (e.key === 'Escape' && full && !document.fullscreenElement) setFull(false);
    };
    const onFsChange = () => setFull(!!document.fullscreenElement);
    window.addEventListener('keydown', onKey);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('fullscreenchange', onFsChange);
    };
  }, [toggleFull, full]);

  const refreshDevices = async () => {
    try {
      const res = await api.devices();
      commit({ devices: Array.isArray(res) ? res : (res && res.devices) || [] });
    } catch (e) { /* the WS device.status push is the other path */ }
  };

  const onConf = async (tile, value) => {
    try {
      const res = await api.setConfThreshold(tile.device_id, tile.stream_id, value);
      // Render what the detector applied, not what was asked: they differ when
      // a platform clamps or rounds, and the slider must show the site's value.
      const applied = (res && res.applied && res.applied.conf_threshold) ?? value;
      toast(t('wall.confOk', { name: tile.name, value: Number(applied).toFixed(2) }), 'ok');
      await refreshDevices();
    } catch (e) {
      toast(e.status === 504 ? t('wall.confTimeout', { name: tile.name })
        : t('wall.confFailed', { name: tile.name, reason: e.message }), 'error');
      await refreshDevices();
    }
  };

  const onRemove = async (tile) => {
    if (!window.confirm(t('wall.removeConfirm', { name: tile.name }))) return;
    try {
      await api.removeStream(tile.device_id, tile.stream_id);
      toast(t('wall.removeOk', { name: tile.name }), 'ok');
    } catch (e) {
      toast(e.status === 504 ? t('wall.removeTimeout', { name: tile.name }) : e.message, 'error');
    }
    await refreshDevices();
  };

  const pickLayout = (value) => {
    setChosen(value);
    if (value) localStorage.setItem(LAYOUT_KEY, String(value));
    else localStorage.removeItem(LAYOUT_KEY);
  };

  return html`
    <div class=${'wall' + (full ? ' wall-full' : '')} ref=${rootRef}>
      <div class="wall-head">
        <div class="wall-summary">
          <span class="sum-big">${summary.live}<small>/${summary.streams}</small></span>
          <span class="sum-label">${t('wall.liveOf')}</span>
          ${summary.down ? html`
            <span class="pill pill-bad">${t('wall.downCount', { n: summary.down })}</span>` : null}
          ${summary.sw_decode ? html`
            <span class="pill pill-warn">${t('wall.swCount', { n: summary.sw_decode })}</span>` : null}
          <span class="pill">${t('wall.fpsTotal', { n: summary.fps.toFixed(1) })}</span>
          ${total > tiles ? html`
            <span class="pill pill-warn">${t('wall.hidden', { n: total - tiles })}</span>` : null}
        </div>
        <div class="wall-actions">
          <div class="seg" role="group" aria-label=${t('wall.layout')}>
            <button class=${chosen === 0 ? 'on' : ''} onClick=${() => pickLayout(0)}
                    title=${t('wall.layoutAutoHint')}>${t('wall.layoutAuto')}</button>
            ${LAYOUT_TILES.map((n) => html`
              <button key=${n} class=${chosen === n ? 'on' : ''}
                      onClick=${() => pickLayout(n)}>${n}</button>`)}
          </div>
          <button class="btn" onClick=${() => setAdding(true)}>
            ${Icon.play({ size: 14 })} ${t('wall.addCamera')}
          </button>
          <button class="btn btn-primary" onClick=${toggleFull} title=${t('wall.fullHint')}>
            ${Icon.arrows({ size: 14 })} ${full ? t('wall.exitFull') : t('wall.enterFull')}
            <kbd>F</kbd>
          </button>
        </div>
      </div>
      ${!visible.length
        ? html`<${Empty} text=${t('wall.empty')} />`
        : html`
          <div class="wall-grid" style=${gridStyle(tiles)}>
            ${visible.map((tile) => html`
              <${Tile} key=${tile.key} tile=${tile} big=${tiles <= 4}
                       live=${live.get(tile.key)} now=${now}
                       rules=${(rules[tile.device_id] || {})[tile.stream_id]
                         && (rules[tile.device_id][tile.stream_id].body
                           || rules[tile.device_id][tile.stream_id])}
                       onConf=${onConf} onRemove=${full ? null : onRemove} />`)}
          </div>`}
      ${adding ? html`
        <${AddCameraDialog} devices=${devices} onClose=${() => setAdding(false)}
                            onAdded=${refreshDevices} />` : null}
    </div>`;
}
