// Alert sound, FRONTEND_SPEC §3.1.
// One packaged local asset (assets/alert.mp3, 2.5 KB). Autoplay policy: the
// AudioContext stays suspended until the user's first click anywhere; until then the
// workbench shows a one-shot "click anywhere to enable alert sound" hint.
import { commit } from './store.js';

const SRC = '/assets/alert.mp3';

let ctx = null;
let buffer = null;
let unlocked = false;
const listeners = new Set();

export function isSoundOn() { return localStorage.getItem('sound') !== 'off'; }
export function setSoundOn(on) {
  localStorage.setItem('sound', on ? 'on' : 'off');
  commit();
}
export function isUnlocked() { return unlocked; }
export function onUnlock(fn) { listeners.add(fn); return () => listeners.delete(fn); }

async function load() {
  if (buffer || !ctx) return;
  const res = await fetch(SRC);
  const bytes = await res.arrayBuffer();
  buffer = await ctx.decodeAudioData(bytes);
}

export function installUnlockHandler() {
  if (unlocked) return;
  const handler = async () => {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      ctx = ctx || new AC();
      if (ctx.state === 'suspended') await ctx.resume();
      await load();
      unlocked = true;
      listeners.forEach((fn) => fn());
      commit();
      window.removeEventListener('pointerdown', handler, true);
      window.removeEventListener('keydown', handler, true);
    } catch (e) { /* keep the hint up; next click retries */ }
  };
  window.addEventListener('pointerdown', handler, true);
  window.addEventListener('keydown', handler, true);
}

export function playAlert() {
  if (!isSoundOn() || !unlocked || !ctx || !buffer) return;
  try {
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    const gain = ctx.createGain();
    gain.gain.value = 0.7;
    src.connect(gain).connect(ctx.destination);
    src.start();
  } catch (e) { /* non-fatal */ }
}
