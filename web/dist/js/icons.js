// Inline SVG only — no icon font, no external sprite file (FRONTEND_SPEC §1).
import { html } from '../vendor/preact-htm.js';

const wrap = (body, extra = {}) => html`
  <svg class=${'ic ' + (extra.cls || '')} viewBox="0 0 24 24" width=${extra.size || 16} height=${extra.size || 16}
       fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
       aria-hidden="true">${body}</svg>`;

export const Icon = {
  check: (o) => wrap(html`<path d="M4 12.5l5 5L20 6.5" />`, o),
  x: (o) => wrap(html`<path d="M6 6l12 12M18 6L6 18" />`, o),
  play: (o) => wrap(html`<path d="M7 4l13 8-13 8z" />`, o),
  bell: (o) => wrap(html`<path d="M6 9a6 6 0 1112 0c0 5 2 6 2 6H4s2-1 2-6z" /><path d="M10 20a2 2 0 004 0" />`, o),
  bellOff: (o) => wrap(html`<path d="M6 9a6 6 0 019.5-4.9" /><path d="M18 12c0 3 2 3 2 3H7" /><path d="M3 3l18 18" />`, o),
  zone: (o) => wrap(html`<path d="M4 8l7-4 9 5-3 9-9 2z" />`, o),
  line: (o) => wrap(html`<path d="M4 20L20 4" /><circle cx="4" cy="20" r="2" /><circle cx="20" cy="4" r="2" />`, o),
  trash: (o) => wrap(html`<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />`, o),
  cursor: (o) => wrap(html`<path d="M5 3l14 8-6 1-2 6z" />`, o),
  download: (o) => wrap(html`<path d="M12 3v12M7 11l5 5 5-5M4 21h16" />`, o),
  upload: (o) => wrap(html`<path d="M12 21V9M7 13l5-5 5 5M4 3h16" />`, o),
  ext: (o) => wrap(html`<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6" />`, o),
  gear: (o) => wrap(html`<circle cx="12" cy="12" r="3.2" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />`, o),
  warn: (o) => wrap(html`<path d="M12 3l9 17H3z" /><path d="M12 9v5M12 17.5v.5" />`, o),
  undo: (o) => wrap(html`<path d="M4 10h9a5 5 0 010 10H8" /><path d="M8 5l-4 5 4 5" />`, o),
  up: (o) => wrap(html`<path d="M12 20V5M5 12l7-7 7 7" />`, o),
  clock: (o) => wrap(html`<circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />`, o),
  shield: (o) => wrap(html`<path d="M12 3l8 3.5V12c0 4.5-3.2 7.5-8 9.5-4.8-2-8-5-8-9.5V6.5z" />`, o),
  filter: (o) => wrap(html`<path d="M3 5h18l-7 8v6l-4-2v-4z" />`, o),
  refresh: (o) => wrap(html`<path d="M20 12a8 8 0 11-2.3-5.7" /><path d="M20 4v4h-4" />`, o),
  arrows: (o) => wrap(html`<path d="M8 5l-4 4 4 4M4 9h16M16 15l4 4-4 4" />`, o),
};

export const eventIcon = (type, o) =>
  type === 'line_cross' ? Icon.line(o) : type === 'loitering' ? Icon.clock(o) : Icon.zone(o);
