#!/usr/bin/env node
/* Verifies zh/en dictionary parity and that every t('key') used in the app exists.
   Run: node web/tools/check-i18n.js  */
'use strict';
const fs = require('fs');
const path = require('path');

const DIST = path.join(__dirname, '..', 'dist');
const src = fs.readFileSync(path.join(DIST, 'js', 'i18n.js'), 'utf8');

function keysOf(lang) {
  const start = src.indexOf('  ' + lang + ': {');
  if (start < 0) throw new Error('dict not found: ' + lang);
  let depth = 0;
  let i = src.indexOf('{', start);
  const from = i;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') { depth--; if (!depth) break; }
  }
  const block = src.slice(from, i);
  const keys = new Set();
  const re = /^\s*'([^']+)':/gm;
  let m;
  while ((m = re.exec(block))) keys.add(m[1]);
  return keys;
}

function walk(dir, out) {
  fs.readdirSync(dir).forEach((f) => {
    const p = path.join(dir, f);
    const st = fs.statSync(p);
    if (st.isDirectory()) { if (f !== 'vendor') walk(p, out); }
    else if (f.endsWith('.js')) out.push(p);
  });
  return out;
}

const zh = keysOf('zh');
const en = keysOf('en');
const files = walk(path.join(DIST, 'js'), []);

const used = new Map();
const dynamic = [];
files.forEach((f) => {
  const txt = fs.readFileSync(f, 'utf8');
  // Literal keys only: t('a.b'). A trailing '+' means it is a dynamic prefix, handled below.
  const re = /\bt\(\s*'([^']+)'(?!\s*\+)/g;
  let m;
  while ((m = re.exec(txt))) used.set(m[1], (used.get(m[1]) || []).concat(path.relative(DIST, f)));
  const re2 = /\bt\(\s*'([^']*)'\s*\+/g;
  while ((m = re2.exec(txt))) dynamic.push({ prefix: m[1], file: path.relative(DIST, f) });
});

let bad = 0;
const missZh = Array.from(used.keys()).filter((k) => !zh.has(k));
const missEn = Array.from(used.keys()).filter((k) => !en.has(k));
const onlyZh = Array.from(zh).filter((k) => !en.has(k));
const onlyEn = Array.from(en).filter((k) => !zh.has(k));

// Dynamic keys like t('alerts.state.' + s) are checked by prefix coverage.
const prefixes = Array.from(new Set(dynamic.map((d) => d.prefix)));
const prefixHits = prefixes.map((p) => ({ p, n: Array.from(zh).filter((k) => k.startsWith(p)).length }));

console.log('zh keys:', zh.size, ' en keys:', en.size, ' literal t() keys used:', used.size);
if (missZh.length) { bad = 1; console.log('MISSING in zh:', missZh); }
if (missEn.length) { bad = 1; console.log('MISSING in en:', missEn); }
if (onlyZh.length) { bad = 1; console.log('zh-only keys:', onlyZh); }
if (onlyEn.length) { bad = 1; console.log('en-only keys:', onlyEn); }
prefixHits.forEach((h) => {
  const line = 'dynamic prefix ' + JSON.stringify(h.p) + ' -> ' + h.n + ' zh keys';
  if (h.p && h.n === 0) { bad = 1; console.log('UNCOVERED ' + line); } else console.log('  ' + line);
});
const unused = Array.from(zh).filter((k) => !used.has(k) && !prefixes.some((p) => p && k.startsWith(p)));
if (unused.length) console.log('note: keys never referenced literally:', unused.length, unused.slice(0, 40));
console.log(bad ? 'FAIL' : 'OK');
process.exit(bad);
