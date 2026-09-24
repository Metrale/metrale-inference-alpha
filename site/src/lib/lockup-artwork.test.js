// SPDX-License-Identifier: AGPL-3.0-only
import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

/**
 * The lockup component draws the brand from ../../../web-shared/brand-art.js,
 * a module written by scripts/brand/lockup.mjs from the kit's own geometry
 * (assets/brand/src/geometry.js and src/paths.json). Three things can drift:
 * the module behind the geometry, the component not drawing every piece, and
 * the tokens the inks read not being the kit's colours. This pins all three,
 * so a redraw is a failing test rather than a logo that is subtly not the logo.
 */
const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const require = createRequire(import.meta.url);
const { ART } = await import('../../../web-shared/brand-art.js');
const lockup = read('../../../web-shared/components/MetraleLockup.svelte');
const tokens = read('../../../web-shared/metrale-tokens.css');

test("brand-art.js is what the kit's geometry draws today", () => {
  const G = require('../../../assets/brand/src/geometry.js');
  G.setPaths(JSON.parse(read('../../../assets/brand/src/paths.json')));
  const w = G.wordmark({ theme: 'dark', id: 'w' });
  const m = G.mark({ theme: 'dark', id: 'i' });
  expect(ART.boxes.wordmark).toEqual({ x0: w.x0, y0: w.y0, width: w.width, height: w.height });
  expect(ART.boxes.mark).toEqual({ x0: m.x0, y0: m.y0, width: m.width, height: m.height });
  expect(ART.base).toBe(G.BASE);
  expect(ART.wordWidth).toBe(G.WORD_W);
  expect(ART.clear).toBe(G.CLEAR);
  // Every path the kit draws is in the module, verbatim.
  for (const d of [...w.body.matchAll(/\bd="([^"]+)"/g)].map((x) => x[1])) expect(ART.wordmark.some((p) => p.d === d)).toBe(true);
  for (const d of [...m.body.matchAll(/\bd="([^"]+)"/g)].map((x) => x[1])) expect(ART.mark.some((p) => p.d === d)).toBe(true);
  expect(ART.colors).toEqual(G.C);
});

test('the module is the shape the component expects: five pieces in the wordmark, four in the mark, three in the compact', () => {
  expect(ART.wordmark.map((p) => `${p.kind}:${p.ink}`)).toEqual(['path:ink', 'path:violet', 'rect:cyan', 'path:ink', 'path:gold']);
  expect(ART.mark.map((p) => `${p.kind}:${p.ink}`)).toEqual(['path:ink', 'path:violet', 'rect:cyan', 'path:gold']);
  expect(ART.compact.map((p) => `${p.kind}:${p.ink}`)).toEqual(['path:ink', 'path:violet', 'rect:cyan']);
  // The letters are stroked as the kit strokes them, so the outline matches the masters.
  expect(ART.wordmark[3].stroke).toBe(3.6);
});

test('the component draws from the module and nothing else, and maps each ink to a gradient of tokens', () => {
  expect(lockup).toContain("import { ART } from '../brand-art.js'");
  expect(lockup).not.toMatch(/\bd="M\d/); // no path literal of its own
  for (const id of ['m-ink', 'm-violet', 'm-cyan', 'm-gold', 'm-wordmark', 'm-mark', 'm-compact']) expect(lockup).toContain(`id="${id}"`);
  for (const t of ['--m-ink-hi', '--m-ink-lo', '--m-lavender', '--m-violet', '--m-cyan-hi', '--m-cyan-lo', '--m-gold-hi', '--m-gold-lo'])
    expect(lockup).toContain(`var(${t})`);
  // The kit's gradient directions, kept.
  expect(lockup).toMatch(/id="m-violet" x1="0" y1="0" x2="0.25" y2="1"/);
  expect(lockup).toMatch(/id="m-cyan" x1="0" y1="0" x2="1" y2="0"/);
  expect(lockup).toMatch(/id="m-gold" x1="0.8" y1="0" x2="0.2" y2="1"/);
});

test("the tokens the inks read are the kit's colours, per ground", () => {
  const block = (sel) => tokens.slice(tokens.indexOf(sel), tokens.indexOf('\n}', tokens.indexOf(sel)));
  const dark = block(':root {');
  const light = block('[data-theme="light"]');
  const C = ART.colors;
  expect(dark).toContain(`--m-ink-hi: ${C.inkHi};`);
  expect(dark).toContain(`--m-ink-lo: ${C.inkLo};`);
  expect(dark).toContain(`--m-lavender: ${C.lavender};`);
  expect(dark).toContain(`--m-violet: ${C.violet};`);
  expect(dark).toContain(`--m-cyan-hi: ${C.cyanHi};`);
  expect(dark).toContain(`--m-cyan-lo: ${C.cyanLo};`);
  expect(dark).toContain(`--m-gold-hi: ${C.goldHi};`);
  expect(dark).toContain(`--m-gold-lo: ${C.goldLo};`);
  expect(dark).toContain(`--m-product: ${C.subDark};`);
  expect(dark).toContain(`--bg: ${C.ground};`);
  expect(light).toContain(`--m-ink-hi: ${C.inkDark};`);
  expect(light).toContain(`--m-ink-lo: ${C.inkDark};`);
  expect(light).toContain(`--m-product: ${C.subLight};`);
  expect(light).toContain(`--bg: ${C.groundLight};`);
});

test('the plain wordmark masters exist and carry the same letters as the module', () => {
  for (const f of ['wordmark-ondark.svg', 'wordmark.svg', 'wordmark-mono-ondark.svg', 'wordmark-mono.svg']) {
    const svg = read(`../../../assets/brand/svg/${f}`);
    expect(svg).toContain(`d="${ART.wordmark[3].d}"`);
    expect(svg).toContain('aria-label="Metrale"');
  }
});

test("the book's menu-bar script carries the on-dark wordmark master verbatim", () => {
  // book/theme/metrale.js is written by scripts/brand/lockup.mjs with the
  // master inlined, because mdBook has no bundler. A kit change that reaches
  // the masters but not the book would leave the docs on an old logo.
  const master = read('../../../assets/brand/svg/wordmark-ondark.svg').trim();
  const book = read('../../../book/theme/metrale.js');
  expect(book).toContain(`var wordmark = ${JSON.stringify(master)};`);
  expect(book).toContain('Engine docs');
  const toml = read('../../../book/book.toml');
  expect(toml).toContain('additional-js = ["theme/metrale.js"]');
  expect(toml).toMatch(/additional-css = \[[^\]]*"theme\/css\/wordmark\.css"\]/);
});
