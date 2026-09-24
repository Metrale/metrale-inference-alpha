// SPDX-License-Identifier: AGPL-3.0-only
//
// The palette is stated twice on purpose: once as data for the kit
// (assets/brand/tokens/brand.json, what the kit's generator writes and a
// designer reads) and once as CSS custom properties (web-shared/metrale-tokens.css,
// what the two sites read). This test keeps them equal, so a swatch change is
// one edit to brand.json and this test then names every token that has to
// follow.
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const brand = JSON.parse(read('../../../assets/brand/tokens/brand.json'));
const css = read('../../../web-shared/metrale-tokens.css');

// The value of a custom property inside one block of the stylesheet: the dark
// theme is `:root {...}`, the light theme is `[data-theme="light"] {...}`.
const block = (selector) => {
  const start = css.indexOf(selector);
  expect(start, `${selector} is in the token file`).toBeGreaterThan(-1);
  const open = css.indexOf('{', start);
  const close = css.indexOf('\n}', open);
  return css.slice(open, close);
};
const token = (selector, name) => {
  const m = block(selector).match(new RegExp(`--${name}:\\s*([^;]+);`));
  expect(m, `--${name} is set in ${selector}`).not.toBeNull();
  return m[1].trim().toUpperCase();
};
const hex = (s) => String(s).toUpperCase();

test('the two grounds in the CSS are the ones in brand.json', () => {
  expect(token(':root', 'bg')).toBe(hex(brand.color.ground.dark));
  expect(token('[data-theme="light"]', 'bg')).toBe(hex(brand.color.ground.light));
});

test('the logo inks in the CSS are the ones in brand.json, per ground', () => {
  const [hi, lo] = brand.color.ink.onDark;
  expect(token(':root', 'm-ink-hi')).toBe(hex(hi));
  expect(token(':root', 'm-ink-lo')).toBe(hex(lo));
  expect(token('[data-theme="light"]', 'm-ink-hi')).toBe(hex(brand.color.ink.onLight));
  expect(token('[data-theme="light"]', 'm-ink-lo')).toBe(hex(brand.color.ink.onLight));
  expect(token(':root', 'm-product')).toBe(hex(brand.color.product.onDark));
  expect(token('[data-theme="light"]', 'm-product')).toBe(hex(brand.color.product.onLight));
});

test('the three hues in the CSS are the ones in brand.json, both ends of each gradient', () => {
  expect(token(':root', 'm-lavender')).toBe(hex(brand.color.violet[0]));
  expect(token(':root', 'm-violet')).toBe(hex(brand.color.violet[1]));
  expect(token(':root', 'm-cyan-hi')).toBe(hex(brand.color.cyan[0]));
  expect(token(':root', 'm-cyan-lo')).toBe(hex(brand.color.cyan[1]));
  expect(token(':root', 'm-gold-hi')).toBe(hex(brand.color.gold[0]));
  expect(token(':root', 'm-gold-lo')).toBe(hex(brand.color.gold[1]));
  // The page's own grammar reads the same hues.
  expect(token(':root', 'ch-violet')).toBe(hex(brand.color.violet[1]));
  expect(token(':root', 'ch-cyan')).toBe(hex(brand.color.cyan[0]));
  expect(token(':root', 'ch-gold')).toBe(hex(brand.color.gold[0]));
  expect(token(':root', 'accent-fill')).toBe(hex(brand.color.violet[1]));
});

test("the text ramp on dark is the kit's inks and gray, and the type is the kit's family", () => {
  expect(token(':root', 't1')).toBe(hex(brand.color.ink.onDark[0]));
  expect(token(':root', 't2')).toBe(hex(brand.color.ink.onDark[1]));
  expect(token(':root', 't3')).toBe(hex(brand.color.product.onDark));
  expect(token('[data-theme="light"]', 't1')).toBe(hex(brand.color.ink.onLight));
  expect(block(':root {')).toMatch(new RegExp(`--font-sans: '${brand.type.family}'`));
});

test('brand.json carries every colour the CSS needs, so a new kit can be dropped in whole', () => {
  for (const path of [
    'color.ground.dark',
    'color.ground.light',
    'color.ink.onDark.0',
    'color.ink.onDark.1',
    'color.ink.onLight',
    'color.violet.0',
    'color.violet.1',
    'color.cyan.0',
    'color.cyan.1',
    'color.gold.0',
    'color.gold.1',
    'color.product.onDark',
    'color.product.onLight',
  ]) {
    const v = path.split('.').reduce((o, k) => o?.[k], brand);
    expect(v, path).toMatch(/^#[0-9A-Fa-f]{6}$/);
  }
  expect(brand.type.family).toBe('Urbanist');
});
