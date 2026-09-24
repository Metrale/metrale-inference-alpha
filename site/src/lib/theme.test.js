// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { THEME_DARK_BG, THEME_KEY, THEME_LIGHT_BG, readTheme } from '../../../web-shared/theme.js';

test('ground colours are the brand pair, not a third canvas', () => {
  expect(THEME_DARK_BG).toBe('#0E1318');
  expect(THEME_LIGHT_BG).toBe('#FFFFFF');
  expect(THEME_KEY).toBe('metrale-theme');
});

test('without a document, readTheme reports dark rather than throwing', () => {
  expect(readTheme()).toBe('dark');
});

/**
 * The blocking boot script in app.html settles data-theme before first paint.
 * The home page's hero is the kit's mark, drawn inline from the shared vector
 * definitions, so the script must not fetch anything, on any page.
 *
 * There is no DOM here, so the script runs against the handful of globals it
 * actually touches. That keeps the test honest about the source in app.html
 * rather than restating it somewhere a copy could drift.
 */
function runBootScript({ stored = null, prefersLight = false, pathname = '/', storageThrows = false } = {}) {
  const html = readFileSync(new URL('../app.html', import.meta.url), 'utf8');
  const source = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
  expect(source, 'no inline boot script found in app.html').toBeTruthy();

  const appended = [];
  const documentElement = {
    attributes: {},
    setAttribute(name, value) {
      this.attributes[name] = value;
    }
  };
  const document = {
    documentElement,
    head: {
      appendChild(node) {
        appended.push(node);
      }
    },
    createElement(tag) {
      return {
        tag,
        setAttribute(name, value) {
          this[name] = value;
        }
      };
    }
  };
  const localStorage = {
    getItem() {
      if (storageThrows) throw new Error('storage unavailable');
      return stored;
    }
  };
  const matchMedia = query => ({ matches: query.includes('light') ? prefersLight : !prefersLight });

  new Function('document', 'localStorage', 'matchMedia', 'location', source)(
    document,
    localStorage,
    matchMedia,
    { pathname }
  );
  return { theme: documentElement.attributes['data-theme'], preloads: appended };
}

test('a stored theme wins over the system preference', () => {
  expect(runBootScript({ stored: 'dark', prefersLight: true }).theme).toBe('dark');
  expect(runBootScript({ stored: 'light', prefersLight: false }).theme).toBe('light');
});

test('an unset or unknown preference follows the media query', () => {
  expect(runBootScript({ prefersLight: true }).theme).toBe('light');
  expect(runBootScript({ prefersLight: false }).theme).toBe('dark');
  expect(runBootScript({ stored: 'sepia', prefersLight: true }).theme).toBe('light');
});

test('when storage is unreadable the page still falls to dark', () => {
  expect(runBootScript({ storageThrows: true, prefersLight: true }).theme).toBe('dark');
});

test('the boot script fetches nothing: no hero image is queued on any page', () => {
  for (const pathname of ['/', '/index.html', '/control']) {
    for (const stored of ['dark', 'light', null]) {
      expect(runBootScript({ stored, pathname }).preloads).toHaveLength(0);
    }
  }
  const html = readFileSync(new URL('../app.html', import.meta.url), 'utf8');
  expect(html).not.toMatch(/rel="preload"[^>]*as="image"/);
});
