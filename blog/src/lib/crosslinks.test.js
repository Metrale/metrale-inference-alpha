// SPDX-License-Identifier: AGPL-3.0-only
import { test, expect } from 'bun:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const script = fileURLToPath(new URL('../../e2e/check-crosslinks.mjs', import.meta.url));

function check(sitePages, blogHtml) {
  const root = mkdtempSync(join(tmpdir(), 'metrale-crosslinks-'));
  const site = join(root, 'site');
  const blog = join(root, 'blog');
  try {
    mkdirSync(site);
    mkdirSync(blog);
    for (const [name, html] of Object.entries(sitePages)) writeFileSync(join(site, name), html);
    writeFileSync(join(blog, 'index.html'), blogHtml);
    return spawnSync(process.execPath, [script, site, blog], { encoding: 'utf8' });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test('a fragment on the control page cannot satisfy a homepage link', () => {
  const result = check(
    { 'index.html': '<main id="home"></main>', 'control.html': '<section id="verified"></section>' },
    '<a href="https://dev.metrale.ai/#verified">Benchmarks</a>'
  );
  expect(result.status).toBe(1);
  expect(result.stderr).toContain('https://dev.metrale.ai/#verified');
});

test('homepage fragments resolve in built markup, including no-JS links', () => {
  const result = check(
    { 'index.html': '<section id="verified"></section><noscript><a id="run" href="/control.html#run">Start Metrale Engine</a></noscript>' },
    '<a href="https://dev.metrale.ai/#verified">Benchmarks</a><a href="https://dev.metrale.ai/index.html#run">Start Metrale Engine</a>'
  );
  expect(result.status).toBe(0);
  expect(result.stdout).toContain('all 2 cross-property fragment links resolve');
});

test('explicit control links resolve against their own built page', () => {
  const result = check(
    { 'index.html': '<main id="home"></main>', 'control.html': '<section id="verified"></section>' },
    '<a href="https://dev.metrale.ai/control.html?source=blog#verified">Benchmarks</a>'
  );
  expect(result.status).toBe(0);
  expect(result.stdout).toContain('all 1 cross-property fragment links resolve');
});

test('URL-encoded fragments match the browser destination', () => {
  const result = check(
    { 'index.html': '<section id="model-detail"></section>' },
    '<a href="https://dev.metrale.ai/#model%2Ddetail">Models</a>'
  );
  expect(result.status).toBe(0);
});

test('extensionless /control resolves against control.html ids', () => {
  const result = check(
    { 'index.html': '<main id="home"></main>', 'control.html': '<section id="verified"></section>' },
    '<a href="https://dev.metrale.ai/control#verified">Benchmarks</a>'
  );
  expect(result.status).toBe(0);
  expect(result.stdout).toContain('all 1 cross-property fragment links resolve');
});

test('a missing fragment on /control still fails after the pretty-URL alias', () => {
  const result = check(
    { 'index.html': '<main id="home"></main>', 'control.html': '<section id="verified"></section>' },
    '<a href="https://dev.metrale.ai/control#nope">Benchmarks</a>'
  );
  expect(result.status).toBe(1);
  expect(result.stderr).toContain('https://dev.metrale.ai/control#nope');
});

test('removing every cross-property fragment link still fails the guard', () => {
  const result = check(
    { 'index.html': '<main id="home"></main>' },
    '<a href="https://dev.metrale.ai">Metrale Engine</a><a href="/#verified">Local section</a>'
  );
  expect(result.status).toBe(1);
  expect(result.stderr).toContain('No https://dev.metrale.ai');
});
