// Metrale brand asset generator.
// Rebuilds every SVG, PNG and token file in this folder from src/geometry.js,
// src/paths.json and src/products.json.
//
//   npm i playwright opentype.js @fontsource/urbanist
//   npx playwright install chromium
//   node gen.js
//
// Adding a product: add  "slug": "Display Name"  to src/products.json and rerun.
// "ai" is special: its lockups use the drawn AI glyph in geometry.js instead of type.
// The product name is cut from Urbanist Regular into src/paths.json, then every
// product asset (-slug files) is rendered.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
const G = require('./src/geometry.js');

const OUT = __dirname;
const P = JSON.parse(fs.readFileSync(path.join(OUT, 'src/paths.json'), 'utf8'));
const PRODUCTS = JSON.parse(fs.readFileSync(path.join(OUT, 'src/products.json'), 'utf8'));
const C = G.C;

// ------------------------------------------------ product names -> outlines
function cutProducts() {
  let opentype;
  try { opentype = require('opentype.js'); } catch { console.warn('opentype.js not installed; using stored product outlines'); return; }
  const fontFile = path.join(OUT, 'node_modules/@fontsource/urbanist/files/urbanist-latin-400-normal.woff');
  if (!fs.existsSync(fontFile)) { console.warn('Urbanist not installed; using stored product outlines'); return; }
  const font = opentype.parse(fs.readFileSync(fontFile).buffer);
  P.products = P.products || {};
  for (const [slug, name] of Object.entries(PRODUCTS)) {
    const st = font.getPath(name, 0, 0, G.PRODUCT.size, { letterSpacing: 0.04 });
    const inl = font.getPath(name, 0, 0, G.PRODUCT.inlineSize, { letterSpacing: 0.01 });
    P.products[slug] = { name, stacked: st.toPathData(2), stackedWidth: +st.getBoundingBox().x2.toFixed(1),
      inline: inl.toPathData(2), inlineWidth: +inl.getBoundingBox().x2.toFixed(1) };
  }
  fs.writeFileSync(path.join(OUT, 'src/paths.json'), JSON.stringify(P, null, 1));
}

// ---------------------------------------------------------------- tokens
function tokenFiles() {
  const json = {
    color: { ground: { dark: C.ground, light: C.groundLight }, ink: { onDark: [C.inkHi, C.inkLo], onLight: C.inkDark },
      violet: [C.lavender, C.violet], cyan: [C.cyanHi, C.cyanLo], gold: [C.goldHi, C.goldLo],
      product: { onDark: C.subDark, onLight: C.subLight }, ui: { grayText: C.gray } },
    type: { family: 'Urbanist', wordmarkWeight: 500, productWeight: 400, source: 'https://fonts.google.com/specimen/Urbanist' },
    geometry: { baseline: G.BASE, mWidth: G.M_W, stem: 47, wordmarkWidth: G.WORD_W, clearSpaceUnit: G.CLEAR },
    minimumSize: { markPx: 24, compactMarkPx: 16, wordmarkPx: 140, productLockupPx: 240 },
  };
  const css = `:root {
  --metrale-ground: ${C.ground};
  --metrale-ink: ${C.inkHi};
  --metrale-ink-2: ${C.inkLo};
  --metrale-violet: ${C.violet};
  --metrale-lavender: ${C.lavender};
  --metrale-cyan: ${C.cyanHi};
  --metrale-cyan-2: ${C.cyanLo};
  --metrale-gold: ${C.goldHi};
  --metrale-gold-2: ${C.goldLo};
  --metrale-product: ${C.subDark};
  --metrale-gray-text: ${C.subDark};
  --metrale-font: Urbanist, "Helvetica Neue", Helvetica, sans-serif;
}

@media (prefers-color-scheme: light) {
  :root {
    --metrale-ground: ${C.groundLight};
    --metrale-ink: ${C.inkDark};
    --metrale-ink-2: ${C.inkDark};
    --metrale-product: ${C.subLight};
    --metrale-gray-text: ${C.gray};
  }
}
`;
  const scss = `$metrale-ground-dark: ${C.ground};
$metrale-ground-light: ${C.groundLight};
$metrale-ink-dark-hi: ${C.inkHi};
$metrale-ink-dark-lo: ${C.inkLo};
$metrale-ink-light: ${C.inkDark};
$metrale-lavender: ${C.lavender};
$metrale-violet: ${C.violet};
$metrale-cyan-hi: ${C.cyanHi};
$metrale-cyan-lo: ${C.cyanLo};
$metrale-gold-hi: ${C.goldHi};
$metrale-gold-lo: ${C.goldLo};
$metrale-product-dark: ${C.subDark};
$metrale-product-light: ${C.subLight};
$metrale-font: Urbanist, "Helvetica Neue", Helvetica, sans-serif;
`;
  const tw = `// tailwind.config.js -> theme.extend.colors
module.exports = {
  metrale: {
    ground: '${C.ground}',
    ink: '${C.inkHi}',
    'ink-2': '${C.inkLo}',
    'ink-light': '${C.inkDark}',
    lavender: '${C.lavender}',
    violet: '${C.violet}',
    cyan: '${C.cyanHi}',
    'cyan-2': '${C.cyanLo}',
    gold: '${C.goldHi}',
    'gold-2': '${C.goldLo}',
    product: '${C.subDark}',
    'product-light': '${C.subLight}',
  },
};
`;
  const manifest = JSON.stringify({ name: 'Metrale', short_name: 'Metrale',
    icons: [{ src: '/icon-192.png', sizes: '192x192', type: 'image/png' }, { src: '/icon-512.png', sizes: '512x512', type: 'image/png' },
      { src: '/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' }],
    theme_color: C.ground, background_color: C.ground, display: 'standalone' }, null, 2) + '\n';
  return { json: JSON.stringify(json, null, 2) + '\n', css, scss, tw, manifest };
}

function emailSignature(label, file, h) {
  return `<!-- ${label} e-mail signature. Host the PNG somewhere public and swap the src. -->
<table cellpadding="0" cellspacing="0" border="0" style="font-family:Urbanist,Arial,Helvetica,sans-serif;font-size:13px;color:${C.gray}">
  <tr>
    <td style="padding:0 16px 0 0;vertical-align:middle">
      <img src="https://example.com/${file}" width="200" height="${h}" alt="${label}" style="display:block;border:0">
    </td>
    <td style="border-left:1px solid #DDE0E6;padding:2px 0 2px 16px;vertical-align:middle">
      <div style="font-size:14px;font-weight:bold;color:${C.inkDark}">Your Name</div>
      <div style="padding-top:2px">Title</div>
      <div style="padding-top:6px"><a href="mailto:you@metrale.com" style="color:${C.gray};text-decoration:none">you@metrale.com</a></div>
    </td>
  </tr>
</table>
`;
}

// ---------------------------------------------------------- spec diagrams
function clearSpaceSpec() {
  const k = G.CLEAR, art = G.wordmark({ id: 'cs' }); const W = art.width + 2 * k, H = art.height + 2 * k;
  const tick = (x1, y1, x2, y2) => `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${C.subDark}" stroke-width="3"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Clear space">
  <title>Clear space</title>
  ${art.defs}<rect width="${W}" height="${H}" fill="${C.ground}"/>
  <rect x="1.5" y="1.5" width="${W - 3}" height="${H - 3}" fill="none" stroke="${C.subDark}" stroke-width="3" stroke-dasharray="14 12" opacity=".6"/>
  <g transform="translate(${k - art.x0} ${k - art.y0})">
  ${art.body}
  </g>
  ${tick(k / 2, 0, k / 2, k)}${tick(k / 2 - 12, 0, k / 2 + 12, 0)}${tick(k / 2 - 12, k, k / 2 + 12, k)}
  <text x="${k / 2 + 18}" y="${k / 2 + 8}" font-family="Urbanist, sans-serif" font-size="30" fill="${C.subDark}">x</text>
  <text x="${k}" y="${H - 22}" font-family="Urbanist, sans-serif" font-size="26" fill="${C.subDark}">x = half the cyan bar (76 units)</text>
</svg>
`;
}
function minSizeSpec(slug) {
  let y = 60, parts = [], defs = '';
  const rows = [[G.wordmark({ id: 'm2' }), 140, 'wordmark', ''], [G.mark({ id: 'm3' }), 24, 'mark', 'below 48 px use mark-compact']];
  if (slug) rows.unshift([G.stacked({ slug, id: 'm1' }), 240, `lockup-${slug}`, 'product name still legible']);
  rows.forEach(([a, px, name, note]) => {
    defs += a.defs; const s = px / a.width, h = a.height * s;
    parts.push(`<g transform="translate(${40 - a.x0 * s} ${y - a.y0 * s}) scale(${s.toFixed(5)})">${a.body}</g><text x="${40 + px + 24}" y="${y + h / 2 + 5}" font-family="Urbanist, sans-serif" font-size="14" fill="${C.subDark}">${name} — min ${px} px wide${note ? ', ' + note : ''}</text>`);
    y += h + 36;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 ${y}" width="820" height="${y}" role="img" aria-label="Minimum sizes">
  <title>Minimum sizes</title>
  ${defs}<rect width="820" height="${y}" fill="${C.ground}"/>
  ${parts.join('\n  ')}
</svg>
`;
}

// ------------------------------------------------------------------- main
(async () => {
  cutProducts();
  G.setPaths(P);
  const { tight, svg } = G;
  for (const d of ['svg', 'svg/spec', 'light', 'dark', 'transparent', 'tokens', 'src']) fs.mkdirSync(path.join(OUT, d), { recursive: true });

  const masters = [
    ['mark.svg', tight(G.mark({ id: 'a' }), 'Metrale mark')], ['mark-onlight.svg', tight(G.mark({ id: 'b', theme: 'light' }), 'Metrale mark')],
    ['mark-mono-ondark.svg', tight(G.mark({ id: 'c', mono: true }), 'Metrale mark')], ['mark-mono.svg', tight(G.mark({ id: 'd', theme: 'light', mono: true }), 'Metrale mark')],
    ['mark-compact.svg', tight(G.markCompact({ id: 'e' }), 'Metrale mark, small-size cut')],
    ['wordmark-ondark.svg', tight(G.wordmark({ id: 'f' }), 'Metrale')], ['wordmark.svg', tight(G.wordmark({ id: 'g', theme: 'light' }), 'Metrale')],
    ['wordmark-mono-ondark.svg', tight(G.wordmark({ id: 'h', mono: true }), 'Metrale')], ['wordmark-mono.svg', tight(G.wordmark({ id: 'i', theme: 'light', mono: true }), 'Metrale')],
    ['icon-dark.svg', svg({ cw: 1024, ch: 1024, art: G.mark({ id: 'p' }), ratio: G.AV, bg: C.ground, label: 'Metrale icon' })],
    ['icon-light.svg', svg({ cw: 1024, ch: 1024, art: G.mark({ id: 'q', theme: 'light' }), ratio: G.AV, bg: C.groundLight, label: 'Metrale icon' })],
    ['app-icon.svg', svg({ cw: 1024, ch: 1024, art: G.mark({ id: 'r' }), ratio: G.AV, bg: C.ground, radius: 224, label: 'Metrale' })],
    ['og-dark.svg', svg({ cw: 1200, ch: 630, art: G.wordmark({ id: 's' }), ratio: 0.62, hRatio: 0.62, bg: C.ground, label: 'Metrale' })],
    ['og-light.svg', svg({ cw: 1200, ch: 630, art: G.wordmark({ id: 't', theme: 'light' }), ratio: 0.62, hRatio: 0.62, bg: C.groundLight, label: 'Metrale' })],
    ['github-social-preview-dark.svg', svg({ cw: 1280, ch: 640, art: G.wordmark({ id: 'w' }), ratio: 0.62, hRatio: 0.62, bg: C.ground, label: 'Metrale' })],
    ['github-social-preview-ondark-transparent.svg', svg({ cw: 1280, ch: 640, art: G.wordmark({ id: 'x' }), ratio: 0.62, hRatio: 0.62, bg: null, label: 'Metrale' })],
    ['header-dark.svg', G.header({ theme: 'dark', id: 'z1' })], ['header-light.svg', G.header({ theme: 'light', id: 'z2' })],
    ['slide-title-dark.svg', G.slide({ theme: 'dark', id: 'z5' })], ['slide-title-light.svg', G.slide({ theme: 'light', id: 'z6' })],
  ];
  const ST = G.productStacked, IN = G.productInline;
  for (const [slug, name] of Object.entries(PRODUCTS)) {
    const PN = `Metrale ${name}`;
    masters.push(
      [`lockup-${slug}-ondark.svg`, tight(ST({ slug, id: 'j' }), PN)], [`lockup-${slug}.svg`, tight(ST({ slug, id: 'k', theme: 'light' }), PN)],
      [`lockup-${slug}-mono-ondark.svg`, tight(ST({ slug, id: 'l', mono: true }), PN)], [`lockup-${slug}-mono.svg`, tight(ST({ slug, id: 'm', theme: 'light', mono: true }), PN)],
      [`lockup-${slug}-inline-ondark.svg`, tight(IN({ slug, id: 'n' }), PN)], [`lockup-${slug}-inline.svg`, tight(IN({ slug, id: 'o', theme: 'light' }), PN)],
      [`og-${slug}-dark.svg`, svg({ cw: 1200, ch: 630, art: ST({ slug, id: 'u' }), ratio: 0.62, hRatio: 0.62, bg: C.ground, label: PN })],
      [`og-${slug}-light.svg`, svg({ cw: 1200, ch: 630, art: ST({ slug, id: 'v', theme: 'light' }), ratio: 0.62, hRatio: 0.62, bg: C.groundLight, label: PN })],
      [`github-social-preview-${slug}-dark.svg`, svg({ cw: 1280, ch: 640, art: ST({ slug, id: 'y' }), ratio: 0.62, hRatio: 0.62, bg: C.ground, label: PN })],
      [`header-${slug}-dark.svg`, G.header({ slug, theme: 'dark', id: 'z3' })], [`header-${slug}-light.svg`, G.header({ slug, theme: 'light', id: 'z4' })],
      [`slide-title-${slug}-dark.svg`, G.slide({ slug, theme: 'dark', id: 'z7' })], [`slide-title-${slug}-light.svg`, G.slide({ slug, theme: 'light', id: 'z8' })],
    );
  }
  if (PRODUCTS.ai) masters.push(
    ['ai-mark.svg', tight(G.aiMark({ id: 'a' }), 'Metrale AI monogram')], ['ai-mark-onlight.svg', tight(G.aiMark({ id: 'b', theme: 'light' }), 'Metrale AI monogram')],
    ['ai-mark-mono-ondark.svg', tight(G.aiMark({ id: 'c', mono: true }), 'Metrale AI monogram')], ['ai-mark-mono.svg', tight(G.aiMark({ id: 'd', theme: 'light', mono: true }), 'Metrale AI monogram')],
    ['app-icon-ai.svg', svg({ cw: 1024, ch: 1024, art: G.aiMark({ id: 'e' }), ratio: 0.66, hRatio: 0.66, bg: C.ground, radius: 224, label: 'Metrale AI' })],
    ['lockup-ai-inline-type-ondark.svg', tight(G.aiTypeInline({ id: 'n2' }), 'Metrale AI')], ['lockup-ai-inline-type.svg', tight(G.aiTypeInline({ id: 'o2', theme: 'light' }), 'Metrale AI')],
  );
  for (const [name, s] of masters) fs.writeFileSync(path.join(OUT, 'svg', name), s + '\n');
  fs.writeFileSync(path.join(OUT, 'svg/spec/clear-space.svg'), clearSpaceSpec());
  fs.writeFileSync(path.join(OUT, 'svg/spec/minimum-sizes.svg'), minSizeSpec(Object.keys(PRODUCTS)[0]));
  { const w = G.wordmark({ id: 'sw' });
    fs.writeFileSync(path.join(OUT, 'svg/spec/swash.svg'), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="1585 272 130 104" width="650" height="520" role="img" aria-label="The swash, cropped from the wordmark">
  <title>The swash</title>
  ${w.defs}<rect x="1585" y="272" width="130" height="104" fill="${C.ground}"/>
  ${w.body}
</svg>
`); }

  const t = tokenFiles();
  fs.writeFileSync(path.join(OUT, 'tokens/brand.json'), t.json);
  fs.writeFileSync(path.join(OUT, 'tokens/brand.css'), t.css);
  fs.writeFileSync(path.join(OUT, 'tokens/brand.scss'), t.scss);
  fs.writeFileSync(path.join(OUT, 'tokens/tailwind.colors.js'), t.tw);
  fs.writeFileSync(path.join(OUT, 'site.webmanifest'), t.manifest);
  const wm = G.wordmark({ id: 'x' });
  fs.writeFileSync(path.join(OUT, 'email-signature.html'), emailSignature('Metrale', 'email-signature-400.png', Math.round(200 * wm.height / wm.width)));
  for (const [slug, name] of Object.entries(PRODUCTS)) {
    const a = ST({ slug, id: 'x' });
    fs.writeFileSync(path.join(OUT, `email-signature-${slug}.html`), emailSignature(`Metrale ${name}`, `email-signature-${slug}-400.png`, Math.round(200 * a.height / a.width)));
  }

  const browser = await chromium.launch();
  const page = await browser.newPage();
  let count = 0;
  async function shoot(source, cw, ch, file, transparent) {
    await page.setViewportSize({ width: cw, height: ch });
    await page.setContent(`<html><body style="margin:0;background:transparent">${source}</body></html>`, { waitUntil: 'load' });
    await page.screenshot({ path: file, omitBackground: transparent });
    count++;
  }
  const bgOf = theme => theme === 'transparent' ? null : theme === 'dark' ? C.ground : C.groundLight;
  const toneOf = theme => theme === 'light' ? 'light' : 'dark';

  // squares: the mark only (shared by company and products)
  for (const [name, size, ratio, which] of G.squares) for (const theme of ['dark', 'light', 'transparent']) {
    if (theme === 'transparent' && (name.startsWith('apple') || name.includes('maskable'))) continue;
    const art = which === 'compact' ? G.markCompact({ theme: toneOf(theme), id: 'q' }) : G.mark({ theme: toneOf(theme), id: 'q' });
    await shoot(svg({ cw: size, ch: size, art, ratio, bg: bgOf(theme) }), size, size, path.join(OUT, theme, name), theme === 'transparent');
  }
  for (const [name, size, ratio] of G.squares.filter(s => ['avatar-1024.png', 'icon-512.png', 'favicon-48.png'].includes(s[0]))) {
    const art = ratio > 0.8 ? G.markCompact({ theme: 'dark', id: 'q' }) : G.mark({ theme: 'dark', id: 'q' });
    await shoot(svg({ cw: size, ch: size, art, ratio, bg: null }), size, size, path.join(OUT, 'transparent', name.replace('.png', '-ondark.png')), true);
  }

  // wides: company (wordmark) and one -slug set per product
  const variants = [['', (t) => G.wordmark({ theme: t, id: 'w' }), null]];
  for (const slug of Object.keys(PRODUCTS)) variants.push(['-' + slug, (t, kind) => kind === 'inline' ? IN({ slug, theme: t, id: 'w' }) : ST({ slug, theme: t, id: 'w' }), slug]);
  for (const [base, cw, ch, ratio, hRatio, kind] of G.wides) for (const [sfx, mk] of variants) for (const theme of ['dark', 'light']) {
    const art = mk(theme, kind); const name = G.withSuffix(base, sfx);
    await shoot(svg({ cw, ch, art, ratio, hRatio, bg: bgOf(theme) }), cw, ch, path.join(OUT, theme, name), false);
    if (G.transparentWides.has(base)) await shoot(svg({ cw, ch, art, ratio, hRatio, bg: null }), cw, ch, path.join(OUT, 'transparent', name.replace('.png', theme === 'dark' ? '-ondark.png' : '.png')), true);
  }

  // lockup bitmaps
  const lockups = [['wordmark', o => G.wordmark(o)], ['mark', o => G.mark(o)]];
  for (const slug of Object.keys(PRODUCTS)) lockups.push([`lockup-${slug}`, o => ST({ slug, ...o })], [`lockup-${slug}-inline`, o => IN({ slug, ...o })]);
  if (PRODUCTS.ai) lockups.push(['ai-mark', o => G.aiMark(o)], ['lockup-ai-inline-type', o => G.aiTypeInline(o)]);
  for (const [base, fn] of lockups) for (const wpx of [1024, 2048]) {
    const a0 = fn({ id: 'x' }); const hpx = Math.round(wpx * a0.height / a0.width);
    for (const theme of ['dark', 'light', 'transparent']) {
      await shoot(svg({ cw: wpx, ch: hpx, art: fn({ theme: toneOf(theme), id: 'x' }), ratio: 1, hRatio: 1, bg: bgOf(theme) }), wpx, hpx,
        path.join(OUT, theme, `${base}-${wpx}${theme === 'transparent' ? '-ondark' : ''}.png`), theme === 'transparent');
    }
    await shoot(svg({ cw: wpx, ch: hpx, art: fn({ theme: 'light', id: 'x' }), ratio: 1, hRatio: 1, bg: null }), wpx, hpx, path.join(OUT, 'transparent', `${base}-${wpx}.png`), true);
  }

  // e-mail signature bitmaps (200 CSS px wide, 1x and 2x)
  const sigs = [['', o => G.wordmark(o)]];
  for (const slug of Object.keys(PRODUCTS)) sigs.push(['-' + slug, o => ST({ slug, ...o })]);
  for (const [sfx, fn] of sigs) for (const wpx of [400, 800]) {
    const a0 = fn({ id: 'x' }); const hpx = Math.round(wpx * a0.height / a0.width);
    for (const [theme, tone] of [['light', 'light'], ['transparent', 'light'], ['dark', 'dark']]) {
      const nm = theme === 'dark' ? `email-signature${sfx}-${wpx}-ondark.png` : `email-signature${sfx}-${wpx}.png`;
      await shoot(svg({ cw: wpx, ch: hpx, art: fn({ theme: tone, id: 'x' }), ratio: 1, hRatio: 1, bg: bgOf(theme) }), wpx, hpx, path.join(OUT, theme, nm), theme === 'transparent');
    }
  }

  if (PRODUCTS.ai) for (const theme of ['dark', 'light', 'transparent']) {
    const tone = theme === 'light' ? 'light' : 'dark';
    await shoot(svg({ cw: 1024, ch: 1024, art: G.aiMark({ theme: tone, id: 'q' }), ratio: 0.66, hRatio: 0.66, bg: bgOf(theme) }), 1024, 1024, path.join(OUT, theme, 'app-icon-ai-1024.png'), theme === 'transparent');
    await shoot(svg({ cw: 1024, ch: 1024, art: G.aiMark({ theme: tone, id: 'q' }), ratio: 0.62, hRatio: 0.62, bg: bgOf(theme) }), 1024, 1024, path.join(OUT, theme, 'avatar-ai-1024.png'), theme === 'transparent');
  }

  // deck / document headers and title slides
  for (const theme of ['dark', 'light']) for (const slug of [null, ...Object.keys(PRODUCTS)]) {
    const sf = slug ? `-${slug}` : '';
    await shoot(G.header({ slug, theme, id: 'h' }), 1920, 160, path.join(OUT, theme, `header${sf}-1920x160.png`), false);
    await shoot(G.slide({ slug, theme, id: 's' }), 1920, 1080, path.join(OUT, theme, `slide-title${sf}-1920x1080.png`), false);
  }

  await browser.close();
  console.log('rendered', count, 'PNGs. Re-cut favicon.ico from dark/favicon-{16,32,48}.png (e.g. `npx png-to-ico`).');
})();
