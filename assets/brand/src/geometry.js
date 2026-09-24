// Metrale brand geometry — shared by gen.js (node) and the in-browser renderer.
// Coordinates are the reference logo's own pixels, shifted so the M's top-left
// point is (0,0). Baseline y = 362.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MetraleGeometry = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  const C = {
    ground: '#0E1318', groundLight: '#FFFFFF',
    inkHi: '#F7F7F9', inkLo: '#D9D9DE', ink: '#EDEDEF',   // white letters on dark, soft vertical sheen
    inkDark: '#15181F',                                    // letters on light
    lavender: '#CDBFF1', violet: '#9F8DD8',                // M right half
    cyanHi: '#6FD9EC', cyanLo: '#5A96BD',                  // bar, left -> right
    goldHi: '#E4C070', goldLo: '#AE8A3F',                  // swash, top -> bottom
    subDark: '#8A8F99', subLight: '#6B7079',               // product name on dark / on light
    gray: '#6B7079',                                       // text-safe gray, 4.9:1 on white
  };
  const BASE = 362, M_W = 378, S = 47, MID = 189, TIP = 265;
  const BAR = { x: 105, y: 326, w: 168, h: 36 };
  const COMMA = { x: 1597, y: 284, w: 106 };
  const WORD_W = COMMA.x + COMMA.w;                        // 1703: right edge of the swash
  const PRODUCT = { size: 240, baselineGap: 64, capHeight: 168, inlineGap: 96, inlineSize: 468 };
  const CLEAR = 76;                                        // clear-space unit: the M's stem, ×1.6

  let P = {};
  const setPaths = p => { P = p; };

  function defs(id, dark) {
    return `<defs>
    <linearGradient id="${id}v" x1="0" y1="0" x2="0.25" y2="1"><stop offset="0" stop-color="${C.lavender}"/><stop offset="1" stop-color="${C.violet}"/></linearGradient>
    <linearGradient id="${id}c" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C.cyanHi}"/><stop offset="1" stop-color="${C.cyanLo}"/></linearGradient>
    <linearGradient id="${id}g" x1="0.8" y1="0" x2="0.2" y2="1"><stop offset="0" stop-color="${C.goldHi}"/><stop offset="1" stop-color="${C.goldLo}"/></linearGradient>
    <linearGradient id="${id}w" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${dark ? C.inkHi : C.inkDark}"/><stop offset="1" stop-color="${dark ? C.inkLo : C.inkDark}"/></linearGradient>
  </defs>`;
  }
  const mLeft = f => `<path d="M0 0L${MID} ${MID}L${MID} ${TIP}L${S} ${S + 76}L${S} ${BASE}L0 ${BASE}Z" fill="${f}"/>`;
  const mRight = f => `<path d="M${M_W} 0L${MID} ${MID}L${MID} ${TIP}L${M_W - S} ${S + 76}L${M_W - S} ${BASE}L${M_W} ${BASE}Z" fill="${f}"/>`;
  const bar = f => `<rect x="${BAR.x}" y="${BAR.y}" width="${BAR.w}" height="${BAR.h}" fill="${f}"/>`;
  // The swash: a sail. Flat top with a rounded outer corner, both edges sweeping
  // from ~58° at the shoulder to 45° at the foot, easing to vertical at the heel,
  // cut flat 2 above the baseline. Edge points traced from the reference.
  function comma(f, dx = 0, dy = 0) {
    const x = COMMA.x + dx, y = COMMA.y + dy, b = BASE + dy - 2;
    const p = (ox, oy) => `${x + ox} ${y + oy}`;
    return `<path d="M${p(62, 0)}L${p(100, 0)}C${p(104, 0)} ${p(106, 2)} ${p(106, 6)}C${p(98, 26)} ${p(66, 58)} ${p(40, 74)}L${x + 36} ${b}L${x + 6} ${b}C${p(2, 76)} ${p(0, 74)} ${p(0, 70)}C${p(30, 58)} ${p(58, 18)} ${p(62, 0)}Z" fill="${f}"/>`;
  }
  const letters = f => `<path d="${P.etrale}" fill="${f}" stroke="${f}" stroke-width="3.6" stroke-linejoin="round"/>`;
  const productPath = (slug, f, dx = 0, dy = 0, inline = false) => {
    const d = inline ? P.products[slug].inline : P.products[slug].stacked;
    return `<g transform="translate(${dx} ${dy})"><path d="${d}" fill="${f}"/></g>`;
  };

  function inks(theme, id, mono) {
    const dark = theme === 'dark';
    const ink = mono ? (dark ? C.ink : C.inkDark) : `url(#${id}w)`;
    return { dark, ink, violet: mono ? ink : `url(#${id}v)`, cyan: mono ? ink : `url(#${id}c)`, gold: mono ? ink : `url(#${id}g)`, sub: dark ? C.subDark : C.subLight };
  }
  const PAD = 12;

  // ---- art pieces: { x0, y0, width, height, defs, body }
  // The mark: icon D — the M with the swash lifted clear of the right shoulder.
  function mark({ theme = 'dark', id = 'i', mono = false } = {}) {
    const k = inks(theme, id, mono);
    const dx = M_W + 24 - COMMA.x, dy = -BASE - 40;
    return { x0: -8, y0: -128, width: M_W + 24 + COMMA.w + 16, height: BASE + 136, defs: defs(id, k.dark),
      body: [mLeft(k.ink), mRight(k.violet), bar(k.cyan), comma(k.gold, dx, dy)].join('\n  ') };
  }
  // Compact cut for favicons at and below 48 px: the M alone, no swash.
  function markCompact({ theme = 'dark', id = 'i', mono = false } = {}) {
    const k = inks(theme, id, mono);
    return { x0: -8, y0: -8, width: M_W + 16, height: BASE + 16, defs: defs(id, k.dark),
      body: [mLeft(k.ink), mRight(k.violet), bar(k.cyan)].join('\n  ') };
  }
  function wordmark({ theme = 'dark', id = 'w', mono = false } = {}) {
    const k = inks(theme, id, mono);
    return { x0: -PAD, y0: -PAD, width: WORD_W + 2 * PAD, height: BASE + 2 * PAD, defs: defs(id, k.dark),
      body: [mLeft(k.ink), mRight(k.violet), bar(k.cyan), letters(k.ink), comma(k.gold)].join('\n  ') };
  }
  // Product lockups. `stacked`: product name under the wordmark, left-aligned to the M.
  // `inline`: product name on the baseline after the swash — "Metrale, Full Stack".
  function stacked({ slug, theme = 'dark', id = 'w', mono = false } = {}) {
    const w = wordmark({ theme, id, mono }); const k = inks(theme, id, mono);
    const by = BASE + PRODUCT.baselineGap + PRODUCT.capHeight;
    const pw = P.products[slug].stackedWidth;
    return { x0: -PAD, y0: -PAD, width: Math.max(WORD_W, pw) + 2 * PAD, height: by + 2 * PAD + 4, defs: w.defs,
      body: w.body + '\n  ' + productPath(slug, mono ? k.ink : k.sub, 0, by) };
  }
  function inline({ slug, theme = 'dark', id = 'w', mono = false } = {}) {
    const w = wordmark({ theme, id, mono }); const k = inks(theme, id, mono);
    const px = WORD_W + PRODUCT.inlineGap, pw = P.products[slug].inlineWidth;
    return { x0: -PAD, y0: -PAD, width: px + pw + 2 * PAD, height: BASE + 2 * PAD, defs: w.defs,
      body: w.body + '\n  ' + productPath(slug, mono ? k.ink : k.sub, px, BASE, true) };
  }

  // ---- Metrale AI. The "AI" is drawn from the M's own parts, not typed:
  // Λ with a sharp apex (ink left leg, violet right leg), the cyan bar floating
  // inside as the crossbar, and the I as a stem with the gold swash standing on
  // its shoulder as the dot. Same stroke (47), same cap height (362).
  const AI = { aW: 378, gap: 84, iW: 47 };
  AI.width = AI.aW + AI.gap + AI.iW;
  AI.markWidth = AI.width + 110;                        // the swash overhangs the I to the right
  // swash: only on the standalone monogram — next to the wordmark the comma already carries the gold.
  function aiGlyph({ theme = 'dark', id = 'ai', mono = false, dx = 0, dy = 0, swash = false } = {}) {
    const k = inks(theme, id, mono);
    const a = AI.aW, m = a / 2, t = 53, yi = t * (BASE / m);   // t: leg thickness measured horizontally; yi: where the inner edges meet
    const L = `<path d="M0 ${BASE}L${m} 0L${m} ${yi}L${t} ${BASE}Z" fill="${k.ink}"/>`;
    const R = `<path d="M${a} ${BASE}L${m} 0L${m} ${yi}L${a - t} ${BASE}Z" fill="${k.violet}"/>`;
    const cb = `<rect x="${m - 67}" y="236" width="134" height="36" fill="${k.cyan}"/>`;
    const ix = a + AI.gap;
    const I = `<rect x="${ix}" y="0" width="${AI.iW}" height="${BASE}" fill="${k.ink}"/>`;
    const sw = swash ? comma(k.gold, ix + AI.iW + 8 - COMMA.x, -BASE - 40) : '';
    return `<g transform="translate(${dx} ${dy})">${[L, R, cb, I, sw].join('')}</g>`;
  }
  // The whole glyph as a mark of its own (AI monogram).
  function aiMark({ theme = 'dark', id = 'ai', mono = false } = {}) {
    return { x0: -8, y0: -128, width: AI.markWidth + 16, height: BASE + 136, defs: defs(id, theme === 'dark'), body: aiGlyph({ theme, id, mono, swash: true }) };
  }
  // Stacked: wordmark on top, the AI glyph beneath at the product cap height (168), left-aligned to the M.
  function aiStacked({ theme = 'dark', id = 'w', mono = false } = {}) {
    const w = wordmark({ theme, id, mono });
    const s = 200 / BASE, top = BASE + PRODUCT.baselineGap;   // the drawn glyph sits a touch larger than a typed product name
    return { x0: -PAD, y0: -PAD, width: WORD_W + 2 * PAD, height: top + BASE * s + 2 * PAD, defs: w.defs,
      body: w.body + `\n  <g transform="translate(0 ${top.toFixed(1)}) scale(${s.toFixed(5)})">${aiGlyph({ theme, id, mono })}</g>` };
  }
  // Inline: "Metrale, AI" — the glyph at full cap height after the swash.
  function aiInline({ theme = 'dark', id = 'w', mono = false } = {}) {
    const w = wordmark({ theme, id, mono });
    const px = WORD_W + PRODUCT.inlineGap + 20;
    return { x0: -PAD, y0: -PAD, width: px + AI.width + 2 * PAD, height: BASE + 2 * PAD, defs: w.defs,
      body: w.body + '\n  ' + aiGlyph({ theme, id, mono, dx: px }) };
  }
  // Typographic alternative: AI in Urbanist Medium filled with the lavender → cyan sweep.
  function aiTypeInline({ theme = 'dark', id = 'w', mono = false } = {}) {
    const w = wordmark({ theme, id, mono }); const k = inks(theme, id, mono);
    const px = WORD_W + PRODUCT.inlineGap + 20;
    const grad = `<linearGradient id="${id}ai" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${C.lavender}"/><stop offset="0.55" stop-color="${C.violet}"/><stop offset="1" stop-color="${C.cyanLo}"/></linearGradient>`;
    return { x0: -PAD, y0: -PAD, width: px + P.aiType.width + 2 * PAD, height: BASE + 2 * PAD, defs: w.defs.replace('</defs>', grad + '</defs>'),
      body: w.body + `\n  <g transform="translate(${px} 0)"><path d="${P.aiType.d}" fill="${mono ? k.ink : `url(#${id}ai)`}"/></g>` };
  }
  // Product dispatch: 'ai' is drawn, everything else is typed.
  const productStacked = o => o.slug === 'ai' ? aiStacked(o) : stacked(o);
  const productInline = o => o.slug === 'ai' ? aiInline(o) : inline(o);

  // ---- canvases
  function tight(art, label) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${art.x0} ${art.y0} ${art.width} ${art.height}" width="${art.width}" height="${art.height}" role="img" aria-label="${label}">
  <title>${label}</title>
  ${art.defs}
  ${art.body}
</svg>`;
  }
  function svg({ cw, ch, art, ratio = 1, hRatio = 0.9, bg = null, label = 'Metrale', radius = 0, align = 'center', margin = 0 }) {
    const s = Math.min((ratio * cw) / art.width, (hRatio * ch) / art.height);
    const tx = align === 'left' ? margin - art.x0 * s : cw / 2 - (art.x0 + art.width / 2) * s;
    const ty = ch / 2 - (art.y0 + art.height / 2) * s;
    const rect = bg ? `<rect width="${cw}" height="${ch}" rx="${radius}" fill="${bg}"/>` : '';
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${cw} ${ch}" width="${cw}" height="${ch}" role="img" aria-label="${label}">
  ${art.defs}${rect}<g transform="translate(${+tx.toFixed(2)} ${+ty.toFixed(2)}) scale(${+s.toFixed(5)})">
  ${art.body}
  </g>
</svg>`;
  }
  // Slide title background: mark small at top-left, lockup large at the lower left.
  function slide({ slug, theme = 'dark', id = 's' }) {
    const cw = 1920, ch = 1080, dark = theme === 'dark';
    const art = slug ? productStacked({ slug, theme, id }) : wordmark({ theme, id });
    const s = 760 / art.width;
    const m = mark({ theme, id: id + 'm' }); const ms = 96 / m.height;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${cw} ${ch}" width="${cw}" height="${ch}" role="img" aria-label="Metrale title slide">
  ${art.defs}<rect width="${cw}" height="${ch}" fill="${dark ? C.ground : C.groundLight}"/>
  <g transform="translate(${120 - m.x0 * ms} ${120 - m.y0 * ms}) scale(${ms.toFixed(5)})">${m.body}</g>
  <g transform="translate(${120 - art.x0 * s} ${ch - 160 - (art.y0 + art.height) * s}) scale(${s.toFixed(5)})">${art.body}</g>
</svg>`;
  }
  // Document / deck header strip.
  function header({ slug, theme = 'dark', id = 'h' }) {
    const art = slug ? productInline({ slug, theme, id }) : wordmark({ theme, id });
    return svg({ cw: 1920, ch: 160, art, ratio: 0.6, hRatio: 0.5, bg: theme === 'dark' ? C.ground : C.groundLight, align: 'left', margin: 80, label: 'Metrale header' });
  }

  // ---- asset lists
  const AV = 0.62;
  const squares = [
    ['avatar-1024.png', 1024, AV, 'mark'], ['discord-avatar-512.png', 512, AV, 'mark'], ['x-avatar-400.png', 400, AV, 'mark'],
    ['github-avatar-460.png', 460, AV, 'mark'], ['slack-icon-512.png', 512, AV, 'mark'], ['linkedin-logo-300.png', 300, AV, 'mark'],
    ['app-icon-1024.png', 1024, AV, 'mark'], ['icon-512.png', 512, AV, 'mark'], ['icon-maskable-512.png', 512, 0.56, 'mark'],
    ['icon-192.png', 192, AV, 'mark'], ['apple-touch-icon-180.png', 180, AV, 'mark'],
    ['bluesky-avatar-1000.png', 1000, AV, 'mark'], ['mastodon-avatar-400.png', 400, AV, 'mark'], ['instagram-threads-avatar-320.png', 320, AV, 'mark'],
    ['facebook-avatar-720.png', 720, AV, 'mark'], ['tiktok-avatar-400.png', 400, AV, 'mark'], ['reddit-icon-256.png', 256, AV, 'mark'],
    ['product-hunt-thumbnail-240.png', 240, AV, 'mark'], ['huggingface-avatar-512.png', 512, AV, 'mark'], ['gravatar-1024.png', 1024, AV, 'mark'],
    ['favicon-48.png', 48, 0.84, 'compact'], ['favicon-32.png', 32, 0.86, 'compact'], ['favicon-16.png', 16, 0.9, 'compact'],
  ];
  // name, w, h, ratio, hRatio, product-lockup kind ('stacked' | 'inline'; company files always use the wordmark)
  const wides = [
    ['og-image-1200x630.png', 1200, 630, 0.62, 0.62, 'stacked'],
    ['social-square-1200.png', 1200, 1200, 0.72, 0.5, 'stacked'],
    ['x-header-1500x500.png', 1500, 500, 0.46, 0.5, 'stacked'],
    ['linkedin-banner-1128x191.png', 1128, 191, 0.5, 0.56, 'inline'],
    ['youtube-channel-art-2560x1440.png', 2560, 1440, 0.24, 0.2, 'stacked'],
    ['github-social-preview-1280x640.png', 1280, 640, 0.62, 0.62, 'stacked'],
    ['bluesky-banner-3000x1000.png', 3000, 1000, 0.46, 0.5, 'stacked'],
    ['mastodon-header-1500x500.png', 1500, 500, 0.46, 0.5, 'stacked'],
    ['facebook-cover-1640x624.png', 1640, 624, 0.5, 0.55, 'stacked'],
    ['reddit-banner-1920x384.png', 1920, 384, 0.5, 0.56, 'inline'],
    ['discord-banner-960x540.png', 960, 540, 0.62, 0.5, 'stacked'],
    ['play-feature-graphic-1024x500.png', 1024, 500, 0.6, 0.55, 'stacked'],
    ['product-hunt-gallery-1270x760.png', 1270, 760, 0.62, 0.5, 'stacked'],
  ];
  const transparentWides = new Set(['github-social-preview-1280x640.png']);
  const withSuffix = (name, suffix) => suffix ? name.replace(/-(\d+(x\d+)?(-ondark)?\.png)$/, `${suffix}-$1`) : name;

  return { C, BASE, M_W, WORD_W, PRODUCT, CLEAR, AV, setPaths, mark, markCompact, wordmark, stacked, inline, slide, header,
    aiGlyph, aiMark, aiStacked, aiInline, aiTypeInline, productStacked, productInline,
    tight, svg, squares, wides, transparentWides, withSuffix };
});
