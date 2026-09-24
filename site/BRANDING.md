# Metrale website branding

The brand is Metrale; this repository is the Metrale Engine. The site, the
blog and the book apply the Metrale kit, whose
guidelines are `assets/brand/BRAND-GUIDELINES.md` (the kit's own text, with a
section on how this repository reads it). This file says how the website
applies them.

## The logo

One component draws every logo on the site and the blog:
`web-shared/components/MetraleLockup.svelte`. It draws from
`web-shared/brand-art.js`, which `scripts/brand/lockup.mjs` writes from the
kit's own geometry (`assets/brand/src/geometry.js` and `src/paths.json`), and
`src/lib/lockup-artwork.test.js` fails if the module is behind the geometry or
the component draws anything of its own. Nothing is redrawn.

- Kinds: `wordmark` (the default, min 140 px), `mark` (the M with the swash
  lifted over its right shoulder, min 24 px), `compact` (the M alone, 48 px and
  below). `horizontal`, `full` and `corp` are accepted as aliases of the
  wordmark.
- The inks are tokens: `--m-ink-hi` and `--m-ink-lo` (the kit's two inks on
  dark, its one ink on light), `--m-lavender` and `--m-violet` (the M's right
  half), `--m-cyan-hi` and `-lo` (the bar), `--m-gold-hi` and `-lo` (the
  swash). The gradients keep the kit's directions, so the theme swap costs no
  second file and no second request.
- **Front page and `/control` nav**: the wordmark at 140 px, the floor.
  **Blog header and footer**: 152 px.
- **The book**: `book/theme/metrale.js` puts the on-dark wordmark in mdBook's
  menu bar with "Engine docs" beside it, and `book/theme/css/wordmark.css`
  places it. `lockup.mjs` writes that script with the master inlined (mdBook
  has no bundler); the lockup test fails if it falls behind.
- **Favicons and icons**: the kit's own cuts in `static/`: the app icon on the
  ground as `favicon.svg`, PNGs at 16, 32, 48, 180, 192, 512 and 1024, the
  maskable 512, and `favicon.ico`. The blog carries the same set.
- **Social card**: `static/og-image.png` (and the blog's) is the kit's
  `assets/brand/social/og-image-dark.png`, the wordmark on the ground. The
  GitHub social preview is in `assets/brand/social/`, uploaded by hand.
- Clear space is 76 units on every side (half the cyan bar), which the
  component applies as margin in proportion to its width.
- One logo per surface. The kit's product lockups are not used: the product is
  named in type ("Metrale Engine") where it appears.

Never generate the logo with an image model, never redraw it, never recolour
it.

## Colour

`web-shared/metrale-tokens.css` is the single source for the site, the blog and
the book, and `src/lib/brand-tokens.test.js` holds it equal to the kit's
`assets/brand/tokens/brand.json`. The ground is `#0E1318`, the only dark
background the kit allows, with the surfaces stepped off it. The inks are the
kit's: `#F7F7F9` for headings, `#D9D9DE` for body, `#8A8F99` for metadata, on
dark; `#15181F` on light. The accent is the M's violet: pale (`#CDBFF1`) as
text on dark, `#9F8DD8` as a fill, and the ink on a filled accent is the
ground, never white (`--on-accent`: white on that violet is 2.9:1, the ground
is 6.5:1). `data-theme` is set before first paint from `localStorage`
(`metrale-theme`) or the system preference; `web-shared/theme.js` and both
`app.html` files pin the two grounds for the theme-color meta, which
`src/lib/theme-color.test.js` holds equal to the tokens.

Each hue has one meaning: violet is speed and the engine, cyan is security and
silicon, gold is community. The kit draws three hues and no green; green stays
as a UI signal for a verified result (`--ch-green`, `--green`). The benchmark
chart series (`--series-*`) are the engine site's own and are not brand hues.
The contrast gates are `.contrast-check.mjs` at the repository root (text over
the chevron field) and `light-text-contrast`, `series-contrast` and
`chart-swatch-css` under `src/lib`.

## Type

Urbanist for everything set in words: Medium for headings, Regular for body,
the way the kit says. IBM Plex Mono for numbers, labels, receipts and code,
which the kit does not cover. Both are self hosted from `static/fonts/` with
their licences beside them: Urbanist as one variable file per style (weights
100 to 900, latin). `src/app.html` attaches `static/fonts/type.css` after first
paint, behind metric matched fallbacks in `src/styles/fonts.css`, so nothing
moves when the faces land. The fallback numbers come from the font files:
instantiate the variable font at the weight (fontTools), average the advance
over English letter frequencies, divide by Arial's, and express Urbanist's
0.95 em ascent and 0.25 em descent in the adjusted em. The kit's own fallback,
Helvetica Neue, follows in the stack.

The blog sets its articles in Charter and its chrome in the shared
`--font-sans` stack; it ships no Urbanist files, so its chrome falls through to
Helvetica Neue, as the kit's blog does. The book links the same faces through
`book/theme/fonts/fonts.css`, whose font files are links to `static/fonts/`.

## The pages

The front page is the engine page: benchmarks, recipes, installation and the
chat tools, with its sections reachable as `/#verified`, `/#news`,
`/#hardware`, `/#models` and `/#run`. `/engine` is a permanent redirect to
`/` (`static/_redirects`). `/control` and `/diligence` retain their existing
functionality.
