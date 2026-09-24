<!--
  The Metrale brand artwork, as vector definitions.

  Every piece is drawn from the kit's own geometry: ../brand-art.js is written
  by site/scripts/brand/lockup.mjs from assets/brand/src/geometry.js and
  src/paths.json, and src/lib/lockup-artwork.test.js fails when it is behind.
  Nothing here is redrawn; the only substitution is that each ink reads a
  token (--m-ink-hi and friends in ../metrale-tokens.css) instead of the kit's
  literal hex, so the light theme swaps the ink for the kit's light-ground ink
  with no second file. The gradients keep the kit's directions.

  Kinds:
    defs      the definitions, rendered once per document by the root layout
    wordmark  the default logo: the M, the letters, the gold swash. Min 140 px
    mark      the M with the swash lifted above its right shoulder. Min 24 px
    compact   the M alone, for favicons at and below 48 px. Min 16 px
  'horizontal', 'full' and 'corp' are accepted as aliases of wordmark.

  Sizing is by width, because the guideline minimums are widths. Clear space
  is 76 units on every side (half the cyan bar), as CSS margin rather than
  viewBox padding, which would silently shrink the artwork below the minimum.
-->
<script>
  import { ART } from '../brand-art.js';

  let { kind = 'wordmark', width = null, label = null, class: klass = '' } = $props();
  const KIND = { horizontal: 'wordmark', full: 'wordmark', corp: 'wordmark' };
  const k = $derived(KIND[kind] ?? kind);
  const NAMES = { wordmark: 'Metrale', mark: 'Metrale', compact: 'Metrale' };
  const name = $derived(label ?? NAMES[k] ?? 'Metrale');
  const box = $derived(ART.boxes[k] ?? ART.boxes.wordmark);
  const viewBox = $derived(`${box.x0} ${box.y0} ${box.width} ${box.height}`);
  // Clear space as a share of the rendered width: 76 units of the box.
  const clear = $derived(ART.clear / box.width);
  const style = $derived(width ? `width:${width}px;margin:${(width * clear).toFixed(2)}px` : null);
  const FILL = { ink: 'url(#m-ink)', violet: 'url(#m-violet)', cyan: 'url(#m-cyan)', gold: 'url(#m-gold)' };
</script>

{#snippet pieces(list)}
  {#each list as p, i (i)}
    {#if p.kind === 'rect'}
      <rect x={p.x} y={p.y} width={p.w} height={p.h} fill={FILL[p.ink]} />
    {:else}
      <path d={p.d} fill={FILL[p.ink]} stroke={p.stroke ? FILL[p.ink] : undefined} stroke-width={p.stroke || undefined} stroke-linejoin={p.stroke ? 'round' : undefined} />
    {/if}
  {/each}
{/snippet}

{#if kind === 'defs'}
  <svg class="metrale-defs" width="0" height="0" aria-hidden="true" focusable="false">
    <defs>
      <linearGradient id="m-ink" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--m-ink-hi)" /><stop offset="1" stop-color="var(--m-ink-lo)" /></linearGradient>
      <linearGradient id="m-violet" x1="0" y1="0" x2="0.25" y2="1"><stop offset="0" stop-color="var(--m-lavender)" /><stop offset="1" stop-color="var(--m-violet)" /></linearGradient>
      <linearGradient id="m-cyan" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="var(--m-cyan-hi)" /><stop offset="1" stop-color="var(--m-cyan-lo)" /></linearGradient>
      <linearGradient id="m-gold" x1="0.8" y1="0" x2="0.2" y2="1"><stop offset="0" stop-color="var(--m-gold-hi)" /><stop offset="1" stop-color="var(--m-gold-lo)" /></linearGradient>
      <g id="m-wordmark">{@render pieces(ART.wordmark)}</g>
      <g id="m-mark">{@render pieces(ART.mark)}</g>
      <g id="m-compact">{@render pieces(ART.compact)}</g>
    </defs>
  </svg>
{:else}
  <svg class="logo logo-{k} {klass}" {style} {viewBox} role="img" aria-label={name}>
    <title>{name}</title>
    <use href={`#m-${k}`} />
  </svg>
{/if}

<style>
  .metrale-defs { position: absolute; }
  .logo { display: block; height: auto; }
  /* Defaults for a caller that passes no width. The header passes 152 and the
     footer 244; both clear the 140 px floor for the wordmark. */
  .logo-wordmark { width: 152px; margin: calc(152px * 0.044); }
  .logo-mark { width: 32px; margin: calc(32px * 0.145); }
  .logo-compact { width: 16px; }
</style>
