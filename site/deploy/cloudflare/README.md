# Cloudflare Pages hosting

The four web properties are served by Cloudflare Pages. There is no origin
server in the request path.

## Projects

| Project | Serves | Built by |
| --- | --- | --- |
| `metrale-engine-site` | `dev.metrale.ai` | `.github/workflows/site.yml` |
| `metrale-engine-blog` | `blog.dev.metrale.ai` | `.github/workflows/site.yml` |
| `metrale-engine-book` | `book.dev.metrale.ai` (mdBook) | `.github/workflows/docs.yml` |
| `metrale-engine-docs` | `docs.dev.metrale.ai` (rustdoc) | `.github/workflows/docs.yml` |

All are **Direct Upload** projects, not Pages' git integration. The site build
needs a `metrale-recipes` checkout and a GitHub token, and it carries four
gates a Pages-native build would bypass — the flagship-recipe check, the
per-route `<title>` checks on both properties, and the blog/site cross-link
check. CI builds, CI uploads the gated output.

`--branch=main` on the upload is load-bearing: a deployment on any other branch
gets a preview URL and does not move the custom domain. That fails as "the
deploy went green and the site is stale".

## What replaced the nginx config

`../nginx/dev.metrale.ai.conf` (and the blog and book vhosts beside it) are
kept as the reference for an origin deployment and are linted by
`.github/scripts/assert-vhost-headers.py`. On Pages the same behaviour comes
from:

- **`static/_headers`** — the security headers and the cache policy. Read the
  note at the top of that file before editing it; Pages concatenates a
  re-declared header rather than replacing it, which silently cost the hashed
  assets their year-long cache once already.
- **`src/routes/404/+page.svelte`** — prerenders to `build/404.html`. Pages has
  no `try_files ... =404`; with no such file it answers every unmatched path
  with index.html and a **200**, so broken links return the front page and
  crawlers index unbounded soft-404s.

## Redirect rules

Path redirects live in `static/_redirects` and ship with the build; Pages
evaluates them before static assets. `/engine` (and its `.html` and trailing
slash spellings) is a 301 to `/`, where that page now lives.

`_redirects` host rules do not fire on these projects, and a hostname attached
to a Pages project never reaches the zone's ruleset engine. A hostname that
must redirect (for example a `www.` alias) therefore stays OFF the Pages
project, keeps a proxied DNS record, and is redirected by a zone-level
Redirect Rule. Writing such a rule over the API needs a token with
**Zone -> Dynamic Redirect -> Edit** on top of the Pages, DNS and Cache Purge
permissions; a token without it fails with `request is not authorized` on the
ruleset write while still listing rulesets.
