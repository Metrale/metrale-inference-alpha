#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every public vhost must carry the same security headers, at server level.

nginx's `add_header` does not accumulate across contexts: a `location` that
declares ANY `add_header` silently discards every one inherited from the server
block. That single rule has now bitten this repo three times, in three files,
and each time it was found by hand after the fact:

  * book.dev.metrale.ai served every HTML document with no
    X-Content-Type-Options, X-Frame-Options or Referrer-Policy, because
    Cache-Control was declared inside `location ~* \\.html$`.
  * the engine site's vhost discarded Alt-Svc on every proxied response, and
    its `location ~ /\\.` dotfile refusal went out with no security headers.
  * that vhost was also simply missing Referrer-Policy, which the other two
    sent -- drift nobody was watching for.

All three are fixed. The site and blog vhosts have since been retired with
their hosts; this keeps a fourth from happening in the ones that remain.

What is pinned:

  1. Every header in CORE is declared at SERVER level in every vhost. Not
     "somewhere in the file" -- inside a location it protects one path and
     silently unprotects the rest.
  2. No `add_header` anywhere inside a `location` block. This is the trap
     itself, and it is worth refusing outright rather than trying to reason
     about which locations would be safe: a config that never does it cannot
     be got wrong later. Per-path values belong in a `map`, which is how the
     docs vhost computes Cache-Control from a single server-level `add_header`.
     Because every vhost must carry all of CORE, the sets cannot drift apart:
     drift is how the missing Referrer-Policy survived, two vhosts having it
     and one not, with no single file wrong on its own.
  3. X-XSS-Protection, if present at all, is exactly "0". The header is
     deprecated; every current browser has removed the auditor it controlled,
     and the OWASP Secure Headers Project recommends "0" precisely because
     the legacy filter it re-enables is itself exploitable (XS-Leaks, and
     selective script-blocking on pages that are otherwise safe).
     `1; mode=block` is the value to refuse, and it was live on the engine
     site when this was written.

Deliberately NOT pinned: Strict-Transport-Security. No vhost sends it,
which is a real gap -- but HSTS is a commitment a browser caches for max-age,
and choosing that value (and whether to preload) is a policy decision for a
person, not a default a linter should install.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
VHOSTS = [
    "book/deploy/nginx/book.dev.metrale.ai.conf",
]
CORE = ["X-Frame-Options", "X-Content-Type-Options", "Referrer-Policy"]
ADD_HEADER = re.compile(r"^\s*add_header\s+([A-Za-z0-9-]+)\s+(.*?);\s*$")

problems: list[str] = []


def parse(path: pathlib.Path):
    """Return (server-level headers, headers found inside a location).

    Comments are stripped before brace counting: this file's own prose talks
    about `add_header` and braces, and counting those would desynchronise the
    depth tracker against the real config.
    """
    server: dict[str, str] = {}
    in_location: list[tuple[int, str]] = []
    depth = 0
    loc_depth = None
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        code = raw.split("#", 1)[0]
        if re.search(r"^\s*location\b", code):
            loc_depth = depth
        m = ADD_HEADER.match(code)
        if m:
            if loc_depth is not None and depth > loc_depth:
                in_location.append((lineno, code.strip()))
            else:
                server[m.group(1)] = m.group(2).strip()
        depth += code.count("{") - code.count("}")
        if loc_depth is not None and depth <= loc_depth:
            loc_depth = None
    return server, in_location


def main() -> None:
    for rel in VHOSTS:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel} is missing; this guard is pinned to a vhost that no longer exists")
            continue
        server, in_location = parse(path)

        for lineno, text in in_location:
            problems.append(
                f"{rel}:{lineno} declares an add_header inside a location: {text}\n"
                f"         nginx drops every inherited add_header for that location. "
                f"Hoist it to server level and use a map for per-path values."
            )

        for header in CORE:
            if header not in server:
                problems.append(f"{rel} does not declare {header} at server level")

        xss = server.get("X-XSS-Protection")
        if xss is not None and xss.split()[0].strip('"') != "0":
            problems.append(
                f"{rel} sets X-XSS-Protection {xss}. The header is deprecated and the OWASP "
                f'Secure Headers Project recommends "0"; re-enabling the legacy auditor is '
                f"itself exploitable."
            )

    if problems:
        for p in problems:
            print(f"REFUSE: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: {len(VHOSTS)} vhosts declare {', '.join(CORE)} at server level, none inside a location")


if __name__ == "__main__":
    main()
