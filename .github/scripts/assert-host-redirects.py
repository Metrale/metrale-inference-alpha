#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The two retired hosts must publish nothing but their redirects.

dev.metrale.ai and blog.dev.metrale.ai are retired. Each is a Cloudflare Pages
project whose whole deployment is one file, a `_redirects`:

  site/_redirects  dev.metrale.ai       -> metrale.ai, the same path where
                                           metrale.ai serves one, else /engine
  blog/_redirects  blog.dev.metrale.ai  -> blog.metrale.ai, every path kept

Static mode (the default) checks the tree:

  1. site/ and blog/ hold `_redirects` and nothing else. Any other file would
     be uploaded next to it, which is how a retired app gets published again.
  2. Every rule is `source target 301`, and every target is on the new host.
  3. site/: every rule but the last keeps its path (a `.html` spelling may drop
     the extension, `*` becomes `:splat`), and the last is the catch-all to
     https://metrale.ai/engine. Pages takes the first match, so a rule after the
     catch-all could never fire.
  4. blog/: exactly the one path-keeping rule.

`--live` asks the deployed hosts instead: each rule answers 301 with the
target it names, and each target answers 200.
"""
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
CATCH_ALL = "/*"
ENGINE_PAGE = "https://metrale.ai/engine"
# tree -> (retired origin, new origin)
HOSTS = {
    "site": ("https://dev.metrale.ai", "https://metrale.ai"),
    "blog": ("https://blog.dev.metrale.ai", "https://blog.metrale.ai"),
}
# Paths the catch-alls must also answer for; each names where it must land.
LIVE_SAMPLES = {
    "site": [("/", ENGINE_PAGE), ("/engine", ENGINE_PAGE), ("/quickstart.sh", ENGINE_PAGE),
             ("/fonts/type.css", "https://metrale.ai/fonts/type.css")],
    "blog": [("/", "https://blog.metrale.ai/"),
             ("/posts/seven-tenets-powering-metrale-inference",
              "https://blog.metrale.ai/posts/seven-tenets-powering-metrale-inference")],
}

problems: list[str] = []


def rules(tree: str) -> list[tuple[int, str, str, str]]:
    out = []
    text = (ROOT / tree / "_redirects").read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            problems.append(f"{tree}/_redirects:{lineno} is not `source target status`: {line}")
            continue
        out.append((lineno, *fields))
    return out


def same_path(source: str) -> str:
    path = source.replace("*", ":splat")
    return path[: -len(".html")] if path.endswith(".html") else path


def check_tree(tree: str) -> list[tuple[str, str]]:
    """Return (source, target) pairs, recording every defect in `problems`."""
    extra = sorted(str(p.relative_to(ROOT)) for p in (ROOT / tree).rglob("*")
                   if p.is_file() and p.name != "_redirects")
    for rel in extra:
        problems.append(f"{rel} would be published beside the redirects; {tree}/ holds _redirects only")
    if not (ROOT / tree / "_redirects").is_file():
        problems.append(f"{tree}/_redirects is missing")
        return []

    new = HOSTS[tree][1]
    found = rules(tree)
    seen: set[str] = set()
    for lineno, source, target, status in found:
        where = f"{tree}/_redirects:{lineno}"
        if status != "301":
            problems.append(f"{where} answers {status}; a retired host redirects permanently (301)")
        if not target.startswith(new + "/"):
            problems.append(f"{where} sends {source} to {target}, which is not on {new}")
        if source in seen:
            problems.append(f"{where} repeats {source}; only the first match is ever used")
        seen.add(source)

    if not found:
        problems.append(f"{tree}/_redirects has no rules")
        return []
    *paths, last = found
    if last[1] != CATCH_ALL:
        problems.append(f"{tree}/_redirects must end with the {CATCH_ALL} catch-all, not {last[1]}")
    for lineno, source, _target, _status in paths:
        if source == CATCH_ALL:
            problems.append(f"{tree}/_redirects:{lineno} is a catch-all before the last line; "
                            f"every rule below it is dead")

    if tree == "site":
        if last[2] != ENGINE_PAGE:
            problems.append(f"site/_redirects sends everything else to {last[2]}, not {ENGINE_PAGE}")
        for lineno, source, target, _status in paths:
            if target != new + same_path(source):
                problems.append(f"site/_redirects:{lineno} moves {source} to {target}; "
                                f"a path metrale.ai serves keeps its path")
    else:
        if paths or last[2] != f"{new}/:splat":
            problems.append(f"blog/_redirects must be the single rule `/* {new}/:splat 301`")
    return [(s, t) for _l, s, t, _st in found]


class NoFollow(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch(url: str, follow: bool) -> tuple[int, str]:
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoFollow)
    req = urllib.request.Request(url, headers={"User-Agent": "metrale-host-redirects-check"})
    try:
        with opener.open(req, timeout=20) as res:
            return res.status, res.headers.get("Location", "")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location", "")
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, str(e)


def check_live(tree: str, pairs: list[tuple[str, str]]) -> None:
    old = HOSTS[tree][0]
    probes = [(s, t) for s, t in pairs if "*" not in s]
    for path, want in probes + LIVE_SAMPLES[tree]:
        code, location = fetch(old + path, follow=False)
        if code != 301 or location != want:
            problems.append(f"{old}{path} answered {code} {location or '(no Location)'}, want 301 {want}")
            continue
        code, _ = fetch(want, follow=True)
        if code != 200:
            problems.append(f"{old}{path} redirects to {want}, which answers {code}")
        else:
            print(f"ok   {old}{path} -> 301 {want} -> 200")


def main() -> None:
    live = "--live" in sys.argv[1:]
    for tree in HOSTS:
        pairs = check_tree(tree)
        if live and pairs and not problems:
            check_live(tree, pairs)
    if problems:
        for p in problems:
            print(f"REFUSE: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: site/ and blog/ publish only their redirects{' and the deployed hosts agree' if live else ''}")


if __name__ == "__main__":
    main()
