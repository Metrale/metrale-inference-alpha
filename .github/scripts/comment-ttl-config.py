#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-09-25: The single reader of the comment-ttl configuration files.

  comment-ttl-config.py ttl              print `comment_ttl` from .comment-ttl.toml
  comment-ttl-config.py enforced-regex   print the `scan_dirs` regex for the
                                         directories in .github/comment-ttl-enforced.txt
                                         (empty when none are enforced yet)
  comment-ttl-config.py check            refuse a configuration that would be
                                         silently wrong in CI

`check` refuses three things, each of which passes every run while being wrong:

  * a key in .comment-ttl.toml that the Action always overrides. The Action's
    inputs with a default (`fail_on_flag: "true"`, `reference_date: commit`,
    `scan_dirs: "."`, ...) reach the binary as set values, and a set input wins
    over the file, so such a key works in a local run and is dead in CI;
  * a third-party file ignored by .licenserc.yaml that exempt_paths does not
    exempt. The two lists name the same files and nothing else keeps them equal;
  * an enforced-directory entry that is missing, lacks its trailing `/`, or lies
    under an exempt path. Such an entry enforces nothing and still reads as done.

Options `--root`, `--config`, `--enforced` and `--licenserc` point it at other
files; the certification self-test uses them for its negative controls.
"""
import argparse
import fnmatch
import pathlib
import re
import subprocess
import sys
import tomllib

ACTION_DEFAULTED = ("fail_on_flag", "reference_date", "scan_dirs", "output_type",
                    "scope", "undated", "annotations")
REGEX_META = re.compile(r"([.\[\](){}*+?^$|\\])")


def load_toml(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def enforced_dirs(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    dirs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            dirs.append(line)
    return dirs


def enforced_regex(dirs: list[str]) -> str:
    if not dirs:
        return ""
    return "^(" + "|".join(REGEX_META.sub(r"\\\1", d) for d in dirs) + ")"


def tracked(root: pathlib.Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                         capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def licenserc_ignores(path: pathlib.Path) -> list[str]:
    import yaml  # 2026-09-25: only `check` needs it; `ttl` and `enforced-regex` do not

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    globs = (doc.get("header") or {}).get("paths-ignore") or []
    return [g for g in globs if not g.startswith("target/")]


def glob_match(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def check(ns: argparse.Namespace) -> list[str]:
    problems = []
    cfg = load_toml(ns.config)
    for key in ACTION_DEFAULTED:
        if key in cfg:
            problems.append(
                f"{ns.config.name} sets `{key}`, which the Action always overrides with its own "
                f"default; set it on the workflow step instead")
    if not cfg.get("comment_ttl"):
        problems.append(f"{ns.config.name} has no `comment_ttl`; the workflows read it from there")
    exempt = re.compile(cfg.get("exempt_paths") or r"(?!)")

    files = tracked(ns.root)
    for pattern in licenserc_ignores(ns.licenserc):
        hits = [p for p in files if glob_match(p, pattern)]
        leaks = [p for p in hits if not exempt.search(p)]
        if leaks:
            problems.append(
                f"third-party path `{pattern}` (.licenserc.yaml paths-ignore) is not exempt in "
                f"{ns.config.name}: {len(leaks)} file(s), e.g. {leaks[0]}")

    dirs = enforced_dirs(ns.enforced)
    tracked_dirs = {p.rsplit("/", 1)[0] + "/" for p in files if "/" in p}
    for d in dirs:
        if not d.endswith("/"):
            problems.append(f"enforced entry `{d}` must end in `/`")
        elif not any(t.startswith(d) for t in tracked_dirs):
            problems.append(f"enforced entry `{d}` holds no tracked file; it would enforce nothing")
        elif exempt.search(d):
            problems.append(f"enforced entry `{d}` lies under an exempt path; it would enforce nothing")
    if len(set(dirs)) != len(dirs):
        problems.append("an enforced entry is listed twice")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("ttl", "enforced-regex", "check"))
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--config", type=pathlib.Path)
    p.add_argument("--enforced", type=pathlib.Path)
    p.add_argument("--licenserc", type=pathlib.Path)
    ns = p.parse_args()
    ns.config = ns.config or ns.root / ".comment-ttl.toml"
    ns.enforced = ns.enforced or ns.root / ".github/comment-ttl-enforced.txt"
    ns.licenserc = ns.licenserc or ns.root / ".licenserc.yaml"

    if ns.command == "ttl":
        ttl = load_toml(ns.config).get("comment_ttl")
        if not ttl:
            print(f"{ns.config}: no comment_ttl", file=sys.stderr)
            return 1
        print(ttl)
        return 0
    if ns.command == "enforced-regex":
        print(enforced_regex(enforced_dirs(ns.enforced)))
        return 0
    problems = check(ns)
    for msg in problems:
        print(f"REFUSE: {msg}", file=sys.stderr)
    if problems:
        return 1
    print(f"ok: {ns.config.name} is consistent with .licenserc.yaml and "
          f"{len(enforced_dirs(ns.enforced))} enforced director(ies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
