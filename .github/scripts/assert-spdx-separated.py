#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-09-26: Refuse anything glued to a file's SPDX line.
#
# comment-ttl merges consecutive line comments with the same marker into one
# comment, and a comment that starts with an SPDX line is a licence comment,
# which it never judges. Text written in the same block as the SPDX line is
# therefore never dated or checked. For every tracked file whose first line
# carries `SPDX-License-Identifier`, the lines after it may be a run of
# metadata lines (`provenance-id:`, `forked-from:`, `SPDX-FileCopyrightText`,
# each as a `//` or `#` comment), and the next line must be blank, the end of
# the file, or a line that is not a comment in the SPDX line's own marker (code
# such as `#![allow(...)]` or `[package]` cannot join the licence comment). A
# comment line there (prose, a bare `//`) is printed as `path:line`, and the
# exit status is 1.
#
# Paths matching `exempt_paths` in .comment-ttl.toml are skipped; the regex is
# read through comment-ttl-config.py, the reader of that file.
#
#   assert-spdx-separated.py [--root DIR]    DIR defaults to the current directory

import argparse
import importlib.util
import pathlib
import re
import subprocess
import sys

MARK = "SPDX-License-Identifier"
METADATA = re.compile(r"^\s*(//|#)\s*(provenance-id:|forked-from:|SPDX-FileCopyrightText)")


def load_config_reader():
    path = pathlib.Path(__file__).resolve().with_name("comment-ttl-config.py")
    spec = importlib.util.spec_from_file_location("comment_ttl_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracked(root: pathlib.Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                         capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def comment_markers(spdx_line: str) -> tuple[str, ...]:
    """The line prefixes that continue the comment the SPDX line opens."""
    marker = spdx_line.split(MARK, 1)[0].strip()
    return (marker, "*") if marker == "/*" else (marker,)


def glued_line(lines: list[str]) -> int | None:
    """The 1-based line after the SPDX line and its metadata when it is a comment, or None."""
    markers = comment_markers(lines[0])
    for n, line in enumerate(lines[1:], start=2):
        if METADATA.match(line):
            continue
        body = line.strip()
        return n if body and body.startswith(markers) else None
    return None


def offenders(root: pathlib.Path, exempt: re.Pattern) -> list[str]:
    found = []
    for rel in tracked(root):
        if exempt.search(rel):
            continue
        path = root / rel
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as f:
            first = f.readline()
            if MARK.encode() not in first:
                continue
            text = (first + f.read()).decode("utf-8", errors="replace")
        n = glued_line(text.splitlines())
        if n is not None:
            found.append(f"{rel}:{n}")
    return found


def main() -> int:
    p = argparse.ArgumentParser(description="Refuse anything glued to a file's SPDX line.")
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    ns = p.parse_args()
    reader = load_config_reader()
    config = ns.root / ".comment-ttl.toml"
    exempt_src = reader.load_toml(config).get("exempt_paths")
    if not exempt_src:
        print(f"{config}: no exempt_paths to read", file=sys.stderr)
        return 2
    found = offenders(ns.root, re.compile(exempt_src))
    for entry in found:
        print(entry)
    if found:
        print(f"{len(found)} file(s) continue the SPDX comment past its "
              f"provenance-id:/forked-from:/SPDX-FileCopyrightText lines", file=sys.stderr)
        return 1
    print("ok: no comment text is glued to an SPDX line or its metadata lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
