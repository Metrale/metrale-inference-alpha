#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every comment command the bot accepts must be documented in the book's
"How a Change Lands" page (book/src/project/landing.md).

Three of the five were not. `/help`, `/review` and `/expedite` existed in the
handler and appeared nowhere a user would look -- `/expedite` in particular is
an administrative override that skips certification, which is the last command
that should be discoverable only by reading a workflow file. All three were
added during this repository's own recent work, which is the point: the drift
was introduced by the same hands that wrote the docs, one commit at a time, and
no single change looked like it was leaving something out.

The verb list is read from the handler's own `case` statement rather than
restated here. A second hand-maintained list would drift from the first exactly
the way the docs drifted from the handler.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
HANDLER = ROOT / ".github" / "workflows" / "certification-commands.yml"
DOC = ROOT / "book" / "src" / "project" / "landing.md"

# the `case` arm that whitelists accepted verbs, e.g. `/help|/stamp|/seal) ;;`
CASE = re.compile(r"^\s*(/[a-z]+(?:\|/[a-z]+)+)\)\s*;;", re.M)


def main() -> None:
    m = CASE.search(HANDLER.read_text())
    if not m:
        print(
            "REFUSE: could not find the verb whitelist in certification-commands.yml. "
            "This guard reads the handler's own `case` arm so the two cannot drift; "
            "if the parser moved, teach this regex rather than hard-coding a list.",
            file=sys.stderr,
        )
        sys.exit(1)

    verbs = m.group(1).split("|")
    doc = DOC.read_text()
    missing = [v for v in verbs if v not in doc]

    if missing:
        for v in missing:
            print(f"REFUSE: the bot accepts `{v}` and book/src/project/landing.md never mentions it", file=sys.stderr)
        print(
            f"\n{len(missing)} of {len(verbs)} commands are undocumented. A command a user "
            f"cannot discover is a command that does not exist for them.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"ok: all {len(verbs)} accepted commands ({', '.join(verbs)}) are documented in book/src/project/landing.md")


if __name__ == "__main__":
    main()
