#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every binary we distribute carries the licence texts and third-party notices.

CUTLASS is BSD-3-Clause, and BSD-3 requires its notice in binary distributions,
so the licence files are part of what a Docker image or a release archive IS.
Each packager copies them explicitly, which means a later edit can drop one
line and ship a non-compliant binary with every job still green. This pins
that line in every place a binary is packaged:

  D1. The final stage of every Dockerfile copies all of NOTICES in. A final
      stage built FROM a local `metrale-*` image is an overlay on an image a
      checked Dockerfile produced, and inherits the files from it.
  D2. The ignore file that governs each Dockerfile does not exclude any of
      them from the build context -- a COPY of an ignored file fails the build
      only when someone runs it, and nobody runs all of these in CI.
  W1. Every workflow job that uploads a `met-*` artifact names all of NOTICES.
  W2. Every workflow step that copies LICENSE-MIT copies the rest too, so a new
      packager that remembers the licence but not the notices is caught.

If a packager is added that none of these recognise, the fix is to teach this
script the new shape -- not to let the packager ship without the notices.
"""
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTICES = ("LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY_NOTICES.md", "LICENSES")
# Workflows known to package the binary. If a rule stops seeing a packager in
# one of them, the rule went blind -- that is a failure, not a pass.
KNOWN_PACKAGERS = ("release-build.yml", "datacenter-binaries.yml")
OVERLAY_BASE = re.compile(r"^metrale-[\w.-]+(:[\w.-]+)?$")


def final_stage(dockerfile: pathlib.Path) -> tuple[str, list[str]]:
    """(base image, COPY source lists) of the last stage, continuations joined."""
    text = re.sub(r"\\\n", " ", dockerfile.read_text(encoding="utf-8"))
    base, copies = "", []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        op = words[0].upper()
        if op == "FROM":
            args = [w for w in words[1:] if not w.startswith("--")]
            base, copies = args[0], []
        elif op in ("COPY", "ADD") and not any(w.startswith("--from") for w in words):
            args = [w for w in words[1:] if not w.startswith("--")]
            copies.append(args[:-1])
    return base, copies


def copied(sources: list[list[str]], item: str) -> bool:
    return any(s.rstrip("/") == item for srcs in sources for s in srcs)


def ignore_regex(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def excluded(rel: str, patterns: list[str]) -> bool:
    """Docker's rule: the last pattern matching the path or a parent wins."""
    parts = rel.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    verdict = False
    for raw in patterns:
        neg = raw.startswith("!")
        pat = raw[1:] if neg else raw
        pat = pat.strip().strip("/")
        if pat and any(ignore_regex(pat).match(c) for c in candidates):
            verdict = not neg
    return verdict


def ignore_patterns(dockerfile: pathlib.Path) -> tuple[str, list[str]]:
    own = dockerfile.with_name(dockerfile.name + ".dockerignore")
    src = own if own.exists() else ROOT / ".dockerignore"
    if not src.exists():
        return "(none)", []
    lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines()]
    return str(src.relative_to(ROOT)), [ln for ln in lines if ln and not ln.startswith("#")]


def check_dockerfiles(errors: list[str]) -> int:
    notice_files = [f for f in NOTICES if f != "LICENSES"]
    notice_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "LICENSES").iterdir())]
    checked = 0
    candidates = sorted(ROOT.glob("Dockerfile*")) + sorted((ROOT / "docker").rglob("Dockerfile*"))
    for df in candidates:
        if df.suffix == ".dockerignore" or not df.is_file():
            continue
        rel = str(df.relative_to(ROOT))
        base, copies = final_stage(df)
        if OVERLAY_BASE.match(base):
            continue
        checked += 1
        for item in NOTICES:
            if not copied(copies, item):
                errors.append(f"D1 {rel}: the final stage (FROM {base}) does not COPY {item}")
        src, patterns = ignore_patterns(df)
        for f in notice_files:
            if excluded(f, patterns):
                errors.append(f"D2 {rel}: {src} excludes {f} from the build context")
    return checked


def job_text(job: dict) -> str:
    return yaml.safe_dump(job.get("steps") or [], sort_keys=False)


def check_workflows(errors: list[str]) -> set[str]:
    packagers: set[str] = set()
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(wf, dict):
            continue
        for key, job in (wf.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                run = str(step.get("run", ""))
                if "LICENSE-MIT" in run:
                    missing = [n for n in NOTICES if n not in run]
                    if missing:
                        errors.append(f"W2 {path.name}:{key}: a step copies LICENSE-MIT but not {missing}")
                uses = str(step.get("uses", ""))
                name = str((step.get("with") or {}).get("name", ""))
                if uses.startswith("actions/upload-artifact") and name.startswith("met-"):
                    packagers.add(path.name)
                    text = job_text(job)
                    missing = [n for n in NOTICES if n not in text]
                    if missing:
                        errors.append(f"W1 {path.name}:{key}: uploads {name} without {missing}")
    return packagers


def main() -> int:
    errors: list[str] = []
    checked = check_dockerfiles(errors)
    if checked == 0:
        errors.append("D1 no Dockerfile was checked; the glob no longer finds them")
    packagers = check_workflows(errors)
    for wf in KNOWN_PACKAGERS:
        if wf not in packagers:
            errors.append(f"W1 {wf}: no `met-*` upload found; teach this script the packager's new shape")
    if errors:
        for e in errors:
            print(f"REFUSE: {e}", file=sys.stderr)
        print(f"\n{len(errors)} binary distribution(s) would ship without the notices.", file=sys.stderr)
        return 1
    print(f"ok: {checked} Dockerfiles and {len(packagers)} release workflows ship {', '.join(NOTICES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
