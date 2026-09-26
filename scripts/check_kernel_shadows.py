#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Kernel Structure Enforcer: validate the kernels/ layout through the resolver.

The build (`crates/kernels/build.rs`) compiles what THE resolver
returns — `crates/closure/src/layout.rs`, mirrored for stdlib Python in
`scripts/lib/kernel_layout.py` and held to it by
`crates/kernels/tests/layout_mirror.rs`. This script runs the mirror
over every target and enforces the structure rules the resolver itself does not
decide, which `crates/kernels/tests/kernels_structure.rs` enforces
identically for `cargo test`:

  RULE 0 (no symlinks): sharing is `[sources] use` in a KERNEL.toml or
    `[hardware] inherits` in a HARDWARE.toml — declarations the resolver
    reads and a diff shows. A symlink is neither.

  RULE R (every target resolves): an undeclared shadow, a `[shadow]` entry
    with nothing to declare, a `use` naming no file, a redirect chain — the
    resolver refuses each, and this script reports the refusal.

  RULE 1 (dead override): a winning entry byte-identical to the entry it
    shadows overrides nothing while masking future changes to the original
    (shadowing is whole-file, not per-symbol — the shadow-drift failure class
    documented in build.rs). Delete it, or `use` the original.

  RULE 2 (duplicate regular file): two regular kernel files with the same
    name and the same bytes anywhere under kernels/ are a divergence-prone
    copy. One is canonical; the rest `use` it.

NOT CHECKED HERE — dropped entry points. A shadow that keeps its namesake's
name but declares FEWER kernels is the third defect of this family, and the one
that actually shipped (the 27B's four multi-sequence GDN decode kernels, gone
until 2026-07-26). Deciding it needs the entry points a source declares, which
means resolving `#define KERNEL_NAME` + `#include` + token-paste macros, and
then filtering by the per-target `[shadow_exempt]` tables. That resolver is
`crates/kernels/build_shadow.rs`, and it is enforced by
`crates/kernels/tests/kernel_shadow_detector.rs` in the same CI run as
this script.

Exit 0 when clean; exit 1 and list every violation otherwise.

Usage: scripts/check_kernel_shadows.py [path/to/kernels]
"""

import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import kernel_layout as kl  # noqa: E402

# Hardware trees this check CLAIMS to cover. Every one must exist: the loop
# below would otherwise scan nothing and print OK, which is how a renamed
# `kernels/gb10` once passed. Adding or retiring hardware edits this list in
# the same commit, which is exactly the moment to notice.
HW_TREES = ("b200", "b300", "gb10", "hopper", "metal", "strix", "strix-hip")
KERNEL_EXTS = (".cu", ".cuh", ".h", ".metal")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path) -> tuple[list[str], list[str]]:
    """Every violation, and one report line per overlay."""
    kernels = root / "kernels"
    violations: list[str] = []
    for link in kl.symlinks(root):
        violations.append(f"RULE0 {link.relative_to(kernels)}: symlink — declare it in [sources] use or inherit the tree")
    try:
        targets = kl.walk(root)
    except kl.LayoutError as e:
        return ([f"RULER {e}"], [])
    seen: set[tuple[str, str, str]] = set()
    overlays: dict[str, list[str]] = {}
    for hw, model, quant in targets:
        try:
            lay = kl.discover(root, hw, model, quant)
        except kl.LayoutError as e:
            violations.append(f"RULER {hw}/{model}/{quant}: {e}")
            continue
        for s in lay.shadows:
            key = (s.role, str(s.winner), str(s.loser))
            if key in seen:
                continue
            seen.add(key)
            if sha(s.winner) == sha(s.loser):
                violations.append(
                    f"RULE1 {hw}/{model}/{quant}: {s.winner.relative_to(kernels)} is byte-identical "
                    f"to {s.loser.relative_to(kernels)} it shadows (dead override)"
                )
        if lay.hardware.inherits and hw not in overlays:
            own = sorted(
                f"{n} ({'replaces' if any(x.role == 'common' and x.name == n for x in lay.shadows) else 'adds'})"
                for n, e in lay.common.items()
                if lay.layers[e.layer].tier == "own"
            )
            overlays[hw] = own
    by_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for dp, _dn, fn in os.walk(kernels):
        for name in fn:
            if name.endswith(KERNEL_EXTS):
                p = Path(dp) / name
                if not p.is_symlink():
                    by_key[(name, sha(p))].append(p)
    for (name, _), paths in sorted(by_key.items()):
        if len(paths) > 1:
            violations.append(
                f"RULE2 {len(paths)} identical regular copies of {name} — keep one canonical file, `use` it from the rest:\n    "
                + "\n    ".join(str(p.relative_to(kernels)) for p in sorted(paths))
            )
    report = []
    for hw, own in sorted(overlays.items()):
        parent = kl.hardware(kernels, hw).inherits
        if own:
            report.append(f"  {hw} is an overlay of {parent} and holds {len(own)} common/ file(s) of its own "
                          f"— `replaces` = {parent} has the same name, `adds` = it does not:")
            report.extend(f"      {e}" for e in own)
        else:
            report.append(f"  {hw} is an overlay of {parent} and holds no common/ file of its own")
    return (violations, report)


def main() -> int:
    kernels_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kernels")
    if not kernels_root.is_dir():
        print(f"error: kernels root not found: {kernels_root}", file=sys.stderr)
        return 1
    # The resolver addresses the tree as `<root>/kernels`, the way build.rs
    # and the gate do, so the directory must carry that name.
    if kernels_root.resolve().name != "kernels":
        print(f"error: {kernels_root} must be a directory named `kernels`", file=sys.stderr)
        return 1
    missing = [hw for hw in HW_TREES if not (kernels_root / hw).is_dir()]
    if missing:
        print(
            f"error: {kernels_root}/ is missing hardware tree(s): {', '.join(missing)}.\n"
            f"       They are listed in HW_TREES, so this check believes it covers\n"
            f"       them -- and it silently scanned nothing instead. If a tree moved or\n"
            f"       was retired, update HW_TREES in the same commit.",
            file=sys.stderr,
        )
        return 1
    root = kernels_root.resolve().parent
    violations, report = check(root)
    if violations:
        print(f"kernel shadow structure: {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v}")
        return 1
    print(f"kernel shadow structure: OK ({len(HW_TREES)} hardware trees scanned: {', '.join(HW_TREES)})")
    # Which kernels each overlay owns, on a clean run: the point of an overlay
    # holding only its own files is that this question has a short answer.
    for line in report:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
