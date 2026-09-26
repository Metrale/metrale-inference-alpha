#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Every GPU-library symbol a crate declares is stubbed for the no-GPU runners.

The `cargo test --workspace` and coverage jobs link against the fail-fast
libraries `scripts/ci_gpu_stubs.sh` generates, not the real libcuda, libnccl,
libcublasLt and libcudart. A crate that declares a new driver entry point in an
`extern "C"` block links on a GPU box and breaks the no-GPU link with
`undefined symbol`, which is seen only after the change lands. This reads every
`extern "C" { ... }` block under crates/ and every `int name(` definition in the
stub script, and refuses any declared symbol its library's stub does not define.

A declaration is skipped only when its `#[cfg(...)]` (on the item or on its
block) is false on every job that uses the stubs. The only cfg known false
there is `metrale_scale`: build.rs sets it for METRALE_TARGET_HW=strix*, and
the jobs that run ci_gpu_stubs.sh never set that variable. Any other cfg is
treated as possibly true, so its symbols must be stubbed.

  assert-gpu-stubs-cover-ffi.py [--root DIR]    DIR defaults to the current directory
"""
import argparse
import pathlib
import re
import sys

STUB_SCRIPT = pathlib.Path("scripts/ci_gpu_stubs.sh")
# Stub C file written by ci_gpu_stubs.sh -> which declarations it must satisfy.
LIBRARIES = {
    "libcuda_stub.c": re.compile(r"^cu(?!da[A-Z]|blas|File)[A-Z]"),
    "libnccl_stub.c": re.compile(r"^nccl[A-Z]"),
    "libcublaslt_stub.c": re.compile(r"^cublasLt[A-Z]"),
    "libcudart_stub.c": re.compile(r"^cuda[A-Z]"),
}
FALSE_ON_STUB_JOBS = {"metrale_scale"}

HEREDOC = re.compile(r"cat > /tmp/(\S+) <<'EOF'\n(.*?)\nEOF\n", re.S)
STUB_DEF = re.compile(r"^int\s+([A-Za-z_]\w*)\s*\(", re.M)
BLOCK = re.compile(r"extern\s+\"C\"\s*\{")
ITEM_FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*[<(]")
LINK_NAME = re.compile(r"#\[link_name\s*=\s*\"([^\"]+)\"\]")


def cfg_may_hold(expr: str) -> bool:
    """False only when `expr` is provably false given FALSE_ON_STUB_JOBS."""
    expr = expr.strip()
    m = re.fullmatch(r"(not|all|any)\((.*)\)", expr, re.S)
    if not m:
        return expr not in FALSE_ON_STUB_JOBS
    op, inner = m.groups()
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    if cur.strip():
        parts.append(cur)
    if op == "not":
        # not(x) is false only when x is provably true; nothing is known true.
        return True
    held = [cfg_may_hold(p) for p in parts]
    return all(held) if op == "all" else any(held)


def attributes_before(text: str, end: int) -> str:
    """The attribute run immediately preceding `end` (text between the last
    item boundary and `end`)."""
    start = max(text.rfind(";", 0, end), text.rfind("{", 0, end), text.rfind("}", 0, end))
    return text[start + 1:end]


def cfgs_hold(attrs: str) -> bool:
    return all(cfg_may_hold(c) for c in re.findall(r"#\[cfg\((.*?)\)\]", attrs, re.S))


def extern_blocks(text: str):
    for m in BLOCK.finditer(text):
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            i += 1
        yield m.start(), m.end(), text[m.end():i - 1]


def declared(root: pathlib.Path) -> dict[str, list[str]]:
    """symbol -> ["path:line", ...] for every live GPU-library declaration."""
    found: dict[str, list[str]] = {}
    for path in sorted((root / "crates").rglob("*.rs")):
        if "target" in path.relative_to(root).parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for start, offset, body in extern_blocks(text):
            head = text[:start].rstrip()
            head = head.removesuffix("unsafe").rstrip()
            if not cfgs_hold(attributes_before(head, len(head))):
                continue
            for f in ITEM_FN.finditer(body):
                attrs = attributes_before(body, f.start())
                if not cfgs_hold(attrs):
                    continue
                link = LINK_NAME.search(attrs)
                name = link.group(1) if link else f.group(1)
                if not any(rx.match(name) for rx in LIBRARIES.values()):
                    continue
                line = text.count("\n", 0, offset + f.start()) + 1
                found.setdefault(name, []).append(f"{path.relative_to(root)}:{line}")
    return found


def stubbed(root: pathlib.Path) -> dict[str, set[str]]:
    """stub C file -> symbols it defines."""
    script = (root / STUB_SCRIPT).read_text(encoding="utf-8")
    out = {name: set(STUB_DEF.findall(body)) for name, body in HEREDOC.findall(script)}
    missing = sorted(set(LIBRARIES) - set(out))
    if missing:
        raise SystemExit(f"{STUB_SCRIPT} no longer writes {', '.join(missing)}; "
                         f"this check cannot read the stubs, so it refuses rather than pass.")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    root = p.parse_args().root
    stubs = stubbed(root)
    decls = declared(root)
    if not decls:
        print("found no GPU-library declarations under crates/; the scan read nothing.")
        return 1
    bad = 0
    for name in sorted(decls):
        lib = next(f for f, rx in LIBRARIES.items() if rx.match(name))
        if name not in stubs[lib]:
            bad += 1
            print(f"{name} is not stubbed in {STUB_SCRIPT} ({lib}); declared at {', '.join(decls[name])}")
    if bad:
        print(f"{bad} declared symbol(s) would be undefined when the no-GPU jobs link. "
              f"Add an `int <name>(...)` returning the library's error code to the "
              f"matching heredoc in {STUB_SCRIPT}.")
        return 1
    print(f"ok: all {len(decls)} GPU-library symbols declared under crates/ are stubbed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
