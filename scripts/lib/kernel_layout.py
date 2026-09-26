#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Mirror of `crates/closure/src/layout.rs` for the stdlib-Python checkers.

The Rust module is THE resolver: `metrale-kernels/build.rs` compiles what it
returns and the benchmark gate hashes what it returns. The Python scripts
(`check_kernel_shadows.py`, `check_cross_hardware.py`, `hopper_ptx_gate.sh`)
run where cargo does not, so they read this mirror, and
`crates/kernels/tests/layout_mirror.rs` holds the two to the same
answer on the real tree (`python3 scripts/lib/kernel_layout.py dump`).

Rules, in one paragraph (the Rust module's doc has the long form): a target
reads up to four layer directories — own leaf `<hw>/<source model>/<quant>`,
parent leaf, own `<hw>/common`, parent common — where the parent is
`[hardware] inherits` and the source model is `[model] kernel_source` or the
model itself. A layer holds its regular kernel files plus its KERNEL.toml
`[sources] use` (paths relative to that layer's `kernels/<hw>`). Own beats
parent by name within a role; leaf beats common by stem. Every such win is
declared in the winner's `[shadow]`. Symlinks are refused.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HEADER_EXTS = ("cuh", "h")


class LayoutError(Exception):
    """A tree the resolver refuses. The message names the file."""


def source_ext(vendor: str) -> str | None:
    if vendor in ("nvidia", "cuda", "amd", "rocm", "scale", "hip"):
        return "cu"
    if vendor in ("apple", "metal"):
        return "metal"
    return None


def stem_of(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def _has_ext(name: str, ext: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1] == ext


def _is_kernel_file(name: str, ext: str) -> bool:
    return _has_ext(name, ext) or any(_has_ext(name, h) for h in HEADER_EXTS)


def _read_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise LayoutError(f"{path}: {e}") from e


@dataclass(frozen=True)
class Hardware:
    name: str
    vendor: str
    arch: str | None
    inherits: str | None
    source_ext: str


def _hardware_raw(kernels: Path, hw: str) -> Hardware:
    path = kernels / hw / "HARDWARE.toml"
    if not path.is_file():
        raise LayoutError(f"kernels/{hw}: no HARDWARE.toml")
    v = _read_toml(path)
    if "overrides" in v.get("kernels", {}):
        raise LayoutError(f"{path}: `[kernels] overrides` is retired; declare replacements in [shadow]")
    h = v.get("hardware", {})
    vendor = h.get("vendor", "nvidia")
    ext = source_ext(vendor)
    if ext is None:
        raise LayoutError(f"{path}: vendor {vendor!r} has no kernel source extension")
    for key in ("arch", "inherits"):
        if key in h and not isinstance(h[key], str):
            raise LayoutError(f"{path}: hardware.{key} must be a string")
    return Hardware(hw, vendor, h.get("arch"), h.get("inherits"), ext)


def hardware(kernels: Path, hw: str) -> Hardware:
    own = _hardware_raw(kernels, hw)
    if own.inherits is not None:
        if own.inherits == hw:
            raise LayoutError(f"kernels/{hw}/HARDWARE.toml: a tree cannot inherit itself")
        parent = _hardware_raw(kernels, own.inherits)
        if parent.inherits is not None:
            raise LayoutError(f"kernels/{hw}/HARDWARE.toml: {own.inherits!r} itself inherits — chains are refused")
        if parent.source_ext != own.source_ext:
            raise LayoutError(f"kernels/{hw}/HARDWARE.toml: parent compiles .{parent.source_ext}, this tree .{own.source_ext}")
    return own


def _kernel_source(model_dir: Path) -> str | None:
    path = model_dir / "MODEL.toml"
    if not path.is_file():
        return None
    src = _read_toml(path).get("model", {}).get("kernel_source")
    if src is None:
        return None
    if not isinstance(src, str) or not src.strip():
        raise LayoutError(f"{path}: [model] kernel_source must name a kernel target directory")
    return src


def kernel_source_dir(model_dir: Path) -> Path:
    src = _kernel_source(model_dir)
    if src is None:
        return model_dir
    src_dir = model_dir.parent / src
    if not (src_dir.is_dir() and (src_dir / "MODEL.toml").is_file()):
        raise LayoutError(f"{model_dir}/MODEL.toml: kernel_source {src!r} does not name a kernel target directory")
    if _kernel_source(src_dir) is not None:
        raise LayoutError(f"{model_dir}/MODEL.toml: kernel_source {src!r} itself redirects — chains are not allowed")
    return src_dir


@dataclass
class KernelManifest:
    path: Path
    uses: list[str]
    shadow: dict[str, str]


def kernel_manifest(d: Path) -> KernelManifest | None:
    path = d / "KERNEL.toml"
    if not path.is_file():
        return None
    v = _read_toml(path)
    uses: list[str] = []
    if "sources" in v:
        s = v["sources"]
        if not isinstance(s, dict):
            raise LayoutError(f"{path}: [sources] must be a table")
        for k in s:
            if k != "use":
                raise LayoutError(f"{path}: [sources] has no key `{k}`; only `use`")
        if "use" in s:
            if not isinstance(s["use"], list) or not all(isinstance(x, str) for x in s["use"]):
                raise LayoutError(f"{path}: [sources] use must be an array of path strings")
            uses = list(s["use"])
    shadow: dict[str, str] = {}
    if "shadow" in v:
        t = v["shadow"]
        if not isinstance(t, dict):
            raise LayoutError(f"{path}: [shadow] must be a table of `stem = \"reason\"`")
        for stem, reason in t.items():
            if not isinstance(reason, str):
                raise LayoutError(f"{path}: [shadow] {stem} must be a reason string")
            if not reason.strip():
                raise LayoutError(f"{path}: [shadow] {stem} needs a stated reason")
            shadow[stem] = reason
    return KernelManifest(path, uses, shadow)


@dataclass
class Layer:
    role: str  # "leaf" | "common"
    tier: str  # "own" | "parent"
    hardware: str
    dir: Path
    manifest: KernelManifest | None


@dataclass
class Entry:
    name: str
    source: Path
    layer: int
    used: bool


@dataclass
class Shadow:
    role: str
    name: str
    winner: Path
    loser: Path
    reason: str


@dataclass
class Layout:
    hardware: Hardware
    target: tuple[str, str, str]
    layers: list[Layer]
    leaf: dict[str, Entry] = field(default_factory=dict)
    common: dict[str, Entry] = field(default_factory=dict)
    leaf_subdirs: dict[str, Path] = field(default_factory=dict)
    common_subdirs: dict[str, Path] = field(default_factory=dict)
    shadows: list[Shadow] = field(default_factory=list)
    source_model: str = ""
    model_dir: Path = Path()

    def configs(self) -> list[Path]:
        order = sorted(self.layers, key=lambda l: (l.role != "common", l.tier != "parent"))
        return [l.manifest.path for l in order if l.manifest is not None]

    def modules(self) -> dict[str, Entry]:
        ext = self.hardware.source_ext
        out: dict[str, Entry] = {}
        for m in (self.common, self.leaf):
            for name, e in m.items():
                if _has_ext(name, ext):
                    out[stem_of(name)] = e
        return dict(sorted(out.items()))

    def sources(self) -> list[Path]:
        return sorted(e.source for e in self.modules().values())

    def inputs(self) -> set[Path]:
        out = {e.source for e in list(self.leaf.values()) + list(self.common.values())}
        for d in list(self.leaf_subdirs.values()) + list(self.common_subdirs.values()):
            for dp, _dn, fn in os.walk(d):
                out.update(Path(dp) / f for f in fn)
        out.update(self.configs())
        return out


def _subdirs(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _leaf_model_dir(hw_dir: Path, model: str) -> Path:
    d = hw_dir / model
    return kernel_source_dir(d) if (d / "MODEL.toml").is_file() else d


def walk(root: Path) -> list[tuple[str, str, str]]:
    kernels = root / "kernels"
    out = []
    for hw in _subdirs(kernels):
        hw_dir = kernels / hw
        if not (hw_dir / "HARDWARE.toml").is_file():
            continue
        h = hardware(kernels, hw)
        for model in _subdirs(hw_dir):
            model_dir = hw_dir / model
            if not (model_dir / "MODEL.toml").is_file():
                continue
            own_src = kernel_source_dir(model_dir)
            quants = set(_subdirs(own_src))
            if h.inherits:
                quants |= set(_subdirs(_leaf_model_dir(kernels / h.inherits, own_src.name)))
            out.extend((hw, model, q) for q in sorted(quants))
    return out


def _normalize(p: Path) -> Path:
    parts: list[str] = []
    for c in p.parts:
        if c == "..":
            if parts:
                parts.pop()
        elif c != ".":
            parts.append(c)
    return Path(*parts)


def _layer_contents(kernels: Path, l: Layer, idx: int, ext: str) -> tuple[dict[str, Entry], dict[str, Path]]:
    entries: dict[str, Entry] = {}
    dirs: dict[str, Path] = {}
    if l.dir.is_dir():
        for p in sorted(l.dir.iterdir()):
            if p.is_symlink():
                raise LayoutError(f"{p}: is a symlink — declare the file in [sources] use instead")
            if p.is_dir():
                dirs[p.name] = p
            elif _is_kernel_file(p.name, ext):
                entries[p.name] = Entry(p.name, p, idx, False)
    if l.manifest is None:
        return entries, dirs
    for raw in l.manifest.uses:
        where = f"{l.manifest.path}: [sources] use {raw!r}"
        if os.path.isabs(raw):
            raise LayoutError(f"{where}: must be relative to kernels/<hardware>")
        path = _normalize(kernels / l.hardware / raw)
        if kernels not in path.parents:
            raise LayoutError(f"{where}: escapes kernels/")
        if path.is_symlink():
            raise LayoutError(f"{path}: is a symlink")
        if not path.is_file():
            raise LayoutError(f"{where}: no such file: {path}")
        if not _is_kernel_file(path.name, ext):
            raise LayoutError(f"{where}: not a .{ext} source or a header")
        if path.name in entries:
            raise LayoutError(f"{l.manifest.path}: [sources] use brings {path.name} but this layer already holds {entries[path.name].source}")
        entries[path.name] = Entry(path.name, path, idx, True)
    return entries, dirs


def _declare(layers: list[Layer], winner: Entry, loser: Path, used: set[tuple[int, str]]) -> Shadow:
    if winner.source == loser:
        raise LayoutError(f"{winner.name} resolves to {winner.source} in two layers — a use of a file the target already reaches")
    l = layers[winner.layer]
    stem = stem_of(winner.name)
    reason = l.manifest.shadow.get(stem) if l.manifest else None
    if reason is None:
        raise LayoutError(
            f"{winner.source} shadows {loser} and {l.dir / 'KERNEL.toml'} does not declare it: "
            f'add `[shadow] {stem} = "<reason>"`'
        )
    used.add((winner.layer, stem))
    return Shadow(l.role, winner.name, winner.source, loser, reason)


def discover(root: Path, hw: str, model: str, quant: str) -> Layout:
    kernels = root / "kernels"
    h = hardware(kernels, hw)
    hw_dir = kernels / hw
    model_dir = hw_dir / model
    if not model_dir.is_dir():
        raise LayoutError(f"{model_dir}: no such model directory")
    own_src = kernel_source_dir(model_dir)
    source_model = own_src.name
    layers = [Layer("leaf", "own", hw, own_src / quant, None)]
    if h.inherits:
        layers.append(Layer("leaf", "parent", h.inherits, _leaf_model_dir(kernels / h.inherits, source_model) / quant, None))
    layers.append(Layer("common", "own", hw, hw_dir / "common", None))
    if h.inherits:
        layers.append(Layer("common", "parent", h.inherits, kernels / h.inherits / "common", None))
    for l in layers:
        if l.dir.is_dir():
            l.manifest = kernel_manifest(l.dir)
    if not any(l.dir.is_dir() for l in layers):
        raise LayoutError(f"{hw}/{model}/{quant}: no kernel directory in any layer")
    lay = Layout(h, (hw, model, quant), layers, source_model=source_model, model_dir=model_dir)
    used: set[tuple[int, str]] = set()
    for idx, l in enumerate(layers):
        entries, dirs = (lay.leaf, lay.leaf_subdirs) if l.role == "leaf" else (lay.common, lay.common_subdirs)
        own_entries, own_dirs = _layer_contents(kernels, l, idx, h.source_ext)
        for name, e in own_entries.items():
            if name in entries:
                lay.shadows.append(_declare(layers, entries[name], e.source, used))
            else:
                entries[name] = e
        for name, d in own_dirs.items():
            dirs.setdefault(name, d)
    for name, e in lay.leaf.items():
        if not _has_ext(name, h.source_ext):
            continue
        for cname, c in lay.common.items():
            if _has_ext(cname, h.source_ext) and stem_of(cname) == stem_of(name):
                lay.shadows.append(_declare(layers, e, c.source, used))
                break
    for idx, l in enumerate(layers):
        if l.manifest:
            for stem in l.manifest.shadow:
                if (idx, stem) not in used:
                    raise LayoutError(f"{l.manifest.path}: [shadow] {stem} is declared but this directory's {stem} shadows nothing")
    lay.shadows.sort(key=lambda s: (s.role != "leaf", s.name, str(s.loser)))
    lay.leaf = dict(sorted(lay.leaf.items()))
    lay.common = dict(sorted(lay.common.items()))
    return lay


def symlinks(root: Path) -> list[Path]:
    out = []
    for dp, dn, fn in os.walk(root / "kernels"):
        for n in dn + fn:
            p = Path(dp) / n
            if p.is_symlink():
                out.append(p)
    return sorted(out)


def dump(root: Path) -> dict:
    """Every target's resolution, repo-relative — what layout_mirror.rs compares."""
    rel = lambda p: os.path.relpath(p, root)  # noqa: E731
    out = {}
    for hw, model, quant in walk(root):
        lay = discover(root, hw, model, quant)
        out[f"{hw}/{model}/{quant}"] = {
            "source_model": lay.source_model,
            "layers": [[l.role, l.tier, rel(l.dir)] for l in lay.layers],
            "leaf": {n: [rel(e.source), e.layer, e.used] for n, e in lay.leaf.items()},
            "common": {n: [rel(e.source), e.layer, e.used] for n, e in lay.common.items()},
            "leaf_subdirs": {n: rel(d) for n, d in sorted(lay.leaf_subdirs.items())},
            "common_subdirs": {n: rel(d) for n, d in sorted(lay.common_subdirs.items())},
            "configs": [rel(p) for p in lay.configs()],
            "modules": {s: rel(e.source) for s, e in lay.modules().items()},
            "shadows": [[s.role, s.name, rel(s.winner), rel(s.loser), s.reason] for s in lay.shadows],
        }
    return out


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "dump":
        root = Path(argv[2]).resolve() if len(argv) > 2 else Path.cwd()
        json.dump(dump(root), sys.stdout, indent=1, sort_keys=True)
        print()
        return 0
    print("usage: kernel_layout.py dump [root]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
