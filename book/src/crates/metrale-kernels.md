# metrale-kernels

**Path:** `crates/kernels/`
**Role:** the bridge between the kernel source tree and the Rust workspace. Every PTX module every other crate launches is embedded here.
**Key files:** `src/lib.rs` (the registry types and the `include!` of the generated file), `src/resolve.rs` (which target serves a checkpoint), `build.rs` and its `build_*.rs` modules (which generate the registry).

## The trick: auto-generated PTX embedding

`crates/kernels/src/lib.rs` includes one generated file:

```rust
include!(concat!(env!("OUT_DIR"), "/target_ptx.rs"));
```

Everything inside `target_ptx.rs` — the per-target PTX bytes, `ptx_modules()` for the primary target, the `all_ptx_sets()` multi-target registry, `TARGET_DEFAULTS` and `TARGET_SM_COUNT` — is produced by `build.rs` at every Cargo build. You will not find `target_ptx.rs` in the repository; it is generated fresh into `OUT_DIR` each time.

`all_ptx_sets()` returns one `TargetPtxSet` per compiled target:

```rust
pub struct TargetPtxSet {
    pub target: KernelTarget,               // (arch, model, quant)
    pub ptx_arch: &'static str,             // HARDWARE.toml arch verbatim: sm_121f, sm_90a, …
    pub modules: Vec<(&'static str, &'static [u8])>,
    pub sampling: SamplingPresets,          // from MODEL.toml
    pub behavior: ModelBehavior,            // from MODEL.toml
    pub model_type_matches: Vec<ModelTypeMatch>,
    pub match_names: &'static [&'static str],
    // …
}
```

At startup, `metrale_kernels::ptx_for_config(model_type, hidden_size, model_refs, pinned)` picks the set that serves the checkpoint (`src/resolve.rs`: exact `(model_type, hidden_size)` declarations first, a collision broken by `match_names`, an unbroken tie an error, `--kernel-target` pins one), and the server passes its `modules` to `MetraleCudaBackend::new`, which uploads each PTX module with `cuModuleLoadData`.

## What `build.rs` actually does

1. **Read `METRALE_TARGET_*`** — `HW` (default `gb10`), `MODEL` (default `*`) and `QUANT` (default `nvfp4`); `*` expands to every matching directory.
2. **Resolve each target's sources** — own leaf, own `common/`, and the parent's pair when `HARDWARE.toml` sets `[hardware] inherits`, plus every `KERNEL.toml` `[sources] use`. The resolver is `metrale-closure`'s layout module; an undeclared shadow or a `[shadow]` entry with nothing to shadow fails the build.
3. **Stage** each role's sources into `OUT_DIR` (`build_stage.rs`), so a quoted `#include` resolves against the headers of the layer that compiles it.
4. **Resolve the compiler** — `resolve_compute_target(vendor)` (`build_target.rs`) returns `NvidiaTarget`, `AppleTarget`, `ScaleTarget` or `HipTarget`.
5. **Compile every source** — flags come from `HARDWARE.toml`, the `common/` and leaf `KERNEL.toml` `[build]` tables, and `METRALE_EXTRA_NVCC_FLAGS`.
6. **Apply module-name overrides** — `KERNEL.toml`'s `[modules]` section lets a file stem (`e2m1_branchless.cu`) expose itself under a shorter module name (`e2m1`).
7. **Parse `MODEL.toml`** into `SamplingPresets` and `ModelBehavior`, and `HARDWARE.toml` `[defaults]` into `TARGET_DEFAULTS` — the serving levers the target runs with.
8. **Write `target_ptx.rs`.**

`rerun-if-changed` and `rerun-if-env-changed` directives on the kernel tree and the `METRALE_TARGET_*` variables mean Cargo re-runs the script only when an input changed.

## `METRALE_SKIP_BUILD=1` — the escape hatch

On a host with no `nvcc`, the crate would fail to build without this. When it is set, `build.rs` invokes no compiler and emits a stub `target_ptx.rs` with no targets, so the crate compiles cleanly. A macOS build without `METRALE_TARGET_HW` takes the same path. CI's lint and test jobs use it; so does a local `cargo clippy`. The [Kernel Dispatch](../architecture/dispatch.md) chapter covers the broader flow.

## What gets added to this crate when you…

- **…add a new `(hw, model, quant)` leaf?** Nothing under `crates/kernels/src/`. `build.rs` picks it up on the next `cargo build`; its `MODEL.toml` declares which `model_type` (and `hidden_size`) it serves.
- **…add a new kernel to an existing leaf?** Drop the source in the leaf directory; if it replaces a `common/` file, declare the shadow in the leaf's `KERNEL.toml` `[shadow]` with a reason. For a non-stem module name, add a `[modules]` entry.
- **…add a new hardware vendor?** Extend `resolve_compute_target(vendor)` in `build_target.rs` to return your new `ComputeTarget` impl.

The rest of the kernel-engineering story is in the [CUDA Kernel Engineering](../deep-dives/kernels.md) deep dive.
