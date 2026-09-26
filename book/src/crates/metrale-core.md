# metrale-core

**Path:** `crates/core/`
**Role:** the shared foundation types and host-side helpers every other Metrale Engine crate builds on.
**Dependencies:** none from the workspace — this is the bottom of the stack. `cudarc` is optional, behind the `cuda` feature, which gates only `device::MetraleDevice` and the `MetraleError::CudaDriver` variant.

## Module map

```text
crates/core/src/
├── lib.rs            one `pub mod` per module below
├── arch.rs           whether PTX built for one SM arch runs on a device (the arch preflight)
├── compute.rs        ComputeTarget trait + Vendor enum (Nvidia/Amd/Apple/Intel) + NvidiaTarget
├── target.rs         KernelTarget — the (arch, model, quant) dispatch key
├── dtype.rs          DType enum: E2M1, FP8E4M3, FP8E5M2, BF16, …
├── tensor.rs         TensorRef — non-owning handle (ptr, shape, strides, dtype)
├── device.rs         GB10 hardware constants (`sm121`) and the cudarc handle `MetraleDevice`
├── error.rs          MetraleError + Result alias
├── fault.rs          process-wide latch for a destroyed CUDA context and its exit status
├── scope.rs          ordered, fallible model teardown (`Teardown`, `ModelResource`)
├── numeric.rs        host-side FP8 E4M3 decode table and BF16 conversions
├── mxfp4_e8m0.rs     host unpack of packed E2M1 nibbles + E8M0 scales
└── safetensors.rs    the one place a safetensors `data_offsets` pair becomes a byte span
```

`ModelConfig` and the capability flags are in [metrale-config](./metrale-config.md) (`crates/config/src/lib.rs`, `crates/config/src/capabilities.rs`). The stream, kernel-descriptor and registry types are in [metrale-gpu-runtime](./metrale-gpu-runtime.md) (`crates/gpu-runtime/src/stream.rs`, `kernel.rs`, `registry.rs`).

## `ComputeTarget`: the hardware-vendor abstraction

```rust
pub trait ComputeTarget {
    fn source_extension(&self) -> &str;       // "cu", "metal", …
    fn output_extension(&self) -> &str;       // "ptx", "metallib", …
    fn output_is_text(&self) -> bool;         // PTX is text; metallib is binary
    // … finding the compiler and compiling one source for one arch
}
```

The trait is consumed at **build time** by the kernels build script, which reads `kernels/<hw>/HARDWARE.toml`, picks the matching `Box<dyn ComputeTarget>` in `crates/kernels/build_target.rs::resolve_compute_target()` and compiles every source of every matching `(model, quant)` target. The build script implements `NvidiaTarget` (nvcc), `AppleTarget` (xcrun), `ScaleTarget` (SCALE, for `amd`) and `HipTarget` (hipcc).

## `Vendor`

```rust
pub enum Vendor { Nvidia, Amd, Apple, Intel }
```

Parsed from `HARDWARE.toml`'s `vendor` field; each variant accepts a few spellings (`nvidia`/`cuda`, `amd`/`rocm`/`hip`, `apple`/`metal`, `intel`/`oneapi`/`sycl`).

## `KernelTarget`: the runtime dispatch key

```rust
pub struct KernelTarget {
    pub arch: &'static str,   // base SM: "sm_121", "sm_90", ...
    pub model: &'static str,  // "qwen3-next-80b-a3b"
    pub quant: &'static str,  // "nvfp4", "fp8", "bf16"
}
```

For CUDA, `arch` is the base SM: the build strips a `HARDWARE.toml` feature suffix (`sm_90a` → `sm_90`). The `metrale-kernels` crate's generated `target_ptx.rs` emits one `TargetPtxSet` per compiled target, each carrying its `KernelTarget`; at startup `metrale_kernels::ptx_for_config` picks the set that serves the checkpoint (see [Kernel Dispatch](../architecture/dispatch.md)).

## `DType`: the supported numeric formats

`DType` names the element types the engine stores: `E2M1` (NVFP4 weights), `FP8E4M3` (block scales and FP8 weights), `FP8E5M2`, `BF16` and the wider formats. `DType::element_size_bits()` returns the per-element bit count — bits, not bytes, because E2M1 is sub-byte.

## `TensorRef`

```rust
pub struct TensorRef {
    pub ptr: u64,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,   // in elements, not bytes
    pub dtype: DType,
}
```

A non-owning view of a device tensor: it neither allocates nor frees, and it is `Clone`.

## What's explicitly not here

- **No actual GPU allocation.** That's `metrale-gpu-runtime`.
- **No model config.** That's `metrale-config`.
- **No weight loader types.** Those are `metrale-model-weights` and `metrale-model-arch`.
- **No HTTP types.** Those are `metrale-server`.

`metrale-core` is small on purpose. It contains only the vocabulary that *every* crate downstream needs.

## Adding a vendor

Implementing a new hardware vendor starts here:

1. Extend `Vendor` if needed (all four major ones are already enumerated).
2. Implement `ComputeTarget` for the vendor's compiler.
3. Register it in `crates/kernels/build_target.rs::resolve_compute_target()`.
4. Continue to the [metrale-gpu-runtime chapter](./metrale-gpu-runtime.md) for the runtime trait (`GpuBackend`).
