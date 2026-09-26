# metrale-gpu-runtime

**Path:** `crates/gpu-runtime/`
**Role:** everything that touches the GPU directly: the `GpuBackend` trait and its CUDA and Metal implementations, streams, buffers, the op cache, pinned host memory, the kernel registry and the cuBLASLt/CUTLASS/FlashInfer bridges.
**Key files:** `gpu.rs`, `gpu/mock.rs`, `cuda_backend.rs`, `metal_backend.rs`, `buffers.rs`, `registry.rs` in `crates/gpu-runtime/src/`.

This chapter also covers the pieces the serving path uses beside it, which live in their own crates: the KV cache and radix tree in `crates/cache/src/` ([metrale-cache](./metrale-cache.md)), the `PrefixCache` trait in `crates/telemetry/src/prefix_cache.rs`, the sampler in `crates/sampling/src/lib.rs` ([metrale-sampling](./metrale-sampling.md)), and the fast weight loader in `crates/model-weights/src/fast_weights/` ([metrale-model-weights](./metrale-model-weights.md)).

## The load-bearing trait: `GpuBackend`

Its methods fall into five concerns:

- **Memory** — `alloc`, `free`, `copy_h2d`/`d2h`/`d2d`, `memset`, `total_memory`, `free_memory`, `alloc_host_pinned`.
- **Kernel launch** — `kernel(module, func)` returns a `KernelHandle`; `launch(handle, grid, block, shared, stream, args)` fires the launch.
- **Streams** — `default_stream`, `create_stream`, `synchronize`, `bind_to_thread`.
- **CUDA graphs** — `begin_capture`, `end_capture` → `GraphHandle`, `launch_graph`.
- **Events** — `create_event`, `record_event`, `stream_wait_event`.

Required methods have no default; optional methods have default panics/no-ops so a partial backend (e.g. a Metal backend without CUDA-graph support yet) still compiles. The trait's [SBIO role](../architecture/sbio.md) is discussed in Part II.

## The production impl: `MetraleCudaBackend`

In `cuda_backend.rs`. Built on `cudarc` (Rust bindings over the CUDA driver API). On construction:

1. `MetraleRegistry::load(ordinal, ptx_modules)` (`registry.rs`) loads every PTX module of the chosen target set with `cuModuleLoadData`. The CUDA context and default stream belong to the process host (`cuda_host.rs`) and are shared by every backend.
2. Kernel lookups (`kernel(module, func)`) resolve a `CUfunction` once and cache it as a `KernelHandle`.

Per-launch the backend unpacks the kernel args into `void*[]`, sets the stream, and calls `cuLaunchKernel`. Typed launch arguments (`gpu_args.rs`, `kernel_args.rs`) keep that conversion out of the layer code.

## The paged KV cache (`kv_cache.rs`)

Metrale Engine uses paged attention (à la vLLM) with block-level allocation. Core types:

```rust
pub enum KvCacheDtype {
    Bf16,    // 2 bytes/element — unquantized baseline
    Fp8,     // 1 byte/element — E4M3 + per-tensor scale
    Nvfp4,   // 0.5 bytes + per-group FP8 scale
    Turbo4,  // 4-bit WHT + Lloyd-Max (TurboQuant) — lower MSE than NVFP4
    Turbo3,  // 3-bit WHT + Lloyd-Max
    Turbo2,  // 2-bit WHT + Lloyd-Max — smallest
    Turbo8,  // WHT + FP8 — outlier-resistant FP8
    // … plus asymmetric K/V pairings such as Turbo4KTurbo3V
}
```

`PagedKvCache` holds a pool of fixed-size blocks (configurable, typically 16 tokens per block). `KvCacheConfig` derives pool sizing from `ModelConfig` + `--max-seq-len` + `--max-batch-size`. Allocation is O(1) from a free list; eviction is handled by the scheduler.

The **TurboQuant** family (`turbo3`, `turbo4`, `turbo8`) is specific to Metrale Engine: Walsh-Hadamard rotation followed by Lloyd-Max quantization to optimal Gaussian codebook levels. For the same bit rate, turbo4 has ~2× lower MSE than NVFP4 on the kinds of activations transformers produce, because WHT flattens outliers before quantization. See `docs/turboquant-plus.md` and [FP8](../deep-dives/fp8.md) / [NVFP4](../deep-dives/nvfp4.md) chapters.

## Prefix caching (`PrefixCache`, `RadixTree`)

RadixAttention: the system prompt shared by every request can be KV-cached once, reused forever. Implementation:

- **`crates/cache/src/radix_tree.rs`** — in-memory radix tree keyed on token sequences. Each node owns the KV pages for its token prefix.
- **`crates/telemetry/src/prefix_cache.rs`** — the `PrefixCache` trait the scheduler calls, implemented by `RadixTree` (and `NoPrefixCaching` when caching is off), plus the hit and miss counters. A lookup walks the tree to the deepest matching node; the KV pages for that prefix are already resident on the GPU.

Hit rates are high in practice — system prompts and few-shot examples dominate, and chat agents reuse most of their tool schemas across turns. TTFT drops ~10× on warm-cache hits. This is the feature enabled by `--enable-prefix-caching`.

**Marconi (SSM snapshots)** extends the idea to SSM layers: a full SSM state is ~GB on a 35B model, so prefix cache hits for hybrid models also need a snapshotted SSM state to be genuinely equivalent. That machinery lives in `metrale-cache` (the snapshot index beside the radix tree) and `metrale-model-engine` (the SSM state pools). See `docs/adr/0003-hybrid-ssm-attention.md` for the SSM-snapshot-cache design.

## Buffer arena (`buffers.rs`)

One `BufferArena` per serve. Allocates every scratch buffer *once* at startup, sized for the worst-case batch × seq_len combination allowed by CLI flags. Includes:

- `hidden_states`, `residual`, `norm_output` — residual stream and its post-norm staging
- `qkv_output`, `attn_output` — attention projection outputs
- `gate_logits`, `moe_output` — MoE intermediates
- `logits` — the final `[M, vocab_size]` output
- `ssm_qkvz`, `ssm_ba`, `ssm_gates` — Mamba/GDN projections
- `expert_gate_out`, `expert_up_out`, `expert_down_out` — per-expert MoE intermediates, sized to cover both speculative decode and batched MoE prefill

The arena never reallocates during serving. This is one of the invariants that makes CUDA graph capture viable — buffer addresses are graph-stable.

## Sampler (`metrale-sampling`)

`SamplingParams` — `temperature`, `top_p`, `top_k`, `top_n_sigma`, `min_p`, `logit_bias`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`. The sampler:

1. Applies penalties (presence, repetition) in-place on the logits buffer.
2. Applies `top_n_sigma` (entropy-based filter).
3. Applies `top_p` + `top_k` + `min_p`.
4. Softmax.
5. Multinomial sampling or argmax (if `temperature == 0`).

The sampler guards the `temperature=0` and `repetition_penalty=0` divisions explicitly. `--adaptive-sampling` toggles an entropy-gated greedy path that avoids the full softmax+sample when the logits are effectively one-hot.

## Fast weight loader (`fast_weights/`)

Metrale Engine's production weight loader. Modeled on `scitix/InstantTensor`:

- Each safetensors shard is opened with `O_DIRECT` (bypasses the page cache — critical on GB10 where the page cache shares physical memory with the GPU).
- One reader thread pre-fetches the next tensor's bytes into a page-aligned buffer.
- The main thread `copy_h2d`'s the current tensor while the next one is being read.
- A per-shard heuristic auto-picks between `O_DIRECT` and buffered reads: shards with > 5000 tensors pay too much per-tensor syscall overhead for `O_DIRECT`, and kernel readahead wins there.
- If the filesystem rejects `O_DIRECT` (tmpfs, overlayfs), falls back to mmap automatically.

Cold-load speedups measured vs mmap (with `posix_fadvise(DONTNEED)` between runs):

| Model | Cold mmap | Cold fast | Speedup |
|---|---:|---:|---:|
| Qwen3.5-27B (1 shard, 2.4k tensors) | 19s | 9s | **2.05×** |
| Qwen3.5-35B-A3B (2 shards, 125k tensors) | 62s | 34s | **1.84×** |
| Qwen3-Next-80B-A3B (11 shards, 298k tensors) | 166s | 110s | **1.51×** |

On by default. `--no-fast-load` reverts to mmap.

## `MockGpuBackend` and testing

The test double lives in `gpu/mock.rs`, beside the trait in `gpu.rs`. Records every launch; returns success for every op. Enables the ~80% of the test suite that doesn't need a real GPU — see [SBIO](../architecture/sbio.md).

## What's explicitly not here

- **No model layers.** That's `metrale-model-layers`.
- **No HTTP.** That's `metrale-server`.
- **No collective ops.** That's `metrale-comm`.

`metrale-gpu-runtime` is the bottom of the "things that move bits on a GPU" stack and the top of the "things a layer is allowed to call directly" stack.
