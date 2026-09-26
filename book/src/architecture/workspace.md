# Workspace Layout

Metrale Engine is a **twenty-one**-member Cargo workspace plus a build-time kernel tree (count them in the root `Cargo.toml` `members` list). This chapter maps every top-level directory to its role, and the crates to the axes of variation they each insulate.

## Repository tree (top level)

```
metrale-inference-alpha/
├── README.md
├── QUICKSTART.md                 per-model Docker recipes
├── CONTRIBUTING.md, AGENTS.md    contributor workflow
├── SECURITY.md                   disclosure
├── CLA.md                        contributor license agreement
├── LICENSE-MIT, LICENSE-APACHE   MIT OR Apache-2.0
├── Cargo.toml                    workspace root (21 members)
├── Cargo.lock
├── rust-toolchain.toml           pins the Rust release
├── deny.toml                     cargo-deny allow/deny lists
├── crates/                       Rust source for every crate
├── kernels/                      CUDA source, organized as (hw, model, quant)
├── docker/                       per-hardware Dockerfiles
├── scripts/                      bench, model-sweep, release helpers
├── tests/                        cross-crate integration tests (run_all_models.py lives here)
├── docs/                         design notes, history, release notes
├── jinja-templates/              chat templates for models that need custom ones
├── bench/                        benchmark harnesses and tracked results
├── governance/                   PR journey ledgers (governance/pr-<n>.jsonl)
├── site/, blog/, web-shared/     web sources
├── book/                         this book (mdBook source)
└── vendor/                       vendored deps (cudarc)
```

## The workspace members

`Cargo.toml` lists:

```toml
members = [
    "crates/core",
    "crates/config",
    "crates/closure",
    "crates/governance",
    "crates/kernels",
    "crates/gpu-sys",
    "crates/telemetry",
    "crates/gpu-runtime",
    "crates/scheduler",
    "crates/cache",
    "crates/sampling",
    "crates/storage",
    "crates/comm",
    "crates/grammar",
    "crates/model-weights",
    "crates/model-layers",
    "crates/model-arch",
    "crates/model-engine",
    "crates/speculative",
    "crates/bench",
    "crates/server",
]
```

Each is its own crate with its own `Cargo.toml`, its own unit tests, and its own responsibility:

| Crate | Role | Consumed by |
|---|---|---|
| `metrale-core` | Shared types (dtype, tensor, arch, device, errors, fault scopes) | `metrale-cache`, `metrale-config`, `metrale-gpu-runtime`, `metrale-kernels`, `metrale-model-engine`, `metrale-model-layers`, `metrale-model-weights`, `metrale-server`, `metrale-storage` |
| `metrale-config` | The typed model config tree (HF config.json parsing, GGUF metadata, capabilities) | `metrale-bench`, `metrale-gpu-runtime`, `metrale-model-arch`, `metrale-model-engine`, `metrale-model-layers`, `metrale-model-weights`, `metrale-server` |
| `metrale-closure` | Content hash of a kernel target's transitive include closure | `metrale-bench`, `metrale-kernels`, `metrale-model-layers` |
| `metrale-governance` | The PR journey ledger: an append-only record of how a change reached main | `metrale-bench` |
| `metrale-kernels` | Metrale Engine CUDA kernel PTX — compiled from `kernels/{hw}/{model}/{quant}/*.cu` | `metrale-gpu-runtime`, `metrale-model-engine`, `metrale-model-layers`, `metrale-server` |
| `metrale-gpu-sys` | Raw FFI only — cuFile (GDS) and NVML dlopen wrappers, NCCL bindings, RDMA verbs C shim | `metrale-comm`, `metrale-model-weights`, `metrale-storage`, `metrale-telemetry` |
| `metrale-telemetry` | Run metrics, kernel audit, launch trace, progress hooks, and the metal-up instruments (NVML device sampling, energy attribution, GPU spans, scheduler/request telemetry) with their exporters | `metrale-cache`, `metrale-gpu-runtime`, `metrale-model-arch`, `metrale-model-engine`, `metrale-model-weights`, `metrale-sampling`, `metrale-server` |
| `metrale-gpu-runtime` | GpuBackend (CUDA/Metal), streams, buffers, op cache, pinned hosts, kernel registry, cuBLASLt/CUTLASS/FlashInfer bridges | `metrale-cache`, `metrale-model-arch`, `metrale-model-engine`, `metrale-model-layers`, `metrale-model-weights`, `metrale-sampling`, `metrale-server` |
| `metrale-scheduler` | The scheduler's I/O contract — step plans, effects, the device/clock/spill router traits, the effect-trace decorator and the one driver loop. No model dependency | `metrale-model-engine`, `metrale-server` |
| `metrale-cache` | KV cache, KV dequant, KV spill, radix-tree prefix cache | `metrale-model-arch`, `metrale-model-engine`, `metrale-model-layers`, `metrale-server` |
| `metrale-sampling` | The sampler | `metrale-model-engine`, `metrale-server` |
| `metrale-storage` | GDS/RDMA tiers, expert paging, cache peers, tiered-cache core (SlotArena/SwapStore/Residency) | `metrale-model-arch`, `metrale-model-engine`, `metrale-model-layers`, `metrale-model-weights`, `metrale-server` |
| `metrale-comm` | Collective-operation backends (NCCL, single-GPU) | `metrale-model-arch`, `metrale-model-engine`, `metrale-model-layers`, `metrale-server` |
| `metrale-grammar` | Pure-Rust grammar-constrained decoding (XGrammar port) | `metrale-server` |
| `metrale-model-weights` | Weight store and loaders (safetensors, fast weights, RDMA weight/LoRA tiers, preflight) | `metrale-model-arch`, `metrale-model-engine`, `metrale-model-layers`, `metrale-server` |
| `metrale-model-layers` | Generic layers (attention, SSM, MoE, FFN, norm, MTP heads, vision, ops), LoRA, weight map, the draft-proposer contract | `metrale-model-arch`, `metrale-model-engine`, `metrale-server`, `metrale-speculative` |
| `metrale-model-arch` | Per-family architectures (GLM-5 Next, DeepSeek V4.1, Nemotron, Kimi K3, DFlash/MTP heads) and their weight loaders | `metrale-model-engine`, `metrale-server` |
| `metrale-model-engine` | The Model trait, the transformer model (prefill, decode, verify, SSM state), the generate engine and the model factory | `metrale-server` |
| `metrale-speculative` | Speculative-decoding policy — MTP gate, adaptive and DFlash rungs, n-gram proposer, spec capacity and stats, scheduler snapshots | `metrale-server` |
| `metrale-bench` | Plugin + benchmark abstraction and registry driven by the metrale-server TUI | `metrale-server` |
| `metrale-server` | Pure Rust LLM inference server (HTTP API, scheduler, TUI, CLI) | n/a — the deliverable (`met`) |

The dependency graph runs strictly downward in the table above — `metrale-core` has no internal deps, every crate above it builds on crates below. There are no cycles.

## The kernel tree

```
kernels/
├── gb10/                                        # One directory per hardware target
│   ├── HARDWARE.toml                            # vendor, arch, memory specs, serving defaults
│   ├── common/                                  # the shared baseline every GB10 target compiles
│   ├── qwen3-next-80b-a3b/                      # One directory per model target
│   │   ├── MODEL.toml                           # layer counts, sampling presets, behavior
│   │   └── nvfp4/                               # One directory per quantization target
│   │       ├── KERNEL.toml                      # compile flags, module names, [sources], [shadow]
│   │       └── *.cu                             # the kernels this target overrides
│   ├── qwen3.6-35b-a3b/
│   │   └── nvfp4/
│   ├── mistral-small-4/
│   │   └── nvfp4/
│   └── ... (one leaf per (model, quant) target)
├── hopper/, b200/                               # inherit gb10 ([hardware] inherits)
├── b300/                                        # Kimi K3 bring-up
└── strix/, strix-hip/, metal/                   # AMD and Apple targets
```

A leaf directory holds only what its `(gb10, model, quant)` target changes relative to `common/`; everything else comes from `common/`. A file in a leaf can use any tile shape, any register budget, any shared-memory layout without affecting a different target, because nothing else compiles it. [Philosophy: AI Kernel HyperCompiling](./philosophy.md) describes the layers, `[sources] use` and `[shadow]`.

This is the mechanism that makes `kernels/` a scalable structure. Adding a new GPU is `kernels/<new-hw>/`. Adding a new model is `kernels/<hw>/<new-model>/`. Adding a new quantization is `kernels/<hw>/<model>/<new-quant>/`. Nothing else moves.

## Docker layout

```
docker/
├── gb10/
│   ├── Dockerfile                         multi-model image — compiles every target
│   ├── Dockerfile.builder                 build sandbox with CUTLASS and FlashInfer pinned
│   ├── qwen3-next-80b-a3b/nvfp4/          per-model slim image
│   ├── qwen3.5-35b-a3b/nvfp4/
│   └── ... (one slim Dockerfile per supported model)
├── hopper/, b200/                         H100/H200 and B200 images
├── k3/                                    Kimi K3 bring-up image
└── docker-guide.md                        build + run instructions
```

The multi-model `Dockerfile` at `docker/gb10/Dockerfile` is what ships as `metrale/metrale-inference-gb10:latest`. Per-model Dockerfiles exist for operators who want a smaller image containing only one target — the kernel registry still uses `KernelTarget` at runtime, but only one target set is baked in.

## Docs, design records, history, releases

Inside `docs/`:

- `adr/` — architecture decision records (licensing, pure-Rust, hybrid SSM/attention, NVFP4/FP8 quantization, TP/EP composition, EP batched decode, etc.). Treat these as the long-form rationale behind code changes; commit messages are deliberately terse and point here. Top-level notes like `ARCHITECTURE.md`, `METRALE_KERNELS.md`, and `HARDWARE.md` sit alongside them.
- `METRALE_JOURNEY.md` — benchmark journey and retrospective of the engine on GB10. Useful context, but not a contract.
- `releases/` — human-readable release notes keyed by release (`README.md` plus per-release files).

The book you're reading in `book/` synthesises all of this into a single narrative — it is *not* a canonical rewrite of those documents. The design records in `docs/adr/` remain the authoritative reference and the book links to them directly from the deep-dive chapters.

## What changes when you add a…

| You added | You touched |
|---|---|
| A new quantization (e.g. MXFP4) | `kernels/<hw>/<model>/<scheme>/*.cu`, a format module under `crates/model-layers/src/quant_format/`, the loader arms in `crates/model-layers/src/weight_map/`, and any new host-side conversion in `crates/core/src/numeric.rs` |
| A new model family (e.g. Phi-4) | `crates/model-arch/src/weight_loader/<family>.rs`, one arm in `crates/model-engine/src/factory.rs`, `kernels/<hw>/<family>/<quant>/MODEL.toml`, optional `jinja-templates/<model_type>.jinja` |
| A new hardware vendor (e.g. MI300X) | `crates/core/src/compute.rs` (new `ComputeTarget` impl), `crates/kernels/build_target.rs::resolve_compute_target()` arm, `crates/gpu-runtime/src/<vendor>_backend.rs` (new `GpuBackend` impl), `crates/comm/src/<name>_backend.rs` if the vendor needs its own collective impl, `kernels/<hw>/HARDWARE.toml`, kernel source under `kernels/<hw>/<model>/<quant>/` |
| A new CLI flag | `crates/server/src/cli/serve_args.rs` (or its `serve_args/` submodules), plumbing wherever it lands |
| A new tool-call format | `crates/server/src/tool_parser.rs` |

Each row touches a small, bounded set of files. That bounded-ness is the architectural payoff of the workspace being split along axes of variation. Read [Kernel Dispatch Pipeline](./dispatch.md) next to see the runtime side, or [SBIO](./sbio.md) to see how the trait layering makes the whole thing testable without a GPU.
