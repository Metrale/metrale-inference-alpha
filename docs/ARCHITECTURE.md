# Metrale Engine Architecture

A 5-minute tour of the crate graph, the build pipeline, and the request lifecycle. The [book](../book/src/SUMMARY.md) goes deeper: *Workspace Layout*, *Kernel Dispatch Pipeline* and one chapter per crate.

## Crate graph

Metrale Engine is a single Cargo workspace with 21 crates (the root `Cargo.toml` `members` list):

```
   HTTP / scheduling / CLI     metrale-server  (the `met` binary)
                               metrale-scheduler (the scheduler's I/O contract)
                               metrale-bench     (benchmark suite + certification gate)
                                     │
   model                       metrale-model-engine  (Model trait, TransformerModel, factory)
                               metrale-model-arch    (per-family architectures + weight loaders)
                               metrale-model-layers  (generic layers, LoRA, weight map, kernel launches)
                               metrale-model-weights (weight store, fast loader, preflight)
                               metrale-speculative   (MTP gate, DFlash, n-gram)
                               metrale-sampling, metrale-grammar
                                     │
   runtime                     metrale-gpu-runtime (GpuBackend: CUDA + Metal, buffers, registry)
                               metrale-cache       (paged KV cache, radix-tree prefix cache)
                               metrale-comm        (CommBackend: NCCL, single-GPU)
                               metrale-storage     (GDS/RDMA tiers, NVMe swap, expert paging)
                               metrale-telemetry   (metrics, kernel audit, NVML instruments)
                                     │
   build / foundation          metrale-kernels  (PTX embedded per (hardware, model, quant))
                               metrale-closure  (kernel layout resolver + closure hash)
                               metrale-config   (ModelConfig, parsers, environment levers)
                               metrale-gpu-sys  (raw FFI: cuFile, NVML, NCCL, RDMA verbs)
                               metrale-core     (shared types: ComputeTarget, KernelTarget, DType, …)
                               metrale-governance (PR journey ledger)
```

## Layer-by-layer

### `metrale-core` and `metrale-config`
`metrale-core` (`crates/core/`) holds the vocabulary every crate shares: `ComputeTarget` and `Vendor` (the build-time compiler abstraction), `KernelTarget` (the `(arch, model, quant)` dispatch key), `DType`, `TensorRef`, the arch preflight and error types. `metrale-config` (`crates/config/`) parses a checkpoint's `config.json` into `ModelConfig` (per-family parsers under `src/parsers/`) and declares every `METRALE_*` environment lever (`src/levers.rs`).

### `metrale-kernels`
The build script (`crates/kernels/build.rs` and its `build_*.rs` modules):

1. Reads `METRALE_TARGET_HW` (default `gb10`), `METRALE_TARGET_MODEL` (default `*`) and `METRALE_TARGET_QUANT` (default `nvfp4`).
2. Resolves each selected target's sources through `metrale-closure`'s layout module: the target's leaf `kernels/<hw>/<model>/<quant>/`, `kernels/<hw>/common/`, the same pair of the tree `HARDWARE.toml` `[hardware] inherits` names, and every `KERNEL.toml` `[sources] use`. A leaf file with a `common/` file's name shadows it and must be declared in `[shadow]`.
3. Compiles every source with the compiler the tree's `vendor` selects (`nvcc` for NVIDIA), applying `HARDWARE.toml` and `KERNEL.toml` flags.
4. Parses `MODEL.toml` (sampling presets, behaviour) and `HARDWARE.toml` `[defaults]` (serving levers), and emits `OUT_DIR/target_ptx.rs`: one `TargetPtxSet` per target, `all_ptx_sets()`, `TARGET_DEFAULTS`.

Skip mode (`METRALE_SKIP_BUILD=1`) emits a stub registry. CI uses it for GPU-free `cargo check` / `clippy` / `test`.

### `metrale-gpu-runtime`, `metrale-cache`, `metrale-comm`, `metrale-storage`
- `metrale-gpu-runtime` — the `GpuBackend` trait (`src/gpu.rs`) with `MetraleCudaBackend` (`src/cuda_backend.rs`), a Metal backend and `MockGpuBackend` for tests; buffer arenas, streams, the kernel registry, and the cuBLASLt/CUTLASS/FlashInfer bridges.
- `metrale-cache` — `PagedKvCache` with BF16 / FP8 / NVFP4 / TurboQuant dtypes, KV dequant and spill, and the radix-tree prefix cache with its SSM snapshot index.
- `metrale-comm` — `CommBackend` with `NcclBackend` (TCP rendezvous, RoCEv2) and the no-op `SingleGpuBackend`.
- `metrale-storage` — the GDS/RDMA tiers, NVMe high-speed swap, expert paging and cache peers.

### The model crates
- `metrale-model-engine` — the `Model` trait the scheduler drives (a composite of `ModelLifecycle`, `ModelForward`, `ModelVerify`, `ModelSsmState`, … in `src/traits/model/`), the concrete `TransformerModel` (`src/model/`), and `factory::loader_for_config`, the one place a `model_type` becomes a loader.
- `metrale-model-arch` — the per-family `ModelWeightLoader`s (`src/weight_loader/`: `qwen3.rs`, `qwen35.rs`, `qwen3_vl.rs`, `gemma4.rs`, `nemotron.rs`, `minimax.rs`, `deepseek_v4.rs`, `glm5_next.rs`, …; `mistral_loader/`) and family-specific layers (Nemotron Mamba-2, GLM-5 Next, DeepSeek V4.1, Kimi K3, the DFlash head).
- `metrale-model-layers` — the `TransformerLayer` trait (`src/layer/`) and the generic layers (`src/layers/`: `qwen3_attention/`, `qwen3_ssm/`, `moe/`, `dense_ffn*`, `vision_encoder/`, MTP heads, `ops/` kernel launches), LoRA, the weight map and quant-format dispatch.
- `metrale-model-weights` — the weight store, the `O_DIRECT` fast loader, the RDMA weight/LoRA tiers and the pre-construction checkpoint preflight.

### `metrale-server`
- `main_modules/serve.rs` — the `met serve` boot sequence (phases under `serve_load/` and `serve_phases/`), the router (`serve_router.rs`) and model hot-swap.
- `scheduler/` — the scheduler loop (`scheduler/mod.rs::run`) and its per-step phases (prefill, decode, verify, MTP/n-gram/DFlash drafting, sampling, emission, lifecycle, rollback).
- `api/` — OpenAI-compatible handlers (chat completions, responses, completions, conversations, LoRA control, tokenize).
- `anthropic/` — Anthropic Messages API; `openai/` — OpenAI request/response types.
- `tool_parser/` — per-format tool-call parsers (Hermes JSON, Qwen3-coder XML, Qwen3 XML, Gemma-4, MiniMax XML, Mistral, bare JSON, Poolside).
- `grammar/` — XGrammar-backed constrained decoding (tool schemas, JSON mode), on `metrale-grammar`.
- `tokenizer/` — chat-template rendering via Jinja, streaming decode.
- `cli/` — `serve`, `benchmark`, `doctor` and `sync-recipes` arguments; `tui/` — the dashboard.

## Request lifecycle (decode path)

1. **HTTP** — Axum receives `POST /v1/chat/completions`, handled in `api/chat/`.
2. **Pre-process** — `chat/msg_entry::build_msg_entries` extracts messages + tools; `chat/template::render_template` renders the chat template; `chat/sampling_setup::build_sampling` resolves the sampling preset, stops and grammar.
3. **Submit** — the request is enqueued onto the scheduler's pending queue.
4. **Scheduler tick** (`scheduler/mod.rs::run`):
   - Admit and prefill new sequences.
   - Draft (MTP / n-gram / DFlash when enabled) and verify the drafts through the model's verify path.
   - Emit accepted tokens via SSE, or buffer them for a blocking response.
5. **Per-token forward** — for each layer, `TransformerLayer::decode`; kernels are looked up with `gpu.kernel(module, fn)` against the embedded `TargetPtxSet` and launched on the model's stream: KV cache write, paged decode attention, SSM state update, MoE routing + expert GEMV.
6. **Sample** — logits are sampled (`metrale-sampling`) and the token is emitted.

## Build pipeline

```
$ METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=qwen3.6-35b-a3b METRALE_TARGET_QUANT=nvfp4 \
    cargo build --release -p metrale-server
```

1. Cargo runs `crates/kernels/build.rs`.
2. It resolves `kernels/gb10/qwen3.6-35b-a3b/nvfp4/` over `kernels/gb10/common/`, invokes `nvcc -arch=sm_121f --ptx` on each source, and emits `OUT_DIR/target_ptx.rs` with the resulting PTX bytes.
3. The Rust crates compile, linking the generated module.
4. At startup `metrale_kernels::ptx_for_config` picks the target for the checkpoint's `model_type` and `hidden_size`, and `MetraleCudaBackend::new` loads its PTX into the CUDA driver.

For a multi-target binary: `METRALE_TARGET_MODEL='*' METRALE_TARGET_QUANT='*'` compiles every kernel target under `kernels/<hw>/`.

## See also

- `docs/HARDWARE.md` — adding a new SM target / model family.
- `docs/DEPLOYMENT.md` — Docker, multi-rank, NVMe swap.
- `docs/adr/` — Architecture Decision Records explaining major design choices.
