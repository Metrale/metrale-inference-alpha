# Kernel Dispatch Pipeline

One Metrale Engine binary contains kernels for every `(Hardware, Model, Quantization)` target it was built for. This chapter traces a single chat completion from the moment the HTTP request arrives to the moment a kernel launches on the GPU, so you know exactly where each piece of dispatch lives.

## The high-level flow

```
             1. HTTP request                                    7. Kernel launch
┌──────────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│ OpenAI       │──►│ axum     │──►│ scheduler │──►│ engine    │──►│ PTX on   │
│ client       │   │ (server) │   │ (server)  │   │ (model)   │   │ GPU      │
└──────────────┘   └──────────┘   └───────────┘   └───────────┘   └──────────┘
                                                        │               ▲
                                                        ▼               │
                                                 ┌──────────────────────┴───┐
                                                 │ KernelTarget → PtxModule │
                                                 │ (metrale-kernels)          │
                                                 └──────────────────────────┘
```

The dispatch decisions happen in two distinct phases:

- **Build time** — `crates/kernels/build.rs` decides *which PTX to embed*.
- **Startup** — `met serve` (`crates/server/src/main_modules/serve_load/`) decides *which embedded PTX to upload to the GPU* based on the model being served.

After startup, the fast path is deterministic: a layer's `forward(ctx)` always calls the same `KernelHandle`s, always on the same GPU stream, always in the same order. There is no per-request dispatch decision. This is the payoff of the specialization thesis — no branching, no polymorphism across kernel variants, no cache miss.

## Phase 1 — build time: which PTX gets embedded

`crates/kernels/build.rs` runs during `cargo build`. Its job:

1. Read the three wildcards:
   - `METRALE_TARGET_HW` (default `gb10`; one hardware directory)
   - `METRALE_TARGET_MODEL` (default `*` — all)
   - `METRALE_TARGET_QUANT` (default `nvfp4`; accepts `*`)
2. Enumerate `kernels/<hw>/<model>/<quant>/` leaves matching the wildcards.
3. For each leaf, resolve its source layers — own leaf, own `common/`, and the parent's when `HARDWARE.toml` sets `[hardware] inherits` (`crates/closure/src/layout.rs`) — and stage them into `OUT_DIR` (`crates/kernels/build_stage.rs`).
4. Read `HARDWARE.toml` to learn the vendor, and call `resolve_compute_target(vendor)` (`crates/kernels/build_target.rs`) to get a `Box<dyn ComputeTarget>`:
   - `nvidia` → `NvidiaTarget { nvcc }`
   - `apple` → `AppleTarget { xcrun }`
   - `amd` → `ScaleTarget` (SCALE), `hip` → `HipTarget { hipcc }`
5. Compile every staged source, applying `KERNEL.toml` flags and module-name overrides (some kernels are compiled from `e2m1_branchless.cu` but exposed at runtime as the `e2m1` module).
6. Emit an auto-generated Rust file, `$OUT_DIR/target_ptx.rs`, that `include!()`'s back into `crates/kernels/src/lib.rs`. The generated file contains one `pub static PTX_<TARGET>: &[PtxModule]` per target plus an `all_ptx_sets()` function that returns the whole set.

The output is one single PTX set per target, embedded in the final `metrale-server` binary as a byte slice. This is why "one Docker image, one binary, zero runtime compilation" is true.

`METRALE_SKIP_BUILD=1` short-circuits the whole phase: `build.rs` emits a stub `target_ptx.rs` with empty constants so that `clippy`, `fmt`, and non-GPU tests can compile on a Linux host with no `nvcc`. The CI in `.github/workflows/ci.yml` uses this.

## Phase 2 — startup: which embedded PTX gets uploaded

When the user runs `met serve <model-id>`, `crates/server/src/main.rs` does the following, roughly in order:

1. **Parse the model config.** `serve_phases::load_model_config(model_dir)` reads `config.json` and hands it to `metrale_config::parse_config`, which fills a `ModelConfig` including its nested text/vision configs.
2. **Resolve the kernel target.** `select_kernel_target` (`serve_load/model_setup.rs`) calls `metrale_kernels::ptx_for_config(model_type, hidden_size, model_refs, --kernel-target)`. Targets that declare the exact `(model_type, hidden_size)` are tried before wildcard ones; a collision is broken by each target's `match_names` against the HF id, `--model-name` and the model directory, and a tie that does not break is an error (`crates/kernels/src/resolve.rs`). No match fails fast, listing the available targets.
3. **Check the quant pair.** `check_kernel_target` refuses a kernel target whose quant does not suit the checkpoint's before any weight loads.
4. **Instantiate the GpuBackend.** `MetraleCudaBackend::new(gpu_ordinal, ptx_modules)` uploads every embedded PTX module for the chosen target to the GPU, via `cuModuleLoadData`. Kernel handles are cached per `(module_name, function_name)` pair.
5. **Instantiate the ModelWeightLoader.** `metrale_model_engine::factory::loader_for_config(&config)` lower-cases `model_type`, turns `-` and `.` into `_`, matches on the result and returns `Box<dyn ModelWeightLoader>`.
6. **Load weights.** The loader translates HF weight names (`model.layers.0.self_attn.q_proj.weight`) into Metrale Engine layer types (`Qwen3AttentionLayer`), going through `WeightStore` (the `O_DIRECT` fast path) and the quantization helpers in `metrale_model_layers::weight_map`.
7. **Build layer trait objects.** Each loaded layer becomes a `Box<dyn TransformerLayer>` stored in the `TransformerModel`.
8. **Capture CUDA graphs.** Decode steps run inside a graph region (`GpuBackend::begin_capture` … `launch_graph`). Subsequent decodes replay the graph — one GPU launch for the whole forward pass.
9. **Bind the HTTP endpoint.** `axum::Router::new()...serve(&addr)` starts listening.

At this point dispatch is frozen. Every request goes through the same kernels, the same graph, the same streams.

## Phase 3 — per-request path

```
POST /v1/chat/completions
 │
 ▼
metrale_server::api::chat_completions  (axum handler)
 │
 ▼  1. Apply jinja chat template
 │  2. Tokenize
 │  3. Enqueue Request {prompt_ids, sampling, stream?, tools?}
 │
 ▼
metrale_server::scheduler              (SLAI or FIFO)
 │
 ▼  1. Allocate KV pages for prefix
 │  2. Chunked prefill through InferenceEngine
 │  3. Enter the decode loop
 │
 ▼
metrale_model_engine::model::TransformerModel  (the Model trait)
 │
 ▼  for layer in layers:
 │      layer.forward(&ctx)      ← dyn dispatch, one per layer
 │          └─ calls into layers/ops kernel launches
 │              └─ GpuBackend::launch(KernelHandle, grid, block, args, stream)
 │                  └─ CUDA cuLaunchKernel    (PTX on GPU)
 │
 ▼
Sampler                              (argmax / top-p / top-n-sigma / min-p)
 │
 ▼
Detokenize → stream chunk → HTTP response
```

Two dynamic-dispatch points:

- **`dyn TransformerLayer`** — one virtual call per layer per step. Layer types (`Qwen3AttentionLayer`, `MoeLayer`, `Qwen3SsmLayer`, `NemotronMamba2Layer`, `VisionEncoder`) hold their own pre-resolved `KernelHandle`s for the ops they need. The virtual call is cheap — typically ~ns — against a forward pass that takes ~0.1–1 ms per token.
- **`&dyn GpuBackend`** — one virtual call per kernel launch. Same argument; the overhead is negligible compared to the kernel itself.

Both virtual calls are unavoidable consequences of the specialization thesis: we want `metrale-server` to not know what `GpuBackend` it's talking to, and we want `TransformerModel` to not know what layer shape it's running. That's how new hardware and new models plug in.

With CUDA graphs enabled (the default in production), steps 5–6 collapse to a single `cuGraphLaunch` — the dynamic dispatch cost disappears into the graph capture phase.

## Where to look in the code

| Question | File |
|---|---|
| "How is the kernel target resolved at startup?" | `crates/kernels/src/resolve.rs` (`ptx_for_config`), called from `crates/server/src/main_modules/serve_load/model_setup.rs` |
| "How does a kernel get compiled at build time?" | `crates/kernels/build.rs`, `crates/kernels/build_target.rs`, `crates/core/src/compute.rs` |
| "How does a layer launch a kernel?" | `crates/model-layers/src/layers/ops/` |
| "How does the model loop over layers?" | `crates/model-engine/src/model/` (`TransformerModel`) |
| "How does the factory pick a `ModelWeightLoader`?" | `crates/model-engine/src/factory.rs` — `loader_for_config()` |
| "How is the HTTP request parsed into a scheduler job?" | `crates/server/src/api/`, `crates/server/src/scheduler/` |

The [metrale-gpu-runtime chapter](../crates/metrale-gpu-runtime.md) expands on `GpuBackend`; [metrale-model-engine](../crates/metrale-model-engine.md) on the layer/factory split; [metrale-kernels](../crates/metrale-kernels.md) on the build-time codegen. The [SBIO chapter](./sbio.md) explains why every arrow in the diagram above goes through a trait.
