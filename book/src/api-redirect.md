# API Reference

<script>window.location.replace("https://docs.dev.metrale.ai/");</script>
<noscript>
<meta http-equiv="refresh" content="0; url=https://docs.dev.metrale.ai/">
</noscript>

If you are not redirected automatically, follow this link to the [Rust API reference](https://docs.dev.metrale.ai/).

The API reference is generated from the crate source with `cargo doc --workspace --no-deps` on every merge to `main`. Top-level crates:

- [`metrale_core`](https://docs.dev.metrale.ai/metrale_core/) — target abstractions, tensor, dtype, host-side FP8/BF16 numerics
- [`metrale_config`](https://docs.dev.metrale.ai/metrale_config/) — the typed model config tree (HF config.json parsing, GGUF metadata, capabilities)
- [`metrale_closure`](https://docs.dev.metrale.ai/metrale_closure/) — content hash of a kernel target's transitive include closure
- [`metrale_governance`](https://docs.dev.metrale.ai/metrale_governance/) — the PR journey ledger: an append-only record of how a change reached main
- [`metrale_kernels`](https://docs.dev.metrale.ai/metrale_kernels/) — embedded PTX registry
- [`metrale_gpu_sys`](https://docs.dev.metrale.ai/metrale_gpu_sys/) — raw FFI only — cuFile (GDS) and NVML dlopen wrappers, NCCL bindings, RDMA verbs C shim
- [`metrale_telemetry`](https://docs.dev.metrale.ai/metrale_telemetry/) — run metrics, kernel audit, launch trace, progress hooks, and the metal-up instruments (NVML device sampling, energy attribution, GPU spans, scheduler/request telemetry) with their exporters
- [`metrale_gpu_runtime`](https://docs.dev.metrale.ai/metrale_gpu_runtime/) — `GpuBackend` (CUDA/Metal), streams, buffers, op cache, pinned hosts, kernel registry, cuBLASLt/CUTLASS/FlashInfer bridges
- [`metrale_scheduler`](https://docs.dev.metrale.ai/metrale_scheduler/) — the scheduler's I/O contract — step plans, effects, the device/clock/spill router traits, the effect-trace decorator and the one driver loop. No model dependency
- [`metrale_cache`](https://docs.dev.metrale.ai/metrale_cache/) — KV cache, KV dequant, KV spill, radix-tree prefix cache
- [`metrale_sampling`](https://docs.dev.metrale.ai/metrale_sampling/) — the sampler
- [`metrale_storage`](https://docs.dev.metrale.ai/metrale_storage/) — GDS/RDMA tiers, expert paging, cache peers, tiered-cache core (SlotArena/SwapStore/Residency)
- [`metrale_comm`](https://docs.dev.metrale.ai/metrale_comm/) — collective-operation backends (NCCL, single-GPU)
- [`metrale_grammar`](https://docs.dev.metrale.ai/metrale_grammar/) — pure-Rust grammar-constrained decoding (XGrammar port)
- [`metrale_model_weights`](https://docs.dev.metrale.ai/metrale_model_weights/) — weight store and loaders (safetensors, fast weights, RDMA weight/LoRA tiers, preflight)
- [`metrale_model_layers`](https://docs.dev.metrale.ai/metrale_model_layers/) — generic layers (attention, SSM, MoE, FFN, norm, MTP heads, vision, ops), LoRA, weight map, the draft-proposer contract
- [`metrale_model_arch`](https://docs.dev.metrale.ai/metrale_model_arch/) — per-family architectures (GLM-5 Next, DeepSeek V4.1, Nemotron, Kimi K3, DFlash/MTP heads) and their weight loaders
- [`metrale_model_engine`](https://docs.dev.metrale.ai/metrale_model_engine/) — the Model trait, the transformer model (prefill, decode, verify, SSM state), the generate engine and the model factory
- [`metrale_speculative`](https://docs.dev.metrale.ai/metrale_speculative/) — speculative-decoding policy — MTP gate, adaptive and DFlash rungs, n-gram proposer, spec capacity and stats, scheduler snapshots
- [`metrale_bench`](https://docs.dev.metrale.ai/metrale_bench/) — plugin + benchmark abstraction and registry driven by the metrale-server TUI
- [`metrale_server`](https://docs.dev.metrale.ai/metrale_server/) — pure Rust LLM inference server (HTTP API, scheduler, TUI, CLI)
