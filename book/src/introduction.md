# Introduction

<div class="metrale-headline">Metrale Engine is a pure-Rust LLM inference engine, built around the belief that every <code>(Hardware, Model<sub>q</sub>)</code> tuple deserves its own hand-tuned kernel set.</div>

On an NVIDIA GB10 Grace-Blackwell Superchip, Metrale Engine reaches **131 tok/s on Qwen3.5-35B-A3B** — **3.6× faster than NVIDIA's vLLM on the same model, same hardware**. It reaches **104 tok/s on Qwen3-Next-80B-A3B** and **46 tok/s on Qwen3.5-122B-A10B** (EP=2, two nodes). On a head-to-head suite of 32 micro-benchmarks against PyTorch — attention, GEMM, SSM, RoPE, RMSNorm — Metrale Engine wins **32 out of 32**, with speedups from 1.04× up to 18.2×.

This book is the canonical long-form documentation for Metrale Engine. It complements — rather than replaces — the source-of-truth material already in the repository:

- [`QUICKSTART.md`](https://github.com/Metrale/metrale-inference-alpha/blob/main/QUICKSTART.md) — Docker recipes for every supported model.
- [`AGENTS.md`](https://github.com/Metrale/metrale-inference-alpha/blob/main/AGENTS.md) / [`CONTRIBUTING.md`](https://github.com/Metrale/metrale-inference-alpha/blob/main/CONTRIBUTING.md) — contributor workflow.
- [`docs/`](https://github.com/Metrale/metrale-inference-alpha/tree/main/docs) — design notes, release history, benchmark journeys.

## Who this book is for

Three audiences, one narrative arc:

- **Operators** who want to serve one of the supported models on a GB10 today (every `(model, quant)` target under `kernels/gb10/` ships in the image; the compatibility matrix is `docs/GB10_DEPLOYMENT_GUIDE.md` §2). Start with *Installation* and *Quickstart*, then jump to *Operating Metrale Engine* for CLI flags, KV-cache dtypes, and multi-GPU bring-up.
- **Model authors** extending Metrale Engine with a new architecture. Read *Architecture* → *metrale-model-engine* → *Engineering Deep Dives* in order, then follow [Adding a new model family](https://github.com/Metrale/metrale-inference-alpha/blob/main/docs/HARDWARE.md#adding-a-new-model-family) alongside `crates/model-arch/src/weight_loader/minimax.rs` as a template.
- **Kernel engineers** porting Metrale Engine to a new hardware target or hyperoptimizing an existing kernel. Read *Philosophy* → *Kernel Dispatch Pipeline* → the *CUDA Kernel Engineering* deep dive, then use the [Adding a new hardware target](https://github.com/Metrale/metrale-inference-alpha/blob/main/docs/HARDWARE.md#adding-a-new-hardware-target) walkthrough with `kernels/gb10/` as a reference implementation.

## What Metrale Engine is not

- **Not a training framework.** Metrale Engine serves; it does not fine-tune. Use vLLM-style baselines, trl, or axolotl upstream.
- **Not a generic kernel.** Metrale Engine does *not* cover the matrix with one templated CUDA kernel. It covers it by specializing per `(hardware, model, quantization)` target and wrapping those kernels in abstractions designed for broad support — `ComputeTarget` (vendor-agnostic build), `GpuBackend` (vendor-agnostic runtime), `CommBackend` (vendor-agnostic collectives). The first hardware we shipped is GB10, with Hopper (H100/H200) and B200 targets beside it; the design is explicitly multi-vendor, and porting to Apple Silicon, AMD or Intel is a well-scoped piece of work, not an architectural change.
- **Not a Python wrapper.** Metrale Engine is pure Rust + GPU source. There is no Python in the serving path — no PyTorch, no Triton JIT, no runtime compilation. Every kernel is compiled to its hardware's native binary (PTX today) at build time and embedded in the Rust binary.

## What you get

A single multi-model Docker image (`metrale/metrale-inference-gb10:latest`), an OpenAI-compatible HTTP server, and the machinery to port Metrale Engine to fresh `(H, M_q)` targets. The model matrix spans Qwen3 through Qwen3.6, Qwen3-VL, Gemma-4, Mistral-Small-4, MiniMax-M2.7, Nemotron-3 Nano and Super, and DeepSeek-V4-Flash — covering dense, hybrid SSM/attention, MoE, vision, MLA, and 256-expert routing. The engine ships with MTP speculative decoding, RadixAttention prefix caching with SSM snapshots, FP8 and NVFP4 KV caches, per-batch CUDA graphs, chunked prefill, tool calling in several wire formats, and RoCEv2-backed expert parallelism for models beyond a single GB10.

## How to read this book

The TOC is linear but the parts are independent. If you came here to run a model, skip straight to [Quickstart](./getting-started/quickstart.md). If you came here to understand *why* Metrale Engine is fast, read [Philosophy](./architecture/philosophy.md) first — the rest of the book is a working demonstration of that claim.
