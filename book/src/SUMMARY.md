# Summary

[Introduction](./introduction.md)

---

# Part I — Getting Started

- [Philosophy](./getting-started/philosophy.md)
- [Installation](./getting-started/installation.md)
- [Quickstart](./getting-started/quickstart.md)
- [Supported Models](./getting-started/models.md)
- [Troubleshooting](./getting-started/troubleshooting.md)

# Part II — Architecture

- [Philosophy: AI Kernel HyperCompiling](./architecture/philosophy.md)
- [Workspace Layout](./architecture/workspace.md)
- [Kernel Dispatch Pipeline](./architecture/dispatch.md)
- [SBIO: Business Logic vs I/O](./architecture/sbio.md)

# Part III — The Crates

- [metrale-core](./crates/metrale-core.md)
- [metrale-config](./crates/metrale-config.md)
- [metrale-closure](./crates/metrale-closure.md)
- [metrale-governance](./crates/metrale-governance.md)
- [metrale-kernels](./crates/metrale-kernels.md)
- [metrale-gpu-sys](./crates/metrale-gpu-sys.md)
- [metrale-telemetry](./crates/metrale-telemetry.md)
- [metrale-gpu-runtime](./crates/metrale-gpu-runtime.md)
- [metrale-scheduler](./crates/metrale-scheduler.md)
- [metrale-cache](./crates/metrale-cache.md)
- [metrale-sampling](./crates/metrale-sampling.md)
- [metrale-storage](./crates/metrale-storage.md)
- [metrale-comm](./crates/metrale-comm.md)
- [metrale-grammar](./crates/metrale-grammar.md)
- [metrale-model-weights](./crates/metrale-model-weights.md)
- [metrale-model-layers](./crates/metrale-model-layers.md)
- [metrale-model-arch](./crates/metrale-model-arch.md)
- [metrale-model-engine](./crates/metrale-model-engine.md)
- [metrale-speculative](./crates/metrale-speculative.md)
- [metrale-bench](./crates/metrale-bench.md)
- [metrale-server](./crates/metrale-server.md)

# Part IV — Engineering Deep Dives

- [CUDA Kernel Engineering](./deep-dives/kernels.md)
- [NVFP4 Quantization](./deep-dives/nvfp4.md)
- [FP8 Native Serving](./deep-dives/fp8.md)
- [Attention & Paged KV Cache](./deep-dives/attention.md)
- [MoE Routing & Experts](./deep-dives/moe.md)
- [SSM / Mamba / GDN Layers](./deep-dives/ssm.md)
- [Speculative Decoding (MTP)](./deep-dives/mtp.md)
- [Constrained Decoding (XGrammar)](./deep-dives/xgrammar.md)

# Part V — Operating Metrale Engine

- [OpenAI-Compatible Server](./operations/server.md)
- [Tool Calling & Streaming](./operations/tools.md)
- [Multi-GPU & EP=2](./operations/multi-gpu.md)
- [Benchmarking](./operations/benchmarks.md)
- [Certification](./operations/certify.md)

# Part VI — The Project

- [Contributing](./project/contributing.md)
- [How a Change Lands](./project/landing.md)
- [The Merge Lattice](./project/merge-lattice.md)
- [Security Policy](./project/security.md)
- [Release Notes](./project/releases.md)

---

# Appendix

- [Paper Summary](./appendix/paper.md)
- [A Category-Theoretic Perspective](./appendix/category-theory.md)
- [Glossary](./appendix/glossary.md)
- [Further Reading](./appendix/reading.md)

---

[API Reference ↗](./api-redirect.md)
