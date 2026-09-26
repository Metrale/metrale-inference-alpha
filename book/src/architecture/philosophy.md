# Philosophy: AI Kernel HyperCompiling

Part I's Philosophy chapter answered *why* Metrale Engine specializes. This chapter answers *how the specialization thesis forces specific design choices in the code*, and what you should see — or, if you're writing a PR, what you should preserve — when you read the codebase.

The single design rule that every choice below derives from is:

> **Specialization is a directory, not a template.** Everything that varies across `(Hardware, Model_q)` targets lives in its own directory. Everything shared across them is abstraction *above* the kernel layer, not parameters *inside* one.

## Consequence 1: the kernel tree is a coordinate system

```
kernels/
  gb10/                               # Hardware
    HARDWARE.toml                     # vendor, arch, memory specs, serving defaults
    common/
      KERNEL.toml
      *.cu                            # the shared baseline
    qwen3-next-80b-a3b/               # Model
      MODEL.toml                      # layer counts, sampling defaults
      nvfp4/                          # Quantization
        KERNEL.toml                   # compiler flags, module names, [sources], [shadow]
        *.cu                          # only the files this target overrides
  hopper/
    HARDWARE.toml                     # [hardware] inherits = "gb10"
    common/                           # Hopper-only kernels
```

Three levels of directory, over a `common/` baseline. A target reads up to four
directories, each a *layer*: its own leaf (`kernels/<hw>/<model>/<quant>/`), its
own `common/`, and — when `HARDWARE.toml` names a parent with `[hardware]
inherits` — the parent's leaf and `common/`. A layer holds the sources in its
directory plus whatever its `KERNEL.toml` `[sources] use` brings in from another
directory of the same tree. The resolver is `metrale-closure`'s layout module
(`crates/closure/src/layout.rs`); the kernels build script compiles what it
returns.

A leaf holds *only its divergences*, not a full kernel set: a leaf file whose
stem matches a `common/` file **shadows** it, whole-file, not per-symbol, and
the own tier wins over the parent tier. Every such win must be declared in the
winner's `KERNEL.toml` `[shadow]` table with a reason; an undeclared shadow is a
build error, and so is a declaration with nothing to shadow. Two leaves
therefore **do** share source for everything neither of them overrides; what
is guaranteed is that where a target *does* diverge, it diverges in a file
nothing else compiles.

The corollary is a real failure class: a shadow that has quietly become
identical to `common/` overrides nothing while masking every later `common/`
improvement, and two models keeping private byte-identical copies of the same
file will drift apart. CI's `kernel-structure` job
(`scripts/check_kernel_shadows.py` and `crates/kernels/tests/kernels_structure.rs`)
rejects both; the sanctioned way for one leaf to compile another's file is
`[sources] use`.

Inheritance is how `kernels/hopper` and `kernels/b200` reuse GB10: they inherit
`gb10`, so every GB10 source is compiled for their arch unless their own tier
supplies a file of the same name, and their `common/` holds only what is
specific to them.

The three `.toml` files are the only metadata the build system consumes. `HARDWARE.toml` tells `crates/kernels/build.rs` which `ComputeTarget` impl to use (`nvidia`, `amd`, `hip`, `apple`), what arch flag to pass the compiler, and — in `[defaults]` — the SERVING levers this target runs with, baked into the binary as `metrale_kernels::TARGET_DEFAULTS` and read before the environment, so "what does this hardware serve with" is answered by a file in the repository rather than by a launch script outside it. `MODEL.toml` is the per-model behavior SSOT — sampling presets, thinking budgets, tool-call parser defaults. `KERNEL.toml` sets compiler flags and module names, and declares `[sources] use` and `[shadow]`.

Adding a model or a hardware target is, at the file-system level, *creating a new directory*. No code elsewhere in the repository needs to move.

## Consequence 2: the runtime crate structure mirrors the axis split

Read the workspace `Cargo.toml` and you'll see twenty-one workspace members. Group them by what axis of variation they insulate:

| Axis they insulate | Crates |
|---|---|
| *Hardware vendor* | `metrale-core` (`ComputeTarget`, `Vendor` enum, `KernelTarget`), `metrale-gpu-runtime` (`GpuBackend`), `metrale-comm` (`CommBackend`) |
| *Model architecture* | `metrale-model-arch` (`ModelWeightLoader` trait, per-family loaders), `metrale-model-layers` (`TransformerLayer` trait) |
| *Quantization format* | `crates/model-layers/src/quant_format/` (per-format modules + runtime dispatch), `crates/core/src/numeric.rs` (host-side FP8/BF16 conversions) |
| *Compiled kernels (one artifact per axis combination)* | `metrale-kernels` (embedded PTX modules, auto-generated from the kernel tree) |
| *Request serving* | `metrale-server` (HTTP, tokenizer, tool parsing) |
| *Measurement* | `metrale-bench` |

Each crate has exactly one reason to change. A new GPU vendor never touches `metrale-model-layers`. A new model family never touches `metrale-gpu-runtime`. A new quantization scheme touches `metrale-model-layers`'s format modules and `metrale-kernels`, but not the layer code. This orthogonality is not a happy accident of the crate layout — it *is* the architectural consequence of the specialization thesis.

## Consequence 3: SBIO — business logic never touches I/O

Business logic — the layer code in `metrale-model-layers`, the scheduler in `metrale-server` — never calls CUDA APIs, never opens a socket, never reads a file. Every such operation goes through a trait:

- GPU memory, launches, graphs → `GpuBackend`
- Collective comms → `CommBackend`
- Weight loading I/O → `WeightStore` (wraps safetensors + `O_DIRECT`)
- HTTP responses → `axum` handlers, tested against a mock channel

This is what the user instructions call **SBIO** (Separation of Business logic from I/O). The payoff is that 80%+ of the codebase is unit-testable without a GPU. `MockGpuBackend` records launches but does not execute them. `SingleGpuBackend` is a no-op `CommBackend` for single-GPU runs. The [SBIO chapter](./sbio.md) shows the pattern in detail.

## Consequence 4: zero runtime compilation

Every general-purpose framework has, somewhere, a codepath that compiles kernels at runtime. PyTorch has `torch.compile`. vLLM has Triton JIT. TensorRT-LLM has TRT engine builds. Each of those is a slow path the first time you hit a new shape, and an ongoing operational surface the ops team has to manage (cache directories, warm-up scripts, cold-start budgets).

Metrale Engine has none of it. `crates/kernels/build.rs` enumerates every `(H, M_q)` target matching the `METRALE_TARGET_*` env vars, compiles every `.cu` file for every matching target, and emits one auto-generated `target_ptx.rs` that is `include!`'d into the crate. The release binary contains every PTX module we ship. Startup is "mmap the binary, upload PTX to the GPU, capture CUDA graphs for a handful of batch sizes, done".

This is what "embedded in the binary" means throughout the book. It is the concrete mechanism by which specialization does not cost operator pain.

## Consequence 5: one binary per installation, N kernel sets

You deploy one Docker image. It contains one `metrale-server` binary. It contains every `(gb10, model, quant)` PTX set embedded in that binary. At startup, the binary reads the model's `config.json`, computes the canonical `model_type`, looks up the matching `KernelTarget`, and uses that set.

The knobs that let this scale:

- **`METRALE_TARGET_*=*` at build time** — compiles every matching target. The default image sets everything to `*` and ships the lot.
- **`METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=qwen3.5-35b-a3b METRALE_TARGET_QUANT=nvfp4`** — compiles exactly one target. Used for per-model slim images in `docker/gb10/<model>/`.
- **`METRALE_SKIP_BUILD=1`** — emits a stub `target_ptx.rs` so that `clippy`, `fmt`, `check`, and any non-GPU test can run on a vanilla Linux host.

The same image works across all supported targets. The startup dispatcher picks the right kernels. Operators don't manage a kernel cache; they don't warm a JIT; they don't think about it.

## What the rest of this book is about

Every subsequent chapter is an elaboration of one of the consequences above. The [workspace layout chapter](./workspace.md) walks the directory tree. The [dispatch chapter](./dispatch.md) traces a single request from HTTP to kernel launch. The [SBIO chapter](./sbio.md) shows how the testability claim actually holds.

The deep-dive chapters in Part IV show what the kernels look like — what a hand-tuned kernel set per target buys you, and how you'd write new ones when you're porting Metrale Engine to your own `(H, M_q)` target.

## Reading the architecture categorically

The design choices above have precise names in category theory. The target set `𝒯 = Hw × Mod × Quant` is a categorical **product**; the crate split is that product made syntactically real, which is why orthogonality of axes is a structural fact and not a convention. The kernel registry is a **coproduct** (disjoint union of per-target PTX sets), which is why adding a summand cannot regress existing summands. The `GpuBackend` trait defines an **algebraic theory** with two ship-worthy models — `MetraleCudaBackend` and `MockGpuBackend` — and that is what makes the test suite runnable without a GPU. A general framework is, in this vocabulary, an engine that factors `Kernels : 𝒯 → 𝐒𝐞𝐭` through a smaller "essence" category; Metrale Engine refuses the factoring, and the 3.6× gap against vLLM is the cost of the factoring that Metrale Engine does not pay.

The appendix [A Category-Theoretic Perspective](../appendix/category-theory.md) works through each of these structures at appendix length. It is a design reference, not a prerequisite.
