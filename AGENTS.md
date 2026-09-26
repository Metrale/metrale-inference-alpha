# AGENTS.md

A contributor guide for AI agents (and humans) working on Metrale Engine. Read this
alongside [CONTRIBUTING.md](CONTRIBUTING.md). The tone is practical: paths,
commands, and the invariants that matter.

## What Metrale Engine is

Metrale Engine is an MIT OR Apache-2.0 inference stack targeting NVIDIA GB10 / DGX Spark. The
moving parts:

- **`crates/server/`** (`metrale-server`, the `met` binary) — OpenAI- and
  Anthropic-compatible HTTP server, request scheduling, tool-call parsing,
  streaming, TUI, CLI.
- **`crates/model-engine/`** — the `Model` trait, the transformer model and
  the model-type dispatch in `src/factory.rs`.
- **`crates/model-arch/`** — per-family architectures and their weight
  loaders (`src/weight_loader/`).
- **`crates/model-layers/`** — generic layers (attention, SSM, MoE, FFN,
  vision, MTP heads), LoRA, the weight map and the kernel launch wrappers.
- **`crates/gpu-runtime/`** — GPU backend (CUDA, Metal), streams, buffers,
  kernel registry; **`crates/cache/`** the KV and prefix caches;
  **`crates/comm/`** collective ops.
- **`crates/kernels/`** — Rust glue over compiled PTX (one artefact
  per `(hw, model, quant)` target).
- **`kernels/<hw>/`** — kernel sources: `common/` plus one leaf per
  `<model>/<quant>/`, with `MODEL.toml` (sampling, behaviour defaults,
  kernel target registration) beside each model.
- **`crates/bench/`** (`metrale-bench`) — the benchmark suite and the
  certification gate.
- The rest of the 21 workspace members (`core`, `config`, `closure`,
  `governance`, `gpu-sys`, `telemetry`, `scheduler`, `sampling`, `storage`,
  `grammar`, `model-weights`, `speculative`) are listed in the root
  `Cargo.toml`; the book's *Workspace Layout* chapter maps each one.

Architecture decision records live in `docs/adr/`; the benchmark journey in
`docs/METRALE_JOURNEY.md`; release notes in `docs/releases/`.

## Ground rules

- **SPDX header on every source file.** `// SPDX-License-Identifier:
  MIT OR Apache-2.0` on line 1 of every `.rs`, `.cu`, `.cuh`, `.h`, `.hpp`,
  `.cpp`. Enforced by Github Pipeline.
- **License is MIT OR Apache-2.0.** Third-party code keeps its own licence
  and is listed in `THIRD_PARTY_NOTICES.md`. `deny.toml` controls what
  dependency licenses are allowed.
- **Don't regress on models already in the support matrix** — Qwen3/Qwen3.5/
  Qwen3.6/Qwen3-Next/Qwen3-VL, Nemotron-3, Mistral-Small-4, Gemma-4, MiniMax-M2.7.
  The complete, current model×quant matrix (the SSOT for what's supported) is
  [`docs/GB10_DEPLOYMENT_GUIDE.md`](docs/GB10_DEPLOYMENT_GUIDE.md) §2; the
  per-model kernel registry is `kernels/gb10/<model>/MODEL.toml`.

## Local checks before a PR

The commands CI will run:

```bash
# 1. Formatting
cargo fmt --all -- --check

# 2. Lints (the build-script gate lets clippy run without CUDA on the host;
#    matches ci.yml — deny-warnings comes from [workspace.lints], not a flag)
METRALE_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests

# 3. License headers (SPDX MIT OR Apache-2.0 line 1; wraps the same apache/skywalking-eyes
#    engine CI runs against .licenserc.yaml):
bash scripts/check-license-headers.sh

# 4. Typos
typos  # crate-ci/typos — install once, `cargo install typos-cli`

# 5. Cross-hardware kernel reach — ONLY if the diff touches kernels/.
#    kernels/<hw>/ looks like one tree per hardware and is not: strix `use`s 98
#    files of kernels/gb10/common/, b300 `use`s 185, and hopper/b200 inherit it. A gb10 edit therefore
#    changes what AMD compiles, and the CI job `cross-hardware kernel reach
#    (CHKI, advisory)` will say so. ADVISORY since 2026-09-11 — AMD is
#    second-tier support, so a red does not block a merge. It is still the only
#    thing that sees this class of reach: read it, do not skip it.
python3 scripts/check_cross_hardware.py --base origin/main --worktree
```

A real build + test cycle requires a CUDA-capable host; see
[CONTRIBUTING.md](CONTRIBUTING.md).

### If the diff touches a perf path

`crates`, `kernels`, `Cargo.*`, `vendor`, `jinja-templates`, `rust-toolchain.toml`
— the gate's `PERF_PATHS` — mean the PR owes a benchmark certification
campaign of **~4.5–5 GPU-hours**, and campaigns have been wasted. Before spending one, and
again before reading `stamp status` / `seal status`, committing `.benchmarks/` records, or
commenting `/stamp` or `/seal`, check these fourteen things: a clean perf tree, a frozen sha,
a binary built from it, `met doctor`, no stray shard dirs, hermetic pins, BFCL scorer imports,
a free box, `campaign-guard.sh`, no other PR mid-certification, top of stack, stamp/seal job
sequencing, one `git_sha` across added records, and one Speed-class signer. `met bench
certify` claims the campaign lock itself. Anything you override, quote in the PR.

★ **A red `stamp status` or `seal status` is not a failure until it is dated.** Those jobs
freeze their outputs for the life of a CI run, so a mark minted *after* they ran leaves them
red until a FULL `gh run rerun <id>` — `--failed` cannot work, because they *succeeded* while
emitting `false`. Compare the mark's time with the run's start before reading the red.

For a `kernels/` change that reaches a second hardware, choose the remedy before pushing
(benign, parameterize in `kernels/<hw>/HARDWARE.toml` **with a reader added in the same
change**, or a separate kernel in the intended tree) and record it in `Hardware:` and
`CHKI-Verdict:` commit trailers.

The CI job is **advisory** (AMD second-tier), so those trailers are not enforced at merge
— which makes the check a judgement you make rather than one CI makes for you. The
reach it reports is real either way: a `kernels/gb10/common/` edit is compiled verbatim by
hipcc through `[sources] use`, and `d584c0c50` was caught only because a release compile leg
happened to fail.

## Adding a new model

High-level walkthrough — the patterns to follow are already in-tree.

1. **Model-type dispatch.** Add a new arm in
   `crates/model-engine/src/factory.rs` that returns a new
   `ModelWeightLoader` impl. Use `crates/model-arch/src/weight_loader/`
   for the loader (study `qwen35.rs`, `minimax.rs`, `nemotron.rs` for the
   three major shapes: dense, SSM+MoE hybrid, attention+MoE).
2. **Kernel target.** Create `kernels/<hw>/<model-slug>/MODEL.toml`
   declaring the model-type matches, sampling presets and behaviour
   defaults, and a `<quant>/` leaf with its `KERNEL.toml` and any kernels it
   overrides. The build picks the new target up automatically under
   `METRALE_TARGET_MODEL=*` (the default).
3. **Behavioural knobs.** `MODEL.toml` is the SSOT for per-model
   sampling/thinking/tool-use policy. The `metrale-kernels` build script parses
   it into `SamplingPresets` + `ModelBehavior` consumed by the server.
4. **Jinja template.** If the model uses a chat template that's not
   covered by `jinja-templates/`, add one, named after the checkpoint's
   `config.json` `model_type` (`jinja-templates/<model_type>.jinja`).

Concrete recent examples worth reading:

- Mistral-Small-4 integration — `crates/model-arch/src/mistral_loader/`
  + `kernels/gb10/mistral-small-4/`.
- MiniMax M2/M2.7 (attention + 256-expert sigmoid-routed MoE) —
  `crates/model-arch/src/weight_loader/minimax.rs` +
  `kernels/gb10/minimax-m2-229b/`.
- Gemma-4 (sliding/full attention alternation) —
  `crates/model-arch/src/weight_loader/gemma4.rs` +
  `kernels/gb10/gemma-4-*/`.

## The kernel target system

Three dimensions: **hardware** × **model** × **quantization**. At build
time, `crates/kernels/build.rs` reads the `METRALE_TARGET_*` env vars
(with `*` meaning "all matching") and produces one PTX artefact per
target. Runtime selects the correct target from the model's
`model_type` and `hidden_size` (`metrale_kernels::ptx_for_config`).

- `METRALE_TARGET_HW` — one hardware directory under `kernels/` (`gb10`,
  the default; `hopper`, `b200`, `b300`, `strix`, `strix-hip`, `metal`).
- `METRALE_TARGET_MODEL` (default `*`) / `METRALE_TARGET_QUANT` (default
  `nvfp4`) — `*` compiles all.
- `METRALE_SKIP_BUILD=1` — emits a stub so clippy/fmt can run without nvcc.

## Writing commits

- One logical change per commit. Don't bundle an unrelated cleanup with a
  bug fix.
- Message format: `<area>: <imperative summary>` — e.g.
  `metrale-server: preserve template-forced thinking through EP=2`.
- If the change affects runtime behaviour, rebuild the Docker image from
  scratch and run the relevant slice of the validation suite
  (`tests/single_gpu_suite.py` for most cases) before opening the PR.

## Failure modes that cost us time

These aren't abstract — they're the classes of bug that have burned days:

- **Protocol drift** between OpenAI and Anthropic paths (`crates/server/src/api/`,
  `anthropic/`). A fix on one surface often needs a matching change on
  the other.
- **Template mismatches** that break tool-calling subtly — different
  `<tool_call>` vs `<minimax:tool_call>` tokens, `<think>` seeded by the
  template vs emitted by the model, `thinking_budget` enforcement.
- **FP8 / KV / quantization edge cases** — BF16 paged cache routed into an
  FP8 kernel, silent NaN. If your change touches numeric paths, verify
  with a real model before claiming success.
- **Docs drift** — CLI flags, release commands, quick-start snippets.
  Verify against the current binary, not your memory.

When you hit a regression, **never assume the model is at fault** — always
look for the Metrale Engine bug first.

## Scope and escalation

If a task is ambiguous, stop and ask in the issue/PR before implementing.
If the scope grows past "one PR", split it. If you're about to modify
something shared (a cross-cutting trait, a build script, CI config), flag
it in the PR description so reviewers catch it.

## Code Principles & Agent Workflow

To ensure high code quality, all agents contributing to Metrale Engine must strictly adhere to these core programming principles:

### Core Directives
- **Minimal Edits:** Make the smallest edit necessary—sufficient but not excessive.
- **TDD & Testing:** Test-driven development is required. Minimize test mocking; maximize production code coverage. Never add test-specific workarounds to production paths.
- **File Size:** Keep Rust source files ≤500 LoC — this is the CI-enforced cap (`.github/workflows/file-size-cap.yml` is the SSOT). Split larger files into sub-modules per the Metrale Engine idiom (see `crates/model-layers/src/layers/qwen3_attention/` for the compute-heavy template, `crates/model-arch/src/weight_loader/` for the variant-dispatch template).
- **Security:** Write secure code adhering to OWASP, CWE, and NIST standards.

### The "Big Three" Invariants (Always Apply)
- **SSOT (Single Source of Truth):** Every data item has exactly one authoritative source. Derive, don't duplicate.
- **PCND (Production Code, No Defaults):** No implicit defaults in production code. Require explicit config or fail fast.
- **SBIO (Strict Boundary for I/O):** Business logic never performs I/O directly. Route through an IORouter abstraction.

### Triggered Principles
- **SDD (Split Driven Design):** Use when multiple implementations are needed, breaking apart large files, or eliminating duplication.
- **CBD (Complex Bug Debugging):** Apply for non-trivial bugs, race conditions, async issues, or unclear failure modes.

### Agent Workflow
- **Plan First:** For any non-trivial task, create a detailed plan before implementation. Use subagents for complex exploration.
- **Verify Before Done:** Never consider a task complete without proving it works (e.g., via tests or logs).
- **Autonomous Fixes:** When given a bug report, fix it autonomously without asking for hand-holding.
- **Self-Improvement:** After user corrections, capture the lesson to prevent the same mistake.
- **Demand Elegance:** For complex fixes, choose the elegant, well-architected solution over a hacky workaround.

See `CONTRIBUTING.md` for coding style and the CLA expectations,
`SECURITY.md` for disclosure, and `docs/adr/` for the authoritative
architecture references.
