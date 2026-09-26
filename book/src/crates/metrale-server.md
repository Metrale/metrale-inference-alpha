# metrale-server

**Path:** `crates/server/`
**Role:** the `met` binary. OpenAI- and Anthropic-compatible HTTP server, request scheduler, tokenizer, tool parsing, streaming, rate limiter, TUI and CLI.
**Key modules** (under `crates/server/src/`): `main.rs` and `main_modules/` (startup, router, model host and hot-swap), `cli.rs` and `cli/` (every subcommand's arguments), `api/`, `openai/`, `anthropic/`, `scheduler/`, `scheduling_policy.rs`, `tool_parser/`, `reasoning_parser/`, `tokenizer/`, `grammar/`, `rate_limiter.rs`, `refusal.rs`, `metrics.rs`, `conversation_store.rs`, `response_store.rs`, `session_manager.rs`, `model_resolver.rs`, `recipe/`, `tui/`. The n-gram proposer is `crates/speculative/src/ngram.rs`.

Building `metrale-server` produces the `met` executable that the Docker images ship. The other crates are libraries (a few carry developer tools as extra bins); `metrale-server` ties them together.

## Subcommands

`met serve` runs the server; `met benchmark` (alias `bench`) runs and inspects the benchmark suite and certification; `met doctor` reports whether this box can run a benchmark; `met sync-recipes` populates the local recipe index. `met --help` and `met <subcommand> --help` are the authoritative flag lists.

## Startup sequence (`met serve`)

The phases live in `main_modules/serve.rs`, `serve_load/` and `serve_phases/`:

1. **Parse the CLI** (`cli::Cli`) and validate the flag combination (`cli/validate.rs`).
2. **Resolve the model path** — an HF id through `HF_HUB_CACHE` / `HF_HOME` / `~/.cache/huggingface/hub`, or `--model-from-path`.
3. **Load `ModelConfig`** from `config.json` (`serve_phases::load_model_config`).
4. **Resolve the kernel target** (`serve_load/model_setup.rs::select_kernel_target`, `metrale_kernels::ptx_for_config`) and check its quant against the checkpoint's.
5. **Instantiate the `GpuBackend`** — `MetraleCudaBackend::new(ordinal, ptx_modules)`.
6. **Instantiate the `CommBackend`** — `NcclBackend` for a multi-rank run, `SingleGpuBackend` otherwise.
7. **Preflight the checkpoint** (`metrale-model-weights`'s preflight) and **build the model** through `factory::loader_for_config`; the weights land on the GPU.
8. **Load the tokenizer** and the chat template (`jinja-templates/<model_type>.jinja` when present, else the checkpoint's own).
9. **Spawn the scheduler** and **bind the HTTP listener** on `--bind`:`--port`.

[Kernel Dispatch](../architecture/dispatch.md) walks through the same path from the kernel side.

## HTTP routes (`main_modules/serve_router.rs`)

| Method | Path | Handler |
|---|---|---|
| GET | `/v1/models`, `/v1/models/{id}` | the one served model |
| POST | `/v1/chat/completions` | OpenAI chat; `GET /v1/chat/completions/{id}` returns a stored completion |
| POST | `/v1/completions` | OpenAI legacy completions |
| POST | `/v1/responses` | OpenAI Responses API (stateful); `/v1/responses/{id}/cancel` |
| POST | `/v1/conversations` | conversation store for the Responses API |
| POST | `/v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages |
| POST | `/v1/lora/active`, `/v1/lora/load` | LoRA adapter rotation and slot loading |
| POST | `/tokenize`, `/detokenize` | helpers, gated behind `--require-auth` when it is set |
| GET | `/health`, `/health/live` | readiness and liveness |
| GET | `/metrics`, `/v1/events` | Prometheus metrics and the telemetry event stream |
| GET | `/hardware`, `/serve-config` | what this box is and how it is serving (read by the benchmark harness) |

OpenAI endpoints the engine does not serve (embeddings, files, audio, images, moderations, batches) are routed to stubs that return a clear "not supported" error rather than a 404.

Streaming is the default for chat; non-streaming aggregates and returns `ChatCompletionResponse`. Tool-call chunks are emitted as `delta.tool_calls` in the SSE stream. Anthropic streaming populates `stop_sequence` on `message_delta` events.

## The scheduler (`scheduler/` + `scheduling_policy.rs`)

Two policies, selected with `--scheduler`:

- **`fifo`** (default) — first come, first served. A decode step picks up to `--max-batch-size` active sequences and runs a batched forward.
- **`slai`** — deadline-aware. New prefills are skipped while any active sequence has waited 80% of `--tbt-deadline-ms` since its last token, and the queue is prefilled front first, then shortest prompts.

`--scheduler-config` picks the execution router: `sync` (every device step completes before the next host decision) or `async` (one plain decode step runs ahead of the host on a model with a device token feed). The I/O contract both routers run under is [metrale-scheduler](./metrale-scheduler.md).

KV pages are claimed on prefill start and released on completion. The scheduler tracks the KV budget and chunks prefills (`--max-prefill-tokens` caps the per-iteration tokens so scratch sizes stay bounded).

Active context compaction (`compact_messages` in `api/compact.rs`) applies at the HTTP level before tokenization when `--auto-compact` is set: as the tokenized prompt approaches `--max-seq-len`, the server progressively truncates middle tool responses, replaces them with pointers, drops the oldest middle pairs, and finally trims the system prompt and keeps only the last messages.

## Tool-call parsing (`tool_parser/`)

The parser comes from `--tool-call-parser`, else `MODEL.toml` `[behavior].tool_call_parser`, else the model type's entry in `crates/server/tool_defaults.toml`:

| Parser | Format |
|---|---|
| `hermes` | JSON in `<tool_call>{...}</tool_call>` |
| `qwen3_coder` | XML, nested `<function=...><parameter=...>` |
| `qwen3_xml` | the Qwen3 XML variant |
| `gemma4` | Gemma-4's tool-call syntax |
| `mistral` | JSON block with an explicit `[TOOL_CALLS]` prefix |
| `minimax_xml` | MiniMax's `<minimax:tool_call>` XML |
| `bare_json` | an unwrapped JSON object |
| `poolside_v1` | Poolside's format |

The parsers tolerate slightly malformed model output (a literal `</tool_call>` inside arguments, a missing `</parameter>`, empty `{}` calls).

## Reasoning / thinking (`reasoning_parser/`)

Models that emit `<think>...</think>` blocks stream the thinking content to the client separately from the answer. `--max-thinking-budget` caps the thinking tokens; `--disable-thinking` is a kill-switch. The parser distinguishes a template-seeded `<think>` from the model's own, treats `<think>\n\n</think>\n\n` as a template no-op rather than a reasoning block, and concatenates multiple blocks.

## Tokenizer (`tokenizer/`)

Wraps the HF `tokenizers` crate and adds chat-template expansion via `minijinja`. Special tokens (`<|im_start|>`, `<think>`, `<minimax:tool_call>`, …) resolve from `tokenizer_config.json`, so the raw token ids match what the model was trained on.

## Rate limiter (`rate_limiter.rs`)

Per-key token bucket, off unless `METRALE_RATE_LIMIT_RPM` / `METRALE_RATE_LIMIT_TPM` is set, with a `MAX_KEYS` bound on the key table. See [OpenAI-Compatible Server](../operations/server.md#rate-limiting-and-auth).

## What's explicitly not here

- **No GPU kernels.** Every GPU call delegates through `metrale-gpu-runtime`.
- **No model-specific weight code.** That's `metrale-model-arch`.

## Adding a new HTTP shape

- A new API endpoint — one handler under `api/`, one route in `main_modules/serve_router.rs`.
- A new tool-call format — one new parser module under `tool_parser/`, one `ToolCallFormat` variant, one `--tool-call-parser` value.
- A new reasoning tag — extend `reasoning_parser/`.
- A new chat template override — a file `jinja-templates/<model_type>.jinja`, picked up by the tokenizer layer for checkpoints whose `config.json` `model_type` matches.
