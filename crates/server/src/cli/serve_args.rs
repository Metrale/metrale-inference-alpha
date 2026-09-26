// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Arguments of `met serve` (`ServeArgs`, re-exported as `cli::ServeArgs`) and the engine fallbacks of three of its flags.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types. The `///` text on `ServeArgs` and its fields is
//! the `--help` output and carries no date.
use clap::Parser;
use std::path::PathBuf;

mod scheduling;
mod service;
#[cfg(test)]
mod stageable_spec_tests;

/// 2026-09-26: Default of `--request-timeout`, in seconds. `AppState::request_deadline`
/// applies the flag's value to every request that sets no `timeout` of its own.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u32 = 300;

/// 2026-09-26: Engine fallback for `--num-drafts` when neither the flag nor MODEL.toml
/// `[behavior].default_num_drafts` gives a value; applied in
/// `serve_phases::config::resolve_num_drafts`.
pub const DEFAULT_NUM_DRAFTS: usize = 1;

/// 2026-09-26: Engine fallback for `--kv-cache-dtype` when neither the flag nor
/// MODEL.toml `[behavior].default_kv_dtype` gives a value; applied in
/// `serve_phases::kv_cache::resolve_kv_dtype_str`.
pub const DEFAULT_KV_CACHE_DTYPE: &str = "fp8";

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    /// or a local model directory.
    /// Optional: with neither this nor `--model-from-path`, the server boots into
    /// the dashboard's Library to pick a model there. Without a dashboard (plain
    /// mode) startup is refused, since there is nothing to pick a model with.
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Load a different model when a request names one.
    ///
    /// Off by default. Only a request whose `model` exactly matches the model id of
    /// a known recipe other than the live model triggers a load; an absent, unknown
    /// or already-live name is served by the current model.
    ///
    /// It covers request-triggered swaps only: the dashboard can change the model
    /// without it.
    #[arg(long)]
    pub auto_swap: bool,

    /// Serve even when kernel lookups this model's dispatch issued did not
    /// resolve.
    ///
    /// By default startup is refused and the unresolved lookups are listed, each
    /// with its dispatch site; kernels the target declares absent are not counted.
    /// With this flag the same report is logged at `warn` and the server starts.
    #[arg(long, default_value_t = false)]
    pub dangerously_allow_unresolved_kernel_lookups: bool,

    /// Resolve all kernels for the model and exit, reporting any that did not
    /// resolve. Does not start the server. The exit code is the number of
    /// unresolved kernels: 0 means every lookup resolved.
    ///
    /// Config, GPU init, weight load and model construction all run first, so the
    /// lookups are the ones a real serve makes; the scheduler never starts and no
    /// port is bound.
    ///
    /// A POSIX status is 8 bits, so the code is CLAMPED at 255 (256 would read as
    /// 0); when the clamp applies, the true count is printed beside it and carried
    /// in the JSON.
    ///
    /// The exit code ignores `--dangerously-allow-unresolved-kernel-lookups`:
    /// passing both still prints the full list and exits with the count.
    ///
    /// A one-line JSON object (`{"metrale_kernel_check": …}`) is printed on stdout
    /// after the human report.
    #[arg(long, default_value_t = false)]
    pub check_kernels: bool,

    /// Load the model from this path instead of MODEL. It is resolved the same
    /// way: a local directory with `config.json` or a `.gguf`, otherwise a
    /// HuggingFace cache id.
    #[arg(long, value_name = "PATH")]
    pub model_from_path: Option<PathBuf>,

    /// Override model name shown in /v1/models and API responses.
    /// Defaults to the positional MODEL argument, then config.json _name_or_path,
    /// then the model directory's name.
    #[arg(long, alias = "served-model-name", value_name = "NAME")]
    pub model_name: Option<String>,

    /// Pin kernel-target resolution to this compiled target directory name
    /// (e.g. "qwen3.8-27b").
    ///
    /// Normally the target is selected by the checkpoint's `(model_type,
    /// hidden_size)`. When several compiled targets declare that pair, each
    /// target's `match_names` are matched against the model id/path, and startup
    /// refuses when that does not single out one target; this flag is the explicit
    /// answer. The pinned target must still declare the checkpoint's
    /// `(model_type, hidden_size)` or a wildcard for its `model_type`: a pin can
    /// choose between compatible targets, never force an incompatible one.
    #[arg(long, value_name = "TARGET")]
    pub kernel_target: Option<String>,

    /// Override HuggingFace cache directory
    /// (default: $HF_HUB_CACHE, $HF_HOME/hub, or ~/.cache/huggingface/hub).
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,

    /// HTTP port.
    #[arg(long, default_value_t = 8888)]
    pub port: u16,

    /// GPU ordinal.
    #[arg(long, default_value_t = 0)]
    pub gpu_ordinal: usize,

    /// Maximum sequence length.
    #[arg(long, default_value_t = 32768)]
    pub max_seq_len: usize,

    /// KV cache block size (tokens per block).
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// KV cache dtype (fp8, bf16, or nvfp4).
    /// Precedence (highest wins): this flag → MODEL.toml
    /// `[behavior].default_kv_dtype` → fp8 (`DEFAULT_KV_CACHE_DTYPE`). An
    /// explicitly passed value always wins, including `fp8` itself.
    ///
    /// The `turbo2`/`turbo3`/`turbo4`/`turbo8` variants (and the asymmetric
    /// `*k_*v` pairs built from them) are experimental: they are not built for
    /// every kernel target, and a target that lacks them fails the kv-cache kernel
    /// preflight at startup.
    #[arg(long)]
    pub kv_cache_dtype: Option<String>,

    /// Storage dtype for the GDN decode h-state: `f32` (default), `f16`, or
    /// `f16-pool`.
    ///
    /// `f16` changes the h-state numerics. It requires `--gdn-fused-norm`, because
    /// the FP16 h-state kernels exist only on the fused-norm decode arm. It is
    /// refused together with `--exact-verify`, and together with `--dflash` when
    /// `--dflash-gamma` is above 15.
    ///
    /// `f16-pool` is `f16` plus h pools sized at 2 bytes per element, which halves
    /// their memory. Prefill keeps its FP32 kernels: they run over an FP32 staging
    /// arena of one h blob per slot, widened before a layer's prefill and narrowed
    /// after it.
    ///
    /// Environment fallback: `METRALE_SSM_H_FP16` (any value) selects f16, never
    /// f16-pool, when no GDN flag is given (`--ssm-h-dtype`, `--gdn-fused-norm`,
    /// `--ssm-batched-recurrent on|off`, `--exact-verify`). Any GDN flag hands the
    /// GDN selection to the command line, except that an absent
    /// `--ssm-batched-recurrent` still takes the target default, which
    /// `METRALE_SSM_BATCHED_RECURRENT` can override; a warning names the GDN
    /// variables that are set.
    #[arg(long)]
    pub ssm_h_dtype: Option<String>,

    /// Experimental SSM verify-rollback mode: `snapshot` (default) or `replay`.
    ///
    /// `snapshot`: every verify writes per-token h/conv state snapshots and a
    /// partial accept restores from them. `replay` allocates no per-token h
    /// snapshots and reserves a verify-window input ring instead, but the replay
    /// itself is not implemented: a replay serve boots, and every speculative
    /// verify step on a model with SSM layers returns an error.
    /// The default is explicit and published on every serve.
    #[arg(long, default_value = "snapshot")]
    pub ssm_rollback_mode: String,

    /// SSM decode-rollback ring depth: `auto` (default) or an explicit slot count
    /// in `0..=8`.
    ///
    /// The ring keeps SSM-state snapshots at boundaries so a watchdog re-steer can
    /// rewind the recurrent state; it reserves `depth x --max-batch-size x the
    /// per-sequence SSM state blob`.
    ///
    /// `auto` starts at depth 8, or 0 under `--speculative`/`--dflash` or with
    /// watchdogs disabled (`METRALE_DISABLE_WATCHDOGS`), and lets preflight shrink
    /// it down `8, 4, 2, 1, 0` until the reserve fits, logging a warning when it
    /// shrinks. Fewer snapshots means fewer re-steer anchors: a rollback that finds
    /// none is declined, and the content-loop and inter-tool prose watchdogs
    /// then end the response instead.
    ///
    /// An explicit `N` pins the depth and skips the fit. `0` disables the ring.
    ///
    /// Environment fallback: `METRALE_SSM_DECODE_RING=1|0` means depth 8 or 0. It
    /// is only consulted when this flag is `auto`.
    #[arg(long, default_value = "auto", value_name = "AUTO_OR_N")]
    pub ssm_decode_ring_slots: String,

    /// Fused GDN output-norm kernel on the decode path (default: off).
    ///
    /// Required by `--ssm-h-dtype f16`: the FP16 h-state kernels live on the
    /// fused-norm arm, and the unfused arm is FP32-only.
    ///
    /// Environment fallback: `METRALE_GDN_FUSED_NORM=1`, on the same terms as
    /// `--ssm-h-dtype`: with no GDN flag on the command line the variable decides;
    /// with any of them an absent `--gdn-fused-norm` means off.
    #[arg(long)]
    pub gdn_fused_norm: bool,

    /// Batched multi-sequence GDN recurrent decode kernel: `auto` (default),
    /// `on` or `off`.
    ///
    /// One strided launch across the batch instead of one per sequence.
    ///
    /// An enum, not a presence flag, because its default is not one value: the
    /// compiled target declares it (`kernels/<hw>/HARDWARE.toml` `[defaults]
    /// ssm_batched_recurrent`: on for `hopper`, off for `gb10`, `b200` and
    /// `b300`), and `METRALE_SSM_BATCHED_RECURRENT` overrides that. `auto` defers
    /// to them; `on`/`off` pin it either way and hand the whole GDN selection to
    /// the command line.
    #[arg(long, default_value = "auto", value_name = "auto|on|off")]
    pub ssm_batched_recurrent: String,

    /// Varlen (ragged) batched prefill, opt-in (default: off).
    ///
    /// Concurrently queued prompts of different lengths are prefilled together,
    /// one forward per wave, so each projection GEMM launches once over the wave's
    /// summed tokens instead of once per request. The scheduler defers chunk 0 of
    /// new requests so they can join a wave, and a wave holds at most
    /// min(`--max-prefill-tokens`, the max batch tokens) tokens. With
    /// `--prefill-codispatch` also set, this path is used.
    ///
    /// Batching changes GEMM row counts, and kernels are selected on row count, so
    /// per-request outputs can differ from the serial path.
    ///
    /// Environment fallback: `METRALE_PREFILL_VARLEN=1` (or `true`) turns it on
    /// when this flag is absent; a given flag wins over it.
    #[arg(long)]
    pub prefill_varlen_batch: bool,

    /// Co-dispatch fresh prompts: when >=2 requests without images are admitted
    /// together with nothing decoding or prefilling, defer their chunk-0 prefill so
    /// they batch into one forward.
    ///
    /// Only with chunked prefill, and not on an expert-parallel model.
    ///
    /// Environment fallback: `METRALE_PREFILL_CODISPATCH=1` (or `true`) turns it on
    /// when this flag is absent.
    #[arg(long)]
    pub prefill_codispatch: bool,

    /// Downcast activations to NVFP4 (W4A4) on the small-M projection paths
    /// (default: false).
    ///
    /// On, the projection sites that call `nvfp4_proj_small_m` (GDN qkvz/out_proj,
    /// attention q/k/v/o, dense-FFN gate/up/down) run up to 32 rows on the FP4
    /// block-scale tensor-core MMA (`w4a4_gemv_mx`), which reads the checkpoint's
    /// NVFP4 weights with no dequant; the activations are quantized per row to
    /// NVFP4. It is a numerics change, so it is off unless a recipe asks for it.
    /// No environment fallback. The lm_head is not affected.
    #[arg(long, default_value_t = false)]
    pub w4a4_downcast: bool,

    /// Widen `--w4a4-downcast` from 1..=32 to 1..=64 rows (default: false).
    ///
    /// Projections of 33 to 64 rows then take the same W4A4 FP4 block-scale GEMV
    /// (the `w4a4_gemv_mx64` entries), with the per-row math of the 1..=32-row
    /// path. A numerics change at those widths, hence opt-in. Requires
    /// `--w4a4-downcast`; without it a warning is logged and it has no effect.
    #[arg(long, default_value_t = false)]
    pub w4a4_downcast_wide: bool,

    /// Sequential-decode-exact GDN/SSM verify chain, opt-in (default: off).
    ///
    /// With it, MTP verify runs, per token, the GDN/SSM kernel chain sequential
    /// decode runs, instead of the WY-chunkwise and fused BF16-conv arms.
    ///
    /// It makes only the GDN/SSM verify chain exact. The FFN and attention
    /// projections still pick their kernels by row count, so speculative output is
    /// not guaranteed to match non-speculative output token for token.
    ///
    /// Refused beside `--ssm-h-dtype f16`: the exact arm's kernels are FP32
    /// readers and must never read the FP16 h-state pool.
    ///
    /// No environment variable. Like the other GDN flags, giving it hands the
    /// GDN selection to the command line; an absent `--ssm-batched-recurrent`
    /// still takes the target default.
    #[arg(long)]
    pub exact_verify: bool,

    /// Turn off mid-chunk SSM tail capture on the prefill path (on by default).
    ///
    /// While it is on, the scheduler does not end a prefill chunk at the SSM tail
    /// boundary, and prefill plans an in-pass capture of the GDN recurrent and
    /// conv state there instead (`prepare_midchunk_capture`).
    ///
    /// Environment fallback: `METRALE_SSM_TAIL_MIDCHUNK=0` disables it when this
    /// flag is absent. An absent flag publishes nothing, so the variable stays
    /// reachable.
    #[arg(long)]
    pub no_ssm_tail_midchunk: bool,

    /// MTP throughput gate: `auto` (default) or `force`.
    ///
    /// `auto` arms the gate, which measures delivered throughput with and without
    /// MTP and switches to whichever is faster. `force` disarms it, so verify steps
    /// keep running even where the gate would switch them off; the server logs it
    /// as a diagnostic setting. To run without speculation at all, omit
    /// `--speculative`.
    ///
    /// Environment fallback: `METRALE_MTP_GATE_FORCE=1` selects `force` when this
    /// flag is absent. There is no clap default, so the variable stays reachable.
    #[arg(long)]
    pub mtp_gate: Option<String>,

    /// LM-head precision: `default` (the clap default: no override, the model
    /// config decides), `bf16` (final vocab projection in BF16), `nvfp4` (force the
    /// model's NVFP4-packed lm_head), or `fp8` (an FP8 E4M3 lm_head decoded with
    /// `w8a16_gemv`: the checkpoint's own FP8 head when it ships one, else the
    /// lm_head quantized with per-row scales when the model is built).
    /// Against BF16, `fp8` reads about half the lm_head bytes per token and `nvfp4`
    /// about a quarter.
    ///
    /// Below BF16 the final vocab projection can flip low-margin argmax choices.
    /// Check long generations per model before enabling `nvfp4` or `fp8`.
    #[arg(long, default_value = "default")]
    pub lm_head_dtype: String,

    /// Boundary attention layers to keep at BF16 KV cache precision (first N + last N);
    /// the other layers use --kv-cache-dtype.
    /// Accepts: number, "auto" (=2, recommended), "max"/"all" (all BF16).
    /// Default: 0, no BF16 boundary layers, except that a turbo KV dtype gets an
    /// automatic count.
    #[arg(long, default_value = "0")]
    pub kv_high_precision_layers: String,

    /// Fraction of total GPU memory this process may consume, in (0.0, 1.0].
    /// Weights, buffers, KV cache and reserves are sized against total memory
    /// times this value.
    #[arg(long, default_value_t = 0.90)]
    pub gpu_memory_utilization: f64,

    /// Capacity of the queue that carries new requests to the scheduler.
    #[arg(long, default_value_t = 128)]
    pub max_num_seqs: usize,

    // 2026-09-26: The remaining flags, in declaration order: `ServeSchedulingArgs`,
    // which flattens `ServeServiceArgs` last. `Deref`/`DerefMut` below reach their
    // fields as `args.<field>`.
    #[command(flatten)]
    pub(crate) scheduling: scheduling::ServeSchedulingArgs,
}

impl std::ops::Deref for ServeArgs {
    type Target = scheduling::ServeSchedulingArgs;

    fn deref(&self) -> &Self::Target {
        &self.scheduling
    }
}

impl std::ops::DerefMut for ServeArgs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.scheduling
    }
}

impl ServeArgs {
    /// 2026-09-26: `--dflash-gamma` when given, else `drafter_block_size`, else 16.
    /// Every caller passes `None`, so this is the flag or 16; the built drafter head
    /// resolves an unset flag itself (`default_dflash_gamma`), and `serve_load` reads
    /// the head's γ back once the model exists.
    pub fn resolved_dflash_gamma(&self, drafter_block_size: Option<usize>) -> usize {
        self.dflash_gamma.or(drafter_block_size).unwrap_or(16)
    }

    /// 2026-09-26: The draft count, valid only after
    /// `serve_phases::apply_model_default_num_drafts` has resolved the
    /// CLI → MODEL.toml → engine-default precedence into `num_drafts`; `serve_load`
    /// calls it before the preflight reserve. Panics when read earlier. Readers that
    /// run before it (argument validation, the dashboard logo) match on the `Option`.
    pub fn resolved_num_drafts(&self) -> usize {
        self.num_drafts
            .expect("num_drafts read before apply_model_default_num_drafts resolved it")
    }
}

/// 2026-09-26: Value parser for `--lora-adapter` and `--lora-stageable-disk`,
/// `NAME=PATH_OR_HF_ID`: split at the first `=`, both parts non-empty.
fn parse_lora_adapter_spec(s: &str) -> Result<(String, String), String> {
    let (name, spec) = s
        .split_once('=')
        .ok_or_else(|| format!("--lora-adapter must be NAME=PATH_OR_HF_ID, got '{s}'"))?;
    if name.is_empty() || spec.is_empty() {
        return Err(format!("--lora-adapter: empty name or path in '{s}'"));
    }
    Ok((name.to_string(), spec.to_string()))
}

/// 2026-09-26: Value parser for `--lora-stageable NAME=PEER_ID=DIR`: split at the first
/// two `=` into three parts, all non-empty. DIR keeps any further `=`. DIR is required
/// because the peft scaling of a promoted adapter is read from its
/// `adapter_config.json` at startup.
fn parse_lora_stageable_spec(s: &str) -> Result<(String, String, String), String> {
    let mut parts = s.splitn(3, '=');
    let name = parts.next().unwrap_or("");
    let peer_id = parts.next().unwrap_or("");
    let dir = parts.next().unwrap_or("");
    if name.is_empty() || peer_id.is_empty() || dir.is_empty() {
        return Err(format!(
            "--lora-stageable must be NAME=PEER_ID=DIR (all three non-empty), got '{s}'"
        ));
    }
    Ok((name.to_string(), peer_id.to_string(), dir.to_string()))
}
