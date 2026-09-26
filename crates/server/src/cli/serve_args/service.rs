// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The last `met serve` flags, from `--request-timeout` to
//! `--lora-stageable-disk`: the request deadline, profiling, FP8 KV calibration,
//! weight loading, the dashboard, vision and video input, the listener, auth and
//! LoRA adapters. `ServeSchedulingArgs` flattens this struct last.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date. clap lists the flags in declaration order, so a flattened
//! struct's fields sit where its `#[command(flatten)]` field is: last.

use clap::Args;

use super::{DEFAULT_REQUEST_TIMEOUT_SECS, parse_lora_adapter_spec, parse_lora_stageable_spec};

// 2026-09-26: `ServeArgs` reaches these fields through `Deref`, via
// `ServeSchedulingArgs`.
#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeServiceArgs {
    /// Server-side deadline for a single request, in seconds. A request
    /// that exceeds it is cut and the response is reported with
    /// `finish_reason="timeout"` (never "length") plus a WARN log naming
    /// the slot, elapsed time and tokens emitted. `0` disables the deadline
    /// entirely. Overridable per request via the OpenAI `timeout` field.
    #[arg(long, default_value_t = DEFAULT_REQUEST_TIMEOUT_SECS)]
    pub request_timeout: u32,

    /// Enable per-kernel profiling: sync + time each operation within layers.
    /// Disables CUDA graphs on the decode path for accurate per-op timing.
    #[arg(long, default_value_t = false)]
    pub profile: bool,

    /// Number of warmup tokens for online FP8 KV cache scale calibration.
    /// Tracks max |K| and max |V| over the first N observed tokens, across
    /// requests, then computes per-tensor scales as amax*headroom/448 (mapping
    /// the observed range to FP8 E4M3 [-448, 448]).
    /// The window's own KV is held in BF16 and requantized at the freeze, so
    /// the write scale always equals the read scale; N is clamped to 4096 to
    /// bound that staging. 1 = freeze on the first observe. 0 = disabled.
    /// Only applies when --kv-cache-dtype is fp8.
    /// Precedence (highest wins): this flag → MODEL.toml
    /// `[behavior].fp8_kv_calibration_tokens` → 0. An explicit value always
    /// wins: passing 0 force-disables calibration even on a model whose
    /// MODEL.toml enables it.
    #[arg(long)]
    pub fp8_kv_calibration_tokens: Option<usize>,

    /// Headroom multiplier applied to the accumulated absmax when the online
    /// FP8 KV scale freezes. The calibration window only sees the first
    /// `--fp8-kv-calibration-tokens` tokens, so the frozen scale covers
    /// headroom× their max, and later tokens whose magnitude grows don't clip.
    /// Must be ≥ 1.0 (below 1.0 guarantees clipping; rejected at startup).
    #[arg(long, default_value_t = 2.0)]
    pub fp8_kv_headroom: f32,

    /// Not implemented: rejected at startup. Nothing reads this: no prompt is
    /// tokenized and no prefill runs. Send one throwaway request after startup
    /// instead.
    #[arg(long)]
    pub warmup_prompt: Option<std::path::PathBuf>,

    /// Enable adaptive sampling: per token, the logits' entropy and the output's
    /// zone (tool call, thinking, grammar) set an effective temperature, and a
    /// greedy gate can switch the token to greedy decoding.
    /// Off by default.
    #[arg(long, default_value_t = false)]
    pub adaptive_sampling: bool,

    /// Disable the fast weight loader and use the mmap loader instead. The fast
    /// loader (O_DIRECT + pipelined reader/copier) is on by default; this flag is
    /// an escape hatch for filesystems that misbehave with O_DIRECT or for A/B
    /// debugging.
    /// Setting `METRALE_FAST_LOAD=0` has the same effect.
    #[arg(long, default_value_t = false)]
    pub no_fast_load: bool,

    /// Disable the interactive TUI dashboard even on a TTY, keeping the plain
    /// log stream. The TUI also auto-disables when stdout/stdin is not an
    /// interactive terminal (pipes, `docker -d`, CI) or `METRALE_NO_TUI=1`.
    #[arg(long, default_value_t = false)]
    pub no_tui: bool,

    /// Ask the fast loader to prefetch each buffered shard before per-tensor
    /// reads. Also enabled by `METRALE_FAST_LOAD_PREFETCH_SHARDS=1` (or `true`).
    #[arg(long, default_value_t = false)]
    pub fast_load_prefetch_shards: bool,

    /// Vision input area bound in pixels, applied before patching. A non-zero
    /// value overrides the checkpoint in both directions: it may raise the bound
    /// as well as lower it.
    ///
    /// 0 (the default): `METRALE_VISION_MAX_PIXELS` when set, else the
    /// checkpoint's own bound, read from `preprocessor_config.json` or
    /// `processor_config.json` (`size.longest_edge`, or `max_pixels`; despite the
    /// name both are pixel counts). When the checkpoint declares none, the
    /// preprocessor clamps the long side to 1280px.
    ///
    /// Raising it raises the vision token count per image, which is charged
    /// against the context budget.
    #[arg(long, default_value_t = 0)]
    pub vision_max_pixels: usize,

    /// Fetch `image_url` parts that carry an http(s) URL, instead of
    /// rejecting them.
    ///
    /// Off by default: enabling it lets anyone who can send the server a chat
    /// request make it issue outbound HTTP to addresses of their choosing, a
    /// server-side request forgery primitive. With the flag off, a URL is refused
    /// with a 400 naming this flag; clients that cannot be changed send a base64
    /// `data:` URI instead.
    ///
    /// Switched on, the fetch is still bounded: loopback/private/link-local
    /// destinations are refused (including across redirects), the body is
    /// capped while being read rather than by trusting `Content-Length`, the
    /// response must declare an image content type, and each HTTP request is
    /// time-limited.
    #[arg(long, default_value_t = false)]
    pub vision_allow_remote_images: bool,

    /// Cap, in MiB, on a single fetched remote image. Enforced against bytes
    /// actually read, so a remote understating its `Content-Length` cannot
    /// exceed it. No effect unless `--vision-allow-remote-images` is set.
    #[arg(long, default_value_t = 20)]
    pub vision_remote_image_max_mb: usize,

    /// Time limit, in seconds, for each HTTP request made to fetch a remote image;
    /// a redirect hop is a separate request. No effect unless
    /// `--vision-allow-remote-images` is set.
    #[arg(long, default_value_t = 10)]
    pub vision_remote_image_timeout_s: u64,

    /// Also permit remote images on loopback, private and link-local
    /// addresses: the destination address check is skipped.
    ///
    /// A second grant on top of `--vision-allow-remote-images`. Only set this
    /// where the image host is internal: it re-opens the path to link-local
    /// cloud instance metadata (169.254.169.254).
    #[arg(long, default_value_t = false)]
    pub vision_remote_image_allow_private: bool,

    /// Decode video content parts with ffmpeg.
    ///
    /// Video support needs ffmpeg on the host for every container except
    /// animated GIF. GIF is decoded in-process in pure Rust; MP4/MOV,
    /// WebM/Matroska and AVI (H.264, H.265, VP9 and AV1) are decoded by running
    /// `ffmpeg`. Without this flag a video part is refused with an error naming
    /// the flag; with it but no usable ffmpeg, the server warns at startup and
    /// each video request fails.
    ///
    /// Install it with `apt install ffmpeg` (Debian/Ubuntu) or
    /// `dnf install ffmpeg` (Fedora/RHEL).
    ///
    /// Off by default because it makes the server execute another program
    /// per video request. The decode is bounded: no shell, no temp file,
    /// capped frames, capped output, capped wall clock.
    #[arg(long, default_value_t = false)]
    pub video_allow_ffmpeg: bool,

    /// Path to the ffmpeg binary. A bare name is resolved on PATH; an
    /// absolute path is used as given, so a deployment can pin a known build
    /// instead of inheriting whatever PATH offers. No effect unless
    /// `--video-allow-ffmpeg` is set.
    #[arg(long, default_value = "ffmpeg")]
    pub video_ffmpeg_path: String,

    /// Frames per second to sample a video at. ffmpeg does the sampling
    /// (`-vf fps=`) rather than decoding every frame. Raising this multiplies the
    /// vision tokens a clip costs.
    #[arg(long, default_value_t = 2.0)]
    pub video_fps: f32,

    /// Hard cap on frames taken from one video. At the default 2 fps this is
    /// 384 seconds of clip.
    #[arg(long, default_value_t = 768)]
    pub video_max_frames: usize,

    /// Wall-clock budget, in seconds, for decoding one video; the ffmpeg child is
    /// killed when it expires.
    #[arg(long, default_value_t = 120)]
    pub video_decode_timeout_s: u64,

    /// Address to bind the HTTP listener to. Defaults to `127.0.0.1` so a
    /// fresh install is reachable only from the local machine; pass
    /// `0.0.0.0` to expose on all interfaces (the server logs a warning
    /// when it does; the CORS policy allows any origin, so the API is then
    /// reachable to anything on the LAN).
    #[arg(long, alias = "host", default_value = "127.0.0.1", value_name = "ADDR")]
    pub bind: String,

    /// Require an `Authorization: Bearer <token>` header on `/v1/*`,
    /// `/tokenize`, and `/detokenize`. The token must match one loaded
    /// via `--auth-tokens-file` or `--auth-token`; without either, startup is
    /// refused. Other paths, such as `/health`, `/health/live` and `/metrics`,
    /// stay open.
    ///
    /// Off by default. Turn it on whenever the server is reachable from anywhere
    /// other than `localhost` (for example with `--bind 0.0.0.0`).
    #[arg(long, default_value_t = false)]
    pub require_auth: bool,

    /// Path to a file containing valid bearer tokens, one per line. Blank
    /// lines and lines starting with `#` are ignored. Permissions should
    /// be `0600`. The file is read once at startup: restart the server to
    /// rotate keys.
    #[arg(long, value_name = "PATH", conflicts_with = "auth_token")]
    pub auth_tokens_file: Option<std::path::PathBuf>,

    /// A single inline bearer token. Convenient for quick starts; not
    /// recommended for production because the token is visible in
    /// `ps`/`/proc/<pid>/cmdline`. Use `--auth-tokens-file` instead.
    #[arg(long, value_name = "TOKEN", conflicts_with = "auth_tokens_file")]
    pub auth_token: Option<String>,

    /// LoRA adapter to serve, as NAME=PATH_OR_HF_ID (e.g.
    /// `holo-sft=/data/adapters/holo-sft` or `holo-sft=org/holo-31-08b-lora`).
    /// Repeatable: each adapter loads into its own pool slot at startup and is
    /// advertised by GET /v1/models; a request selects one by name in its
    /// `adapter` field.
    #[arg(long, value_name = "NAME=PATH_OR_HF_ID", value_parser = parse_lora_adapter_spec)]
    pub lora_adapter: Vec<(String, String)>,

    /// NLLB/M2M-100 only: source-language token for translation (e.g.
    /// `eng_Latn`). Required when serving an `m2m_100`/`nllb` checkpoint;
    /// ignored otherwise.
    #[arg(long)]
    pub src_lang: Option<String>,

    /// NLLB/M2M-100 only: target-language token (e.g. `fra_Latn`, `gvn_Latn`).
    /// Required when serving an `m2m_100`/`nllb` checkpoint; ignored otherwise.
    #[arg(long)]
    pub tgt_lang: Option<String>,

    /// Maximum LoRA adapter rank. The adapter slot pool is sized for this rank at
    /// startup; an adapter whose `r` exceeds it is rejected at load.
    ///
    /// Unset (default): the largest `r` among the `--lora-adapter` adapters, so
    /// a small adapter is not padded to a larger rank; 64 when a stageable adapter
    /// (`--lora-stageable`, `--lora-stageable-disk`) is configured, since its rank
    /// is not known at startup.
    ///
    /// Set it explicitly only to reserve headroom for a larger adapter staged in
    /// later: the pool layout is fixed at startup, so a stage-in above the pool's
    /// rank is rejected.
    #[arg(long)]
    pub max_lora_rank: Option<usize>,

    /// Maximum number of LoRA adapter slots in the pool. Slots beyond the
    /// startup-resident adapters are cache headroom for demand promotion
    /// (`--lora-stageable`, `--lora-stageable-disk`).
    #[arg(long, default_value_t = 8)]
    pub max_loras: usize,

    /// A stageable (promotable-but-not-resident) LoRA adapter, as
    /// `NAME=PEER_STAGE_ID=CONFIG_DIR` (repeatable). NAME is what a request's
    /// `adapter` field asks for; PEER_STAGE_ID is the adapter's id on the
    /// `$METRALE_LORA_PEER` weight peer; CONFIG_DIR is a local dir with
    /// `adapter_config.json` (the peft scaling is read from there at startup). A
    /// request naming a stageable adapter triggers an on-miss RDMA promotion into
    /// a cache pool slot. Requires `$METRALE_LORA_PEER` and at least one
    /// `--lora-adapter`.
    #[arg(long, value_name = "NAME=PEER_ID=DIR", value_parser = parse_lora_stageable_spec)]
    pub lora_stageable: Vec<(String, String, String)>,

    /// A disk-stageable (promotable-but-not-resident, no peer) LoRA adapter, as
    /// `NAME=PATH_OR_HF_ID` (repeatable). A request naming NAME triggers an
    /// on-miss disk load into a cache pool slot. Needs `METRALE_LORA_ROTATE=1`
    /// (which arms runtime adapter rotation, so the disk swap can re-point a
    /// cache slot) unless
    /// `$METRALE_LORA_PEER` is set, at least one `--lora-adapter`, and
    /// `--max-loras` above the resident count for cache headroom. Its rank must
    /// not exceed `--max-lora-rank` (64 when unset).
    #[arg(long, value_name = "NAME=PATH_OR_HF_ID", value_parser = parse_lora_adapter_spec)]
    pub lora_stageable_disk: Vec<(String, String)>,
}
