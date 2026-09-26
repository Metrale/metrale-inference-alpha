// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Online per-tensor FP8 KV scale calibration for one attention layer.
//!
//! The running max of |K| and |V| accumulates across requests over the first
//! `window_tokens` observed tokens (`--fp8-kv-calibration-tokens`, clamped to
//! `MAX_STAGED_TOKENS`). The observe that reaches the window freezes
//! `scale = amax * headroom / 448`, over every observed token including its
//! own batch (`state.rs`).
//!
//! Attention dequantizes a sequence's whole history with one
//! `k_scale`/`v_scale`, so an entry is read correctly only at the scale that
//! wrote it. Entries written inside the window use `PROVISIONAL_SCALE`; their
//! BF16 K/V and slot mappings are staged and, at the freeze, rewritten through
//! `ops::reshape_and_cache_fp8` at the frozen scale (`staging.rs`).
//!
//! Owner: model-layers (FP8 KV cache).
//! Invariants:
//! - The scales change once, at the freeze, and afterwards only on the opt-in
//!   `METRALE_FP8_KV_EMA_RECAL` path.
//! - The scale set at the freeze is at least `amax / 448`, where `amax` covers
//!   every token observed up to the freeze (`headroom` is clamped to `>= 1.0`
//!   in `Fp8KvCalibration::new`).

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use parking_lot::Mutex;

mod staging;
mod state;

pub use staging::Fp8KvWriteTarget;
use staging::{KvStaging, MAX_STAGED_TOKENS};
use state::{CalibrationState, CalibrationStep, POST_FREEZE_OBSERVE_PERIOD};

/// 2026-09-25: Scale of the window's own writes, before the freeze: `2.0 * 448`
/// covers |x| <= 896. Staged writes are rewritten at the frozen scale when the
/// window closes, so this only has to avoid clipping.
const PROVISIONAL_SCALE: f32 = 2.0;

/// 2026-09-25: Whether this KV dtype's write path calls [`Fp8KvCalibration::observe`].
///
/// Only the `KvCacheDtype::Fp8` arm of `qwen3_attention/decode/write_kv_cache.rs`
/// does (`write_kv_cache_fp8`). A calibrator on a layer that never observes
/// would never freeze, and [`graphs_ready_after_fp8_kv_cal`] would then keep
/// CUDA graphs suppressed for the life of the process.
pub fn dtype_runs_online_fp8_kv_calibration(kv_dtype: KvCacheDtype) -> bool {
    matches!(kv_dtype, KvCacheDtype::Fp8)
}

/// 2026-09-25: True when every calibrating layer has frozen, which is when the
/// model lifts its CUDA-graph suppression (`trait_impl/decode_a.rs`).
///
/// Per layer: `None` = no calibrator, `Some(false)` = still inside the window,
/// `Some(true)` = frozen. True when no layer calibrates. Graphs therefore stay
/// eager for the whole window.
pub fn graphs_ready_after_fp8_kv_cal<I>(states: I) -> bool
where
    I: IntoIterator<Item = Option<bool>>,
{
    states.into_iter().all(|s| s.unwrap_or(true))
}

struct CalibrationInner {
    state: CalibrationState,
    staging: KvStaging,
    /// 2026-09-25: Tokens of batches `KvStaging::stage` declined, for the freeze log.
    unstaged_tokens: usize,
}

/// 2026-09-25: The calibrator of one attention layer. `observe` takes `&self`;
/// the state sits behind a `Mutex`.
pub struct Fp8KvCalibration {
    inner: Mutex<CalibrationInner>,
    attn_layer_idx: usize,
    /// 2026-09-25: Tokens the window was asked for, before the `MAX_STAGED_TOKENS` clamp.
    requested_tokens: usize,
    /// 2026-09-25: 8 device bytes: `[k_absmax: f32, v_absmax: f32]`.
    absmax_buf: DevicePtr,
    absmax_kernel: KernelHandle,
}

// 2026-09-25: SAFETY: every field is a `Mutex` or a plain integer handle
// (`DevicePtr(u64)`, `KernelHandle(u64)`, `usize`), so no field is shared
// through a raw pointer; the state that changes is behind the `Mutex`.
unsafe impl Send for Fp8KvCalibration {}
unsafe impl Sync for Fp8KvCalibration {}

impl Fp8KvCalibration {
    /// 2026-09-25: Build the calibrator for attention layer `attn_layer_idx`.
    ///
    /// `window_tokens` is `--fp8-kv-calibration-tokens`, clamped to
    /// `MAX_STAGED_TOKENS` with a warning. 1 freezes on the first observe and
    /// stages nothing. The attention initializer builds no calibrator for 0
    /// (`qwen3_attention/init.rs`).
    ///
    /// `headroom` is `--fp8-kv-headroom`; a value below 1.0 is clamped to 1.0
    /// with a warning, because it would clip the window's own amax.
    ///
    /// Errors if the `bf16_absmax` kernel is missing or the 8-byte absmax
    /// buffer cannot be allocated or zeroed.
    pub fn new(
        attn_layer_idx: usize,
        window_tokens: usize,
        headroom: f32,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let headroom = if headroom >= 1.0 {
            headroom
        } else {
            tracing::warn!("fp8-kv headroom {headroom} < 1.0 guarantees clipping; clamped to 1.0");
            1.0
        };
        let effective = window_tokens.min(MAX_STAGED_TOKENS);
        if effective != window_tokens && attn_layer_idx == 0 {
            tracing::warn!(
                "--fp8-kv-calibration-tokens {window_tokens} exceeds the {MAX_STAGED_TOKENS}-token \
                 staging cap (the window's KV is held in BF16 so it can be requantized at the \
                 freeze); calibrating on {effective} tokens instead."
            );
        }
        let absmax_kernel = gpu.kernel("reshape_and_cache", "bf16_absmax")?;
        let absmax_buf = gpu.alloc(8)?;
        let zeros = [0u8; 8];
        gpu.copy_h2d(&zeros, absmax_buf)?;

        Ok(Self {
            inner: Mutex::new(CalibrationInner {
                state: CalibrationState::new(effective, headroom, PROVISIONAL_SCALE),
                staging: KvStaging::default(),
                unstaged_tokens: 0,
            }),
            attn_layer_idx,
            requested_tokens: window_tokens,
            absmax_buf,
            absmax_kernel,
        })
    }

    /// 2026-09-25: True until the scales freeze.
    pub fn is_calibrating(&self) -> bool {
        !self.inner.lock().state.frozen
    }

    /// 2026-09-25: `(k_scale, v_scale)`: `PROVISIONAL_SCALE` inside the window,
    /// the frozen scales after it.
    pub fn scales(&self) -> (f32, f32) {
        let inner = self.inner.lock();
        (inner.state.k_scale, inner.state.v_scale)
    }

    /// 2026-09-25: Fold this batch's BF16 K/V (`[num_tokens, num_kv_heads, head_dim]`)
    /// into the window.
    ///
    /// Inside the window it reduces the absmax on the device, synchronizes the
    /// stream to read it back, and stages the batch. The observe that reaches
    /// the window freezes the scales and replays the staged batches into
    /// `target`'s pools at them. Call it after the K/V projections and before
    /// the cache write, which then uses [`Self::scales`]. After the freeze it
    /// reduces only when `CalibrationState::should_observe` asks, for the
    /// opt-in EMA path.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
        target: &Fp8KvWriteTarget,
    ) -> Result<()> {
        {
            let inner = self.inner.lock();
            if !inner.state.should_observe(num_tokens as usize) {
                return Ok(());
            }
        }

        let (k_max, v_max) = self.absmax(
            gpu,
            k_data,
            v_data,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
        )?;

        let mut inner = self.inner.lock();
        match inner.state.record(k_max, v_max, num_tokens as usize) {
            CalibrationStep::Stage => {
                let capacity = inner.state.window_tokens;
                let elems = (num_kv_heads * head_dim) as usize;
                let staged = inner.staging.stage(
                    gpu, k_data, v_data, num_tokens, elems, target, capacity, stream,
                )?;
                if !staged {
                    inner.unstaged_tokens += num_tokens as usize;
                }
            }
            CalibrationStep::Freeze {
                k_scale,
                v_scale,
                tokens_seen,
            } => {
                let staged_tokens = inner.staging.used_tokens();
                let inner = &mut *inner;
                inner.staging.replay_and_release(
                    gpu,
                    target,
                    num_kv_heads,
                    head_dim,
                    k_scale,
                    v_scale,
                    stream,
                )?;
                tracing::info!(
                    "FP8 KV scales frozen after {} tokens (requested {}) on attn layer {}: \
                     k_scale={:.6} (amax={:.3}), v_scale={:.6} (amax={:.3}), headroom={:.2}; \
                     requantized {} staged tokens in {} batches, {} unstaged",
                    tokens_seen,
                    self.requested_tokens,
                    self.attn_layer_idx,
                    k_scale,
                    inner.state.k_running_max,
                    v_scale,
                    inner.state.v_running_max,
                    inner.state.headroom,
                    staged_tokens,
                    inner.state.staged_batches,
                    inner.unstaged_tokens,
                );
            }
            CalibrationStep::Frozen => {
                if inner.state.tokens_seen % POST_FREEZE_OBSERVE_PERIOD < num_tokens as usize
                    && ema_recal_enabled()
                {
                    // 2026-09-25: Opt-in only (`METRALE_FP8_KV_EMA_RECAL`). Moving
                    // a frozen scale changes the basis every already-written entry
                    // is read through, and nothing rewrites entries after the window.
                    inner.state.ema_recalibrate(k_max, v_max);
                    tracing::info!(
                        "FP8 KV EMA-recalibrated after {} tokens on attn layer {}: \
                         k_scale={:.6} (amax={:.2}), v_scale={:.6} (amax={:.2})",
                        inner.state.tokens_seen,
                        self.attn_layer_idx,
                        inner.state.k_scale,
                        inner.state.k_running_max,
                        inner.state.v_scale,
                        inner.state.v_running_max,
                    );
                }
            }
        }
        Ok(())
    }

    /// 2026-09-25: Absmax of the K and V buffers, read back after a stream synchronize.
    #[allow(clippy::too_many_arguments)]
    fn absmax(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
    ) -> Result<(f32, f32)> {
        let n_elems = num_tokens * num_kv_heads * head_dim;
        // 2026-09-25: `bf16_absmax` max-accumulates into its output, so zero it first.
        gpu.memset_async(self.absmax_buf, 0, 8, stream)?;
        let k_out = self.absmax_buf;
        super::ops::bf16_absmax(gpu, self.absmax_kernel, k_data, k_out, n_elems, stream)?;
        let v_out = self.absmax_buf.offset(4);
        super::ops::bf16_absmax(gpu, self.absmax_kernel, v_data, v_out, n_elems, stream)?;

        gpu.synchronize(stream)?;
        let mut buf = [0u8; 8];
        gpu.copy_d2h(self.absmax_buf, &mut buf)?;
        Ok((
            f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        ))
    }
}

fn ema_recal_enabled() -> bool {
    std::env::var("METRALE_FP8_KV_EMA_RECAL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;
