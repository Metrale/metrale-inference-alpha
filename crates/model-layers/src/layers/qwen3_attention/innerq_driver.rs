// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-side driver for InnerQ, the per-channel K equalization of
//! the WHT-rotated (turbo) KV dtypes.
//!
//! `TURBO_INNERQ=N` (N > 0 calibration tokens) turns it on;
//! `TURBO_INNERQ_STRENGTH` in (0, 1] sets the exponent, 0.5 otherwise. The
//! state is `__device__` globals in namespace `tq_plus`, defined in
//! `kernels/gb10/common/tq_plus_innerq_apply.cu`. The driver reaches them
//! through the registry: `device_symbol` for the address, async copies to read
//! and write.
//!
//! `start` zeroes the accumulators and sets `d_innerq_calibrating = 1`.
//! `maybe_finalize` reads `d_innerq_count`; once it reaches `target_tokens` it
//! clears `d_innerq_calibrating`, derives per-channel `scale`/`scale_inv` from
//! `d_innerq_sq_accum`, and either stops (every channel ratio within 1.2x) or
//! uploads them and sets `d_innerq_active = 1`.
//!
//! Owner: model-layers (attention).
//! Invariants: the `d_innerq_active = 1` write is issued only after the scale
//! uploads were synchronised on the driver's stream.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use std::sync::Arc;

use metrale_gpu_runtime::registry::MetraleRegistry;

// 2026-09-25: Itanium-mangled names of the `tq_plus::d_innerq_*` globals,
// defined in `kernels/gb10/common/tq_plus_innerq_apply.cu`. The module must be
// that file's: each `.cu` is its own PTX module, built without `-rdc`, so each
// module has its own copy of a `__device__` global, and only this module's
// kernels (`tq_plus_innerq_apply_q`, `tq_plus_innerq_apply_k`) read them.
const MODULE: &str = "tq_plus_innerq_apply";
const SYM_SCALE: &str = "_ZN7tq_plus14d_innerq_scaleE";
const SYM_SCALE_INV: &str = "_ZN7tq_plus18d_innerq_scale_invE";
const SYM_SQ_ACCUM: &str = "_ZN7tq_plus17d_innerq_sq_accumE";
const SYM_COUNT: &str = "_ZN7tq_plus14d_innerq_countE";
const SYM_ACTIVE: &str = "_ZN7tq_plus15d_innerq_activeE";
const SYM_CALIBRATING: &str = "_ZN7tq_plus20d_innerq_calibratingE";

// 2026-09-25: Equal to `INNERQ_MAX_CHANNELS` in `tq_plus_innerq.cuh`; the
// copies below also check each symbol's real size.
const MAX_CHANNELS: usize = 128;

pub struct InnerQDriver {
    /// 2026-09-25: This model's kernel registry. Every device symbol is
    /// resolved against it, so a driver writes only its own model's globals.
    registry: Arc<MetraleRegistry>,
    pub target_tokens: i32,
    pub strength: f32,
    pub calibrating: AtomicBool,
    pub finalized: AtomicBool,
}

impl InnerQDriver {
    /// 2026-09-25: Reads `TURBO_INNERQ` and `TURBO_INNERQ_STRENGTH`. Returns
    /// `None` if `TURBO_INNERQ` is unset, not an `i32`, or `<= 0`. A strength
    /// outside (0, 1] or unparsable is 0.5.
    pub fn from_env(registry: Arc<MetraleRegistry>) -> Option<Self> {
        let n = std::env::var("TURBO_INNERQ")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&n| n > 0)?;
        let strength: f32 = std::env::var("TURBO_INNERQ_STRENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s: &f32| s > 0.0 && s <= 1.0)
            .unwrap_or(0.5);
        Some(Self {
            registry,
            target_tokens: n,
            strength,
            calibrating: AtomicBool::new(false),
            finalized: AtomicBool::new(false),
        })
    }

    /// 2026-09-25: Enters calibration: zeroes `d_innerq_sq_accum`,
    /// `d_innerq_count` and `d_innerq_active`, sets `d_innerq_calibrating = 1`,
    /// and resets the host flags. Calling it again gives the same state.
    pub fn start(&self) -> Result<()> {
        let reg = &self.registry;
        let stream = reg.raw_stream();

        let zeros_f32 = [0.0f32; MAX_CHANNELS];
        let zero_i32: i32 = 0;
        let one_i32: i32 = 1;

        let (sq_ptr, sq_bytes) = reg
            .device_symbol(MODULE, SYM_SQ_ACCUM)
            .with_context(|| format!("resolve {MODULE}::{SYM_SQ_ACCUM}"))?;
        let (count_ptr, _) = reg.device_symbol(MODULE, SYM_COUNT)?;
        let (active_ptr, _) = reg.device_symbol(MODULE, SYM_ACTIVE)?;
        let (calib_ptr, _) = reg.device_symbol(MODULE, SYM_CALIBRATING)?;

        let copy_bytes = sq_bytes.min(std::mem::size_of_val(&zeros_f32));
        // 2026-09-25: SAFETY (all four copies): `copy_h2d_async` requires (a) a
        // valid device destination, (b) `bytes` readable from `src`, and (c)
        // `src` alive until the next sync on `stream`.
        //   (a) every `*_ptr` comes from `device_symbol` on this model's
        //       registry.
        //   (b) `copy_bytes = min(sq_bytes, size_of_val(&zeros_f32))` is bounded
        //       by the symbol's reported length and the 512-byte host array. The
        //       three scalar copies move `size_of::<i32>()` bytes from `i32`
        //       locals into `d_innerq_count/_active/_calibrating`, declared
        //       `int` in tq_plus_innerq.cuh.
        //   (c) the sources are locals of this function, and the
        //       `stream_synchronize` after the block retires every copy before
        //       they go out of scope. If a later copy returns an error, the
        //       function returns before that sync.
        unsafe {
            reg.copy_h2d_async(
                sq_ptr,
                zeros_f32.as_ptr() as *const c_void,
                copy_bytes,
                stream,
            )?;
            reg.copy_h2d_async(
                count_ptr,
                &zero_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
            reg.copy_h2d_async(
                active_ptr,
                &zero_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
            reg.copy_h2d_async(
                calib_ptr,
                &one_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        // 2026-09-25: The stack sources must live until the copies retire.
        reg.stream_synchronize(stream)?;

        self.calibrating.store(true, Ordering::Release);
        self.finalized.store(false, Ordering::Release);
        tracing::info!(
            "InnerQ calibration started: target={} tokens, strength={:.2}",
            self.target_tokens,
            self.strength,
        );
        Ok(())
    }

    /// 2026-09-25: Polls `d_innerq_count`. Once it reaches `target_tokens`,
    /// reads `d_innerq_sq_accum`, computes per-channel `scale`/`scale_inv`,
    /// uploads them and sets `d_innerq_active = 1`. Returns `Ok(true)` on the
    /// call that activates and `Ok(false)` otherwise, including when the
    /// channels are already balanced and nothing is uploaded. Errors when
    /// `group_size` is outside `1..=MAX_CHANNELS` or larger than a symbol.
    pub fn maybe_finalize(&self, group_size: i32) -> Result<bool> {
        if self.finalized.load(Ordering::Acquire) {
            return Ok(false);
        }
        let gs = group_size as usize;
        if gs == 0 || gs > MAX_CHANNELS {
            bail!("group_size {group_size} out of range (1..={MAX_CHANNELS})");
        }

        let reg = &self.registry;
        let stream = reg.raw_stream();

        let (count_ptr, _) = reg.device_symbol(MODULE, SYM_COUNT)?;
        let mut count: i32 = 0;
        // 2026-09-25: SAFETY: `count_ptr` is `d_innerq_count`, an `int` in this
        // model's module; the copy is `size_of::<i32>()` bytes into `count`, a
        // local that the `stream_synchronize` on the next line keeps alive until
        // the copy retires. No other reference to `count` exists.
        unsafe {
            reg.copy_d2h_async(
                &mut count as *mut i32 as *mut c_void,
                count_ptr,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        reg.stream_synchronize(stream)?;

        if count < self.target_tokens {
            return Ok(false);
        }

        let (sq_ptr, sq_bytes) = reg.device_symbol(MODULE, SYM_SQ_ACCUM)?;
        let mut sq_accum = [0.0f32; MAX_CHANNELS];
        let accum_bytes = gs * std::mem::size_of::<f32>();
        // 2026-09-25: The host side is bounded by the `gs <= MAX_CHANNELS` check
        // above. The device side is bounded by the symbol's real length from
        // `device_symbol`, so the check does not depend on `MAX_CHANNELS`
        // agreeing with `INNERQ_MAX_CHANNELS`.
        if accum_bytes > sq_bytes {
            bail!(
                "{MODULE}::{SYM_SQ_ACCUM} is {sq_bytes} bytes but group_size {group_size} \
                 needs {accum_bytes} — INNERQ_MAX_CHANNELS and MAX_CHANNELS disagree"
            );
        }
        // 2026-09-25: SAFETY: `copy_d2h_async` needs a valid device source,
        // `bytes` writable at `dst`, and `dst` alive until the sync. `sq_ptr` is
        // `d_innerq_sq_accum` from this model's module; `accum_bytes = gs * 4` is
        // `<= sq_bytes` (checked above) and `<= size_of_val(&sq_accum)` because
        // `gs <= MAX_CHANNELS`. `sq_accum` is an initialised local that lives to
        // the end of the function, and the `stream_synchronize` below retires the
        // copy before it is read.
        unsafe {
            reg.copy_d2h_async(
                sq_accum.as_mut_ptr() as *mut c_void,
                sq_ptr,
                accum_bytes,
                stream,
            )?;
        }
        reg.stream_synchronize(stream)?;

        // 2026-09-25: `scale[i] = (mean_rms / rms[i]) ^ strength`, clamped to
        // [0.5, 2.0] (ratio 1 for a channel with rms <= 1e-10). Nothing is
        // uploaded when every ratio is inside (1/1.2, 1.2).
        let count_f = count as f32;
        let mut rms = [0.0f32; MAX_CHANNELS];
        let mut mean_rms = 0.0f32;
        for i in 0..gs {
            rms[i] = (sq_accum[i] / count_f).sqrt();
            mean_rms += rms[i];
        }
        mean_rms /= gs as f32;

        let mut scale = [1.0f32; MAX_CHANNELS];
        let mut scale_inv = [1.0f32; MAX_CHANNELS];
        let mut max_ratio = 0.0f32;
        let mut min_ratio = 1e30f32;
        for i in 0..gs {
            let ratio = if rms[i] > 1e-10 {
                mean_rms / rms[i]
            } else {
                1.0
            };
            let s = ratio.powf(self.strength).clamp(0.5, 2.0);
            scale[i] = s;
            scale_inv[i] = 1.0 / s;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
            if ratio < min_ratio {
                min_ratio = ratio;
            }
        }

        let (calib_ptr, _) = reg.device_symbol(MODULE, SYM_CALIBRATING)?;
        let zero_i32: i32 = 0;
        // 2026-09-25: SAFETY: `calib_ptr` is `d_innerq_calibrating`, an `int` in
        // this model's module; the copy is `size_of::<i32>()` bytes from
        // `zero_i32`, a local that stays in scope to the end of the function. The
        // auto-disable return and the success path both synchronise the stream
        // first. The error returns between here and the first sync below (a
        // failed `device_symbol`, the size `bail!`, a failed upload) do not.
        unsafe {
            reg.copy_h2d_async(
                calib_ptr,
                &zero_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }

        if max_ratio < 1.2 && min_ratio > (1.0 / 1.2) {
            reg.stream_synchronize(stream)?;
            self.calibrating.store(false, Ordering::Release);
            self.finalized.store(true, Ordering::Release);
            tracing::info!(
                "InnerQ auto-disabled (channels already balanced: max_ratio={max_ratio:.3}, \
                 min_ratio={min_ratio:.3})"
            );
            return Ok(false);
        }

        let (scale_ptr, scale_bytes) = reg.device_symbol(MODULE, SYM_SCALE)?;
        let (scale_inv_ptr, scale_inv_bytes) = reg.device_symbol(MODULE, SYM_SCALE_INV)?;
        let (active_ptr, _) = reg.device_symbol(MODULE, SYM_ACTIVE)?;
        let one_i32: i32 = 1;
        let copy_bytes = gs * std::mem::size_of::<f32>();
        // 2026-09-25: These two are writes into device globals, where an overrun
        // would silently overwrite the next global. They are bounded by the
        // symbols' real lengths, not by `MAX_CHANNELS`.
        if copy_bytes > scale_bytes || copy_bytes > scale_inv_bytes {
            bail!(
                "{MODULE} scale symbols are {scale_bytes}/{scale_inv_bytes} bytes but \
                 group_size {group_size} needs {copy_bytes} — INNERQ_MAX_CHANNELS and \
                 MAX_CHANNELS disagree"
            );
        }
        // 2026-09-25: SAFETY: `scale_ptr`/`scale_inv_ptr` are `d_innerq_scale` /
        // `d_innerq_scale_inv` from this model's module. `copy_bytes = gs * 4` is
        // `<= scale_bytes`/`scale_inv_bytes` (checked above) and
        // `<= size_of_val(&scale)` because `gs <= MAX_CHANNELS`. `scale` and
        // `scale_inv` are initialised locals that stay in scope past the
        // `stream_synchronize` that retires these copies. If the second copy
        // returns an error, the function returns before that sync.
        unsafe {
            reg.copy_h2d_async(
                scale_ptr,
                scale.as_ptr() as *const c_void,
                copy_bytes,
                stream,
            )?;
            reg.copy_h2d_async(
                scale_inv_ptr,
                scale_inv.as_ptr() as *const c_void,
                copy_bytes,
                stream,
            )?;
        }
        // 2026-09-25: The scale uploads are synchronised before the
        // `d_innerq_active` write is issued.
        reg.stream_synchronize(stream)?;
        // 2026-09-25: SAFETY: `active_ptr` is `d_innerq_active`, an `int` in this
        // model's module; the copy is `size_of::<i32>()` bytes from `one_i32`, a
        // local that outlives the `stream_synchronize` after the block.
        unsafe {
            reg.copy_h2d_async(
                active_ptr,
                &one_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        reg.stream_synchronize(stream)?;

        self.calibrating.store(false, Ordering::Release);
        self.finalized.store(true, Ordering::Release);
        tracing::info!(
            "InnerQ scales activated (group_size={group_size}, max_ratio={max_ratio:.3}, \
             strength={:.2})",
            self.strength,
        );
        Ok(true)
    }
}
