// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The FP8 KV-cache write for decode: the calibration window's observation, the write,
//! and the fused k-norm + RoPE + FP8 write with its eligibility rule.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - The unfused write reads `effective_fp8_scales()` after the calibration observation (which is
//!   skipped under graph capture), so it writes with the scales that observation settled on.
//! - `write_kv_cache_fp8_fused` returns an error, and launches nothing, when
//!   `fused_fp8_kv_decode_eligible` is false.

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use crate::layers::fp8_calibration::Fp8KvWriteTarget;
use crate::layers::ops;

/// 2026-09-25: Largest `head_dim` the fused decode kernel's shared-memory row holds
/// (`FUSED_KFP8_MAX_HEAD_DIM` in `reshape_and_cache_fused_k_fp8.cu`). The two must move together.
const FUSED_FP8_MAX_HEAD_DIM: u32 = 256;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Feed this write to the layer's FP8 calibration, which accumulates the amax over
    /// the `--fp8-kv-calibration-tokens` window across requests and, when the window closes,
    /// rewrites the staged part of the window with the frozen scales; so it is told where this
    /// write goes. Called before the write, so `effective_fp8_scales()` then returns the scales
    /// this batch is written with. A layer with no calibration does nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_fp8_kv_write(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(ref cal) = self.fp8_calibration else {
            return Ok(());
        };
        let target = Fp8KvWriteTarget {
            kernel: self.reshape_cache_k,
            k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
            v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
            block_size,
            cache_stride: kv_cache.cache_stride() as u64,
            key_stride,
            value_stride,
            slot,
        };
        cal.observe(
            gpu,
            k,
            v,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
            &target,
        )
    }
    /// 2026-09-25: Observe (skipped under graph capture), then write with the current scales.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_kv_cache_fp8(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
        graph_capture: bool,
    ) -> Result<()> {
        if !graph_capture {
            self.observe_fp8_kv_write(
                gpu,
                k,
                v,
                kv_cache,
                slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
                stream,
            )?;
        }
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::reshape_and_cache_fp8(
            gpu,
            self.reshape_cache_k,
            k,
            v,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            slot,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            k_scale,
            v_scale,
            key_stride,
            value_stride,
            kv_cache.cache_stride() as u64,
            stream,
        )
    }

    /// 2026-09-25: Whether this layer's single-token decode may take the fused k_norm + RoPE + FP8
    /// write kernel instead of the unfused chain. The kernel checks none of these itself; when this
    /// is false the caller runs the unfused chain.
    ///
    /// - FP8 KV.
    /// - The handle is loaded. `reshape_and_cache_fused_k_fp8.cu` is in the gb10 tree, which the
    ///   hopper and b200 targets inherit; b300 leaves it out, and metal, strix and strix-hip have
    ///   their own trees.
    /// - No MLA, YaRN, proportional or MRoPE rotation: the kernel implements `rope_forward`'s
    ///   rotate-half with a `theta`-derived frequency only.
    /// - A per-head `k_norm`, not `k_norm_full`: the kernel reduces over `head_dim`, one CTA per
    ///   (token, kv_head).
    /// - Not `norm_vanilla`: the kernel applies `rms_norm`'s `(1 + w)`.
    /// - No FP8 calibration: the calibration observes the normed, rotated K, which the fused path
    ///   never writes to memory.
    /// - `head_dim` a positive multiple of 32 and at most [`FUSED_FP8_MAX_HEAD_DIM`]: the block is
    ///   `head_dim` threads.
    /// - `rotary_dim` positive, even and at most `head_dim`: `rope_forward` pairs
    ///   `(d, d + rotary_dim/2)`.
    pub(in super::super) fn fused_fp8_kv_decode_eligible(
        &self,
        head_dim: u32,
        rotary_dim: u32,
    ) -> bool {
        self.kv_dtype == KvCacheDtype::Fp8
            && self.fused_k_norm_rope_cache_write_fp8_kv_k.0 != 0
            && self.mla.is_none()
            && self.yarn_inv_freq.is_null()
            && !self.rope_proportional
            && !self.mrope_interleaved
            && self.attn.k_norm_full.is_none()
            && !self.attn.k_norm.weight.is_null()
            && !self.norm_vanilla
            && self.fp8_calibration.is_none()
            && head_dim > 0
            && head_dim.is_multiple_of(32)
            && head_dim <= FUSED_FP8_MAX_HEAD_DIM
            && rotary_dim > 0
            && rotary_dim.is_multiple_of(2)
            && rotary_dim <= head_dim
    }

    /// 2026-09-25: The fused write. The caller must have skipped both the K-side `rms_norm` and the
    /// K half of the RoPE launch: this kernel reads the raw K projection and does both itself.
    /// Returns an error when [`Self::fused_fp8_kv_decode_eligible`] is false.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn write_kv_cache_fp8_fused(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        positions: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        rms_eps: f32,
        theta: f32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: `attention_forward` takes this path only when the same predicate holds. An
        // `ensure!` rather than a `debug_assert!` because a fused write with its preconditions
        // unmet would corrupt the KV cache silently.
        anyhow::ensure!(
            self.fused_fp8_kv_decode_eligible(head_dim, rotary_dim),
            concat!(
                "write_kv_cache_fp8_fused called on an ineligible layer ",
                "(kv_dtype={:?}, head_dim={}, rotary_dim={})"
            ),
            self.kv_dtype,
            head_dim,
            rotary_dim,
        );
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::fused_k_norm_rope_cache_write_fp8_kv(
            gpu,
            self.fused_k_norm_rope_cache_write_fp8_kv_k,
            k,
            v,
            self.attn.k_norm.weight,
            positions,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            slot,
            num_tokens,
            num_kv_heads,
            head_dim,
            rotary_dim,
            block_size,
            k_scale,
            v_scale,
            key_stride,
            value_stride,
            kv_cache.cache_stride() as u64,
            rms_eps,
            theta,
            stream,
        )
    }
}
