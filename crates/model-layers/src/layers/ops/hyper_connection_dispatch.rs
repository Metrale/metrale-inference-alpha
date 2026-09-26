// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One entry per mHC stage (`hc_pre`, `hc_post`, `hc_head`) that selects between the Sinkhorn and the low-rank hyper-connection launchers.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Both families run over the same FP32 `[T, hc_mult, H]` streams and use the same kernel
//!   names, because a model's own `hyper_connection.cu` replaces the whole file
//!   (kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu), but their argument lists
//!   differ. The signatures here are the union of both.
//! - The variant is chosen from the weights (`lowrank.is_some()`), never from the model name;
//!   a site with low-rank weights takes the low-rank launcher.
//! - `post_out` / `post` carries the low-rank injection vector: both are `[T, hc]` FP32 and read
//!   by the matching `hc_post`, so one buffer serves both variants. The low-rank path neither
//!   writes nor reads `comb`.
//!
//! The DeepSeek-V4 mixer is Sinkhorn-normalized over `hc_fn` / `hc_scale` / `hc_base` and emits
//! a `[hc, hc]` combine matrix; the Qwen3.8-Flash-Next mixer is a low-rank pair with a grouped
//! RMSNorm and emits one scalar per stream.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::hyper_connection as sinkhorn;
use super::hyper_connection_lowrank as lowrank;
use crate::layers::qwen3_attention::{HcHeadWeights, HcSiteWeights, HcWeights};

/// 2026-09-25: Which family a site's weights select. Callers use it to decide whether the
/// block's `input_norm` runs on `hc_pre`'s output ([`HcVariant::applies_block_input_norm`]): in
/// the low-rank family `hc_norm` inside `hc_pre` is the block's input norm, and a second RMS pass
/// with a ones-filled weight is not an identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HcVariant {
    /// 2026-09-25: DeepSeek-V4: Sinkhorn mix, `[hc, hc]` combine matrix.
    Sinkhorn,
    /// 2026-09-25: Qwen3.8-Flash-Next: low-rank pair, one injection scalar per stream.
    LowRank,
}

impl HcVariant {
    pub fn of_site(site: &HcSiteWeights) -> Self {
        if site.lowrank.is_some() {
            Self::LowRank
        } else {
            Self::Sinkhorn
        }
    }

    pub fn of(hc: &HcWeights) -> Self {
        Self::of_site(&hc.attn)
    }

    /// 2026-09-25: Whether the block's own `input_norm` runs on `hc_pre`'s output: true only for
    /// [`HcVariant::Sinkhorn`].
    pub fn applies_block_input_norm(self) -> bool {
        self == Self::Sinkhorn
    }
}

/// 2026-09-25: Collapse the streams to one and emit what the matching `hc_post` needs: `post` and
/// `comb` for Sinkhorn, the injection vector in `post_out` for low-rank.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    site: &HcSiteWeights,
    hc: &HcWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    match &site.lowrank {
        Some(w) => lowrank::hc_pre_lowrank(
            gpu,
            kernel,
            streams,
            w,
            y_out,
            post_out,
            scratch,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            norm_eps,
            stream,
        ),
        None => sinkhorn::hc_pre(
            gpu,
            kernel,
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            y_out,
            post_out,
            comb_out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            hc.sinkhorn_iters as u32,
            norm_eps,
            hc.hc_eps,
            stream,
        ),
    }
}

/// 2026-09-25: Inject the block output back into every stream. `out` may alias `residual`.
///
/// Takes the whole [`HcWeights`] rather than a site because neither variant's `hc_post` reads
/// site weights: Sinkhorn reads the `comb` its `hc_pre` emitted, low-rank the injection vector.
/// The variant is taken from the layer's attention site.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hc: &HcWeights,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    match HcVariant::of(hc) {
        // 2026-09-25: `post` is the injection vector here; `comb` is not read.
        HcVariant::LowRank => lowrank::hc_post_lowrank(
            gpu,
            kernel,
            block_out,
            residual,
            post,
            out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            stream,
        ),
        HcVariant::Sinkhorn => sinkhorn::hc_post(
            gpu,
            kernel,
            block_out,
            residual,
            post,
            comb,
            out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            stream,
        ),
    }
}

/// 2026-09-25: The model-level final collapse before the LM head. In the low-rank family this is
/// also the model's final normalization: the Qwen3.8-Flash-Next config parser sets
/// `final_norm_identity` (config/src/parsers/qwen4_exp.rs), and the mixer's `hc_norm` normalizes.
#[allow(clippy::too_many_arguments)]
pub fn hc_head_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head: &HcHeadWeights,
    hc: &HcWeights,
    y_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    match &head.lowrank {
        Some(w) => lowrank::hc_head_lowrank(
            gpu,
            kernel,
            streams,
            w,
            y_out,
            scratch,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            norm_eps,
            stream,
        ),
        None => sinkhorn::hc_head(
            gpu,
            kernel,
            streams,
            head.hc_fn,
            head.hc_scale,
            head.hc_base,
            y_out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            norm_eps,
            hc.hc_eps,
            stream,
        ),
    }
}
