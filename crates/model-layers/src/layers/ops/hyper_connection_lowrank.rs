// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the Qwen3.8-Flash-Next low-rank hyper-connection (mHC) kernels.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The kernels come from kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu, which
//!   uses the same kernel names as the DeepSeek-V4 file launched by `hyper_connection.rs` but
//!   different argument lists; the two sets of launchers are kept apart for that reason.
//! - `hc_expand` behaves the same in both kernel files, so it has no launcher here.
//! - `hyper_connection_dispatch.rs` routes a site here when `HcSiteWeights::lowrank` is `Some`.
//! - With `T <= 64` and a non-null scratch, [`hc_pre_lowrank`] and [`hc_head_lowrank`] run the
//!   cuBLASLt GEMM path (or the three-launch split path when `METRALE_HC_DECODE_SPLIT=1`); with
//!   larger `T` and a scratch they run the `dense_gemm_bf16_pipelined` GEMM path unless
//!   `METRALE_QWEN4EXP_NO_HC_GEMM=1`; otherwise the fused kernel, one 1024-thread block per token.

use anyhow::Result;
#[path = "hyper_connection_lowrank_gemm.rs"]
mod gemm;
pub(crate) use gemm::hc_pre_gemm;

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::layers::qwen3_attention::HcLowRank;

/// 2026-09-25: `METRALE_QWEN4EXP_NO_HC_GEMM` equal to `"1"`, read once per process: the large-T
/// collapse runs the fused FP32 kernel instead of the GEMM path, which rounds `normed` to BF16
/// before the projections (`hc_pre_stage_bf16`).
fn hc_gemm_disabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_QWEN4EXP_NO_HC_GEMM").as_deref() == Ok("1"))
}

/// 2026-09-25: `METRALE_HC_DECODE_SPLIT` equal to `"1"`, read once per process: at `T <= 64` run
/// the three-launch split path instead of the cuBLASLt GEMM path.
fn hc_decode_split_forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_HC_DECODE_SPLIT").as_deref() == Ok("1"))
}

/// 2026-09-25: Collapse the `hc_mult` streams to one, and emit the per-stream injection weights
/// the matching [`hc_post_lowrank`] needs: `streams [T, hc, H] -> y_out [T, H]` BF16,
/// `inj_out [T, hc]` FP32. Fails when `w.inject_w` is null (a site without an injection weight
/// is the model-level mixer, which uses [`hc_head_lowrank`]).
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !w.inject_w.is_null(),
        "hc_pre_lowrank needs block_inject_weight; a site loaded without one \
         is the model-level mixer and must use hc_head_lowrank"
    );
    // 2026-09-25: Small T: the fused kernel's grid is [T], a single block at T = 1, so small T
    // uses multi-block launches through the scratch instead.
    if num_tokens <= 64 && !scratch.is_null() {
        if !hc_decode_split_forced() {
            return hc_pre_gemm(
                gpu,
                streams,
                w,
                y_out,
                inj_out,
                scratch,
                num_tokens,
                hidden_size,
                hc_mult,
                norm_eps,
                true,
                true,
                stream,
            );
        }
        return hc_pre_split(
            gpu,
            streams,
            w,
            y_out,
            inj_out,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            true,
            stream,
        );
    }
    // 2026-09-25: Large T: the GEMM formulation on `dense_gemm_bf16_pipelined`, unless the kill
    // switch sends it to the fused kernel below.
    if !scratch.is_null() && !hc_gemm_disabled() {
        return hc_pre_gemm(
            gpu,
            streams,
            w,
            y_out,
            inj_out,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            true,
            false,
            stream,
        );
    }
    // 2026-09-25: 1024 threads, and dynamic shared memory for the staged `normed` vector
    // `[hc*H]` and the rank vector, both FP32.
    let smem = (hc_mult * hidden_size + w.rank as u32) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .shared_mem(smem)
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(w.inject_w)
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// 2026-09-25: The model-level mixer (`use_combine=False` in the reference
/// bench/qwen4_exp/ref/modeling_qwen4_exp.py): the same collapse with no injection vector, on
/// the same path choice as [`hc_pre_lowrank`]. It is also the model's final normalization: the
/// config parser sets `final_norm_identity` (config/src/parsers/qwen4_exp.rs).
#[allow(clippy::too_many_arguments)]
pub fn hc_head_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    if num_tokens <= 64 && !scratch.is_null() {
        if !hc_decode_split_forced() {
            return hc_pre_gemm(
                gpu,
                streams,
                w,
                y_out,
                DevicePtr::NULL,
                scratch,
                num_tokens,
                hidden_size,
                hc_mult,
                norm_eps,
                false,
                true,
                stream,
            );
        }
        return hc_pre_split(
            gpu,
            streams,
            w,
            y_out,
            DevicePtr::NULL,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            false,
            stream,
        );
    }
    // 2026-09-25: The GEMM path without the injection GEMM; `hc_pre_mix` skips the injection
    // when `inj_pre` is null.
    if !scratch.is_null() && !hc_gemm_disabled() {
        return hc_pre_gemm(
            gpu,
            streams,
            w,
            y_out,
            DevicePtr::NULL,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            false,
            false,
            stream,
        );
    }
    let smem = (hc_mult * hidden_size + w.rank as u32) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .shared_mem(smem)
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// 2026-09-25: Inject the block output back into every stream:
/// `out[t, s*H + d] = residual[t, s*H + d] + block_out[t, d] * inj[t, s]`. There is no `comb`
/// argument: this family scales each stream by one scalar.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    inj: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(inj)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: The three-launch collapse (`hc_pre_stage`, `hc_pre_down`, `hc_pre_finish`) for
/// `T <= 64`, the row count its scratch layout reserves. `hyper_connection_lowrank_tests.rs`
/// calls it directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_split(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    stream: u64,
) -> Result<()> {
    let hc_dim = hc_mult * hidden_size;
    // 2026-09-25: Scratch layout: normed [64, hc_dim] then low [64, rank], FP32.
    let normed = scratch;
    let low = scratch.offset(64 * hc_dim as usize * 4);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage")?;
    let k_down = gpu.kernel("hyper_connection", "hc_pre_down")?;
    let k_fin = gpu.kernel("hyper_connection", "hc_pre_finish")?;

    KernelLaunch::new(gpu, k_stage)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(normed)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)?;

    // 2026-09-25: Grid y splits the rank rows so that small T still launches several blocks.
    let dsplit = (48 / num_tokens.max(1)).clamp(1, 10);
    KernelLaunch::new(gpu, k_down)
        .grid([num_tokens, dsplit, 1])
        .block([1024, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down_w)
        .arg_ptr(low)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .launch(stream)?;

    let fsplit = (48 / num_tokens.max(1)).clamp(1, 10);
    KernelLaunch::new(gpu, k_fin)
        .grid([num_tokens, fsplit, 1])
        .block([256, 1, 1])
        .shared_mem(w.rank as u32 * 4)
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up_w)
        .arg_ptr(if inject { w.inject_w } else { DevicePtr::NULL })
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .launch(stream)
}
