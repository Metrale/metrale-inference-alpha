// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The chunked GDN prefill launcher (three kernels) and the guard
//! that decides whether its tensor-core state spine runs.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.
#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: The tensor-core spine's compile-time tile: `K_DIM == V_DIM` in
/// `kernels/hopper/common/gated_delta_rule_chunk_tc.cu`.
pub(crate) const GDN_TC_DIM: u32 = 128;
/// 2026-09-25: That kernel's `CHUNK`.
pub(crate) const GDN_TC_CHUNK: u32 = 64;
/// 2026-09-25: Mirror of `TCF_SMEM` in the same file:
/// `St[128][136] + Wp[64][136] + Up[64][136] + ducT[128][72]` bf16 plus
/// `dec[65]` f32, `34816 + 17408 + 17408 + 18432 + 260 = 88 324` bytes.
/// The padded 136/72 row strides make the MMA fragment reads
/// bank-conflict-free. Under-sizing this reads a tile out of bounds, so the
/// launcher and the kernel must agree on it.
pub(crate) const GDN_TC_SMEM: u32 = GDN_TC_DIM * 136 * 2
    + 2 * (GDN_TC_CHUNK * 136 * 2)
    + GDN_TC_DIM * 72 * 2
    + (GDN_TC_CHUNK + 1) * 4;

/// 2026-09-25: Why the tensor-core state spine is not running; `None` means it
/// is.
///
/// Pure, so the grammar is testable without a GPU or the process environment.
/// Each refusal names its guard, so a run that asked for the spine and fell
/// back says which guard refused.
///
/// The tile guards are required, not defensive: the kernel's shared-memory
/// layout and fragment maps are compile-time 128/128/64, and its K staging
/// reads 16 bytes at a time, so a narrower head or a `qk_stride` that is not a
/// multiple of 8 would load the wrong columns.
pub(crate) fn gdn_tc_spine_reject(
    requested: bool,
    kernel_present: bool,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
    qk_stride: u32,
) -> Option<&'static str> {
    if !requested {
        Some("not requested")
    } else if !kernel_present {
        Some("kernel absent from this image")
    } else if k_dim != GDN_TC_DIM || v_dim != GDN_TC_DIM || chunk != GDN_TC_CHUNK {
        Some("head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)")
    } else if !qk_stride.is_multiple_of(8) {
        Some("qk_stride is not a multiple of 8 (the K staging uses 16-byte vector loads)")
    } else {
        None
    }
}

#[cfg(test)]
#[path = "ssm_gdn_tc_tests.rs"]
mod ssm_gdn_tc_tests;

/// 2026-09-25: Chunked GDN prefill in three launches, ordered by `stream`:
///   1. `recompute_wu`, or its Hopper twin: per chunk, W and U (bf16) and the
///      cumulative gate `gc` (f32);
///   2. a state spine, chosen below: walks the chunks in order, writes each
///      chunk's entry state `S` and `uc` (bf16), and updates `h_state` in place;
///   3. `chunk_fwd_o`, or its Hopper twin: the bf16 output.
///
/// `w_out`, `u_out`, `s_out`, `uc_out` and `gc_out` are caller-sized scratch.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_fla(
    gpu: &dyn GpuBackend,
    k_recompute_wu: KernelHandle,
    // 2026-09-25: Hopper twins of kernels 1 and 3, `KernelHandle(0)` on every
    // other target. `gdn_hopper_remnants` chooses between twin and parent.
    k_recompute_wu_hopper: KernelHandle,
    k_chunk_fwd_o_hopper: KernelHandle,
    k_chunk_delta_h: KernelHandle,
    // 2026-09-25: The DV-block-split spine
    // (`gated_delta_rule_chunk_delta_h_tc_vblock`). It runs only when the fused
    // spine does not, the handle is non-zero and `METRALE_GDN_TC_VBLOCK=1`.
    k_chunk_delta_h_tc_vblock: KernelHandle,
    // 2026-09-25: The tensor-core spine (`GDN_TC_SPINE_ENTRY`). `KernelHandle(0)`
    // unless the resolved `gdn_prefill_tc` lever is on and the target compiles
    // the kernel. Same arguments as the fused spine, launched on grid
    // [nv, batch] with 256 threads and `GDN_TC_SMEM` bytes of shared memory.
    k_chunk_delta_h_tcfuse: KernelHandle,
    k_chunk_delta_h_fused: KernelHandle,
    k_chunk_delta_h_tma: KernelHandle,
    k_chunk_fwd_o: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    w_out: DevicePtr,
    u_out: DevicePtr,
    s_out: DevicePtr,
    uc_out: DevicePtr,
    gc_out: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_chunks: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // 2026-09-25: true: `h_state` is a device table of per-sequence state
    // pointers, each `[nv, kd, vd]`. false: one contiguous base.
    h_state_is_table: bool,
    // 2026-09-25: true: streams of different lengths. `cu_seqlens` holds
    // `batch + 1` device token offsets, and `num_chunks` must be the maximum over
    // the streams (it is grid x). The kernels derive each stream's chunk offset
    // from `cu_seqlens` and do not read `cu_chunks`. false: pass NULL for both.
    cu_seqlens: DevicePtr,
    cu_chunks: DevicePtr,
    is_varlen: bool,
    profile: bool,
    stream: u64,
) -> Result<()> {
    const C: u32 = 64; // 2026-09-25: the kernels' `CHUNK`.
    let (kd, vd) = (k_dim, v_dim);
    // 2026-09-25: Shared-memory bytes per kernel. In `recompute_wu`, L aliases
    // the kk Gram (they use disjoint triangles), so one C*C*4 buffer holds both.
    let smem_wu = C * kd * 2 + C * C * 4 + C * 4;
    let smem_dh = 2 * (C * (2 * kd + vd) * 2) + 2 * C * 4 + 2 * (C + 1) * 4;
    let smem_fo = C * kd * 2 + C * kd * 2 + C * C * 4 + C * vd * 2 + kd * vd * 2 + 2 * C * 4;

    let mut t0: Option<std::time::Instant> = if profile {
        gpu.synchronize(stream)?;
        Some(std::time::Instant::now())
    } else {
        None
    };

    macro_rules! prof {
        ($label:expr, $t0:expr) => {
            if let Some(t0) = $t0.take() {
                gpu.synchronize(stream)?;
                let elapsed = t0.elapsed().as_micros();
                tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, seq_len, elapsed);
                *$t0 = Some(std::time::Instant::now());
            }
        };
    }

    // 2026-09-25: The tensor-core prefill lever, read once for the three kernels
    // it selects: the two twins here and the state spine below. It is the
    // target's `[defaults] gdn_prefill_tc`, overridden by the value of
    // `METRALE_GDN_PREFILL_TC` (`=0` turns it off).
    // `METRALE_NO_GDN_PREFILL_TC_REMNANTS=1` keeps the spine and pins the twins
    // to their parents.
    let tc_requested = super::target_defaults::resolved().gdn_prefill_tc.value;
    let (wu, fo) = gdn_hopper_remnants(
        tc_requested,
        k_recompute_wu,
        smem_wu,
        k_chunk_fwd_o,
        smem_fo,
        k_recompute_wu_hopper,
        k_chunk_fwd_o_hopper,
        kd,
        vd,
        C,
    );

    // 2026-09-25: Kernel 1: recompute_wu, or its Hopper twin.
    KernelLaunch::new(gpu, wu.kernel)
        .grid([num_chunks, num_v_heads, batch_size])
        .block([wu.block, 1, 1])
        .shared_mem(wu.smem)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(w_out)
        .arg_ptr(u_out)
        .arg_ptr(gc_out)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_chunks)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(kd)
        .arg_u32(vd)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_ptr(cu_seqlens)
        .arg_ptr(cu_chunks)
        .arg_u32(is_varlen as u32)
        .launch(stream)?;
    prof!("gdn_fla_recompute_wu", &mut t0);

    // 2026-09-25: Kernel 2, the state spine. The non-TMA candidates take the
    // same arguments; grid y, block size and shared memory differ. In order of
    // precedence: TMA and the tensor-core spine (both below); the fused spine
    // that `init_kernels::fused_spine_kernel` loaded (`_vfused` by default,
    // `_vtile` under `METRALE_GDN_VTILE=1`, `_pipe` under `METRALE_GDN_PIPE=1`),
    // unless its handle is 0 or `METRALE_GDN_VTILE=0`; `_tc_vblock` under
    // `METRALE_GDN_TC_VBLOCK=1`; else `_ksplit`. The fused spine folds the two
    // per-chunk passes into one, so at kd = vd = 128 its shared memory is
    // 49,412 B against ksplit's 99,336 B.
    let use_fused = k_chunk_delta_h_fused.0 != 0
        && std::env::var("METRALE_GDN_VTILE").ok().as_deref() != Some("0");
    let use_tcvb = !use_fused
        && k_chunk_delta_h_tc_vblock.0 != 0
        && std::env::var("METRALE_GDN_TC_VBLOCK").ok().as_deref() == Some("1");
    const DV_BLK: u32 = 64; // 2026-09-25: the kernel's compile-time `DV_BLK`.
    let num_dv_blk = (vd / DV_BLK).max(1);
    // 2026-09-25: tc_vblock smem: St[DV_BLK*kd] + ws[C*DV_BLK] f32
    // + buf[2][C*kd + C*DV_BLK] + gcb[2][C] f32 + decb[2][C+1] f32.
    let smem_tcvb = DV_BLK * kd * 2
        + C * DV_BLK * 4
        + 2 * (C * kd + C * DV_BLK) * 2
        + 2 * C * 4
        + 2 * (C + 1) * 4;
    // 2026-09-25: `_vfused` and `_vtile` stage W, K and U single-buffered plus one
    // decay row and do not split the DV axis, so grid y stays `batch_size`; they
    // differ only in thread count. `_pipe` double-buffers W, K and U, so it needs
    // `smem_dh`, the ksplit footprint. The env reads here must select the same
    // build that `init_kernels::fused_spine_kernel` loaded, or the second buffer
    // is read out of bounds.
    let pipe = std::env::var("METRALE_GDN_PIPE").ok().as_deref() == Some("1");
    let smem_fused = if pipe {
        smem_dh
    } else {
        C * kd * 2 + C * kd * 2 + C * vd * 2 + (C + 1) * 4
    };
    let fused_block = match std::env::var("METRALE_GDN_VTILE").ok().as_deref() {
        Some("1") if !pipe => 512u32,
        _ => 256u32,
    };
    // 2026-09-25: The tensor-core spine puts both per-chunk products on
    // `mma.sync.m16n8k16`: bf16 operands (S_c and duc as two bf16 limbs each
    // in the `GDN_TC_SPINE_ENTRY` build), f32 accumulate. The recurrent state
    // stays in the f32 accumulator. It runs when the resolved `gdn_prefill_tc`
    // lever is on (only `kernels/hopper` declares it true) and every guard in
    // `gdn_tc_spine_reject` passes; a refused request is logged with its guard.
    let smem_tcfuse = GDN_TC_SMEM;
    let tc_reject = gdn_tc_spine_reject(
        tc_requested,
        k_chunk_delta_h_tcfuse.0 != 0,
        kd,
        vd,
        C,
        qk_stride,
    );
    if tc_requested && let Some(why) = tc_reject {
        tracing::warn!("METRALE_GDN_PREFILL_TC set but tensor-core spine is NOT running: {why}");
    }
    let tc_ok = tc_reject.is_none();
    if tc_ok {
        // 2026-09-25: Built from `GDN_TC_SPINE_ENTRY`, the constant
        // `init_kernels` binds the handle with.
        tracing::info!(
            "{}",
            gdn_tc_spine_route_line(num_v_heads, batch_size, smem_tcfuse)
        );
    }
    // 2026-09-25: TMA state spine (`METRALE_GDN_TMA=1`). Every precondition is
    // checked here. The descriptors encode the compile-time tile
    // (K_DIM = V_DIM = 128, CHUNK = 64), so a narrower head would load the wrong
    // columns.
    let tma_requested = std::env::var("METRALE_GDN_TMA").ok().as_deref() == Some("1");
    let tma_reject: Option<&str> = if !tma_requested {
        Some("not requested")
    } else if tc_ok {
        // 2026-09-25: With both levers set, TMA yields to the tensor-core spine
        // and logs that it did.
        Some("METRALE_GDN_PREFILL_TC is active and takes precedence")
    } else if k_chunk_delta_h_tma.0 == 0 {
        Some("kernel absent from this image")
    } else if is_varlen {
        Some(
            "varlen: choff comes from cu_chunks, so the descriptor's flat row count is unknown host-side",
        )
    } else if kd != 128 || vd != 128 || C != 64 {
        Some("head/chunk differs from the compile-time tile the descriptors encode")
    } else if !qk_stride.is_multiple_of(8) {
        Some("qk_stride is not a multiple of 8 (bf16 row pitch must be 16-byte aligned)")
    } else {
        None
    };
    if tma_requested && let Some(why) = tma_reject {
        tracing::warn!("METRALE_GDN_TMA=1 but the TMA spine is NOT running: {why}");
    }
    let tma_ok = tma_reject.is_none();
    if tma_ok {
        tracing::info!("GDN state spine: gated_delta_rule_chunk_delta_h_tma");
    }
    // 2026-09-25: `cuda_backend`, and with it `TensorMap`, exists only under the
    // `cuda` feature. Without it the TMA handle is absent and `tma_ok` is false,
    // but the `use` below is resolved at compile time, so it needs the `cfg`.
    #[cfg(feature = "cuda")]
    if tma_ok {
        use metrale_gpu_runtime::cuda_backend::tensormap::TensorMap;
        // 2026-09-25: W/U are [total_blocks][CHUNK][tile] flattened; as a 2-D
        // tensor that is (total_blocks * CHUNK) rows of `tile` columns, contiguous.
        let blocks = (batch_size as u64) * (num_chunks as u64) * (num_v_heads as u64);
        let w_map =
            TensorMap::tiled_2d_bf16(w_out, blocks * C as u64, kd as u64, kd as u64, C, kd)?;
        let u_map =
            TensorMap::tiled_2d_bf16(u_out, blocks * C as u64, vd as u64, vd as u64, C, vd)?;
        // 2026-09-25: K is a view into the packed conv output: rows are tokens
        // at a `qk_stride` pitch, and the kernel supplies `kh * K_DIM` as the
        // column origin.
        let k_map = TensorMap::tiled_2d_bf16(
            key,
            (batch_size as u64) * (seq_len as u64),
            qk_stride as u64,
            qk_stride as u64,
            C,
            kd,
        )?;
        KernelLaunch::new(gpu, k_chunk_delta_h_tma)
            .grid([num_v_heads, batch_size, 1])
            .block([256, 1, 1])
            .shared_mem(smem_dh)
            .arg_ptr(h_state)
            .arg_tensormap(w_map.bytes())
            .arg_tensormap(u_map.bytes())
            .arg_tensormap(k_map.bytes())
            .arg_ptr(gc_out)
            .arg_ptr(s_out)
            .arg_ptr(uc_out)
            .arg_u32(batch_size)
            .arg_u32(seq_len)
            .arg_u32(num_chunks)
            .arg_u32(num_k_heads)
            .arg_u32(num_v_heads)
            .arg_u32(vd)
            .arg_u32(h_state_is_table as u32)
            .arg_ptr(cu_seqlens)
            .arg_ptr(cu_chunks)
            .arg_u32(is_varlen as u32)
            .launch(stream)?;
        prof!("gdn_fla_chunk_delta_h", &mut t0);
    }

    // 2026-09-25: Kernel 2 without TMA. Both paths write s_out/uc_out, and
    // kernel 3 is the same either way.
    if !tma_ok {
        let (k_cdh, cdh_grid_y, cdh_smem, cdh_block) = if tc_ok {
            (k_chunk_delta_h_tcfuse, batch_size, smem_tcfuse, 256u32)
        } else if use_fused {
            (k_chunk_delta_h_fused, batch_size, smem_fused, fused_block)
        } else if use_tcvb {
            (
                k_chunk_delta_h_tc_vblock,
                batch_size * num_dv_blk,
                smem_tcvb,
                256u32,
            )
        } else {
            (k_chunk_delta_h, batch_size, smem_dh, 256u32)
        };
        KernelLaunch::new(gpu, k_cdh)
            .grid([num_v_heads, cdh_grid_y, 1])
            .block([cdh_block, 1, 1])
            .shared_mem(cdh_smem)
            .arg_ptr(h_state)
            .arg_ptr(w_out)
            .arg_ptr(u_out)
            .arg_ptr(key)
            .arg_ptr(gate)
            .arg_ptr(gc_out)
            .arg_ptr(s_out)
            .arg_ptr(uc_out)
            .arg_u32(batch_size)
            .arg_u32(seq_len)
            .arg_u32(num_chunks)
            .arg_u32(num_k_heads)
            .arg_u32(num_v_heads)
            .arg_u32(kd)
            .arg_u32(vd)
            .arg_u32(qk_stride)
            .arg_u32(gb_stride)
            .arg_u32(h_state_is_table as u32)
            .arg_ptr(cu_seqlens)
            .arg_ptr(cu_chunks)
            .arg_u32(is_varlen as u32)
            .launch(stream)?;
        prof!("gdn_fla_chunk_delta_h", &mut t0);
    }

    // 2026-09-25: Kernel 3: chunk_fwd_o, or its Hopper twin.
    KernelLaunch::new(gpu, fo.kernel)
        .grid([num_chunks, num_v_heads, batch_size])
        .block([fo.block, 1, 1])
        .shared_mem(fo.smem)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(gate)
        .arg_ptr(gc_out)
        .arg_ptr(s_out)
        .arg_ptr(uc_out)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_chunks)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(kd)
        .arg_u32(vd)
        .arg_u32(qk_stride)
        .arg_u32(gb_stride)
        .arg_ptr(cu_seqlens)
        .arg_ptr(cu_chunks)
        .arg_u32(is_varlen as u32)
        .launch(stream)?;
    if let Some(t0) = t0 {
        gpu.synchronize(stream)?;
        let elapsed = t0.elapsed().as_micros();
        tracing::info!(
            "  SSM prefill [gdn_fla_chunk_fwd_o] N={}: {}µs",
            seq_len,
            elapsed
        );
    }
    Ok(())
}
