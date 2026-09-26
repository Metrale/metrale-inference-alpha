// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The single-token decode step split into a host half
//! (`decode_prep`: the per-token uploads) and a capturable device half
//! (`decode_body`), the projection and attend-and-project bodies both paths
//! share, and the check that says whether a layer's decode can be captured
//! into a CUDA graph.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - `decode_prep` and `decode_body` refuse kv-source and index-source layers.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{
    AttnMat, AttnV41, AttnV41LayerState, AttnV41LayerWeights, SharedV41, upload_i32_async,
};
use crate::deepseek_v41_ref::attn::window_topk_idxs;
use metrale_model_layers::layers::ops::{kquant_mmvq_pair_w, kquant_q8_1_rows};

/// 2026-09-25: `METRALE_DS41_SPARSE_SPLIT`, read once: the blocks per (token,
/// head) that `attn_v41_sparse_attn` splits the output dimension across,
/// 1..=8, default 2 (other values fall back to 2). Every block recomputes the
/// same scores and sums its outputs in the same order, so the bytes do not
/// depend on the value.
fn sparse_split() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("METRALE_DS41_SPARSE_SPLIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s| (1..=8).contains(&s))
            .unwrap_or(2)
    })
}

impl AttnV41 {
    /// 2026-09-25: q (low-rank, normed, up-projected, rotated) and kv (one
    /// latent row per token, normed, rotated, fp8-rounded) from the normed
    /// input `x`. Device work only: the positions are read from `pos` /
    /// `head_pos`.
    pub(super) fn q_kv_projections(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        x: DevicePtr,
        m: usize,
        yarn: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        // 2026-09-25: `wq_a` and `wkv` read the same input: one q8_1
        // quantisation and one launch for both when both are Q2_K and m <= 8.
        if let (AttnMat::Q2K(a), AttnMat::Q2K(kv), true) = (w.wq_a, w.wkv, m <= 8) {
            kquant_q8_1_rows(
                gpu,
                self.k.q8_rows,
                x,
                self.a_q8,
                m as u32,
                dim as u32,
                stream,
            )?;
            kquant_mmvq_pair_w(
                gpu,
                self.k.mmvq_q2k_pair_w,
                (a, self.qr_raw, c.q_rank as u32),
                (kv, self.kv_raw, hd as u32),
                self.a_q8,
                dim as u32,
                m as u32,
                stream,
            )?;
        } else {
            self.gemm(gpu, x, w.wq_a, self.qr_raw, m, c.q_rank, dim, stream)?;
            self.gemm(gpu, x, w.wkv, self.kv_raw, m, hd, dim, stream)?;
        }
        self.rmsnorm(
            gpu,
            false,
            self.qr_raw,
            w.q_norm,
            self.qr,
            m,
            c.q_rank,
            stream,
        )?;
        self.gemm(gpu, self.qr, w.wq_b, self.q, m, nh * hd, c.q_rank, stream)?;
        self.rope(gpu, self.q, self.head_pos, m * nh, hd, yarn, false, stream)?;
        self.rmsnorm(gpu, false, self.kv_raw, w.kv_norm, self.kv, m, hd, stream)?;
        self.rope(gpu, self.kv, self.pos, m, hd, yarn, false, stream)?;
        self.act_quant(gpu, self.kv, m * hd, stream)
    }

    /// 2026-09-25: Sparse attention over the selection in `idx` (`topk` entries
    /// a token, -1 = absent), the inverse rotation and the grouped output
    /// projection into `self.out`. Device work only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_and_project(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        rows_a: DevicePtr,
        rows_a_len: usize,
        rows_b: Option<DevicePtr>,
        idx: DevicePtr,
        topk: usize,
        m: usize,
        yarn: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        let scale = (hd as f32).powf(-0.5);
        KernelLaunch::new(gpu, self.k.sparse_attn)
            .grid([m as u32, nh as u32, sparse_split()])
            .block([256, 1, 1])
            .arg_ptr(self.q)
            .arg_ptr(rows_a)
            .arg_ptr(rows_b.unwrap_or(rows_a))
            .arg_u32(rows_a_len as u32)
            .arg_ptr(idx)
            .arg_ptr(w.sink)
            .arg_ptr(self.o)
            .arg_u32(nh as u32)
            .arg_u32(hd as u32)
            .arg_u32(topk as u32)
            .arg_f32(scale)
            .launch(stream)?;
        // 2026-09-25: `self.o` keeps the pre-rotation output (the reference's
        // `sa_o`, returned as `AttnV41Run::o`); the inverse rotation runs on a copy.
        let o_copy = self.o_rot;
        gpu.copy_d2d_async(self.o, o_copy, m * nh * hd * 2, stream)?;
        self.rope(gpu, o_copy, self.head_pos, m * nh, hd, yarn, true, stream)?;

        // 2026-09-25: og[t, g*o_rank + r] = o_g . wo_a[g*o_rank + r]
        if let (AttnMat::Q2K(blocks), true) = (w.wo_a, m <= 8) {
            self.wo_a_grouped(gpu, blocks, o_copy, m, stream)?;
        } else {
            self.wo_a_per_group(gpu, w, o_copy, m, stream)?;
        }
        self.gemm(
            gpu,
            self.og,
            w.wo_b,
            self.out,
            m,
            dim,
            c.groups * c.o_rank,
            stream,
        )?;
        Ok(())
    }

    /// 2026-09-25: The per-group `wo_a` path (bf16 `wo_a`, or m > 8): slice the
    /// group's columns out of the rotated attention output, project, scatter
    /// into `og`.
    pub(super) fn wo_a_per_group(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        o_copy: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd) = (c.n_heads, c.head_dim);
        let gw = c.gw();
        for g in 0..c.groups {
            KernelLaunch::new(gpu, self.k.slice_cols)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(o_copy)
                .arg_ptr(self.slice_in)
                .arg_u32((nh * hd) as u32)
                .arg_u32((g * gw) as u32)
                .arg_u32(gw as u32)
                .launch(stream)?;
            self.gemm(
                gpu,
                self.slice_in,
                w.wo_a.at_rows(g * c.o_rank, gw),
                self.slice_out,
                m,
                c.o_rank,
                gw,
                stream,
            )?;
            KernelLaunch::new(gpu, self.k.scatter_cols)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.slice_out)
                .arg_ptr(self.og)
                .arg_u32((c.groups * c.o_rank) as u32)
                .arg_u32((g * c.o_rank) as u32)
                .arg_u32(c.o_rank as u32)
                .launch(stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: Whether this layer's single-token step can be captured into
    /// a CUDA graph: false on kv and index sources, whose forward decides the
    /// compressor group and the index top-k on the host mid-forward (position
    /// parity, a score download). Every other layer is kernels and copies once
    /// its per-token inputs are read from the device.
    pub fn decode_capturable(&self, w: &AttnV41LayerWeights) -> bool {
        !w.role.is_kv_source && !w.role.is_index_source
    }

    /// 2026-09-25: The `topk` a captured step launches `sparse_attn` with: the
    /// widest selection of the layer's class, so the launch argument is
    /// constant; `decode_prep_role` pads the selection with -1. A -1 entry adds
    /// an exact zero to its thread's denominator sum and nothing to the output,
    /// so the result equals the eager step's bit for bit.
    pub fn decode_fixed_topk(&self, w: &AttnV41LayerWeights) -> usize {
        self.decode_fixed_topk_of(w.role.ratio > 0)
    }

    fn decode_fixed_topk_of(&self, ratio_positive: bool) -> usize {
        let c = &self.cfg;
        if ratio_positive {
            (c.window + c.index_topk).min(2048)
        } else {
            c.window
        }
    }

    /// 2026-09-25: Forget what the captured decode step last uploaded; the next
    /// [`Self::decode_prep`] re-uploads. Call after any eager `forward` (it
    /// writes `pos` / `head_pos` / `idx_dev` for itself) and at a new step.
    pub fn invalidate_decode_uploads(&mut self) {
        self.decode_pos = None;
        self.decode_idx = None;
        self.decode_idx_win = None;
    }

    /// 2026-09-25: The selection buffer a captured step reads for a layer of
    /// this class: `idx_dev` for ratio > 0, `idx_dev_win` for ratio 0.
    fn decode_idx_of(&self, ratio_positive: bool) -> DevicePtr {
        if ratio_positive {
            self.idx_dev
        } else {
            self.idx_dev_win
        }
    }

    /// 2026-09-25: The host half of a capturable layer's single-token step at
    /// `start_pos`: the position into `pos` / `head_pos` and the padded
    /// selection (window slots, then the shared index selection, then -1) into
    /// the class's selection buffer, each only when it differs from what is
    /// already there. Returns the compressed rows the captured `sparse_attn`
    /// reads (`None` on ratio-0 layers): a pointer the graph bakes, which the
    /// callers compare on every replay.
    pub fn decode_prep(
        &mut self,
        w: &AttnV41LayerWeights,
        shared: &SharedV41,
        gpu: &dyn GpuBackend,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        ensure!(
            self.decode_capturable(w),
            "attn_v41: decode_prep on a kv/index source layer"
        );
        self.decode_prep_role(w.role.ratio > 0, shared, gpu, start_pos, stream)
    }

    /// 2026-09-25: [`Self::decode_prep`] by selection class: `ratio_positive`
    /// layers read the window plus the shared index selection from `idx_dev`,
    /// the others the window alone from `idx_dev_win`; both classes can be
    /// prepared for one multi-layer capture. Fails for `start_pos` outside
    /// `1..max_seq`.
    pub fn decode_prep_role(
        &mut self,
        ratio_positive: bool,
        shared: &SharedV41,
        gpu: &dyn GpuBackend,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let c = &self.cfg;
        ensure!(
            start_pos >= 1 && start_pos < c.max_seq,
            "attn_v41: position {start_pos} outside the decode range 1..{}",
            c.max_seq
        );
        if self.decode_pos != Some(start_pos) {
            let p = start_pos as i32;
            upload_i32_async(gpu, self.pos, &[p], stream)?;
            upload_i32_async(gpu, self.head_pos, &vec![p; c.n_heads], stream)?;
            self.decode_pos = Some(start_pos);
        }
        let fixed = self.decode_fixed_topk_of(ratio_positive);
        let (mut idx, topk) = window_topk_idxs(c.window, 1, start_pos);
        debug_assert_eq!(topk, c.window);
        let rows_b = if ratio_positive {
            ensure!(
                shared.topk_idxs.len() == shared.topk,
                "attn_v41: shared index selection is {} for one token x {}",
                shared.topk_idxs.len(),
                shared.topk
            );
            idx.extend_from_slice(&shared.topk_idxs);
            Some(
                shared
                    .compress_kv
                    .context("compressed layer before any kv source published")?,
            )
        } else {
            None
        };
        ensure!(
            idx.len() <= fixed,
            "attn_v41: selection {} exceeds the fixed {fixed}",
            idx.len()
        );
        idx.resize(fixed, -1);
        let (buf, cache) = if ratio_positive {
            (self.idx_dev, &mut self.decode_idx)
        } else {
            (self.idx_dev_win, &mut self.decode_idx_win)
        };
        if cache.as_deref() != Some(&idx[..]) {
            upload_i32_async(gpu, buf, &idx, stream)?;
            *cache = Some(idx);
        }
        Ok(rows_b)
    }

    /// 2026-09-25: The device half of a capturable layer's single-token step:
    /// the projections, the window ring write at `pos % window` (slot read on
    /// the device), sparse attention over the padded selection and the output
    /// projection into the returned buffer. No host work and no per-token
    /// scalar arguments, so a captured graph replays for any position once
    /// `decode_prep` has updated the device inputs.
    pub fn decode_body(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        x: DevicePtr,
        rows_b: Option<DevicePtr>,
        stream: u64,
    ) -> Result<DevicePtr> {
        let c = &self.cfg;
        ensure!(
            self.decode_capturable(w),
            "attn_v41: decode_body on a kv/index source layer"
        );
        ensure!(
            (w.role.ratio > 0) == rows_b.is_some(),
            "attn_v41: compressed rows {} on a ratio-{} layer",
            if rows_b.is_some() { "given" } else { "missing" },
            w.role.ratio
        );
        let yarn = w.role.ratio > 0;
        self.q_kv_projections(gpu, w, x, 1, yarn, stream)?;
        KernelLaunch::new(gpu, self.k.ring_put)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.kv)
            .arg_ptr(st.window)
            .arg_ptr(self.pos)
            .arg_u32(c.window as u32)
            .arg_u32(c.head_dim as u32)
            .launch(stream)?;
        self.attend_and_project(
            gpu,
            w,
            st.window,
            c.window,
            rows_b,
            self.decode_idx_of(yarn),
            self.decode_fixed_topk(w),
            1,
            yarn,
            stream,
        )?;
        Ok(self.out)
    }
}
