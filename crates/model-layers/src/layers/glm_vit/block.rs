// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One GLM ViT block, and the GEMM, norm and attention launches it uses.
//!
//! A block is `h += attn(rmsnorm(h))` then `h += swiglu_mlp(rmsnorm(h))`:
//! pre-norm with unscaled residual adds.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{GlmVit, GlmVitBlock};

/// 2026-09-25: Shared-memory floats a block-wide reduction needs: one per warp
/// of a block of at most 1024 threads.
const RED_SLOTS: u32 = 32;

impl GlmVit {
    /// 2026-09-25: `C[M,N] = A[M,K] · B[N,K]^T (+ bias[N])`.
    ///
    /// `dense_gemm_bf16_pipelined` has no bias argument, so a bias is a second
    /// launch (`glm_vit_add_bias`) on the same stream. The merger's
    /// proj/gate/up/down GEMMs pass `None`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        bias: Option<DevicePtr>,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        let Some(bias) = bias else { return Ok(()) };
        KernelLaunch::new(gpu, self.k_add_bias)
            .grid([div_ceil(m * n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(bias)
            .arg_u32(m)
            .arg_u32(n)
            .launch(stream)
    }

    /// 2026-09-25: Weight-only RMSNorm, in place over `rows × dim`.
    pub(super) fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        rows: u32,
        dim: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_rmsnorm)
            .grid([rows, 1, 1])
            .block([dim.min(1024), 1, 1])
            .shared_mem(RED_SLOTS * 4)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_u32(rows)
            .arg_u32(dim)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)
    }

    /// 2026-09-25: Mean-subtracting LayerNorm with bias, in place. Used only for
    /// the merger's `post_projection_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layernorm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        b: DevicePtr,
        rows: u32,
        dim: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_layernorm)
            .grid([rows, 1, 1])
            .block([dim.min(1024), 1, 1])
            .shared_mem(RED_SLOTS * 4)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(b)
            .arg_u32(rows)
            .arg_u32(dim)
            .arg_f32(eps)
            .launch(stream)
    }

    /// 2026-09-25: `out = silu(min(gate, limit)) * clamp(up, ±limit)`, elementwise.
    pub(super) fn swiglu(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        out: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_swiglu)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(out)
            .arg_u32(n)
            .arg_f32(self.swiglu_limit)
            .launch(stream)
    }

    pub(super) fn copy(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(n)
            .launch(stream)
    }

    pub(super) fn add_inplace(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n)
            .launch(stream)
    }

    /// 2026-09-25: Unmasked attention over one image's `[seq, 3*H*D]` QKV rows,
    /// with the per-head QK-RMSNorm and the axial RoPE folded into the
    /// deinterleave. `block` calls it once per image, so no token attends
    /// across images.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        gpu: &dyn GpuBackend,
        blk: &GlmVitBlock,
        qkv: DevicePtr,
        o: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        let d = self.head_dim as u32;
        let h_n = self.num_heads as u32;
        let hd = self.hidden_size as u32;

        // 2026-09-25: (1) QK-norm + RoPE into head-contiguous Qr/Kr, and V
        // transposed, for all heads.
        KernelLaunch::new(gpu, self.k_qknorm_rope)
            .grid([seq, h_n, 1])
            .block([d, 1, 1])
            .shared_mem((2 * d + RED_SLOTS) * 4)
            .arg_ptr(qkv)
            .arg_ptr(blk.q_norm_w)
            .arg_ptr(blk.k_norm_w)
            .arg_ptr(self.scratch().buf_qr)
            .arg_ptr(self.scratch().buf_kr)
            .arg_ptr(self.scratch().buf_vt)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)?;

        let qk_head = (seq * d) as usize;
        for head in 0..self.num_heads {
            let qr = self.scratch().buf_qr.offset(head * qk_head * 2);
            let kr = self.scratch().buf_kr.offset(head * qk_head * 2);
            let vt = self.scratch().buf_vt.offset(head * qk_head * 2);
            let o_h = o.offset(head * self.head_dim * 2);

            // 2026-09-25: (2) Raw scores `S[seq,seq] = Qr · Kr^T`, f32.
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr)
                .arg_ptr(kr)
                .arg_ptr(self.scratch().buf_scores)
                .arg_u32(seq)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;
            // 2026-09-25: (3) Row softmax, with `rsqrt(head_dim)` applied inside.
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_scores)
                .arg_ptr(self.scratch().buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;
            // 2026-09-25: (4) `O_stage[seq,D] = P[seq,seq] · Vt[D,seq]^T`.
            KernelLaunch::new(gpu, self.k_gemm)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_probs)
                .arg_ptr(vt)
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(seq)
                .launch(stream)?;
            // 2026-09-25: (5) Scatter into this head's columns of the `[seq, H*D]` output.
            KernelLaunch::new(gpu, self.k_scatter_head)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_ptr(o_h)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: One block over `p_total` packed rows: the GEMMs and norms run
    /// once over all rows, the attention once per image.
    pub(super) fn block(
        &self,
        blk: &GlmVitBlock,
        p_total: usize,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let pt = p_total as u32;
        let qkv_n = 3 * h;
        let inter = self.intermediate_size as u32;
        let n_h = (p_total * self.hidden_size) as u32;
        let s = self.scratch();

        self.copy(gpu, s.buf_h1, s.buf_h2, n_h, stream)?;
        self.rmsnorm(gpu, s.buf_h1, blk.norm1_w, pt, h, stream)?;
        self.gemm(
            gpu,
            s.buf_h1,
            blk.qkv_w,
            Some(blk.qkv_b),
            s.buf_gate,
            pt,
            qkv_n,
            h,
            stream,
        )?;
        for (i, &p) in p_i.iter().enumerate() {
            let qkv = s.buf_gate.offset(p_off[i] * qkv_n as usize * 2);
            let o = s.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let cos = s.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin = s.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            self.attention(gpu, blk, qkv, o, cos, sin, p as u32, stream)?;
        }
        self.gemm(
            gpu,
            s.buf_h1,
            blk.proj_w,
            Some(blk.proj_b),
            s.buf_gate,
            pt,
            h,
            h,
            stream,
        )?;
        self.add_inplace(gpu, s.buf_gate, s.buf_h2, n_h, stream)?;
        self.copy(gpu, s.buf_gate, s.buf_h1, n_h, stream)?;

        self.copy(gpu, s.buf_h1, s.buf_h2, n_h, stream)?;
        self.rmsnorm(gpu, s.buf_h1, blk.norm2_w, pt, h, stream)?;
        self.gemm(
            gpu,
            s.buf_h1,
            blk.gate_w,
            Some(blk.gate_b),
            s.buf_gate,
            pt,
            inter,
            h,
            stream,
        )?;
        self.gemm(
            gpu,
            s.buf_h1,
            blk.up_w,
            Some(blk.up_b),
            s.buf_up,
            pt,
            inter,
            h,
            stream,
        )?;
        self.swiglu(gpu, s.buf_gate, s.buf_up, s.buf_gate, pt * inter, stream)?;
        self.gemm(
            gpu,
            s.buf_gate,
            blk.down_w,
            Some(blk.down_b),
            s.buf_h1,
            pt,
            h,
            inter,
            stream,
        )?;
        self.add_inplace(gpu, s.buf_h1, s.buf_h2, n_h, stream)
    }
}
