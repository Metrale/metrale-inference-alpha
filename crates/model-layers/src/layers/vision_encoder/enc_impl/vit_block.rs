// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One ViT block: norm → QKV → RoPE attention → proj → +residual →
//! norm → fc1 → GELU → fc2 → +residual.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{ViTBlock, VisionEncoder};

mod attn_gemm;

impl VisionEncoder {
    /// 2026-09-25: `C[m,n] = A[m,k] @ B[n,k]^T + bias[n]` (BF16): `dense_gemm_bf16_pipelined`
    /// plus `vision_add_bias` when both handles are set, else `vision_gemm_bias`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vit_gemm_bias(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        bias: DevicePtr,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.k_gemm_pipelined.0 != 0 && self.k_add_bias.0 != 0 {
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.k_add_bias)
                .grid([div_ceil(m * n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(c)
                .arg_ptr(bias)
                .arg_u32(m)
                .arg_u32(n)
                .launch(stream)
        } else {
            KernelLaunch::new(gpu, self.k_gemm)
                .grid([div_ceil(n, 32), div_ceil(m, 32), 1])
                .block([32, 32, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(bias)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        }
    }

    /// 2026-09-25: Run one ViT block in place on buf_h1; buf_h2 and buf_wide are scratch.
    pub(super) fn vit_block(
        &self,
        blk: &ViTBlock,
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let p32 = p as u32;
        let qkv_n = (3 * self.num_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let n_h = p * self.hidden_size;
        // 2026-09-25: `vision_attention_rope` shared memory: scores[p] + q_rope[head_dim] f32.
        let sm_bytes = (p + self.head_dim) * std::mem::size_of::<f32>();

        // 2026-09-25: --- Attention sub-block ---
        // 1. Save the residual in buf_h2.
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p32, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(blk.norm1_w)
            .arg_ptr(blk.norm1_b)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_h1,
            blk.qkv_w,
            blk.qkv_b,
            self.scratch().buf_wide,
            p32,
            qkv_n,
            h,
            stream,
        )?;
        // 2026-09-25: 4. Attention: the GEMM-based path, or `vision_attention_rope`
        // when METRALE_VISION_ATTN_LEGACY is set (any value) or `k_rope_deint` is
        // null (trees built from the qwen3-vl-30b-a3b source lack the GEMM-based
        // kernels).
        if std::env::var("METRALE_VISION_ATTN_LEGACY").is_ok() || self.k_rope_deint.0 == 0 {
            KernelLaunch::new(gpu, self.k_attn)
                .grid([p32, self.num_heads as u32, 1])
                .block([32, 1, 1])
                .shared_mem(sm_bytes as u32)
                .arg_ptr(self.scratch().buf_wide)
                .arg_ptr(self.scratch().buf_h1)
                .arg_ptr(self.scratch().buf_rope_cos)
                .arg_ptr(self.scratch().buf_rope_sin)
                .arg_u32(p32)
                .arg_u32(self.num_heads as u32)
                .arg_u32(self.head_dim as u32)
                .launch(stream)?;
        } else {
            self.vit_attention_gemm(
                gpu,
                self.scratch().buf_wide,
                self.scratch().buf_h1,
                self.scratch().buf_rope_cos,
                self.scratch().buf_rope_sin,
                p32,
                stream,
            )?;
        }
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_h1,
            blk.proj_w,
            blk.proj_b,
            self.scratch().buf_wide,
            p32,
            h,
            h,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_wide)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_wide)
            .arg_ptr(self.scratch().buf_h1)
            .arg_u32(n_h as u32)
            .launch(stream)?;

        // 2026-09-25: --- FFN sub-block ---
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p32, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(blk.norm2_w)
            .arg_ptr(blk.norm2_b)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_h1,
            blk.fc1_w,
            blk.fc1_b,
            self.scratch().buf_wide,
            p32,
            inter,
            h,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(p32 * inter, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(p32 * inter)
            .launch(stream)?;
        // 2026-09-25: fc2 writes buf_h1, whose normed rows fc1 has already consumed.
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            blk.fc2_w,
            blk.fc2_b,
            self.scratch().buf_h1,
            p32,
            h,
            inter,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }

    /// 2026-09-25: Batched ViT block over N images packed at row `p_off[i]`,
    /// `p_total` rows in all. The kernel sequence is `vit_block`'s, except that
    /// the element and GEMM counts use `p_total` and the attention loops per
    /// image over its rows `p_off[i] .. p_off[i] + p_i[i]`, so attention never
    /// crosses an image boundary. METRALE_VISION_NOATTN skips that loop.
    pub(super) fn vit_block_batched(
        &self,
        blk: &ViTBlock,
        p_total: usize,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let pt = p_total as u32;
        let qkv_n = (3 * self.num_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let n_h = p_total * self.hidden_size;

        // 2026-09-25: --- Attention sub-block ---
        // 1. Save the residual in buf_h2.
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_norm)
            .grid([pt, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(blk.norm1_w)
            .arg_ptr(blk.norm1_b)
            .arg_u32(pt)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_h1,
            blk.qkv_w,
            blk.qkv_b,
            self.scratch().buf_wide,
            pt,
            qkv_n,
            h,
            stream,
        )?;
        // 2026-09-25: 4. Attention per image. buf_wide (QKV) is only read here, and
        // each image writes its own buf_h1 rows. METRALE_VISION_NOATTN (any
        // value) skips the loop, which leaves the output wrong; it is a
        // timing diagnostic.
        let skip_attn = std::env::var("METRALE_VISION_NOATTN").is_ok();
        // 2026-09-25: `vision_attention_rope` when METRALE_VISION_ATTN_LEGACY is set
        // or `k_rope_deint` is null.
        let legacy_attn =
            std::env::var("METRALE_VISION_ATTN_LEGACY").is_ok() || self.k_rope_deint.0 == 0;
        for (i, &p) in p_i.iter().enumerate() {
            if skip_attn {
                break;
            }
            let p32 = p as u32;
            let qkv = self
                .scratch()
                .buf_wide
                .offset(p_off[i] * qkv_n as usize * 2);
            let o = self
                .scratch()
                .buf_h1
                .offset(p_off[i] * self.hidden_size * 2);
            let cos = self
                .scratch()
                .buf_rope_cos
                .offset(p_off[i] * self.head_dim * 2);
            let sin = self
                .scratch()
                .buf_rope_sin
                .offset(p_off[i] * self.head_dim * 2);
            if legacy_attn {
                let sm_bytes = ((p + self.head_dim) * std::mem::size_of::<f32>()) as u32;
                KernelLaunch::new(gpu, self.k_attn)
                    .grid([p32, self.num_heads as u32, 1])
                    .block([32, 1, 1])
                    .shared_mem(sm_bytes)
                    .arg_ptr(qkv)
                    .arg_ptr(o)
                    .arg_ptr(cos)
                    .arg_ptr(sin)
                    .arg_u32(p32)
                    .arg_u32(self.num_heads as u32)
                    .arg_u32(self.head_dim as u32)
                    .launch(stream)?;
            } else {
                self.vit_attention_gemm(gpu, qkv, o, cos, sin, p32, stream)?;
            }
        }
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_h1,
            blk.proj_w,
            blk.proj_b,
            self.scratch().buf_wide,
            pt,
            h,
            h,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_wide)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_wide)
            .arg_ptr(self.scratch().buf_h1)
            .arg_u32(n_h as u32)
            .launch(stream)?;

        // 2026-09-25: --- FFN sub-block ---
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_norm)
            .grid([pt, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(blk.norm2_w)
            .arg_ptr(blk.norm2_b)
            .arg_u32(pt)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_h1,
            blk.fc1_w,
            blk.fc1_b,
            self.scratch().buf_wide,
            pt,
            inter,
            h,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(pt * inter, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(pt * inter)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            blk.fc2_w,
            blk.fc2_b,
            self.scratch().buf_h1,
            pt,
            h,
            inter,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }
}
