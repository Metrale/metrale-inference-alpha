// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The GEMM-based ViT attention (`vit_attention_gemm`), which `vit_block` and
//! `vit_block_batched` run unless the warp-per-query `vision_attention_rope` is selected.
//!
//! Owner: model-layers (vision).
//! Invariants: every launch is on the caller's `stream`, so each head finishes with the shared
//! `buf_scores`, `buf_probs` and `buf_o_stage` before the next head writes them.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::super::VisionEncoder;

impl VisionEncoder {
    /// 2026-09-25: GEMM-based attention for one image's `[seq, 3*H*D]` QKV slice →
    /// `O[seq, H*D]`, the alternative to the warp-per-query `vision_attention_rope`.
    /// Once per call: rope + deinterleave + V-transpose (all heads). Then per
    /// head: GEMM1 raw QKᵀ (f32 out) → row softmax (1/sqrt(D) applied there) →
    /// GEMM2 P·V → scatter into the interleaved O head slot. `qkv`, `o`, `cos`
    /// and `sin` are per-image base pointers: `qkv` `[seq, 3*H*D]`, `o`
    /// `[seq, H*D]`, `cos`/`sin` `[seq, D]`. All launches share `stream`, so
    /// each head finishes before the next reuses `buf_scores`, `buf_probs` and
    /// `buf_o_stage`; splitting heads across streams would need per-head
    /// buffers. `seq` must be at most `p_max`, the side of `buf_scores`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::vision_encoder::enc_impl) fn vit_attention_gemm(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr,
        o: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(
            seq <= 1024,
            "ViT SDPA seq {seq} exceeds buf_scores cap (1024)"
        );
        let h_n = self.num_heads as u32;
        let d = self.head_dim as u32;
        let hd = self.hidden_size as u32;

        // 2026-09-25: (1) rope + deinterleave + V-transpose → buf_qr/buf_kr/buf_vt (all heads).
        KernelLaunch::new(gpu, self.k_rope_deint)
            .grid([div_ceil(seq * d, 256), h_n, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(self.scratch().buf_qr)
            .arg_ptr(self.scratch().buf_kr)
            .arg_ptr(self.scratch().buf_vt)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .launch(stream)?;

        // 2026-09-25: Per-head strides in elements: Qr/Kr `seq*D`, Vt `D*seq`.
        let qk_head = (seq * d) as usize;
        let v_head = (d * seq) as usize;
        for head in 0..self.num_heads {
            let qr_h = self.scratch().buf_qr.offset(head * qk_head * 2);
            let kr_h = self.scratch().buf_kr.offset(head * qk_head * 2);
            let vt_h = self.scratch().buf_vt.offset(head * v_head * 2);
            let o_h = o.offset(head * self.head_dim * 2);

            // 2026-09-25: (2) GEMM1: S[seq,seq] = Qr_h[seq,D] @ Kr_h[seq,D]ᵀ, f32 out.
            // `dense_gemm_bf16_f32out` uses 16x16 tiles: block (16,16),
            // grid (ceil(N/16), ceil(M/16)).
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr_h)
                .arg_ptr(kr_h)
                .arg_ptr(self.scratch().buf_scores)
                .arg_u32(seq)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            // 2026-09-25: (3) Row softmax with the 1/sqrt(D) scale → buf_probs[seq,seq] bf16.
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_scores)
                .arg_ptr(self.scratch().buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            // 2026-09-25: (4) GEMM2: O_stage[seq,D] = P[seq,seq] @ Vt_h[D,seq]ᵀ = P·V,
            // with the pipelined kernel's grid (ceil(N/128), ceil(M/128)).
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_probs)
                .arg_ptr(vt_h)
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(seq)
                .launch(stream)?;

            // 2026-09-25: (5) Scatter O_stage[seq,D] into the head's slot of o,
            // whose row stride is H*D.
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
}
