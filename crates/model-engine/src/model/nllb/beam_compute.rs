// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched beam-search GPU primitives: the decode step over all rows (one
//! tensor-core GEMM per projection) and the batched attention, scatter, gather and
//! top-k launches behind it. Uses `linear` (with its LoRA delta), `layer_norm_to`,
//! `layer_norm`, `add`, `relu`, `scale`, `gemm` and `embed_rows` from
//! [`super::compute`].
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NllbGpuModel;
use super::beam::{CrossBatch, DecBuf};
use super::util::u32_bytes;

thread_local! {
    /// 2026-09-25: Accumulated lm_head (tied-embedding projection) device time in
    /// ns, measured in `beam_forward_step` when `METRALE_NLLB_BEAM_PROFILE=1`.
    pub(super) static LMHEAD_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl NllbGpuModel {
    /// 2026-09-25: Write `src[B,d]` into batch-major cache `[B,stride,d]` at row `pos`.
    fn scatter(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        pos: usize,
        b: usize,
        stride: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.scatter)
            .grid([div_ceil((b * self.d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(pos as u32)
            .arg_u32(b as u32)
            .arg_u32(stride as u32)
            .arg_u32(self.d as u32)
            .launch(self.stream())
    }

    /// 2026-09-25: Batched attention over `b` rows; `tk` holds each row's key
    /// length. Row `i` reads K/V cache slab `i / group`: self-attention passes
    /// `group = 1` (one cache per row), cross-attention passes the beams per
    /// request, so a request's beams share its cross slab.
    #[allow(clippy::too_many_arguments)]
    fn attn_batched(
        &self,
        q: DevicePtr,
        kc: DevicePtr,
        vc: DevicePtr,
        out: DevicePtr,
        b: usize,
        stride: usize,
        group: usize,
        tk: DevicePtr,
        sh: u32,
    ) -> Result<()> {
        debug_assert!(group >= 1);
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.attn_bdecode)
            .grid([(b * self.heads) as u32, 1, 1])
            .block([self.head_dim as u32, 1, 1])
            .shared_mem(sh)
            .arg_ptr(q)
            .arg_ptr(kc)
            .arg_ptr(vc)
            .arg_ptr(out)
            .arg_u32(b as u32)
            .arg_u32(stride as u32)
            .arg_u32(group as u32)
            .arg_ptr(tk)
            .arg_u32(self.heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_f32(self.attn_scale)
            .launch(self.stream())
    }

    /// 2026-09-25: Reorder beam caches: `dst[i] = src[perm[i]]` over rows `0..used`.
    pub(super) fn gather(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        perm: DevicePtr,
        b: usize,
        used: usize,
        stride: usize,
    ) -> Result<()> {
        let n = (b * used * self.d) as u32;
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.gather)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_ptr(perm)
            .arg_u32(b as u32)
            .arg_u32(used as u32)
            .arg_u32(stride as u32)
            .arg_u32(self.d as u32)
            .launch(self.stream())
    }

    /// 2026-09-25: One batched decode step over `b` rows (Σ beams of the fused
    /// requests), `cur` holding one token per row: fill self-KV row `pos` for
    /// every row and write `logits[b, vocab]` (bf16). Each projection and the
    /// lm_head is one M=b GEMM; cross-attention reads `xb` with a per-request
    /// group divisor.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn beam_forward_step(
        &self,
        cur: &[u32],
        pos: usize,
        b: usize,
        sk: &[DevicePtr],
        sv: &[DevicePtr],
        xb: &CrossBatch,
        buf: &DecBuf,
    ) -> Result<()> {
        let (d, ffn) = (self.d, self.ffn);
        let sh = ((self.head_dim + buf.cache_rows) * 4) as u32;
        self.gpu.copy_h2d(u32_bytes(cur), buf.id)?;
        self.embed_rows(buf.id, buf.dh, b)?;
        self.scale(buf.dh, b * d)?;
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.add_row)
            .grid([div_ceil((b * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buf.dh)
            .arg_ptr(buf.pos_table.offset(pos * d * 2))
            .arg_u32((b * d) as u32)
            .arg_u32(d as u32)
            .launch(self.stream())?;
        self.gpu
            .copy_h2d(u32_bytes(&vec![(pos + 1) as u32; b]), buf.selftk)?;

        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            // 2026-09-25: `dh` carries the running residual in place across all
            // three blocks: nothing writes it between a block's start and its
            // post-block add, so no separate residual copy is needed. `normed`
            // is `dh` normalized out of place.
            self.layer_norm_to(&format!("{p}.self_attn_layer_norm"), buf.dh, buf.normed, b)?;
            self.linear(&format!("{p}.self_attn.q_proj"), buf.normed, buf.q, b, d, d)?;
            self.linear(
                &format!("{p}.self_attn.k_proj"),
                buf.normed,
                buf.knew,
                b,
                d,
                d,
            )?;
            self.linear(
                &format!("{p}.self_attn.v_proj"),
                buf.normed,
                buf.vnew,
                b,
                d,
                d,
            )?;
            self.scatter(buf.knew, sk[l], pos, b, buf.cache_rows)?;
            self.scatter(buf.vnew, sv[l], pos, b, buf.cache_rows)?;
            self.attn_batched(
                buf.q,
                sk[l],
                sv[l],
                buf.attn,
                b,
                buf.cache_rows,
                1,
                buf.selftk,
                sh,
            )?;
            self.linear(
                &format!("{p}.self_attn.out_proj"),
                buf.attn,
                buf.proj,
                b,
                d,
                d,
            )?;
            self.add(buf.dh, buf.proj, b * d)?;
            self.layer_norm_to(
                &format!("{p}.encoder_attn_layer_norm"),
                buf.dh,
                buf.normed,
                b,
            )?;
            self.linear(
                &format!("{p}.encoder_attn.q_proj"),
                buf.normed,
                buf.q,
                b,
                d,
                d,
            )?;
            // 2026-09-25: Grouped cross-attention: one launch over all rows. Row
            // `i` reads slab `i / group` (its request), bounded to its own
            // enc_len by the per-row `crosstk`.
            let sh_cross = ((self.head_dim + xb.max_enc) * 4) as u32;
            self.attn_batched(
                buf.q,
                xb.kpad[l],
                xb.vpad[l],
                buf.attn,
                b,
                xb.max_enc,
                xb.group,
                buf.crosstk,
                sh_cross,
            )?;
            self.linear(
                &format!("{p}.encoder_attn.out_proj"),
                buf.attn,
                buf.proj,
                b,
                d,
                d,
            )?;
            self.add(buf.dh, buf.proj, b * d)?;
            self.layer_norm_to(&format!("{p}.final_layer_norm"), buf.dh, buf.normed, b)?;
            self.linear(&format!("{p}.fc1"), buf.normed, buf.ff, b, ffn, d)?;
            self.relu(buf.ff, b * ffn)?;
            self.linear(&format!("{p}.fc2"), buf.ff, buf.proj, b, d, ffn)?;
            self.add(buf.dh, buf.proj, b * d)?;
        }
        self.layer_norm("model.decoder.layer_norm", buf.dh, b)?;
        let t_lm = std::env::var("METRALE_NLLB_BEAM_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
            .then(std::time::Instant::now);
        self.gemm(buf.dh, self.embed_table, buf.logits, b, self.vocab, d)?; // 2026-09-25: tied lm_head
        if let Some(t) = t_lm {
            self.gpu.synchronize(self.stream())?;
            LMHEAD_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
        }
        Ok(())
    }

    /// 2026-09-25: On-device candidate reduction: for each of `rows` logit rows,
    /// its log-sum-exp over the full vocab and its top-`k` `(value, token)` pairs,
    /// descending: the inputs the host path computes from a full-vocab D2H. The
    /// D2H is `rows·(4 + 8k)` bytes instead of `rows·vocab·2`. `k` must be ≤
    /// `NLLB_TOPK_KMAX`.
    pub(super) fn beam_cands_device(
        &self,
        buf: &DecBuf,
        rows: usize,
        k: usize,
    ) -> Result<Vec<(f32, Vec<(f32, u32)>)>> {
        let sh = (128 * k * 8) as u32; // 2026-09-25: 128 threads · k · (f32 val + u32 id)
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.beam_topk)
            .grid([rows as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(sh)
            .arg_ptr(buf.logits)
            .arg_ptr(buf.topk_lse)
            .arg_ptr(buf.topk_val)
            .arg_ptr(buf.topk_idx)
            .arg_u32(rows as u32)
            .arg_u32(self.vocab as u32)
            .arg_u32(k as u32)
            .launch(self.stream())?;
        self.gpu.synchronize(self.stream())?;
        // 2026-09-25: The kernel packs outputs as [rows, k] (stride k), so the
        // first rows*k elements are contiguous.
        let (mut lse, mut val, mut idx) = (
            vec![0u8; rows * 4],
            vec![0u8; rows * k * 4],
            vec![0u8; rows * k * 4],
        );
        self.gpu.copy_d2h(buf.topk_lse, &mut lse)?;
        self.gpu.copy_d2h(buf.topk_val, &mut val)?;
        self.gpu.copy_d2h(buf.topk_idx, &mut idx)?;
        let f32_at =
            |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let u32_at =
            |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..rows)
            .map(|r| {
                let top = (0..k)
                    .map(|j| (f32_at(&val, r * k + j), u32_at(&idx, r * k + j)))
                    .collect();
                (f32_at(&lse, r), top)
            })
            .collect())
    }

    /// 2026-09-25: Copy the device `[B,vocab]` bf16 logits to host as per-row `f32` vectors.
    pub(super) fn beam_logits_host(&self, buf: &DecBuf, b: usize) -> Result<Vec<Vec<f32>>> {
        self.gpu.synchronize(self.stream())?;
        let mut raw = vec![0u8; b * self.vocab * 2];
        self.gpu.copy_d2h(buf.logits, &mut raw)?;
        Ok((0..b)
            .map(|bi| {
                raw[bi * self.vocab * 2..(bi + 1) * self.vocab * 2]
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            })
            .collect())
    }
}
