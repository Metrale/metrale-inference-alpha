// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode FFN: residual add, post-attention norm and
//! the MoE or dense FFN for all n rows, then the residual add of its output.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - A layer with a shortcut MoE carry and no MLA fails before any launch
//!   (the `ensure!` at the top of `ms_phase_ffn`).
//!
//! `ms_phase_ffn` takes the first arm whose condition holds: n = 3
//! (`forward_k3`), n = 2 (`forward_k2`), the small-batch dense FFN
//! (`try_forward_km`), the grouped FP8 MoE (`forward_fp8_grouped_decode`),
//! `forward_prefill` for a dense FFN or a wide enough MoE, the opt-in grouped
//! routed MoE, the pairwise `forward_k2` walk, and otherwise one row at a time.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// 2026-09-25: The pairwise MoE decode arm is on unless
/// `METRALE_MOE_PAIRWISE_DECODE=0`, read once per process.
fn pairwise_moe_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_PAIRWISE_DECODE").as_deref() != Ok("0"))
}

/// 2026-09-25: The opt-in grouped routed MoE arm (`forward_prefill`),
/// `METRALE_MOE_GROUPED_ROUTED_DECODE=1`, read once per process. It applies from
/// `grouped_routed_decode_min()` rows: `METRALE_MOE_GROUPED_ROUTED_DECODE_MIN`,
/// or 2 when unset or unparsable.
fn grouped_routed_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_GROUPED_ROUTED_DECODE").as_deref() == Ok("1"))
}
fn grouped_routed_decode_min() -> usize {
    use std::sync::OnceLock;
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("METRALE_MOE_GROUPED_ROUTED_DECODE_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_ffn(&self, c: &MultiSeqCtx<'_>, o_out: DevicePtr) -> Result<()> {
        // 2026-09-25: Only the one-row-at-a-time loop at the end runs the shortcut
        // MoE (`shortcut_carry_out` / `shortcut_carry_in`, set by the LongCat
        // loader); the other arms would skip it without an error. That loop is
        // forced when `mla` is set, so a shortcut layer without MLA is refused.
        anyhow::ensure!(
            (self.shortcut_carry_out.is_none() && self.shortcut_carry_in.is_none())
                || self.mla.is_some(),
            "shortcut-MoE model reached the batched FFN ladder, which does not              implement the shortcut; only the per-token branch does"
        );
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            eps,
            bf16,
            hidden,
            residual,
            ..
        } = *c;

        if self.ffn.is_none() {
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                o_out,
                (n * h) as u32,
                stream,
            )?;
            return Ok(());
        }
        // 2026-09-25: MLA layers skip every batched arm and take the final branch.
        let force_seq_ffn = self.mla.is_some();
        // 2026-09-25: With the grouped routed arm on and eligible, the n = 2 and
        // n = 3 arms are skipped. The km, FP8 grouped and `forward_prefill` arms
        // are still checked before it.
        let use_grouped = !force_seq_ffn
            && n >= grouped_routed_decode_min()
            && grouped_routed_decode_enabled()
            && self.ffn.moe_grouped_decode_ok();
        if !use_grouped && n == 3 && !force_seq_ffn {
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                3,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_k3(normed2, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (3 * h) as u32,
                stream,
            )?;
        } else if !use_grouped && n == 2 && !force_seq_ffn {
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                2,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_k2(normed2, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else if (4..=ops::w4a4_proj::ffn_proj_max_rows() as usize).contains(&n)
            && !force_seq_ffn
            && self.ffn.can_forward_km(n as u32)
        {
            // 2026-09-25: A dense FFN for which `can_forward_km(n)` holds:
            // `forward_km` runs a batched GEMV per projection, or
            // `forward_prefill` for BF16 or native FP8 weights
            // (`native_small_batch_uses_prefill`). `try_forward_km` checks the
            // same predicate, so it returns `true` here.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            let used = self.ffn.try_forward_km(normed2, n as u32, fwd, stream)?;
            debug_assert!(used, "can_forward_km checked at branch entry");
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else if !force_seq_ffn && self.ffn.fp8_grouped_decode_ok(n, fwd) {
            // 2026-09-25: FP8 MoE grouped by expert across all n rows, when
            // `fp8_grouped_decode_ok` holds. Checked before the `forward_prefill`
            // and pairwise arms.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn
                .forward_fp8_grouped_decode(normed2, n, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else if !force_seq_ffn
            && (self.ffn.is_dense() || crate::layers::moe_grouped_decode_for(n))
        {
            // 2026-09-25: `forward_prefill` over all n rows, reading each weight
            // once instead of once per row: always for a dense FFN, and for MoE
            // when `moe_grouped_decode_for(n)` holds (n >= 16 unless
            // `METRALE_NO_MOE_GROUPED_DECODE` is set; `METRALE_MOE_GROUPED_DECODE=1`
            // forces it below 16).
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_prefill(normed2, n, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else if use_grouped {
            // 2026-09-25: The opt-in grouped routed arm: `forward_prefill` over all
            // n rows, as in the arm above. Reached only for a MoE whose
            // `grouped_decode_ok` holds (no BF16 or FP8 expert gate pointers, no
            // token-id routing table).
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_prefill(normed2, n, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else if !force_seq_ffn && n > 2 && n % 2 == 0 && pairwise_moe_decode_enabled() {
            // 2026-09-25: Even n above 2: walk the batch two rows at a time with
            // `forward_k2`, adding each pair's `moe_output` into the residual
            // before the next pair overwrites it. `forward_k2` falls back to
            // `forward_batched` for layouts without a fused 2-row kernel.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            for pair in 0..(n / 2) {
                let off = pair * 2 * h;
                self.ffn
                    .forward_k2(normed2.offset(off * bf16), fwd, stream)?;
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden.offset(off * 2),
                    fwd.buffers.moe_output(),
                    (2 * h) as u32,
                    stream,
                )?;
            }
        } else {
            // 2026-09-25: Every other case, MLA included: the residual add and
            // norm run one row at a time, with BF16 hidden and residual rows.
            let residual_elem = 2usize;
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let o_out_i = o_out.offset(i * h * bf16);
                let residual_i = residual.offset(i * h * residual_elem);
                let normed2_i = fwd.buffers.norm_output().offset(i * h * bf16);
                ops::residual_add_rms_norm(
                    fwd.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_i,
                    o_out_i,
                    &self.post_attn_norm,
                    normed2_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            // 2026-09-25: Each `forward()` returns the one-row `moe_output`, which
            // is added into the residual before the next row overwrites it.
            let normed_base = fwd.buffers.norm_output();
            if std::env::var("METRALE_MOE_BATCHED_DECODE").ok().as_deref() == Some("1") {
                // 2026-09-25: `METRALE_MOE_BATCHED_DECODE=1`: all n rows in one
                // `forward_batched` call, whose per-row map folds a MoE LoRA
                // adapter; the per-row `forward` refuses a routed adapter when the
                // step has more than one sequence (`moe/forward.rs`).
                self.ffn.forward_batched(normed_base, n, fwd, stream)?;
                let moe_out = fwd.buffers.moe_output();
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden,
                    moe_out,
                    (n * h) as u32,
                    stream,
                )?;
            } else {
                for i in 0..n {
                    let hidden_i = hidden.offset(i * h * residual_elem);
                    let normed2_i = normed_base.offset(i * h * bf16);
                    // 2026-09-25: LongCat shortcut MoE, producer side, as in
                    // `decode_inner`: this row's shortcut output is copied into
                    // the carry before `self.ffn` runs, because both write
                    // `moe_output`.
                    if let (Some(moe_ffn), Some((carry, cap))) =
                        (&self.moe_ffn, self.shortcut_carry_out)
                    {
                        anyhow::ensure!(
                            n <= cap,
                            "shortcut carry capacity {cap} < decode batch {n}"
                        );
                        let sc_out = moe_ffn.forward(normed2_i, fwd, stream)?;
                        if let crate::layers::FfnComponent::Moe(m) = moe_ffn {
                            m.apply_zero_expert(sc_out, normed2_i, 1, fwd, stream)?;
                        }
                        fwd.gpu.copy_d2d_async(
                            sc_out,
                            carry.offset(i * h * bf16),
                            h * bf16,
                            stream,
                        )?;
                    }
                    let moe_out = self.ffn.forward(normed2_i, fwd, stream)?;
                    ops::residual_add(
                        fwd.gpu,
                        self.residual_add_k,
                        hidden_i,
                        moe_out,
                        h as u32,
                        stream,
                    )?;
                    // 2026-09-25: LongCat shortcut carry, consumer side: add this
                    // row of the paired sublayer's stashed shortcut output.
                    if let Some((carry, _cap)) = self.shortcut_carry_in {
                        ops::residual_add(
                            fwd.gpu,
                            self.residual_add_k,
                            hidden_i,
                            carry.offset(i * h * bf16),
                            h as u32,
                            stream,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}
