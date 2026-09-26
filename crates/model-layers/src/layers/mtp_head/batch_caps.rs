// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched-propose width policy. `propose_batch_max` derives how many
//! sequences one drafter forward can carry from the weight layout, the resolved
//! kernels (the batched LM head in particular) and the byte sizes of the arena
//! buffers the forward writes; this file also sizes the per-sequence stride of
//! the dedicated `propose_meta` allocation.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `propose_batch_max` returns a value in `1..=PROPOSE_META_SEQS`.
//! - Every `propose_meta` stride is a multiple of 8 within
//!   `[PROPOSE_META_STRIDE_FLOOR, PROPOSE_META_STRIDE_CAP]`, env override
//!   included.

use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::KernelHandle;

use metrale_config::ModelConfig;

use super::{MtpHead, MtpQuantization, ProjectionWeight};

/// 2026-09-25: Floor for the per-sequence stride of `propose_meta`. Sequence `i`'s slab
/// at `propose_meta + i * propose_meta_stride` has the layout `forward_one`
/// uses: position u32 at 0, slot i64 at 8, seq_len i32 at 16, block table
/// from [`PROPOSE_META_HEADER`].
pub(crate) const PROPOSE_META_STRIDE_FLOOR: usize = 2048;

/// 2026-09-25: Bytes of the fixed header ahead of the block table in each meta slab.
pub(crate) const PROPOSE_META_HEADER: usize = 256;

/// 2026-09-25: Sequences the `propose_meta` allocation holds, the same count as
/// `VERIFY_WY_TABLE_SEQS`. The allocation is this many strides.
pub(crate) const PROPOSE_META_SEQS: usize = 32;

/// 2026-09-25: Stride for a `max_seq_len`-token drafter sequence: the header plus one
/// i32 block-table entry per KV block, 8-byte aligned and clamped to
/// [`PROPOSE_META_STRIDE_FLOOR`]..=[`PROPOSE_META_STRIDE_CAP`]. The `+ 1` matches
/// the batched forward's `blocks_needed = seq_len / bs + 1`, which needs one more
/// entry at `seq_len == max_seq_len`.
pub(crate) fn propose_meta_stride_bytes(max_seq_len: usize, kv_block_size: usize) -> usize {
    let entries = (max_seq_len / kv_block_size.max(1)).saturating_add(1);
    let raw = PROPOSE_META_HEADER.saturating_add(entries.saturating_mul(4));
    let aligned = raw.saturating_add(7) & !7;
    aligned.clamp(PROPOSE_META_STRIDE_FLOOR, PROPOSE_META_STRIDE_CAP)
}

/// 2026-09-25: Stride ceiling (16 MiB). It keeps `PROPOSE_META_SEQS * stride` from
/// overflowing `usize`, which in release would wrap to a small allocation.
pub(crate) const PROPOSE_META_STRIDE_CAP: usize = 1 << 24;

/// 2026-09-25: Stride for a new head: [`propose_meta_stride_bytes`], unless
/// `METRALE_PROPOSE_META_STRIDE=<bytes>` parses as a `usize`. An override is
/// aligned up to 8 and clamped to the floor and the cap; an empty or unparsable
/// value is ignored.
pub(crate) fn propose_meta_stride_env(max_seq_len: usize, kv_block_size: usize) -> usize {
    let value = std::env::var("METRALE_PROPOSE_META_STRIDE").ok();
    if let Some(stride) = parse_stride_override(value.as_deref()) {
        return stride;
    }
    propose_meta_stride_bytes(max_seq_len, kv_block_size)
}

fn parse_stride_override(value: Option<&str>) -> Option<usize> {
    value?
        .trim()
        .parse::<usize>()
        .ok()
        .map(clamp_stride_override)
}

/// 2026-09-25: Align up to 8, then clamp to the floor and the cap. Saturating, so no
/// input, `usize::MAX` included, panics or wraps.
pub(crate) fn clamp_stride_override(bytes: usize) -> usize {
    (bytes.saturating_add(7) & !7).clamp(PROPOSE_META_STRIDE_FLOOR, PROPOSE_META_STRIDE_CAP)
}

/// 2026-09-25: A projection the batched propose's row dispatch (`proj_rows`) can run:
/// BF16 or NVFP4.
pub(super) fn is_row_proj(p: &ProjectionWeight) -> bool {
    matches!(p, ProjectionWeight::Bf16(_) | ProjectionWeight::Nvfp4(_))
}

/// 2026-09-25: The attention-projection half of the batched-propose scope: a BF16
/// forward with BF16 KV, BF16 fc/k/v, and q/o each accepted by [`is_row_proj`].
/// `proj` is `[fc, q, k, v, o]`. `propose_ffn_arm` decides the FFN separately.
fn batch_weight_layout_ok(
    quant: MtpQuantization,
    kv_bf16: bool,
    proj: [&ProjectionWeight; 5],
) -> bool {
    let bf16_proj = |p: &ProjectionWeight| matches!(p, ProjectionWeight::Bf16(_));
    let [fc, q, k, v, o] = proj;
    matches!(quant, MtpQuantization::Bf16)
        && kv_bf16
        && bf16_proj(fc)
        && is_row_proj(q)
        && bf16_proj(k)
        && bf16_proj(v)
        && is_row_proj(o)
}

impl MtpHead {
    /// 2026-09-25: Batched LM-head kernel for `n` rows: `w4a16_batchm.kernel(n)`
    /// when non-zero, else the first resolved of `w4a16_gemv_batch16` and
    /// `w4a16_gemv_batch32` that covers `n`, else a 0 handle.
    pub(crate) fn lm_head_batch_kernel(&self, n: usize) -> KernelHandle {
        let narrow = self.w4a16_batchm.kernel(n as u32);
        if narrow.0 != 0 {
            return narrow;
        }
        for (max_m, k) in [
            (16usize, self.w4a16_gemv_batch16_k),
            (32, self.w4a16_gemv_batch32_k),
        ] {
            if n <= max_m && k.0 != 0 {
                return k;
            }
        }
        KernelHandle(0)
    }

    /// 2026-09-25: Whether the weight layout, an FFN arm (`propose_ffn_arm`) and the
    /// width-independent kernels and buffers of the batched propose are all
    /// present. [`Self::propose_batch_max`] decides the width.
    fn propose_batch_scope_ok(&self) -> bool {
        batch_weight_layout_ok(
            self.quant,
            self.kv_bf16,
            [
                &self.fc,
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                &self.o_proj,
            ],
        ) && self.propose_ffn_arm().is_some()
            && self.dense_gemm_pipelined_k.0 != 0
            && self.dense_gemv_k.is_some()
            && self.deinterleave_qg_k.is_some()
            && !self.propose_meta.is_null()
    }

    /// 2026-09-25: The widest batched propose this head can run; `1` means
    /// per-sequence only.
    pub(crate) fn propose_batch_max(&self, buffers: &BufferArena, config: &ModelConfig) -> usize {
        if !self.propose_batch_scope_ok() {
            return 1;
        }
        let h = config.hidden_size;
        let bf16 = 2usize;
        let sizes = buffers.sizes();
        // 2026-09-25: Rows each buffer can hold for this forward's use of it: `ssm_ba`
        // holds the `[n, 2h]` concat, the others `[n, h]` BF16 rows.
        let rows = |bytes: usize, per_row: usize| {
            if per_row == 0 { 0 } else { bytes / per_row }
        };
        let mut cap = PROPOSE_META_SEQS
            .min(rows(sizes.ssm_ba, 2 * h * bf16))
            .min(rows(sizes.ssm_gates, h * bf16))
            .min(rows(sizes.ssm_qkvz, h * bf16))
            .min(rows(sizes.ssm_deinterleaved, h * bf16))
            .min(rows(sizes.hidden_states, h * bf16))
            .min(rows(sizes.residual, h * bf16))
            .min(rows(sizes.norm_output, h * bf16))
            .min(buffers.max_batch_tokens());
        // 2026-09-25: Shrink to a width some resolved LM-head kernel covers.
        while cap > 1 && self.lm_head_batch_kernel(cap).0 == 0 {
            cap -= 1;
        }
        // 2026-09-25: The native-FP8 MoE arm runs the grouped decode on the `n` rows,
        // so its row range and arena needs (`fp8_grouped_decode_arena_ok`) bound
        // the width too. `propose_batch` checks the context terms (kill switch,
        // FP32 routing, EP) and returns `None` to fall back per sequence.
        if let Some(moe) = self.moe_fp8.as_ref() {
            while cap > 1 && !moe.fp8_grouped_decode_arena_ok(cap, config, buffers) {
                cap -= 1;
            }
        }
        cap.max(1)
    }

    /// 2026-09-25: Whether the batched propose can run for `n` sequences:
    /// `2 <= n <= propose_batch_max`.
    pub(crate) fn can_propose_batch(
        &self,
        n: usize,
        buffers: &BufferArena,
        config: &ModelConfig,
    ) -> bool {
        n >= 2 && n <= self.propose_batch_max(buffers, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};
    use metrale_gpu_runtime::gpu::DevicePtr;

    type Ffn = (ProjectionWeight, ProjectionWeight, ProjectionWeight);

    /// 2026-09-25: A null weight of the variant `MtpHead::quantize_proj` produces for
    /// `q`; the predicates read only the variant.
    fn at(q: MtpQuantization) -> ProjectionWeight {
        match q {
            MtpQuantization::Nvfp4 => ProjectionWeight::Nvfp4(QuantizedWeight::null()),
            MtpQuantization::Fp8 => ProjectionWeight::Fp8(Fp8DenseWeight {
                weight: DevicePtr::NULL,
                row_scale: DevicePtr::NULL,
            }),
            MtpQuantization::Bf16 => ProjectionWeight::Bf16(DenseWeight {
                weight: DevicePtr::NULL,
            }),
        }
    }

    /// 2026-09-25: A dense-FFN head's layout under `requested`: fc/k/v at the forward
    /// precision (`effective_for_head`), q/o and the dense FFN at the requested
    /// one.
    fn dense_head(requested: MtpQuantization) -> (MtpQuantization, [ProjectionWeight; 5], Ffn) {
        let fwd = requested.effective_for_head(true);
        (
            fwd,
            [at(fwd), at(requested), at(fwd), at(fwd), at(requested)],
            (at(requested), at(requested), at(requested)),
        )
    }

    /// 2026-09-25: The weight terms of `propose_batch_scope_ok` for a dense head: the
    /// attention layout and `dense_rows_ok`.
    fn layout_ok(
        quant: MtpQuantization,
        kv_bf16: bool,
        p: &[ProjectionWeight; 5],
        ffn: Option<&Ffn>,
    ) -> bool {
        batch_weight_layout_ok(quant, kv_bf16, [&p[0], &p[1], &p[2], &p[3], &p[4]])
            && super::super::forward_batch_ffn::dense_rows_ok(ffn)
    }

    #[test]
    fn effective_for_head_rewrites_only_dense_nvfp4() {
        use MtpQuantization::*;
        assert_eq!(Nvfp4.effective_for_head(true), Bf16);
        assert_eq!(Nvfp4.effective_for_head(false), Nvfp4);
        for q in [Fp8, Bf16] {
            assert_eq!(q.effective_for_head(true), q);
            assert_eq!(q.effective_for_head(false), q);
        }
    }

    #[test]
    fn dense_nvfp4_head_keeps_the_bf16_forward_and_drafter_prefill() {
        // 2026-09-25: The KV-pool reserve and the prompt-capture allocation key off
        // the forward precision, so a dense NVFP4 head must report BF16 there.
        let fwd = MtpQuantization::Nvfp4.effective_for_head(true);
        assert!(fwd.supports_drafter_prefill());
        assert!(
            !MtpQuantization::Nvfp4
                .effective_for_head(false)
                .supports_drafter_prefill()
        );
    }

    #[test]
    fn batched_propose_admits_the_dense_nvfp4_layout() {
        let (fwd, p, ffn) = dense_head(MtpQuantization::Nvfp4);
        assert!(
            matches!(p[1], ProjectionWeight::Nvfp4(_)),
            "q is the NVFP4 stream"
        );
        assert!(
            matches!(p[4], ProjectionWeight::Nvfp4(_)),
            "o is the NVFP4 stream"
        );
        assert!(layout_ok(fwd, true, &p, Some(&ffn)));
    }

    #[test]
    fn batched_propose_admits_the_default_bf16_layout() {
        let (fwd, p, ffn) = dense_head(MtpQuantization::Bf16);
        assert_eq!(fwd, MtpQuantization::Bf16);
        assert!(layout_ok(fwd, true, &p, Some(&ffn)));
    }

    #[test]
    fn batched_propose_refuses_fp8_and_non_dense_layouts() {
        let (fwd, p, ffn) = dense_head(MtpQuantization::Fp8);
        assert!(!layout_ok(fwd, true, &p, Some(&ffn)), "FP8 head");
        let (fwd, p, _) = dense_head(MtpQuantization::Bf16);
        assert!(
            !layout_ok(fwd, true, &p, None),
            "no dense FFN, no dense arm"
        );
        let (_, p, ffn) = dense_head(MtpQuantization::Nvfp4);
        // 2026-09-25: An MoE head under NVFP4 keeps `Nvfp4` and FP8 KV.
        assert!(!layout_ok(MtpQuantization::Nvfp4, false, &p, Some(&ffn)));
        assert!(
            !layout_ok(MtpQuantization::Bf16, false, &p, Some(&ffn)),
            "FP8 KV"
        );
    }

    #[test]
    fn batched_propose_refuses_nvfp4_fc_k_v() {
        for slot in [0usize, 2, 3] {
            let (fwd, mut p, ffn) = dense_head(MtpQuantization::Nvfp4);
            p[slot] = at(MtpQuantization::Nvfp4);
            assert!(!layout_ok(fwd, true, &p, Some(&ffn)), "slot {slot}");
        }
        let (fwd, p, mut ffn) = dense_head(MtpQuantization::Nvfp4);
        ffn.2 = at(MtpQuantization::Fp8);
        assert!(!layout_ok(fwd, true, &p, Some(&ffn)), "FP8 down_proj");
    }

    #[test]
    fn propose_meta_stride_floor_at_4k() {
        // 2026-09-25: 256 + 257 * 4 = 1284, aligned 1288, below the floor.
        assert_eq!(
            propose_meta_stride_bytes(4096, 16),
            PROPOSE_META_STRIDE_FLOOR
        );
    }

    #[test]
    fn propose_meta_stride_16k() {
        // 2026-09-25: 256 + 1025 * 4 = 4356, aligned 4360.
        assert_eq!(propose_meta_stride_bytes(16 * 1024, 16), 4360);
    }

    #[test]
    fn propose_meta_stride_64k() {
        // 2026-09-25: 256 + 4097 * 4 = 16644, aligned 16648.
        assert_eq!(propose_meta_stride_bytes(64 * 1024, 16), 16648);
    }

    #[test]
    fn propose_meta_stride_never_below_floor() {
        assert_eq!(propose_meta_stride_bytes(0, 16), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(
            propose_meta_stride_bytes(1024, 16),
            PROPOSE_META_STRIDE_FLOOR
        );
        // 2026-09-25: A block size of 0 is treated as 1.
        assert!(propose_meta_stride_bytes(1024, 0) >= PROPOSE_META_STRIDE_FLOOR);
    }

    #[test]
    fn propose_meta_stride_covers_boundary_and_alignment() {
        for max in [
            4096usize,
            10 * 1024,
            16 * 1024,
            20 * 1024,
            64 * 1024,
            128 * 1024,
        ] {
            let s = propose_meta_stride_bytes(max, 16);
            assert_eq!(s % 8, 0, "stride {s} not 8-aligned for max={max}");
            let bt_len = (max / 16 + 1) * 4;
            assert!(
                PROPOSE_META_HEADER + bt_len <= s,
                "stride {s} cannot hold {bt_len}B block table at max={max}"
            );
        }
    }

    #[test]
    fn computed_stride_caps_extreme_contexts_without_overflow() {
        assert_eq!(
            propose_meta_stride_bytes(usize::MAX, 1),
            PROPOSE_META_STRIDE_CAP
        );
        assert!(
            PROPOSE_META_SEQS
                .checked_mul(propose_meta_stride_bytes(usize::MAX, 1))
                .is_some()
        );
    }

    #[test]
    fn stride_override_is_panic_and_overflow_safe() {
        for hostile in [usize::MAX, 1usize << 60, (1usize << 60) + 2048] {
            let s = clamp_stride_override(hostile);
            assert_eq!(s, PROPOSE_META_STRIDE_CAP);
            assert!(PROPOSE_META_SEQS.checked_mul(s).is_some());
        }
        assert_eq!(clamp_stride_override(0), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(clamp_stride_override(2047), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(clamp_stride_override(2048), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(clamp_stride_override(2049), 2056);
        assert_eq!(clamp_stride_override(4361), 4368);
        assert_eq!(
            clamp_stride_override(PROPOSE_META_STRIDE_CAP - 1),
            PROPOSE_META_STRIDE_CAP
        );
        assert_eq!(
            clamp_stride_override(PROPOSE_META_STRIDE_CAP),
            PROPOSE_META_STRIDE_CAP
        );
        assert_eq!(
            clamp_stride_override(PROPOSE_META_STRIDE_CAP + 1),
            PROPOSE_META_STRIDE_CAP
        );
    }

    #[test]
    fn stride_override_parser_accepts_trimmed_bytes_and_rejects_invalid_values() {
        assert_eq!(parse_stride_override(Some(" 4361 ")), Some(4368));
        assert_eq!(
            parse_stride_override(Some("0")),
            Some(PROPOSE_META_STRIDE_FLOOR)
        );
        assert_eq!(parse_stride_override(None), None);
        assert_eq!(parse_stride_override(Some("")), None);
        assert_eq!(parse_stride_override(Some("not-bytes")), None);
        assert_eq!(parse_stride_override(Some("-1")), None);
    }
}
