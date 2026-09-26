// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which derived weight copies the native-FP8 dense route builds,
//! and a tally of the derived bytes the loader kept, skipped and freed.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - `DenseFp8Plan::resolve` reads no environment and allocates nothing.
//! - With `keep_nvfp4` set, the plan builds every NVFP4 copy, and every FP8
//!   twin on a layer with an FP8 attention overlay.
//!
//! With the native FP8 overlay installed, the dense FFN reads no NVFP4
//! weight: `DenseFfnLayer::forward` and `forward_prefill_inner` return inside
//! their `fp8_weights` arm, `forward_k2`, `forward_k3` and `forward_km` send an
//! FP8 layer to `forward_prefill` (`native_small_batch_uses_prefill`), and the
//! prefill `w8_gemm!` is given `None` for every transposed operand, so each of
//! its arms reads the untransposed E4M3 weight. `METRALE_FFN_W8A16_ONLY` only
//! turns off the W8A8 arm of that macro, so it is not an input here.
//!
//! The loader runs before a `ForwardContext` exists, so the route comes from
//! `GemmDispatch::from_env`, the constructor the model uses for the context's
//! dispatch. A lever that can send a native-FP8 layer to an NVFP4 kernel must
//! be an input of [`DenseFp8Plan::resolve`] as well as of that dispatch site,
//! or the loader skips a copy the site reads. `METRALE_DENSE_FP8_KEEP_NVFP4`
//! builds every copy while such a gap is diagnosed.

use metrale_model_layers::layers::ops::GemmDispatch;
use metrale_model_layers::layers::qwen3_attention::Fp8TwinSet;

/// 2026-09-25: Whether `METRALE_DENSE_FP8_KEEP_NVFP4` is set, which makes the
/// plan build every NVFP4 copy even where the dispatch cannot reach it.
/// Presence, like `METRALE_FFN_W8A16_ONLY`: any value, including empty and
/// `0`, turns it on. Read once per process.
pub fn keep_nvfp4_fallback() -> bool {
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| std::env::var_os("METRALE_DENSE_FP8_KEEP_NVFP4").is_some())
}

/// 2026-09-25: Which derived copies one dense layer builds beyond the
/// checkpoint bytes. Each field means "build this copy".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseFp8Plan {
    /// 2026-09-25: The NVFP4 gate/up/down weights and their transposed twins
    /// (`fp8_lut::load_dense_ffn`).
    pub ffn_nvfp4: bool,
    /// 2026-09-25: The NVFP4 q/k/v/o weights, their transposed twins, and the
    /// fused `[q|k|v]` transposed twin.
    pub attn_nvfp4: bool,
    /// 2026-09-25: Which FP8 prefill twins `transpose_fp8_for_prefill_selected`
    /// builds.
    pub attn_fp8_twins: Fp8TwinSet,
}

/// 2026-09-25: The inputs of [`DenseFp8Plan::resolve`]. A plain struct, so the
/// decision-table tests set every clause without the process environment,
/// which `keep_nvfp4_fallback` caches in a `OnceLock`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseFp8Inputs {
    /// 2026-09-25: The native FP8 dense-FFN overlay is installed on this layer
    /// (`qwen35_dense::ffn_fp8_arm_selected`: `METRALE_DENSE_FP8=1`, tp size 1,
    /// `Fp8Dequanted`, a native FP8 `gate_proj`).
    pub ffn_fp8: bool,
    /// 2026-09-25: The native FP8 attention overlay is installed on this layer.
    pub attn_fp8: bool,
    /// 2026-09-25: `METRALE_DENSE_FP8_KEEP_NVFP4` (`keep_nvfp4_fallback`).
    pub keep_nvfp4: bool,
    /// 2026-09-25: `GemmDispatch::from_env()`, the value the model stores as its
    /// dispatch.
    pub dispatch: GemmDispatch,
    /// 2026-09-25: Both W8A8 prefill kernels are loaded
    /// (`per_token_group_quant_fp8` and `fp8_gemm_t_blockscaled`). The W8A8 arms
    /// of `prefill/paged_qkv.rs` and `prefill/paged_oproj.rs` need both; without
    /// them those projections fall through to the transposed W8A16 kernels,
    /// which read the FP8 twins.
    pub w8a8_kernels: bool,
    /// 2026-09-25: `METRALE_ATTN_W4A4` is set. The W4A4 arm of
    /// `prefill/paged_oproj.rs` checks no weight type and reads
    /// `self.attn.o_proj`, the NVFP4 o_proj, so this lever keeps the NVFP4
    /// attention weights under an FP8 overlay. The Q/K/V W4A4 arm in
    /// `prefill/paged_qkv.rs` requires an NVFP4 `q_weight`, so it never runs on
    /// an FP8 layer.
    pub attn_w4a4: bool,
    /// 2026-09-25: `METRALE_ATTN_PREFILL_Q_T=1`. `prefill/cache_skip_qkv.rs` reads
    /// it on every call and, when set, lets Q use `q_fp8w_t`.
    pub attn_prefill_q_t: bool,
}

impl DenseFp8Plan {
    /// 2026-09-25: The decision table.
    ///
    /// - FFN NVFP4: built only when the layer has no FFN overlay, or under
    ///   `keep_nvfp4` (the module docs say why the FP8 FFN never reads it).
    /// - Attention NVFP4: `set_fp8_weights` replaces `q_weight`, `k_weight`,
    ///   `v_weight` and `o_weight` with `QuantWeight::Fp8`. The NVFP4 copies
    ///   are kept when a `METRALE_CUTLASS_NVFP4_*` attention arm
    ///   (`cutlass_nvfp4_*` in the dispatch) or the W4A4 o_proj arm
    ///   (`attn_w4a4`) can read them.
    /// - FP8 twins K and V: built on every FP8 attention layer unless the
    ///   umbrella flag is set (last item). A single
    ///   stream's first prefill chunk (`seq_len_start == 0`) runs
    ///   `prefill/cache_skip_qkv.rs`, whose only W8A8 arm is the cuBLASLt one
    ///   (`cache_skip_qkv_cublas_selected`, which needs `METRALE_CUBLAS_GEMM`
    ///   to cover `attn`). When it declines, K and V read `k_fp8w_t` and
    ///   `v_fp8w_t` if present.
    /// - FP8 twins Q and O: Q is read on that chain only with
    ///   `attn_prefill_q_t`, and on the paged chain after its W8A8 arm declines;
    ///   both chains run O through `prefill/paged_oproj.rs`, which reads the O
    ///   twin after its W8A8 arm declines. The W8A8 arms decline when
    ///   block-scaled prefill is off or a W8A8 kernel is missing.
    /// - Under the umbrella `cutlass_nvfp4_gemm` no FP8 twin is planned, because
    ///   `transpose_fp8_for_prefill_selected` builds none under that flag.
    pub fn resolve(i: DenseFp8Inputs) -> Self {
        if i.keep_nvfp4 {
            return Self {
                ffn_nvfp4: true,
                attn_nvfp4: true,
                attn_fp8_twins: if i.attn_fp8 {
                    Fp8TwinSet::ALL
                } else {
                    Fp8TwinSet::NONE
                },
            };
        }
        let cutlass_nvfp4_attn = i.dispatch.cutlass_nvfp4_gemm
            || i.dispatch.cutlass_nvfp4_attn_q
            || i.dispatch.cutlass_nvfp4_attn_kv
            || i.dispatch.cutlass_nvfp4_attn_o;
        let w8a8_covers_prefill = i.dispatch.fp8_blockscaled_prefill && i.w8a8_kernels;
        let fp8_twins = if !i.attn_fp8 || i.dispatch.cutlass_nvfp4_gemm {
            Fp8TwinSet::NONE
        } else {
            Fp8TwinSet {
                q: i.attn_prefill_q_t || !w8a8_covers_prefill,
                k: true,
                v: true,
                o: !w8a8_covers_prefill,
            }
        };
        Self {
            ffn_nvfp4: !i.ffn_fp8,
            attn_nvfp4: !i.attn_fp8 || cutlass_nvfp4_attn || i.attn_w4a4,
            attn_fp8_twins: fp8_twins,
        }
    }
}

/// 2026-09-25: The environment half of [`DenseFp8Inputs`]. `load_layers`
/// resolves it once, so every layer of one load is planned from the same
/// values.
#[derive(Clone, Copy, Debug)]
pub struct RouteEnv {
    pub keep_nvfp4: bool,
    pub dispatch: GemmDispatch,
    /// 2026-09-25: The `METRALE_ATTN_W4A4` check of `prefill/paged_oproj.rs`
    /// and `prefill/paged_qkv.rs`, which read the variable on every call.
    pub attn_w4a4: bool,
    /// 2026-09-25: The `METRALE_ATTN_PREFILL_Q_T` check of
    /// `prefill/cache_skip_qkv.rs`, also read on every call.
    pub attn_prefill_q_t: bool,
}

impl RouteEnv {
    pub fn from_env() -> Self {
        Self {
            keep_nvfp4: keep_nvfp4_fallback(),
            dispatch: GemmDispatch::from_env(),
            // 2026-09-25: The dispatch sites' predicates: `is_ok()` for W4A4,
            // `== "1"` for the Q transpose.
            attn_w4a4: std::env::var("METRALE_ATTN_W4A4").is_ok(),
            attn_prefill_q_t: std::env::var("METRALE_ATTN_PREFILL_Q_T").ok().as_deref()
                == Some("1"),
        }
    }

    /// 2026-09-25: One layer's plan. `w8a8_kernels` comes from the layer's
    /// resolved kernel handles, so it is an argument, not a field.
    pub fn plan(&self, ffn_fp8: bool, attn_fp8: bool, w8a8_kernels: bool) -> DenseFp8Plan {
        DenseFp8Plan::resolve(DenseFp8Inputs {
            ffn_fp8,
            attn_fp8,
            keep_nvfp4: self.keep_nvfp4,
            dispatch: self.dispatch,
            w8a8_kernels,
            attn_w4a4: self.attn_w4a4,
            attn_prefill_q_t: self.attn_prefill_q_t,
        })
    }

    /// 2026-09-25: The attention NVFP4 decision. The loader asks it before the
    /// layer exists and passes `w8a8_kernels = true`, which is sound because
    /// `attn_nvfp4` does not depend on it
    /// (`attn_nvfp4_does_not_depend_on_the_w8a8_kernels`).
    pub fn attn_nvfp4(&self, attn_fp8: bool) -> bool {
        self.plan(false, attn_fp8, true).attn_nvfp4
    }

    /// 2026-09-25: Which FP8 prefill twins this layer needs. The loader passes
    /// the constructed layer's `has_w8a8_prefill_kernels()`.
    pub fn attn_fp8_twins(&self, attn_fp8: bool, w8a8_kernels: bool) -> Fp8TwinSet {
        self.plan(false, attn_fp8, w8a8_kernels).attn_fp8_twins
    }
}

/// 2026-09-25: Bytes of one NVFP4 `QuantizedWeight`: `n * k / 2` packed E2M1
/// bytes plus `n * k / 16` group-scale bytes, the two allocations of
/// `quantize_to_nvfp4` and of `QuantizedWeight::transpose_for_gemm_gs` with 16-wide groups.
pub fn nvfp4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / 16
}

/// 2026-09-25: NVFP4 bytes of one dense-FFN layer: gate, up and down, each
/// with a transposed twin of the same size.
pub fn dense_ffn_nvfp4_bytes(hidden: usize, inter: usize) -> usize {
    // 2026-09-25: gate/up are `[inter, hidden]` and down is `[hidden, inter]`:
    // the same element count.
    3 * 2 * nvfp4_bytes(inter, hidden)
}

/// 2026-09-25: NVFP4 bytes of one full-attention layer: q/k/v/o, each with a
/// transposed twin, plus the fused `[q|k|v]` transposed twin
/// (`transpose_concat_for_gemm`).
///
/// `q_n` is `num_attention_heads * head_dim`, doubled when `attn_gated`;
/// `kv_n` is `num_key_value_heads * head_dim`; `o_k` is the o_proj contraction
/// width `num_attention_heads * head_dim`. These are the loader's arguments.
pub fn attn_nvfp4_bytes(q_n: usize, kv_n: usize, o_k: usize, hidden: usize) -> usize {
    let qkv = nvfp4_bytes(q_n, hidden) + 2 * nvfp4_bytes(kv_n, hidden);
    let o = nvfp4_bytes(hidden, o_k);
    2 * (qkv + o) + nvfp4_bytes(q_n + 2 * kv_n, hidden)
}

/// 2026-09-25: Bytes of one FP8 transposed twin: `n * k` E4M3 bytes plus the
/// FP32 block-scale grid over 128x128 blocks, the two allocations of
/// `Fp8Weight::transpose_for_gemm`.
pub fn fp8_twin_bytes(n: usize, k: usize) -> usize {
    n * k + n.div_ceil(128) * k.div_ceil(128) * 4
}

/// 2026-09-25: Bytes of the FP8 prefill twins `want` selects, for one
/// full-attention layer.
pub fn attn_fp8_twin_bytes(
    want: Fp8TwinSet,
    q_n: usize,
    kv_n: usize,
    o_k: usize,
    hidden: usize,
) -> usize {
    let mut b = 0;
    if want.q {
        b += fp8_twin_bytes(q_n, hidden);
    }
    if want.k {
        b += fp8_twin_bytes(kv_n, hidden);
    }
    if want.v {
        b += fp8_twin_bytes(kv_n, hidden);
    }
    if want.o {
        b += fp8_twin_bytes(hidden, o_k);
    }
    b
}

/// 2026-09-25: Bytes of one fused dense-FFN gate+up weight: the two
/// `[inter, hidden]` E4M3 blocks side by side, plus their two FP32 block-scale
/// grids side by side. Two grids, not one grid over `2 * inter`, because the
/// concat copies the two source grids.
pub fn ffn_gateup_fused_bytes(hidden: usize, inter: usize) -> usize {
    let (w, s) = ffn_gateup_fused_parts(hidden, inter);
    w + s
}

/// 2026-09-25: [`ffn_gateup_fused_bytes`] as `(weight bytes, scale-grid
/// bytes)`. The loader adopts the two buffers separately.
pub fn ffn_gateup_fused_parts(hidden: usize, inter: usize) -> (usize, usize) {
    (
        2 * inter * hidden,
        2 * (inter.div_ceil(128) * hidden.div_ceil(128) * 4),
    )
}

/// 2026-09-25: Running tally, kept by the loader as it goes, of the derived
/// (non-checkpoint) device bytes it built, freed and declined to build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DerivedResidency {
    /// 2026-09-25: Bytes the loader recorded with `keep`: derived copies it
    /// keeps after the load.
    pub kept: u64,
    /// 2026-09-25: Derived bytes the plan declined to build.
    pub skipped: u64,
    /// 2026-09-25: Derived bytes allocated and freed during the load.
    pub freed: u64,
    /// 2026-09-25: Which derived families were built, for the summary line.
    pub twins: TwinsBuilt,
}

/// 2026-09-25: Which derived families the loader built, named in the summary
/// line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TwinsBuilt {
    pub ffn_nvfp4: bool,
    pub attn_nvfp4: bool,
    pub attn_fp8: bool,
    pub ssm_fp8_concat: bool,
    /// 2026-09-25: The dense-FFN `[2*inter, hidden]` gate+up concat. Its bytes
    /// count in `kept`, while `prune_after_load` releases the two store
    /// tensors it copied.
    pub ffn_gateup_fused: bool,
}

impl TwinsBuilt {
    /// 2026-09-25: `none`, or a comma-separated list, for the summary line.
    pub fn describe(self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.ffn_nvfp4 {
            parts.push("ffn-nvfp4+t");
        }
        if self.attn_nvfp4 {
            parts.push("attn-nvfp4+t");
        }
        if self.attn_fp8 {
            parts.push("attn-fp8-t");
        }
        if self.ssm_fp8_concat {
            parts.push("ssm-qkvz-fp8");
        }
        if self.ffn_gateup_fused {
            parts.push("ffn-gateup-fp8");
        }
        if parts.is_empty() {
            "none".to_owned()
        } else {
            parts.join(", ")
        }
    }
}

impl DerivedResidency {
    pub fn keep(&mut self, bytes: usize) {
        self.kept += bytes as u64;
    }

    pub fn skip(&mut self, bytes: usize) {
        self.skipped += bytes as u64;
    }

    pub fn free(&mut self, bytes: usize) {
        self.freed += bytes as u64;
    }

    /// 2026-09-25: The summary line `load_layers` logs at its end. `weights`
    /// is the argument (the store's resident bytes), `derived` is `kept`, and
    /// `not built` is `skipped`.
    pub fn summary(&self, weight_bytes: usize) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        format!(
            "native FP8 dense residency: weights {:.2} GB, derived {:.2} GB \
             (twins: {}), freed {:.2} GB, not built {:.2} GB",
            weight_bytes as f64 / 1e9,
            gb(self.kept),
            self.twins.describe(),
            gb(self.freed),
            gb(self.skipped),
        )
    }
}

#[cfg(test)]
#[path = "fp8_residency_tests.rs"]
mod tests;
