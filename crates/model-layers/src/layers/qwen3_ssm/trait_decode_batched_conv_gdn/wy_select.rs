// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The verify WY kernel selectors (`wy2_kernel`, `wy3_kernel`, `wy4_kernel`),
//! their register-resident kill switches and width gate, and `require_wy_f16`, moved
//! whole from `trait_decode_batched_conv_gdn.rs`.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.

use anyhow::Result;

use crate::layers::qwen3_ssm::Qwen3SsmLayer;

/// 2026-09-25: Kill switch for the register-resident wy2 twin:
/// `METRALE_NO_GDN_WY2_RESIDENT` set to any value, `0` included, disables it. Read once
/// per process.
fn wy2_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_GDN_WY2_RESIDENT").is_none())
}

/// 2026-09-25: Kill switch for the register-resident wy3 twin, separate from wy2's:
/// `METRALE_NO_GDN_WY3_RESIDENT` set to any value, `0` included, disables it. Read once
/// per process.
fn wy3_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_GDN_WY3_RESIDENT").is_none())
}

/// 2026-09-25: Minimum launch width (sequences) at which `wy2_kernel` and `wy3_kernel`
/// select the register-resident twins, which are `__launch_bounds__(128, 1)`
/// (kernels/gb10/common/gated_delta_rule_wy{2,3}_resident.cu). Narrower launches run the
/// base kernel.
fn wy_resident_min_width() -> usize {
    16
}

impl Qwen3SsmLayer {
    /// 2026-09-25: The K=2 verify WY kernel for a launch of `n` sequences. With an FP32
    /// h-state: the register-resident `gated_delta_rule_wy2_resident` when kd == vd ==
    /// 128, `n >= wy_resident_min_width()`, its handle is linked and
    /// `METRALE_NO_GDN_WY2_RESIDENT` is absent; the base `gated_delta_rule_wy2` otherwise.
    /// The two take the same arguments, and the resident-parity leg of the
    /// `gdn_wy_verify_microtest` example compares them bit for bit. The first call of
    /// each FP32 outcome logs it with the handle.
    pub(in crate::layers::qwen3_ssm) fn wy2_kernel(
        &self,
        kd: usize,
        vd: usize,
        n: usize,
    ) -> metrale_gpu_runtime::gpu::KernelHandle {
        let wide_enough = n >= wy_resident_min_width();
        let eligible = kd == 128
            && vd == 128
            && wide_enough
            && self.gdn_wy2_resident_k.0 != 0
            && wy2_resident_enabled();
        static LOGGED_ENGAGED: std::sync::Once = std::sync::Once::new();
        static LOGGED_BASE: std::sync::Once = std::sync::Once::new();
        // 2026-09-25: With an FP16 h-state only an FP16 twin is correct, chosen by the
        // same rule. A zero handle is returned as is, never replaced by an FP32 kernel;
        // `decode_batched_conv_gdn_multi` turns it into an error (`require_wy_f16`).
        if super::super::ssm_h_fp16_enabled() {
            return if eligible && self.gdn_wy2_resident_f16_k.0 != 0 {
                self.gdn_wy2_resident_f16_k
            } else {
                self.gdn_wy2_f16_k
            };
        }
        if eligible {
            LOGGED_ENGAGED.call_once(|| {
                tracing::info!(
                    target: "metrale_model_layers::layers::qwen3_ssm::trait_decode_batched_conv_gdn",
                    "GDN wy2 REGISTER-RESIDENT ENGAGED (handle {:#x}, n={n}): K=2 verify \
                     Pass 2 served from registers — state traffic 2R+2W -> 1R+2W; \
                     width-gated n >= {}; kill switch METRALE_NO_GDN_WY2_RESIDENT (presence)",
                    self.gdn_wy2_resident_k.0,
                    wy_resident_min_width(),
                );
            });
            self.gdn_wy2_resident_k
        } else {
            LOGGED_BASE.call_once(|| {
                tracing::info!(
                    target: "metrale_model_layers::layers::qwen3_ssm::trait_decode_batched_conv_gdn",
                    "GDN wy2 register-resident twin NOT engaged at this dispatch (kd={kd}, \
                     vd={vd}, n={n} vs min width {}, handle {:#x}, kill_switch_present={}): \
                     base gated_delta_rule_wy2 in use (wider K=2 launches re-decide)",
                    wy_resident_min_width(),
                    self.gdn_wy2_resident_k.0,
                    !wy2_resident_enabled(),
                );
            });
            self.gdn_wy2_k
        }
    }

    /// 2026-09-25: The K=3 verify WY kernel, chosen by the `wy2_kernel` rule with the wy3
    /// handles and `METRALE_NO_GDN_WY3_RESIDENT`. K=3 rows come from 2-draft rungs: an
    /// explicit `METRALE_MTP_K_LADDER` step or the adaptive n=16 rung
    /// (`metrale_speculative::adaptive_rung`); the default static ladder
    /// (`4:3,8:3,16:1,32:1`, `speculative/ladder.rs`) has none.
    pub(in crate::layers::qwen3_ssm) fn wy3_kernel(
        &self,
        kd: usize,
        vd: usize,
        n: usize,
    ) -> metrale_gpu_runtime::gpu::KernelHandle {
        let wide_enough = n >= wy_resident_min_width();
        let eligible = kd == 128
            && vd == 128
            && wide_enough
            && self.gdn_wy3_resident_k.0 != 0
            && wy3_resident_enabled();
        static LOGGED_ENGAGED: std::sync::Once = std::sync::Once::new();
        static LOGGED_BASE: std::sync::Once = std::sync::Once::new();
        // 2026-09-25: FP16 h-state: as in `wy2_kernel`.
        if super::super::ssm_h_fp16_enabled() {
            return if eligible && self.gdn_wy3_resident_f16_k.0 != 0 {
                self.gdn_wy3_resident_f16_k
            } else {
                self.gdn_wy3_f16_k
            };
        }
        if eligible {
            LOGGED_ENGAGED.call_once(|| {
                tracing::info!(
                    target: "metrale_model_layers::layers::qwen3_ssm::trait_decode_batched_conv_gdn",
                    "GDN wy3 REGISTER-RESIDENT ENGAGED (handle {:#x}, n={n}): K=3 verify \
                     Pass 2 served from registers — state traffic 2R+3W -> 1R+3W; \
                     width-gated n >= {}; kill switch METRALE_NO_GDN_WY3_RESIDENT (presence)",
                    self.gdn_wy3_resident_k.0,
                    wy_resident_min_width(),
                );
            });
            self.gdn_wy3_resident_k
        } else {
            LOGGED_BASE.call_once(|| {
                tracing::info!(
                    target: "metrale_model_layers::layers::qwen3_ssm::trait_decode_batched_conv_gdn",
                    "GDN wy3 register-resident twin NOT engaged at this dispatch (kd={kd}, \
                     vd={vd}, n={n} vs min width {}, handle {:#x}, kill_switch_present={}): \
                     base gated_delta_rule_wy3 in use (wider K=3 launches re-decide)",
                    wy_resident_min_width(),
                    self.gdn_wy3_resident_k.0,
                    !wy3_resident_enabled(),
                );
            });
            self.gdn_wy3_k
        }
    }

    /// 2026-09-25: The K=4 verify WY kernel: `gated_delta_rule_wy4`, or its FP16 twin under
    /// an FP16 h-state; there is no register-resident K=4 twin. K=4 (3 drafts) is the
    /// default ladder's shape up to 8 sequences (`4:3,8:3`, `speculative/ladder.rs`).
    pub(in crate::layers::qwen3_ssm) fn wy4_kernel(
        &self,
    ) -> metrale_gpu_runtime::gpu::KernelHandle {
        if super::super::ssm_h_fp16_enabled() {
            return self.gdn_wy4_f16_k;
        }
        self.gdn_wy4_k
    }

    /// 2026-09-25: Error when the h-state is FP16 and `wy_k`, the FP16 twin for verify
    /// width `kk`, did not resolve. `decode_batched_conv_gdn_multi` calls it before a zero
    /// handle would make it decline to the per-sequence loop.
    pub(in crate::layers::qwen3_ssm) fn require_wy_f16(
        &self,
        kk: usize,
        wy_k: metrale_gpu_runtime::gpu::KernelHandle,
    ) -> Result<()> {
        if super::super::ssm_h_fp16_enabled() && wy_k.0 == 0 {
            anyhow::bail!(
                "METRALE_SSM_H_FP16: no FP16 h-state twin resolved for the K={kk} MTP verify \
                 WY kernel. Falling back to the FP32 kernel would read the FP16 pool as \
                 floats and emit fluent garbage, so this refuses instead. Run without \
                 --speculative, or unset METRALE_SSM_H_FP16."
            );
        }
        Ok(())
    }
}
