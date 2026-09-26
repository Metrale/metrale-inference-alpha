// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel lookups that `Qwen3SsmLayer::new` delegates: the mHC
//! probes, the wyN verify families, the GDN prefill state spines and the
//! Hopper-only prefill twins.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - Entry `i` of `wyn_kernels`, `wyn_f16_kernels`, `WYN_TABLE_NAMES` and
//!   `WYN_F16_TABLE_NAMES` is the K = i + 5 kernel.

use super::*;

/// 2026-09-25: Look up `hyper_connection::{func}` only when
/// `config.hc_mult > 0`; otherwise return `KernelHandle(0)` without a lookup,
/// so a model without the highway leaves no failed row in the boot kernel
/// audit.
#[track_caller]
pub(super) fn hc_kernel(
    config: &metrale_config::ModelConfig,
    gpu: &dyn GpuBackend,
    func: &str,
) -> KernelHandle {
    if config.hc_mult > 0 {
        crate::layers::try_kernel(gpu, "hyper_connection", func)
    } else {
        KernelHandle(0)
    }
}

/// 2026-09-25: The chain-verify WY kernels `gated_delta_rule_wy5` ..
/// `gated_delta_rule_wy16` of module `gated_delta_rule_wyn`, index = K - 5.
/// With an FP32 h-state, a 0 handle makes `wyn_kernel` return `None` and the
/// verify dispatch takes its sequential per-token path.
pub(super) fn wyn_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    [
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy5"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy6"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy7"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy8"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy9"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy10"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy11"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy12"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy13"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy14"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy15"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy16"),
    ]
}

/// 2026-09-25: The FP16 h-state twins (`gated_delta_rule_wy{K}_f16`) of
/// [`wyn_kernels`], same module and index. Under the FP16 h-state `wyn_kernel`
/// reads these, and when one is 0 the single-sequence verify dispatch
/// (`decode_batched_conv_gdn`) returns an error instead of running FP32
/// kernels on the FP16 state.
pub(super) fn wyn_f16_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    [
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy5_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy6_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy7_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy8_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy9_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy10_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy11_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy12_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy13_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy14_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy15_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy16_f16"),
    ]
}

/// 2026-09-25: The tensor-core GDN prefill state spine,
/// [`ops::GDN_TC_SPINE_ENTRY`] in module `GDN_TC_SPINE_MODULE`. It is looked
/// up only when the target default `gdn_prefill_tc` resolves true
/// (`[defaults] gdn_prefill_tc` in the target's HARDWARE.toml,
/// `METRALE_GDN_PREFILL_TC` overriding; see `ops::target_defaults`), and
/// through `try_target_kernel`, so a target without the module issues no
/// lookup. Otherwise the handle is 0. `kernels/hopper` declares the default
/// true; gb10, b200 and b300 declare it false.
pub(super) fn gdn_prefill_tc_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if !crate::layers::ops::target_defaults::resolved()
        .gdn_prefill_tc
        .value
    {
        return KernelHandle(0);
    }
    crate::layers::try_target_kernel(
        gpu,
        crate::layers::ops::GDN_TC_SPINE_MODULE,
        crate::layers::ops::GDN_TC_SPINE_ENTRY,
    )
}

/// 2026-09-25: The scalar fused GDN state-spine handle from
/// `gated_delta_rule_fla`: `GDN_SCALAR_SPINE_PIPE` when `METRALE_GDN_PIPE=1`,
/// else `GDN_SCALAR_SPINE_VTILE` when `METRALE_GDN_VTILE=1`, else
/// `GDN_SCALAR_SPINE_VFUSED`. `ops::gdn_prefill_fla` derives the launch's block
/// size from the same two variables.
///
/// Logs one route line (`gdn_init_spine_line`): the tensor-core entry when
/// `tc_spine` (the handle [`gdn_prefill_tc_kernel`] returned) is non-zero,
/// else this scalar entry.
pub(super) fn fused_spine_kernel(gpu: &dyn GpuBackend, tc_spine: KernelHandle) -> KernelHandle {
    use crate::layers::ops::{
        GDN_SCALAR_SPINE_PIPE, GDN_SCALAR_SPINE_VFUSED, GDN_SCALAR_SPINE_VTILE, gdn_init_spine_line,
    };
    let scalar = match (
        std::env::var("METRALE_GDN_PIPE").ok().as_deref(),
        std::env::var("METRALE_GDN_VTILE").ok().as_deref(),
    ) {
        (Some("1"), _) => GDN_SCALAR_SPINE_PIPE,
        (_, Some("1")) => GDN_SCALAR_SPINE_VTILE,
        _ => GDN_SCALAR_SPINE_VFUSED,
    };
    tracing::info!("{}", gdn_init_spine_line(tc_spine.0 != 0, scalar));
    crate::layers::try_kernel(gpu, "gated_delta_rule_fla", scalar)
}

// 2026-09-25: The Hopper prefill twins below use `try_target_kernel`: their
// sources exist only under `kernels/hopper/common`, so any other target
// issues no lookup, gets `KernelHandle(0)`, and its launcher keeps the parent
// kernel.

/// 2026-09-25: The Hopper twin of the FLA prefill's `recompute_wu` kernel
/// (`gdn_recompute_wu_hopper.cu`). The probe is not gated on
/// `gdn_prefill_tc`; `ops::ssm_gdn_hopper_prefill` selects the twin only when
/// that family default is on and `METRALE_NO_GDN_PREFILL_TC_REMNANTS=1` is
/// not set.
pub(super) fn prefill_wu_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_target_kernel(
        gpu,
        "gdn_recompute_wu_hopper",
        "gated_delta_rule_recompute_wu_hopper",
    )
}

/// 2026-09-25: The Hopper twin of the FLA prefill's `chunk_fwd_o` kernel
/// (`gdn_fwd_o_hopper.cu`), probed and selected like the `recompute_wu` twin.
pub(super) fn prefill_fwd_o_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_target_kernel(
        gpu,
        "gdn_fwd_o_hopper",
        "gated_delta_rule_chunk_fwd_o_hopper",
    )
}

/// 2026-09-25: The Hopper twin of the prefill BA-gates kernel
/// (`ssm_ba_gates_hopper.cu`). The probe is not gated on
/// `[defaults] ssm_ba_gates_hopper`: `ops::ba_gates_pick` reads that lever
/// and the shape guards at dispatch, and returns the `ssm_preprocess` parent
/// when the twin is 0 or refused.
pub(super) fn ba_gates_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_target_kernel(
        gpu,
        "ssm_ba_gates_hopper",
        "dense_gemm_ba_gates_prefill_hopper",
    )
}

// 2026-09-25: Pointer-table twins of `wyn_kernels` (`state_is_table`) for the
// cross-sequence batched verify, which reads them through `wyn_table_kernel`.
// provenance-id: 526f6e616c6420522e205374657369616b
/// 2026-09-25: Symbol names of the pointer-table twins, index = K - 5; the
/// tests below check each entry against its K.
pub(super) const WYN_TABLE_NAMES: [&str; 12] = [
    "gated_delta_rule_wy5_table",
    "gated_delta_rule_wy6_table",
    "gated_delta_rule_wy7_table",
    "gated_delta_rule_wy8_table",
    "gated_delta_rule_wy9_table",
    "gated_delta_rule_wy10_table",
    "gated_delta_rule_wy11_table",
    "gated_delta_rule_wy12_table",
    "gated_delta_rule_wy13_table",
    "gated_delta_rule_wy14_table",
    "gated_delta_rule_wy15_table",
    "gated_delta_rule_wy16_table",
];

pub(super) fn wyn_table_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    WYN_TABLE_NAMES.map(|n| crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", n))
}

/// 2026-09-25: Symbol names of the FP16 pointer-table twins, index = K - 5.
pub(super) const WYN_F16_TABLE_NAMES: [&str; 12] = [
    "gated_delta_rule_wy5_f16_table",
    "gated_delta_rule_wy6_f16_table",
    "gated_delta_rule_wy7_f16_table",
    "gated_delta_rule_wy8_f16_table",
    "gated_delta_rule_wy9_f16_table",
    "gated_delta_rule_wy10_f16_table",
    "gated_delta_rule_wy11_f16_table",
    "gated_delta_rule_wy12_f16_table",
    "gated_delta_rule_wy13_f16_table",
    "gated_delta_rule_wy14_f16_table",
    "gated_delta_rule_wy15_f16_table",
    "gated_delta_rule_wy16_f16_table",
];

pub(super) fn wyn_f16_table_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    WYN_F16_TABLE_NAMES.map(|n| crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", n))
}

#[cfg(test)]
mod table_registry_tests {
    use super::{WYN_F16_TABLE_NAMES, WYN_TABLE_NAMES};

    /// 2026-09-25: Entry i is the K = i + 5 symbol, for both dtypes.
    #[test]
    fn table_names_hold_the_index_contract() {
        for (i, (n, f)) in WYN_TABLE_NAMES.iter().zip(WYN_F16_TABLE_NAMES).enumerate() {
            let k = i + 5;
            assert_eq!(
                *n,
                format!("gated_delta_rule_wy{k}_table"),
                "fp32 index {i}"
            );
            assert_eq!(
                f,
                format!("gated_delta_rule_wy{k}_f16_table"),
                "f16 index {i}"
            );
        }
    }

    /// 2026-09-25: The tables run from K = 5 to K = 16, the range
    /// `wyn_table_kernel` accepts.
    #[test]
    fn table_registry_covers_k5_through_k16() {
        assert_eq!(WYN_TABLE_NAMES.len(), 12);
        assert!(WYN_TABLE_NAMES[0].contains("wy5_"));
        assert!(WYN_TABLE_NAMES[11].contains("wy16_"));
    }
}
