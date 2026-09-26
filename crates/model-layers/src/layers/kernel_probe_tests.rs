// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that `try_target_kernel` issues no lookup for an absent module.
//!
//! Owner: model-layers (kernel lookup).
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: A module the backend does not carry is never looked up; one it
/// does carry is looked up as `try_kernel` does.
#[test]
fn a_module_the_target_never_built_is_not_looked_up() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent("gdn_fwd_o_hopper");
    let h = try_target_kernel(
        &gpu,
        "gdn_fwd_o_hopper",
        "gated_delta_rule_chunk_fwd_o_hopper",
    );
    assert_eq!(h.0, 0);
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no lookup may reach the backend for an absent module"
    );
    // 2026-09-25: Negative control: `try_kernel` does issue the lookup.
    let _ = try_kernel(
        &gpu,
        "gdn_fwd_o_hopper",
        "gated_delta_rule_chunk_fwd_o_hopper",
    );
    assert_eq!(gpu.kernel_lookups_snapshot().len(), 1);
    let h = try_target_kernel(&gpu, "ssm_preprocess", "dense_gemm_ba_gates_prefill");
    assert_ne!(h.0, 0);
    assert_eq!(gpu.kernel_lookups_snapshot().len(), 2);
}
