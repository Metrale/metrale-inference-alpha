// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Mock-backend tests of the Kimi K3 MLA decode launches in
//! `mla_cuda.rs`: kernel lookup, launch geometry and arguments, and how the
//! host-KV and device-resident paths scale their copies with history.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

#[test]
fn resolve_looks_up_k3_entries() {
    let gpu = MockGpuBackend::new();
    let _ = K3MlaDecodeKernels::resolve(&gpu).unwrap();
    assert_eq!(
        gpu.kernel_lookups_snapshot(),
        vec![
            (MODULE.to_string(), ROPE_ENTRY.to_string()),
            (MODULE.to_string(), SDPA_ENTRY.to_string()),
        ]
    );
}

#[test]
fn mock_launch_contract_twin_geometry() {
    let gpu = MockGpuBackend::new();
    let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
    let cfg = MlaConfig::twin_0_40b();
    let mut q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
    let mut k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
    let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
    let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
    let mut kv = MlaKv::default();
    let _ = launch_k3_mla_decode_token(
        &gpu, &kernels, &mut q, &mut k, &v, &g, &mut kv, &cfg, 3, 10000.0, 3,
    )
    .unwrap();
    assert_eq!(kv.seq_len, 1);
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 2, "rope then sdpa_gate");
    assert_eq!(
        launches[0].grid,
        [div_ceil(cfg.heads as u32, ROPE_BLOCK), 1, 1]
    );
    assert_eq!(launches[0].block, [ROPE_BLOCK, 1, 1]);
    assert_eq!(launches[0].args.len(), 8);
    assert_eq!(
        launches[0].args[7],
        MockArg::Bytes(1u32.to_le_bytes().to_vec()),
        "twin is NoPE"
    );
    let sdpa = &launches[1];
    assert_eq!(sdpa.grid, [cfg.heads as u32, 1, 1]);
    assert_eq!(sdpa.block, [SDPA_BLOCK, 1, 1]);
    assert_eq!(sdpa.shared_mem, 0);
    assert_eq!(sdpa.stream, 3);
    assert_eq!(sdpa.args.len(), 10);
    assert_eq!(
        sdpa.args[9],
        MockArg::Bytes(1u32.to_le_bytes().to_vec()),
        "twin uses output gate"
    );
}

#[test]
fn device_kv_second_token_d2h_is_output_only() {
    let gpu = MockGpuBackend::new();
    let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
    let cfg = MlaConfig::twin_0_40b();
    let q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
    let k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
    let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
    let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
    let mut kv = MlaDeviceKv::alloc(
        &gpu,
        8,
        cfg.heads * cfg.qk_head_dim(),
        cfg.heads * cfg.v_head_dim,
    )
    .unwrap();
    let _ = launch_k3_mla_decode_token_on_device(
        &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 0, 10000.0, 0,
    )
    .unwrap();
    let before = gpu.d2h_blocking_count();
    let _ = launch_k3_mla_decode_token_on_device(
        &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 1, 10000.0, 0,
    )
    .unwrap();
    let pulled = gpu.d2h_blocking_count() - before;
    assert_eq!(
        pulled, 1,
        "resident MLA must not re-download KV; D2H is output only (got {pulled})"
    );
    assert_eq!(kv.seq_len, 2);
}

#[test]
fn host_kv_wrapper_h2d_grows_with_history_known_bad() {
    // 2026-09-25: `launch_k3_mla_decode_token` uploads the whole history
    // each token; this pins that its upload grows, as the contrast for
    // `resident_kv_h2d_does_not_grow_with_history`.
    let gpu = MockGpuBackend::new();
    let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
    let cfg = MlaConfig::twin_0_40b();
    let mut q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
    let mut k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
    let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
    let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
    let mut kv = MlaKv::default();
    let _ = launch_k3_mla_decode_token(
        &gpu, &kernels, &mut q, &mut k, &v, &g, &mut kv, &cfg, 0, 10000.0, 0,
    )
    .unwrap();
    let after_t0 = gpu.h2d_bytes();
    let _ = launch_k3_mla_decode_token(
        &gpu, &kernels, &mut q, &mut k, &v, &g, &mut kv, &cfg, 1, 10000.0, 0,
    )
    .unwrap();
    let t1 = gpu.h2d_bytes() - after_t0;
    assert!(
        t1 > after_t0,
        "host-KV wrapper must re-upload history (t0={after_t0} t1={t1})"
    );
}

#[test]
fn resident_kv_h2d_does_not_grow_with_history() {
    let gpu = MockGpuBackend::new();
    let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
    let cfg = MlaConfig::twin_0_40b();
    let q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
    let k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
    let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
    let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
    let mut kv = MlaDeviceKv::alloc(
        &gpu,
        8,
        cfg.heads * cfg.qk_head_dim(),
        cfg.heads * cfg.v_head_dim,
    )
    .unwrap();
    let _ = launch_k3_mla_decode_token_on_device(
        &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 0, 10000.0, 0,
    )
    .unwrap();
    let t0 = gpu.h2d_bytes();
    let _ = launch_k3_mla_decode_token_on_device(
        &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 1, 10000.0, 0,
    )
    .unwrap();
    let t1 = gpu.h2d_bytes() - t0;
    assert_eq!(
        t1, t0,
        "resident MLA token-append H2D must stay O(1) in T (t0={t0} t1={t1})"
    );
}

#[test]
fn failed_second_buffer_allocation_frees_k() {
    let gpu = MockGpuBackend::new();
    // 2026-09-25: K (8 floats) fits the 32-byte limit; V (16 floats) does not.
    gpu.set_max_allocation_bytes(8 * 4);
    assert!(MlaDeviceKv::alloc(&gpu, 1, 8, 16).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}
