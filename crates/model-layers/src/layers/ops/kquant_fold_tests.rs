// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: GPU test that the fused `kquant_swiglu_q8_1_rows_bf16` matches `moe_v41_swiglu` followed by `kquant_q8_1_rows_bf16` byte for byte.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The fused kernel writes the SwiGLU output and its q8_1 rows in one launch. Both outputs are
//! compared, with and without the routing weights and the swiglu limit, at 1, 5 and 6 rows.
//!
//! The test is `#[ignore]` and needs a GPU. Run with
//! ```text
//! cargo test -p metrale-model-layers --release kquant_fold -- --ignored --nocapture
//! ```

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::kquant_mmq::*;

const INTER: u32 = 1024;

fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v).unwrap();
    v
}

/// 2026-09-25: `n` bf16 values in [-4, 4) from an LCG, as little-endian bytes.
fn bf16_bytes(n: usize, mut state: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let x = ((state >> 8) as f32 / (1u32 << 24) as f32) * 8.0 - 4.0;
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

#[test]
#[ignore]
fn kquant_swiglu_q8_1_rows_matches_the_two_launches_bitwise() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_swiglu = g.kernel("moe_v41", "moe_v41_swiglu").unwrap();
    let k_rows = g.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16").unwrap();
    let k_fused = g
        .kernel(KQUANT_MODULE, "kquant_swiglu_q8_1_rows_bf16")
        .unwrap();
    for (rows, limit, weighted) in [(1u32, 0.0f32, false), (6, 1.5, true), (5, 0.0, true)] {
        let n = (rows * INTER) as usize;
        let gate = upload(g, &bf16_bytes(n, 0x51A7_0001 + rows));
        let up = upload(g, &bf16_bytes(n, 0x51A7_1001 + rows));
        let w_host: Vec<u8> = (0..rows)
            .flat_map(|r| (0.35 + 0.4 * r as f32).to_le_bytes())
            .collect();
        let w = if weighted {
            upload(g, &w_host)
        } else {
            DevicePtr(0)
        };
        let h_want = g.alloc(n * 2).unwrap();
        let y_want = g.alloc(kquant_q8_1_rows_bytes(rows, INTER)).unwrap();
        KernelLaunch::new(g, k_swiglu)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(w)
            .arg_ptr(h_want)
            .arg_u32(rows)
            .arg_u32(INTER)
            .arg_f32(limit)
            .launch(stream)
            .unwrap();
        kquant_q8_1_rows(g, k_rows, h_want, y_want, rows, INTER, stream).unwrap();
        let h_got = g.alloc(n * 2).unwrap();
        let y_got = g.alloc(kquant_q8_1_rows_bytes(rows, INTER)).unwrap();
        kquant_swiglu_q8_1_rows(
            g, k_fused, gate, up, w, h_got, y_got, rows, INTER, limit, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let (hw, hg) = (download(g, h_want, n * 2), download(g, h_got, n * 2));
        let yb = kquant_q8_1_rows_bytes(rows, INTER);
        let (yw, yg) = (download(g, y_want, yb), download(g, y_got, yb));
        assert!(
            hw == hg,
            "fused swiglu differs in h at rows={rows} limit={limit}"
        );
        assert!(
            yw == yg,
            "fused swiglu differs in q8_1 at rows={rows} limit={limit}"
        );
        assert!(
            hw.iter().any(|&b| b != 0),
            "degenerate all-zero h at rows={rows}"
        );
        println!(
            "  swiglu+q8_1 rows={rows} limit={limit} weighted={weighted}: {} + {} bytes identical",
            hw.len(),
            yw.len()
        );
        for p in [gate, up, h_want, y_want, h_got, y_got] {
            g.free(p).unwrap();
        }
        if weighted {
            g.free(w).unwrap();
        }
    }
}
