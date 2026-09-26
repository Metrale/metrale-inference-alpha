// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Probe: resolve the Qwen3.6-35B-A3B kernel target through `ptx_for_config`, as
//! the server does, then report whether each pipelined GEMM module is in the target's set and
//! whether each kernel resolves, printing the error when it does not. No model load.
//!
//! Owner: model-arch examples (kernel registry).
//! Invariants: none beyond the types. An unresolved kernel is printed, not fatal.

use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;

fn main() -> anyhow::Result<()> {
    // 2026-09-25: `serve_load` resolves its target with `ptx_for_config(model_type,
    // hidden_size, refs, pin)`; `(qwen3_6_moe, 2048)` is the pair the qwen3.6-35b-a3b
    // MODEL.toml declares.
    let set = metrale_kernels::ptx_for_config("qwen3_6_moe", 2048, &[], None)
        .expect("unambiguous")
        .expect("no ptx set for (qwen3_6_moe, 2048)");
    eprintln!(
        "target.model={} quant={} modules={}",
        set.target.model,
        set.target.quant,
        set.modules.len()
    );

    for m in [
        "w8a16_gemm_pipelined",
        "gemm",
        "w8a16_gemm_t",
        "moe_fp8_grouped_gemm",
    ] {
        let present = set.modules.iter().any(|(n, _)| *n == m);
        eprintln!("  module '{m}' in set: {present}");
    }

    // 2026-09-25: The backend gets the same module set the server would load.
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let probes = [
        ("w8a16_gemm_pipelined", "w8a16_gemm_pipelined"),
        ("gemm", "dense_gemm_bf16_pipelined"),
        ("w8a16_gemm_t", "w8a16_gemm_t_pipelined"),
        ("gemm", "dense_gemm_bf16_router"),
        ("gemm", "dense_gemm_bf16"),
        ("w8a16_gemm_t", "w8a16_gemm_t"),
    ];
    for (m, f) in probes {
        match gpu.kernel(m, f) {
            Ok(h) => eprintln!("OK    {m}::{f} -> handle {}", h.0),
            Err(e) => eprintln!("ERR   {m}::{f} -> {e}"),
        }
    }
    Ok(())
}
