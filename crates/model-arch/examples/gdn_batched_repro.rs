// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One launch of `gated_delta_rule_prefill_wy64_batched` on synthetic inputs
//! at Qwen3.6-35B-A3B GDN dimensions, small enough to run under compute-sanitizer.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types. The run only reports that the launch completed;
//! it checks no output values.
//!
//! `BATCH` (default 2), `SEQ` (default 96) and `SMEM` (bytes; default computed below) set
//! the launch. The launch arguments and geometry are those of
//! `ops::gdn_prefill_persistent_smem_batched`.
//!   compute-sanitizer --tool memcheck \
//!     cargo run -p metrale-model-arch --release --example gdn_batched_repro

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

// 2026-09-25: GDN key/value heads and head dims of kernels/gb10/qwen3.6-35b-a3b/MODEL.toml.
const NK: usize = 16;
const NV: usize = 32;
const KD: usize = 128;
const VD: usize = 128;

fn main() -> Result<()> {
    let batch_size: u32 = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let seq_len: u32 = std::env::var("SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    let bs = batch_size as usize;
    let sl = seq_len as usize;

    let key_dim = NK * KD;
    let value_dim = NV * VD;
    let conv_dim = key_dim * 2 + value_dim;
    let gb_stride = NV * 2;
    let h_numel = NV * KD * VD;

    eprintln!(
        "gdn_batched_repro: batch={batch_size} seq_len={seq_len} key_dim={key_dim} \
         value_dim={value_dim} conv_dim={conv_dim} h_numel={h_numel}"
    );

    let set = metrale_kernels::ptx_for_config("qwen3_6_moe", 2048, &[], None)
        .expect("unambiguous")
        .expect("no ptx set");
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k: KernelHandle = g.kernel(
        "gated_delta_rule_wy64_prefill",
        "gated_delta_rule_prefill_wy64_batched",
    )?;

    // 2026-09-25: qkv is [bs*sl, conv_dim] BF16, each row [Q (key_dim) | K (key_dim) |
    // V (value_dim)].
    let qkv_n = bs * sl * conv_dim;
    let qkv_host: Vec<u8> = (0..qkv_n)
        .flat_map(|i| {
            bf16::from_f32(((i % 17) as f32 - 8.0) * 0.01)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let qkv = g.alloc(qkv_host.len())?;
    g.copy_h2d(&qkv_host, qkv)?;

    // 2026-09-25: gate_beta is [bs*sl, NV*2] FP32, each row [gate (NV) | beta (NV)].
    let gb_n = bs * sl * gb_stride;
    let gb_host: Vec<u8> = (0..gb_n)
        .flat_map(|i| (0.5f32 + ((i % 7) as f32) * 0.01).to_le_bytes())
        .collect();
    let gate_beta = g.alloc(gb_host.len())?;
    g.copy_h2d(&gb_host, gate_beta)?;

    let out = g.alloc(bs * sl * value_dim * 2)?;
    g.memset(out, 0, bs * sl * value_dim * 2)?;

    let mut h_ptrs: Vec<u64> = Vec::with_capacity(bs);
    for _ in 0..bs {
        let h = g.alloc(h_numel * 4)?;
        g.memset(h, 0, h_numel * 4)?;
        h_ptrs.push(h.0);
    }
    let hp_host: Vec<u8> = h_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let h_state_ptrs = g.alloc(hp_host.len())?;
    g.copy_h2d(&hp_host, h_state_ptrs)?;

    let q_ptr = qkv;
    let k_ptr = qkv.offset(key_dim * 2);
    let v_ptr = qkv.offset(key_dim * 2 * 2);
    let gate_ptr = gate_beta;
    let beta_ptr = gate_beta.offset(NV * 4);

    let default_smem = (KD * VD * 4 + 32 * KD * 2 + 32 * KD * 2 + 32 * 32 * 4 + 256) as u32;
    let smem: u32 = std::env::var("SMEM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_smem);

    eprintln!(
        "launching gated_delta_rule_prefill_wy64_batched grid=[{NV},{batch_size},1] smem={smem}"
    );
    KernelLaunch::new(g, k)
        .grid([NV as u32, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(q_ptr)
        .arg_ptr(k_ptr)
        .arg_ptr(v_ptr)
        .arg_ptr(gate_ptr)
        .arg_ptr(beta_ptr)
        .arg_ptr(out)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(conv_dim as u32)
        .arg_u32(conv_dim as u32)
        .arg_u32(gb_stride as u32)
        .launch(0)?;

    g.synchronize(0)?;
    eprintln!("gdn_batched_repro: kernel completed cleanly (no fault at batch={batch_size})");
    Ok(())
}
