// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `METRALE_DUMP_EXPERT_IDS=1` diagnostic dumps of the MoE routing
//! and outputs, called from the MoE prefill paths (`forward_prefill*.rs`) and
//! the batched decode paths (`forward_batched*.rs`).
//!
//! Owner: model-layers (MoE).
//! Invariants: when the variable is not exactly `1`, every helper returns
//! before synchronizing or copying. When it is, each synchronizes the stream
//! before reading, so the values are post-kernel.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

#[inline]
pub fn enabled() -> bool {
    std::env::var("METRALE_DUMP_EXPERT_IDS").ok().as_deref() == Some("1")
}

/// 2026-09-25: Read `num_elements` BF16 values at `ptr + offset_bytes` as `f32`.
/// A failed copy is ignored and reads as zeros.
fn read_bf16_row(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    offset_bytes: usize,
    num_elements: usize,
) -> Vec<f32> {
    let mut buf = vec![0u8; num_elements * 2];
    let _ = gpu.copy_d2h(ptr.offset(offset_bytes), &mut buf);
    buf.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// 2026-09-25: L2 norm and first five values of the last token's BF16 row.
fn last_tok_stats(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, width: usize) -> (f32, Vec<f32>) {
    let offset = (n - 1) * width * 2;
    let v = read_bf16_row(gpu, ptr, offset, width);
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let first5 = v.iter().take(5).copied().collect();
    (mag, first5)
}

/// 2026-09-25: Log the last token's router input: L2 norm and first five values.
pub fn dump_gate_input(
    gpu: &dyn GpuBackend,
    stream: u64,
    router_in: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, router_in, n as usize, h as usize);
    tracing::info!(
        "METRALE_GATE_INPUT last_tok: |x|={:.4}  first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// 2026-09-25: Log the last token's top-10 BF16 gate logits, before top-k, with
/// their mean and standard deviation.
pub fn dump_gate_logits(
    gpu: &dyn GpuBackend,
    stream: u64,
    gate_logits: DevicePtr,
    n: u32,
    num_experts: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (n - 1) as usize * num_experts as usize * 2;
    let logits = read_bf16_row(gpu, gate_logits, offset, num_experts as usize);
    let mut idx_val: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top10: Vec<(usize, f32)> = idx_val.iter().take(10).copied().collect();
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let var: f32 = logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;
    tracing::info!(
        "METRALE_GATE_LOGITS last_tok: top10_(idx,val)={:?} mean={:.4} std={:.4}",
        top10,
        mean,
        var.sqrt()
    );
    Ok(())
}

/// 2026-09-25: Log the last token's top-k expert indices and routing weights.
pub fn dump_expert_ids(
    gpu: &dyn GpuBackend,
    stream: u64,
    indices_dev: DevicePtr,
    weights_dev: DevicePtr,
    n: u32,
    top_k: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (n - 1) as usize * top_k as usize * 4;
    let mut idx_buf = vec![0u8; top_k as usize * 4];
    let mut w_buf = vec![0u8; top_k as usize * 4];
    let _ = gpu.copy_d2h(indices_dev.offset(offset), &mut idx_buf);
    let _ = gpu.copy_d2h(weights_dev.offset(offset), &mut w_buf);
    let ids: Vec<u32> = idx_buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let ws: Vec<f32> = w_buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    tracing::info!(
        "METRALE_EXPERT_IDS last_tok: indices={:?} weights={:?} sum={:.4}",
        ids,
        ws,
        ws.iter().sum::<f32>()
    );
    Ok(())
}

/// 2026-09-25: Log per-expert token counts from `expert_offsets`, once per
/// `GpuBackend`. `truncated` reports whether the busiest expert exceeds
/// `max_m_tiles * 64` rows.
pub fn dump_expert_load(
    gpu: &dyn GpuBackend,
    stream: u64,
    expert_offsets: DevicePtr,
    num_experts: usize,
    num_tokens: usize,
    avg_per_expert: usize,
    max_m_tiles: u32,
) {
    if !enabled() {
        return;
    }
    // 2026-09-25: `OpCache::once` latches per backend; a new backend logs again.
    if gpu.op_cache().once("dump:moe_expert_load") {
        if gpu.synchronize(stream).is_err() {
            return;
        }
        let dump_n = num_experts + 1;
        let mut eo_buf = vec![0u8; dump_n * 4];
        let _ = gpu.copy_d2h(expert_offsets, &mut eo_buf);
        let eo: Vec<u32> = eo_buf
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let counts: Vec<u32> = (0..num_experts).map(|i| eo[i + 1] - eo[i]).collect();
        let max_cnt = *counts.iter().max().unwrap_or(&0);
        let min_cnt = *counts.iter().min().unwrap_or(&0);
        let max_idx = counts.iter().position(|&x| x == max_cnt).unwrap_or(0);
        let kernel_max = max_m_tiles * 64;
        tracing::info!(
            "METRALE_EXPERT_LOAD: n_tokens={} avg={} max={} (expert {}) min={} max_m_tiles={} kernel_cap={} truncated={}",
            num_tokens,
            avg_per_expert,
            max_cnt,
            max_idx,
            min_cnt,
            max_m_tiles,
            kernel_max,
            max_cnt > kernel_max
        );
    }
}

/// 2026-09-25: Log the last token's routed-only MoE output, before the shared
/// expert is blended in.
pub fn dump_routed_only(
    gpu: &dyn GpuBackend,
    stream: u64,
    output: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, output, n as usize, h as usize);
    tracing::info!(
        "METRALE_ROUTED_ONLY last_tok: |x|={:.4} first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// 2026-09-25: Log the last token's shared-expert output, before its sigmoid gate.
pub fn dump_shared_out(
    gpu: &dyn GpuBackend,
    stream: u64,
    shared_down_out: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, shared_down_out, n as usize, h as usize);
    tracing::info!(
        "METRALE_SHARED_OUT last_tok: |x|={:.4} first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// 2026-09-25: Log the last token's shared-expert gate: the dot product with
/// `gate_weight` and its sigmoid, computed on the host.
pub fn dump_shared_gate(
    gpu: &dyn GpuBackend,
    stream: u64,
    input: DevicePtr,
    gate_weight: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (n - 1) as usize * h as usize * 2;
    let v_in = read_bf16_row(gpu, input, offset, h as usize);
    let v_g = read_bf16_row(gpu, gate_weight, 0, h as usize);
    let dot: f32 = v_in.iter().zip(v_g.iter()).map(|(a, b)| a * b).sum();
    let sig = 1.0 / (1.0 + (-dot).exp());
    tracing::info!(
        "METRALE_SHARED_GATE last_tok: dot={:.4} sigmoid={:.6}",
        dot,
        sig
    );
    Ok(())
}

/// 2026-09-25: Log the last token's MoE output, after the shared-expert blend
/// when the pass runs one.
pub fn dump_moe_out(
    gpu: &dyn GpuBackend,
    stream: u64,
    output: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, output, n as usize, h as usize);
    tracing::info!(
        "METRALE_MOE_OUT last_tok: |x|={:.4} first5={:?}",
        mag,
        first5
    );
    Ok(())
}
