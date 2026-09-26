// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DEBUG-level device dumps of `MoeLayer::forward`: the routed experts and
//! weights, and the first eight expert and shared-expert intermediates. The caller checks
//! `tracing::enabled!(DEBUG)` and `!ctx.graph_capture` first.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

pub(super) fn debug_routing(
    ctx: &ForwardContext,
    indices_dev: DevicePtr,
    weights_dev: DevicePtr,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    ctx.gpu.synchronize(stream)?;
    let k = top_k as usize;
    let mut idx_buf = vec![0u8; k * 4];
    let mut wt_buf = vec![0u8; k * 4];
    ctx.gpu.copy_d2h(indices_dev, &mut idx_buf)?;
    ctx.gpu.copy_d2h(weights_dev, &mut wt_buf)?;
    let indices: Vec<u32> = (0..k)
        .map(|i| {
            u32::from_le_bytes([
                idx_buf[i * 4],
                idx_buf[i * 4 + 1],
                idx_buf[i * 4 + 2],
                idx_buf[i * 4 + 3],
            ])
        })
        .collect();
    let weights: Vec<f32> = (0..k)
        .map(|i| {
            f32::from_le_bytes([
                wt_buf[i * 4],
                wt_buf[i * 4 + 1],
                wt_buf[i * 4 + 2],
                wt_buf[i * 4 + 3],
            ])
        })
        .collect();
    tracing::info!(
        target: "metrale_model_layers::layers::moe::forward",
        "  MoE experts: {:?}, weights: {:.4?}",
        indices,
        weights
    );
    Ok(())
}

pub(super) fn debug_expert_intermediates(
    ctx: &ForwardContext,
    expert_gate_out: DevicePtr,
    expert_up_out: DevicePtr,
    shared_gate_scratch: DevicePtr,
    shared_up_scratch: DevicePtr,
    stream: u64,
) -> Result<()> {
    ctx.gpu.synchronize(stream)?;
    let mut gate_buf = vec![0u8; 16];
    ctx.gpu.copy_d2h(expert_gate_out, &mut gate_buf)?;
    let gate_vals: Vec<f32> = (0..8)
        .map(|i| {
            let bits = u16::from_le_bytes([gate_buf[i * 2], gate_buf[i * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect();
    tracing::info!(
        target: "metrale_model_layers::layers::moe::forward",
        "  MoE gate_out[slot0,0..8]: {:?}",
        gate_vals
    );
    let mut up_buf = vec![0u8; 16];
    ctx.gpu.copy_d2h(expert_up_out, &mut up_buf)?;
    let up_vals: Vec<f32> = (0..8)
        .map(|i| {
            let bits = u16::from_le_bytes([up_buf[i * 2], up_buf[i * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect();
    tracing::info!(
        target: "metrale_model_layers::layers::moe::forward",
        "  MoE up_out[slot0,0..8]: {:?}",
        up_vals
    );
    let mut sg_buf = vec![0u8; 16];
    ctx.gpu.copy_d2h(shared_gate_scratch, &mut sg_buf)?;
    let sg_vals: Vec<f32> = (0..8)
        .map(|i| {
            let bits = u16::from_le_bytes([sg_buf[i * 2], sg_buf[i * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect();
    tracing::info!(
        target: "metrale_model_layers::layers::moe::forward",
        "  MoE shared_gate_scratch[0..8]: {:?}",
        sg_vals
    );
    let mut su_buf = vec![0u8; 16];
    ctx.gpu.copy_d2h(shared_up_scratch, &mut su_buf)?;
    let su_vals: Vec<f32> = (0..8)
        .map(|i| {
            let bits = u16::from_le_bytes([su_buf[i * 2], su_buf[i * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect();
    tracing::info!(
        target: "metrale_model_layers::layers::moe::forward",
        "  MoE shared_up_scratch[0..8]: {:?}",
        su_vals
    );
    Ok(())
}
