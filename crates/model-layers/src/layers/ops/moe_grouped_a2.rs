// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the MoE expert sort, the unpermute reduce and the shared-expert blend, and the host side of the CUTLASS grouped NVFP4 gate_up and down GEMMs.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The two CUTLASS calls take the weight tables from [`MoeCutlassHostTables`]; per call they
//!   copy at most `expert_offsets` from the device.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: Host copies of the per-expert pointer and scale tables for the CUTLASS grouped
/// path, which takes them on the host. `MoeLayer::build_cutlass_grouped_sfb` builds them at load
/// into the layer's `cutlass_grouped_host`, so they are dropped with the layer.
pub struct MoeCutlassHostTables {
    pub gate_packed: Vec<u64>,
    pub gate_sfb: Vec<u64>,
    pub gate_scale2: Vec<f32>,
    pub up_packed: Vec<u64>,
    pub up_sfb: Vec<u64>,
    pub up_scale2: Vec<f32>,
    /// 2026-09-25: `None` when the layer has no down-projection scale table; the grouped down
    /// branch of `forward_prefill_routed` is then not taken.
    pub down: Option<MoeCutlassDownHostTables>,
}

/// 2026-09-25: The down-projection tables of [`MoeCutlassHostTables`].
pub struct MoeCutlassDownHostTables {
    pub packed: Vec<u64>,
    pub sfb: Vec<u64>,
    pub scale2: Vec<f32>,
}

/// 2026-09-25: One blocking device-to-host copy of an `[n]` u64 pointer table.
pub fn read_expert_ptrs_u64(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u64>> {
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(8)
        .map(|x| u64::from_le_bytes(x.try_into().expect("8")))
        .collect())
}

/// 2026-09-25: One blocking device-to-host copy of an `[n]` f32 scale table.
pub fn read_expert_scales_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().expect("4")))
        .collect())
}

impl MoeCutlassHostTables {
    /// 2026-09-25: Snapshot the grouped-path tables. The SFB pointer vectors are taken by value
    /// because `build_cutlass_grouped_sfb` builds them on the host; the packed and scale2 tables
    /// are copied from the device here.
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot(
        gpu: &dyn GpuBackend,
        num_experts: usize,
        gate_packed: DevicePtr,
        gate_sfb: Vec<u64>,
        gate_scale2: DevicePtr,
        up_packed: DevicePtr,
        up_sfb: Vec<u64>,
        up_scale2: DevicePtr,
        down: Option<(DevicePtr, Vec<u64>, DevicePtr)>,
    ) -> Result<Self> {
        Ok(Self {
            gate_packed: read_expert_ptrs_u64(gpu, gate_packed, num_experts)?,
            gate_sfb,
            gate_scale2: read_expert_scales_f32(gpu, gate_scale2, num_experts)?,
            up_packed: read_expert_ptrs_u64(gpu, up_packed, num_experts)?,
            up_sfb,
            up_scale2: read_expert_scales_f32(gpu, up_scale2, num_experts)?,
            down: match down {
                Some((packed, sfb, scale2)) => Some(MoeCutlassDownHostTables {
                    packed: read_expert_ptrs_u64(gpu, packed, num_experts)?,
                    sfb,
                    scale2: read_expert_scales_f32(gpu, scale2, num_experts)?,
                }),
                None => None,
            },
        })
    }
}

/// 2026-09-25: Counting sort of the `total_expanded` (token, slot) routes by expert, in one block:
/// writes `sorted_token_ids` and `sorted_expert_ids` in expert order, `expert_offsets` (the
/// `[num_experts + 1]` prefix sum) and `token_to_perm` (each route's sorted position).
#[allow(clippy::too_many_arguments)]
pub fn moe_sort_by_expert(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    topk_ids: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    expert_offsets: DevicePtr,
    token_to_perm: DevicePtr,
    total_expanded: u32,
    num_experts: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(expert_offsets)
        .arg_ptr(token_to_perm)
        .arg_u32(total_expanded)
        .arg_u32(num_experts)
        .arg_u32(topk)
        .launch(stream)
}

/// 2026-09-25: For each token, `output[t] = sum_k topk_weights[t, k] * expert_output[token_to_perm[t, k]]`,
/// one block per token.
#[allow(clippy::too_many_arguments)]
pub fn moe_unpermute_reduce_indexed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_output: DevicePtr,
    output: DevicePtr,
    token_to_perm: DevicePtr,
    topk_weights: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_output)
        .arg_ptr(output)
        .arg_ptr(token_to_perm)
        .arg_ptr(topk_weights)
        .arg_u32(hidden_size)
        .arg_u32(num_tokens)
        .arg_u32(topk)
        .launch(stream)
}

/// 2026-09-25: `output[t] += sigmoid(dot(normed[t], gate_weight)) * shared_out[t]`, one block per
/// token. A null `gate_weight` blends at weight 1.
pub fn moe_batched_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    shared_out: DevicePtr,
    normed: DevicePtr,
    gate_weight: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(shared_out)
        .arg_ptr(normed)
        .arg_ptr(gate_weight)
        .arg_u32(hidden_size)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: Single-launch CUTLASS grouped NVFP4 gate_up GEMM through
/// [`metrale_gpu_runtime::cutlass::nvfp4_grouped_gate_up_fused`], with the weights from `host`.
/// `expert_offsets` is the device i32 `[num_experts + 1]` prefix sum; the host entry needs it on
/// the host, so it is copied and the stream synchronised first. Returns that host copy, which
/// [`moe_grouped_down_cutlass`] accepts to skip its own copy and synchronise.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gate_up_cutlass(
    gpu: &dyn GpuBackend,
    host: &MoeCutlassHostTables,
    a: DevicePtr,
    sorted_token_ids: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    inter: u32,
    hidden: u32,
    stream: u64,
) -> Result<Vec<i32>> {
    let num_experts = host.gate_packed.len();
    let mut off_raw = vec![0u8; (num_experts + 1) * 4];
    gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
    gpu.synchronize(stream)?;
    let eoff: Vec<i32> = off_raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    metrale_gpu_runtime::cutlass::nvfp4_grouped_gate_up_fused(
        a.0,
        sorted_token_ids.0,
        &host.gate_packed,
        &host.gate_sfb,
        &host.gate_scale2,
        &host.up_packed,
        &host.up_sfb,
        &host.up_scale2,
        c_gate.0,
        c_up.0,
        &eoff,
        inter,
        hidden,
        stream,
    )?;
    Ok(eoff)
}

/// 2026-09-25: Single-launch CUTLASS grouped NVFP4 down GEMM through
/// [`metrale_gpu_runtime::cutlass::nvfp4_grouped_down`]. `a` is the expert-sorted post-SiLU
/// intermediate `[total_expanded, inter]`; `expert_offsets` is the device prefix sum.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_down_cutlass(
    gpu: &dyn GpuBackend,
    // 2026-09-25: The host `expert_offsets` returned by the paired gate_up call. When supplied,
    // no device copy or synchronise happens here.
    eoff_cached: Option<&[i32]>,
    host: &MoeCutlassDownHostTables,
    a: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    hidden: u32,
    inter: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = host.packed.len();
    let eoff: Vec<i32> = if let Some(e) = eoff_cached {
        e.to_vec()
    } else {
        let mut off_raw = vec![0u8; (num_experts + 1) * 4];
        gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
        gpu.synchronize(stream)?;
        off_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4")))
            .collect()
    };
    metrale_gpu_runtime::cutlass::nvfp4_grouped_down(
        a.0,
        &host.packed,
        &host.sfb,
        &host.scale2,
        c.0,
        &eoff,
        hidden,
        inter,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

    fn upload_u64(gpu: &MockGpuBackend, vals: &[u64]) -> DevicePtr {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, p).unwrap();
        p
    }

    fn upload_f32(gpu: &MockGpuBackend, vals: &[f32]) -> DevicePtr {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, p).unwrap();
        p
    }

    /// 2026-09-25: A read returns what the device holds at the time of the call: new contents
    /// written at the same address are observed, so nothing is cached by address.
    #[test]
    fn read_reflects_current_device_contents_at_a_reused_address() {
        let gpu = MockGpuBackend::new();
        let old_model = [0x1111_u64, 0x2222, 0x3333];
        let p = upload_u64(&gpu, &old_model);
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 3).unwrap(), old_model);

        let new_model = [0xaaaa_u64, 0xbbbb, 0xcccc];
        let bytes: Vec<u8> = new_model.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&bytes, p).unwrap();
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 3).unwrap(), new_model);

        let old_scales = [1.0_f32, 2.0];
        let ps = upload_f32(&gpu, &old_scales);
        assert_eq!(read_expert_scales_f32(&gpu, ps, 2).unwrap(), old_scales);
        let new_scales = [3.0_f32, 4.0];
        let sbytes: Vec<u8> = new_scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&sbytes, ps).unwrap();
        assert_eq!(read_expert_scales_f32(&gpu, ps, 2).unwrap(), new_scales);
    }

    /// 2026-09-25: A read of `n` elements returns exactly `n` elements.
    #[test]
    fn read_honors_the_requested_length() {
        let gpu = MockGpuBackend::new();
        let vals = [1_u64, 2, 3, 4];
        let p = upload_u64(&gpu, &vals);
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 2).unwrap(), vals[..2]);
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 4).unwrap(), vals);
    }

    /// 2026-09-25: Each of the nine tables `snapshot` takes lands in its own field; a swapped
    /// argument of the same type would still compile.
    #[test]
    fn snapshot_maps_every_table_to_its_field() {
        let gpu = MockGpuBackend::new();
        let n = 2;
        let gate_packed = upload_u64(&gpu, &[10, 11]);
        let gate_scale2 = upload_f32(&gpu, &[0.5, 0.25]);
        let up_packed = upload_u64(&gpu, &[20, 21]);
        let up_scale2 = upload_f32(&gpu, &[2.0, 4.0]);
        let down_packed = upload_u64(&gpu, &[30, 31]);
        let down_scale2 = upload_f32(&gpu, &[8.0, 16.0]);

        let t = MoeCutlassHostTables::snapshot(
            &gpu,
            n,
            gate_packed,
            vec![100, 101],
            gate_scale2,
            up_packed,
            vec![200, 201],
            up_scale2,
            Some((down_packed, vec![300, 301], down_scale2)),
        )
        .unwrap();

        assert_eq!(t.gate_packed, [10, 11]);
        assert_eq!(t.gate_sfb, [100, 101]);
        assert_eq!(t.gate_scale2, [0.5, 0.25]);
        assert_eq!(t.up_packed, [20, 21]);
        assert_eq!(t.up_sfb, [200, 201]);
        assert_eq!(t.up_scale2, [2.0, 4.0]);
        let d = t.down.expect("down tables were supplied");
        assert_eq!(d.packed, [30, 31]);
        assert_eq!(d.sfb, [300, 301]);
        assert_eq!(d.scale2, [8.0, 16.0]);
    }
}
