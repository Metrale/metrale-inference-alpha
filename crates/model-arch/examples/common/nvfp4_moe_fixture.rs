// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Random NVFP4 and FP8 expert matrices with f64 host oracles, the kernel
//! sequences of the three grouped-decode configurations, and the small helpers
//! `nvfp4_moe_grouped_microtest` uses.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};

use super::legs::{Leg, hi_lo_rows};
use super::{H, INTER, TOP_K};

pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub(crate) fn unit(&mut self) -> f64 {
        self.next() as f64 / u32::MAX as f64
    }
}

pub(crate) fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|a| a.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

pub(crate) fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-09-27: E4M3 byte to value (no NaN bytes are drawn).
fn e4m3(b: u8) -> f64 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    let v = if e == 0 {
        m as f64 / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

/// 2026-09-27: A weight matrix `[n, k]` whose host copy can dot a row with an f64 vector.
pub(crate) trait HostDot {
    fn dot(&self, row: usize, x: &[f64]) -> f64;
}

/// 2026-09-27: One NVFP4 matrix `[n, k]` on host and device.
pub(crate) struct Mat {
    packed: Vec<u8>,
    scale: Vec<u8>,
    pub(crate) s2: f32,
    pub(crate) w: QuantizedWeight,
    k: usize,
}

impl Mat {
    pub(crate) fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Self> {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        // 2026-09-27: Scale bytes 0x28..0x3f: E4M3 values 1/32 .. 15/16.
        let scale: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x28 + (rng.next() % 24) as u8)
            .collect();
        let s2 = 0.02 + 0.01 * rng.unit() as f32;
        let mut w = QuantizedWeight::null();
        w.weight = upload(g, &packed)?;
        w.weight_scale = upload(g, &scale)?;
        w.weight_scale_2 = s2;
        Ok(Self {
            packed,
            scale,
            s2,
            w,
            k,
        })
    }
}

impl HostDot for Mat {
    fn dot(&self, row: usize, x: &[f64]) -> f64 {
        (0..self.k)
            .map(|c| {
                let byte = self.packed[(row * self.k + c) / 2];
                let nib = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
                E2M1[nib as usize]
                    * e4m3(self.scale[(row * self.k + c) / 16])
                    * self.s2 as f64
                    * x[c]
            })
            .sum()
    }
}

/// 2026-09-27: One FP8 E4M3 matrix `[n, k]` with f32 block scales `[ceil(n/128), ceil(k/128)]`.
pub(crate) struct Fp8Mat {
    bytes: Vec<u8>,
    scales: Vec<f32>,
    pub(crate) w: Fp8Weight,
    k: usize,
}

impl Fp8Mat {
    pub(crate) fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Self> {
        let bytes: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 120) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let scales: Vec<f32> = (0..n.div_ceil(128) * k.div_ceil(128))
            .map(|_| ((rng.next() % 16 + 1) as f32) / 4096.0)
            .collect();
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let w = Fp8Weight {
            weight: upload(g, &bytes)?,
            row_scale: upload(g, &scale_bytes)?,
            n: n as u32,
            k: k as u32,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };
        Ok(Self {
            bytes,
            scales,
            w,
            k,
        })
    }
}

impl HostDot for Fp8Mat {
    fn dot(&self, row: usize, x: &[f64]) -> f64 {
        let kb = self.k.div_ceil(128);
        (0..self.k)
            .map(|c| {
                e4m3(self.bytes[row * self.k + c])
                    * self.scales[(row / 128) * kb + c / 128] as f64
                    * x[c]
            })
            .sum()
    }
}

/// 2026-09-27: Device pointer tables of NVFP4 projections, one entry per expert.
pub(crate) fn nvfp4_table(g: &dyn GpuBackend, mats: &[&Mat]) -> Result<ops::Nvfp4ExpertTables> {
    let ptrs = |f: &dyn Fn(&Mat) -> u64| -> Vec<u8> {
        mats.iter().flat_map(|m| f(m).to_le_bytes()).collect()
    };
    Ok(ops::Nvfp4ExpertTables {
        packed_ptrs: upload(g, &ptrs(&|m| m.w.weight.0))?,
        scale_ptrs: upload(g, &ptrs(&|m| m.w.weight_scale.0))?,
        scale2_vals: upload(
            g,
            &mats
                .iter()
                .flat_map(|m| m.s2.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?,
    })
}

/// 2026-09-27: Device pointer tables (weights, scales) of FP8 projections, one entry per
/// expert.
pub(crate) fn fp8_table(g: &dyn GpuBackend, mats: &[&Fp8Mat]) -> Result<(DevicePtr, DevicePtr)> {
    let ptrs = |f: &dyn Fn(&Fp8Mat) -> u64| -> Vec<u8> {
        mats.iter().flat_map(|m| f(m).to_le_bytes()).collect()
    };
    Ok((
        upload(g, &ptrs(&|m| m.w.weight.0))?,
        upload(g, &ptrs(&|m| m.w.row_scale.0))?,
    ))
}

pub(crate) fn bf16_to_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect()
}

pub(crate) fn f32s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

/// 2026-09-27: The SiLU product the gate+up kernels write, from the f64 projections rounded
/// to BF16 as they round them.
pub(crate) fn silu_product(
    gate: &dyn HostDot,
    up: &dyn HostDot,
    x: &[f64],
    inter: usize,
) -> Vec<f64> {
    (0..inter)
        .map(|c| {
            let gv = bf16::from_f64(gate.dot(c, x)).to_f64();
            let uv = bf16::from_f64(up.dot(c, x)).to_f64();
            gv / (1.0 + (-gv).exp()) * uv
        })
        .collect()
}

/// 2026-09-27: `got` within 2 % of the largest |want| of its row, element by element.
pub(crate) fn close(got: &[f64], want: &[f64], what: &str) -> Result<()> {
    let scale = want.iter().fold(1e-30f64, |a, v| a.max(v.abs()));
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        ensure!(
            x.is_finite() && (x - y).abs() <= 0.02 * scale,
            "{what}: element {i} is {x}, the reference {y} (row max {scale})"
        );
    }
    Ok(())
}

pub(crate) struct Kernels {
    pub(crate) gate_up: KernelHandle,
    pub(crate) down: KernelHandle,
    pub(crate) gate_up_tc: KernelHandle,
    pub(crate) down_tc: KernelHandle,
    pub(crate) fp8_gate_up: KernelHandle,
    pub(crate) fp8_down: KernelHandle,
    pub(crate) sort: KernelHandle,
    pub(crate) blend: KernelHandle,
}

/// 2026-09-27: The weights: NVFP4 routed gate/up/down, FP8 routed down, and the shared expert
/// in both formats.
pub(crate) struct Weights {
    pub(crate) gate: Vec<Mat>,
    pub(crate) up: Vec<Mat>,
    pub(crate) down: Vec<Mat>,
    pub(crate) down8: Vec<Fp8Mat>,
    pub(crate) sh: [Mat; 3],
    pub(crate) sh8: [Fp8Mat; 3],
    pub(crate) gate_t: ops::Nvfp4ExpertTables,
    pub(crate) up_t: ops::Nvfp4ExpertTables,
    pub(crate) down_t: ops::Nvfp4ExpertTables,
    pub(crate) down8_t: (DevicePtr, DevicePtr),
}

/// 2026-09-27: Device buffers shared by every run.
pub(crate) struct Bufs {
    pub(crate) input: DevicePtr,
    pub(crate) idx: DevicePtr,
    pub(crate) slot_w: DevicePtr,
    pub(crate) sh_gate_vec: DevicePtr,
    pub(crate) sort: ops::Fp8GroupedSortOut,
    pub(crate) act: DevicePtr,
    pub(crate) sh_act: DevicePtr,
    pub(crate) down_out: DevicePtr,
    pub(crate) sh_out: DevicePtr,
    pub(crate) output: DevicePtr,
}

/// 2026-09-27: What one run returns: blended output, routed act, routed down output, shared
/// output, sorted token ids, expert offsets, and the expert kernels' mean time in us.
pub(crate) type Run = (Vec<u8>, Vec<f64>, Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>, f64);

/// 2026-09-27: The expert kernels of `leg` for the first `m` rows, in the order and with the
/// arguments `forward_nvfp4_grouped_decode` uses.
fn experts(
    g: &dyn GpuBackend,
    k: &Kernels,
    w: &Weights,
    b: &Bufs,
    leg: Leg,
    m: usize,
    e: usize,
) -> Result<()> {
    let s = g.default_stream();
    let n = m as u32;
    let cap = ops::fp8_grouped_active_cap(n, TOP_K as u32, e as u32);
    let so = &b.sort;
    let null = DevicePtr::NULL;
    if leg.fp8_shared() {
        ops::moe_expert_gate_up_act_fp8_grouped(
            g,
            k.fp8_gate_up,
            ops::FP8_GROUPED_GATE_UP_SCALAR,
            b.input,
            null,
            null,
            null,
            null,
            b.act,
            so.expert_offsets,
            so.sorted_token_ids,
            so.active_experts,
            so.active_count,
            &w.sh8[0].w,
            &w.sh8[1].w,
            b.sh_act,
            INTER as u32,
            H as u32,
            0,
            n,
            s,
        )?;
    }
    let nv_shared = if leg.fp8_shared() { 0 } else { n };
    let (gu, gu_geo, dn, dn_geo) = if leg.tc() {
        (
            k.gate_up_tc,
            ops::NVFP4_GROUPED_GATE_UP_TC,
            k.down_tc,
            ops::NVFP4_GROUPED_DOWN_TC,
        )
    } else {
        (
            k.gate_up,
            ops::NVFP4_GROUPED_GATE_UP_SCALAR,
            k.down,
            ops::NVFP4_GROUPED_DOWN_SCALAR,
        )
    };
    ops::moe_expert_gate_up_act_nvfp4_grouped(
        g,
        gu,
        gu_geo,
        b.input,
        w.gate_t,
        w.up_t,
        b.act,
        so.expert_offsets,
        so.sorted_token_ids,
        so.active_experts,
        so.active_count,
        &w.sh[0].w,
        &w.sh[1].w,
        b.sh_act,
        INTER as u32,
        H as u32,
        cap,
        nv_shared,
        s,
    )?;
    if leg.fp8_down() {
        return ops::moe_expert_down_act_fp8_grouped(
            g,
            k.fp8_down,
            ops::FP8_GROUPED_DOWN_SCALAR,
            b.act,
            w.down8_t.0,
            w.down8_t.1,
            b.down_out,
            so.expert_offsets,
            so.active_experts,
            so.active_count,
            b.sh_act,
            &w.sh8[2].w,
            b.sh_out,
            H as u32,
            INTER as u32,
            cap,
            n,
            s,
        );
    }
    if leg.fp8_shared() {
        ops::moe_expert_down_act_fp8_grouped(
            g,
            k.fp8_down,
            ops::FP8_GROUPED_DOWN_SCALAR,
            b.act,
            null,
            null,
            b.down_out,
            so.expert_offsets,
            so.active_experts,
            so.active_count,
            b.sh_act,
            &w.sh8[2].w,
            b.sh_out,
            H as u32,
            INTER as u32,
            0,
            n,
            s,
        )?;
    }
    ops::moe_expert_down_act_nvfp4_grouped(
        g,
        dn,
        dn_geo,
        b.act,
        w.down_t,
        b.down_out,
        so.expert_offsets,
        so.active_experts,
        so.active_count,
        b.sh_act,
        &w.sh[2].w,
        b.sh_out,
        H as u32,
        INTER as u32,
        cap,
        nv_shared,
        s,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    g: &dyn GpuBackend,
    k: &Kernels,
    w: &Weights,
    b: &Bufs,
    leg: Leg,
    m: usize,
    e: usize,
    iters: usize,
) -> Result<Run> {
    let s = g.default_stream();
    let (te, n) = (m * TOP_K, m as u32);
    ops::moe_fp8_grouped_sort(
        g,
        k.sort,
        ops::Fp8GroupedSortOut { ..b.sort },
        b.idx,
        te as u32,
        e as u32,
        TOP_K as u32,
        s,
    )?;
    experts(g, k, w, b, leg, m, e)?;
    g.synchronize(s)?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        experts(g, k, w, b, leg, m, e)?;
    }
    g.synchronize(s)?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64;
    ops::moe_weighted_sum_blend_fp8_grouped(
        g,
        k.blend,
        b.output,
        b.down_out,
        b.slot_w,
        b.sort.token_to_perm,
        b.sh_out,
        b.input,
        b.sh_gate_vec,
        H as u32,
        TOP_K as u32,
        H as u32,
        n,
        s,
    )?;
    g.synchronize(s)?;
    let u32s = |v: Vec<u8>| -> Vec<u32> {
        v.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    Ok((
        read(g, b.output, m * H * 2)?,
        if leg.tc() {
            hi_lo_rows(&read(g, b.act, te * INTER * 4)?, INTER)
        } else {
            f32s(&read(g, b.act, te * INTER * 4)?)
        },
        read(g, b.down_out, te * H * 2)?,
        read(g, b.sh_out, m * H * 2)?,
        u32s(read(g, b.sort.sorted_token_ids, te * 4)?),
        u32s(read(g, b.sort.expert_offsets, (e + 1) * 4)?),
        us,
    ))
}
