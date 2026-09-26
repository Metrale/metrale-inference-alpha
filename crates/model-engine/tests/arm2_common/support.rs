// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Helpers shared by the native-MXFP4 (E8M0) kernel tests
//! `arm2_leg2_decode.rs` and `arm2_leg2_prefill.rs`.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Both binaries include this file with `#[path]` and each uses a subset of
//! it, so `dead_code` is allowed. The decode kernel is a GEMV in f32 (BF16
//! activations widened, weights = E2M1 value x scale), so its test compares
//! against the host GEMV `host_gemv` within one BF16 ULP. The prefill tests
//! compare each `_e8m0` kernel bit for bit with its NVFP4 twin, fed the same
//! nibbles and scales that encode the same power of two: E4M3 encodes 2^e
//! exactly for e in [-6, 8], both 16-row halves of a 32-row group get the same
//! scale, and scale2 is 1.0, so `mx_block_scale` returns the same f32 for
//! both formats.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops;

/// 2026-09-25: Create the CUDA backend on device 0 with this build's PTX
/// modules, and one stream on it.
pub fn setup() -> Result<(MetraleCudaBackend, u64)> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let st = backend.create_stream()?;
    Ok((backend, st))
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub fn nibble(&mut self) -> u8 {
        (self.next_u64() & 0xF) as u8
    }
}

// 2026-09-25: E2M1 values, the same table as the kernels' E2M1_LUT_T (decode)
// and E2M1_LUT_MOE (prefill).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// 2026-09-25: Round-to-nearest-even f32 -> BF16 bits; a NaN stays a quiet NaN.
pub fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// 2026-09-25: E8M0 byte -> f32 = 2^(sb-127), with sb 0 and 255 -> 0. The same
// bits as `metrale_core::mxfp4_e8m0::fp8_e8m0_to_f32` and the kernels'
// `mx_block_scale<true>`.
pub fn e8m0_to_f32(sb: u8) -> f32 {
    if sb == 0 || sb == 255 {
        0.0
    } else {
        f32::from_bits((sb as u32) << 23)
    }
}
// 2026-09-25: E4M3 byte for exactly 2^e (normal, mantissa 0), e in [-6, 8].
pub fn e4m3_pow2_byte(e: i32) -> u8 {
    assert!((-6..=8).contains(&e), "e {e} outside E4M3 exact-pow range");
    (((e + 7) as u8) & 0x0F) << 3
}

pub fn up_u8(g: &dyn GpuBackend, v: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(v.len().max(1))?;
    if !v.is_empty() {
        g.copy_h2d(v, p)?;
    }
    Ok(p)
}
pub fn up_u16(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_u32(g: &dyn GpuBackend, v: &[u32]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_u64(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn rd_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

// 2026-09-25: Bitwise compare of two BF16 buffers over `a.len()` elements.
// Returns (all_equal, n_diff, first differing index or -1).
pub fn cmp_bits(a: &[u16], b: &[u16]) -> (bool, usize, isize) {
    let mut n = 0usize;
    let mut first = -1isize;
    for i in 0..a.len() {
        if a[i] != b[i] {
            n += 1;
            if first < 0 {
                first = i as isize;
            }
        }
    }
    (n == 0, n, first)
}
// 2026-09-25: Tolerance compare against the host reference: passes when every
// pair of BF16 bit patterns differs by at most 1 as integers (one ULP when the
// signs agree). Returns (pass, exact_matches, max_ulp, worst_idx).
pub fn cmp_tol(kern: &[u16], href: &[u16]) -> (bool, usize, u32, usize) {
    let mut exact = 0usize;
    let mut max_ulp = 0u32;
    let mut worst = 0usize;
    for i in 0..kern.len() {
        if kern[i] == href[i] {
            exact += 1;
            continue;
        }
        let ulp = (kern[i] as i32 - href[i] as i32).unsigned_abs();
        if ulp > max_ulp {
            max_ulp = ulp;
            worst = i;
        }
    }
    (max_ulp <= 1, exact, max_ulp, worst)
}

// 2026-09-25: A generated FP4 weight. Transposed (`t = true`, the `_t`
// kernels): packed [K/2, N], scales [K/GS, N]. Not transposed (the `ptrtable`
// kernel): packed [N, K/2], scales [N, K/GS]. `s_e8m0` has one byte per 32 K
// rows, `s_nvfp4` one per 16 (empty from `gen_wt_fullrange`), and `nib` holds
// the nibbles at [k * N + n].
pub struct Wt {
    pub packed: Vec<u8>,
    pub s_e8m0: Vec<u8>,
    pub s_nvfp4: Vec<u8>,
    pub nib: Vec<u8>,
}

// 2026-09-25: Weight for the prefill bit-exact check, with E8M0 bytes in
// [121, 135] (e in [-6, 8], exact in E4M3) and the matching NVFP4 scales.
pub fn gen_wt_bitexact(rng: &mut Rng, k: usize, n: usize, t: bool) -> Wt {
    let g32 = k / 32;
    let g16 = k / 16;
    let mut nib = vec![0u8; k * n];
    for x in nib.iter_mut() {
        *x = rng.nibble();
    }
    let mut packed = vec![0u8; k / 2 * n];
    for kh in 0..k / 2 {
        for col in 0..n {
            let lo = nib[(2 * kh) * n + col];
            let hi = nib[(2 * kh + 1) * n + col];
            let byte = (lo & 0xF) | ((hi & 0xF) << 4);
            let idx = if t {
                kh * n + col // 2026-09-25: [K/2, N]
            } else {
                col * (k / 2) + kh // 2026-09-25: [N, K/2]
            };
            packed[idx] = byte;
        }
    }
    // 2026-09-25: e = -6 + (g + col) % 15 spans E4M3's exact window. At a fixed
    // column consecutive groups always differ, so a group-index slip (per-16
    // vs per-32) changes the scale; with K = 64, 128 or 448 (2, 4 or 14 groups)
    // every group of a column is distinct.
    let sb_of = |g: usize, col: usize| -> u8 {
        let e = -6 + (((g + col) % 15) as i32);
        (127 + e) as u8
    };
    let mut s_e8m0 = vec![0u8; g32 * n];
    for g in 0..g32 {
        for col in 0..n {
            let idx = if t { g * n + col } else { col * g32 + g };
            s_e8m0[idx] = sb_of(g, col);
        }
    }
    // 2026-09-25: Each per-16 NVFP4 scale is the E4M3 encoding of the same 2^e
    // as its 32-row group (g32 = g16 / 2); the tests launch with scale2 = 1.0.
    let mut s_nvfp4 = vec![0u8; g16 * n];
    for g in 0..g16 {
        for col in 0..n {
            let sb = sb_of(g / 2, col);
            let e = sb as i32 - 127;
            let idx = if t { g * n + col } else { col * g16 + g };
            s_nvfp4[idx] = e4m3_pow2_byte(e);
        }
    }
    Wt {
        packed,
        s_e8m0,
        s_nvfp4,
        nib,
    }
}

// 2026-09-25: Transposed weight for the decode host-reference check, with
// exponents e = -14 + (5g + 3col) % 29 in [-14, 14], beyond E4M3's exact
// window; `s_nvfp4` is empty.
pub fn gen_wt_fullrange(rng: &mut Rng, k: usize, n: usize) -> Wt {
    let g32 = k / 32;
    let mut nib = vec![0u8; k * n];
    for x in nib.iter_mut() {
        *x = rng.nibble();
    }
    let mut packed = vec![0u8; k / 2 * n];
    for kh in 0..k / 2 {
        for col in 0..n {
            let lo = nib[(2 * kh) * n + col];
            let hi = nib[(2 * kh + 1) * n + col];
            packed[kh * n + col] = (lo & 0xF) | ((hi & 0xF) << 4);
        }
    }
    let mut s_e8m0 = vec![0u8; g32 * n];
    for g in 0..g32 {
        for col in 0..n {
            let e = -14 + (((g * 5 + col * 3) % 29) as i32);
            s_e8m0[g * n + col] = (127 + e) as u8;
        }
    }
    Wt {
        packed,
        s_e8m0,
        s_nvfp4: vec![],
        nib,
    }
}

// 2026-09-25: Launch `moe_expert_gate_up_shared_t` or its `_e8m0` twin: grid
// (ceil(n / 32), top_k + 1, 2), 32 threads. Slot `top_k` is the shared
// expert; z selects gate (0) or up (1).
pub fn launch_decode_gate_up(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_s2: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_s2: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_p: DevicePtr,
    sh_gate_s: DevicePtr,
    sh_gate_s2: f32,
    sh_gate_out: DevicePtr,
    sh_up_p: DevicePtr,
    sh_up_s: DevicePtr,
    sh_up_s2: f32,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let bx = n.div_ceil(32);
    KernelLaunch::new(g, kern)
        .grid([bx, top_k + 1, 2])
        .block([32, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_s2)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_s2)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_p)
        .arg_ptr(sh_gate_s)
        .arg_f32(sh_gate_s2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_p)
        .arg_ptr(sh_up_s)
        .arg_f32(sh_up_s2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

// 2026-09-25: Host f32 GEMV over a transposed E8M0 weight ([K/2, N], scales
// [K/32, N]), rounded to BF16.
pub fn host_gemv(a_bf16: &[u16], w: &Wt, k: usize, n: usize) -> Vec<u16> {
    let g32 = k / 32;
    let mut out = vec![0u16; n];
    for col in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            let a = bf16_bits_to_f32(a_bf16[kk]);
            let nib = w.nib[kk * n + col] as usize;
            let sb = w.s_e8m0[(kk / 32).min(g32 - 1) * n + col];
            acc += a * E2M1[nib] * e8m0_to_f32(sb);
        }
        out[col] = f32_to_bf16_bits(acc);
    }
    out
}

// 2026-09-25: Prefill launchers: each variant selects one
// `ops::moe_w4a16_*` wrapper, which launches the kernel handle it is given.
#[derive(Clone, Copy)]
pub enum GOp {
    Ptr64,
    PtrN128,
    PtrK64N128,
}
#[derive(Clone, Copy)]
pub enum FOp {
    FusedN128,
    FusedK64N128,
}

pub fn launch_grouped(
    g: &dyn GpuBackend,
    op: GOp,
    kern: KernelHandle,
    a: DevicePtr,
    bp: DevicePtr,
    bs: DevicePtr,
    s2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    sti: DevicePtr,
    ne: u32,
    n: u32,
    k: u32,
    mt: u32,
    st: u64,
) -> Result<()> {
    match op {
        GOp::Ptr64 => ops::moe_w4a16_grouped_gemm_ptrtable(
            g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st,
        ),
        GOp::PtrN128 => ops::moe_w4a16_grouped_gemm_ptrtable_n128(
            g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st,
        ),
        GOp::PtrK64N128 => ops::moe_w4a16_grouped_gemm_ptrtable_k64_n128(
            g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st,
        ),
    }
}
pub fn launch_fused(
    g: &dyn GpuBackend,
    op: FOp,
    kern: KernelHandle,
    a: DevicePtr,
    gp: DevicePtr,
    gs: DevicePtr,
    gs2: DevicePtr,
    upp: DevicePtr,
    ups: DevicePtr,
    ups2: DevicePtr,
    cg: DevicePtr,
    cu: DevicePtr,
    off: DevicePtr,
    sti: DevicePtr,
    ne: u32,
    n: u32,
    k: u32,
    mt: u32,
    st: u64,
) -> Result<()> {
    match op {
        FOp::FusedN128 => ops::moe_w4a16_fused_gate_up_n128(
            g, kern, a, gp, gs, gs2, upp, ups, ups2, cg, cu, off, sti, ne, n, k, mt, st,
        ),
        FOp::FusedK64N128 => ops::moe_w4a16_fused_gate_up_k64_n128(
            g, kern, a, gp, gs, gs2, upp, ups, ups2, cg, cu, off, sti, ne, n, k, mt, st,
        ),
    }
}
