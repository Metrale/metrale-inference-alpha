// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Timing probe: does the routed-MoE decode kernel get L2 reuse when two tokens
//! route to the same experts?
//!
//! Owner: model-arch examples (MoE kernels).
//! Invariants: none beyond the types. The run prints timings; it checks no values.
//!
//! `moe_expert_gate_up_shared_batch2` launches one CTA per (token, expert slot) with no explicit
//! cross-token reuse, so every slot issues its own weight reads. Three arms at the Laguna shape
//! (H = K = 3072, inter = N = 1024, top_k = 10):
//!   m1          moe_expert_gate_up_shared        10 distinct experts
//!   m2_distinct moe_expert_gate_up_shared_batch2 20 slots, 20 distinct experts
//!   m2_dup      moe_expert_gate_up_shared_batch2 20 slots, 10 distinct (both tokens identical)
//! Reading: `m2_dup` near `m1` means L2 catches the duplicates; `m2_dup` near `m2_distinct`
//! means it does not, and a kernel that loads each distinct expert once would save the overlap.
//!
//! Each iteration moves the expert base 37 places round a 128-expert pool, so consecutive
//! iterations read mostly different experts.
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example moe_l2_reuse_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const H: usize = 3072;
const INTER: usize = 1024;
const TOP_K: usize = 10;
/// 2026-09-25: Pool size; the program prints its footprint (128 experts of gate + up).
const POOL: usize = 128;
const ITERS: usize = 60;
const WARMUP: usize = 10;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
}

fn up_u8(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
fn up_u64(g: &dyn GpuBackend, d: &[u64]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    up_u8(g, &b)
}
fn up_u32(g: &dyn GpuBackend, d: &[u32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    up_u8(g, &b)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    up_u8(g, &b)
}
fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    up_u8(g, &b)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    gp: DevicePtr,
    gs: DevicePtr,
    g2: DevicePtr,
    gout: DevicePtr,
    upp: DevicePtr,
    ups: DevicePtr,
    u2: DevicePtr,
    uout: DevicePtr,
    idx: DevicePtr,
    grid_y: u32,
) -> Result<()> {
    let nul = DevicePtr(0);
    KernelLaunch::new(g, kern)
        .grid([div_ceil(INTER as u32, 8), grid_y, 2])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gp)
        .arg_ptr(gs)
        .arg_ptr(g2)
        .arg_ptr(gout)
        .arg_ptr(upp)
        .arg_ptr(ups)
        .arg_ptr(u2)
        .arg_ptr(uout)
        .arg_ptr(idx)
        // 2026-09-25: The shared-expert arguments. grid.y covers only the routed CTAs, so the
        // shared branch never runs and these NULL pointers are never read.
        .arg_ptr(nul)
        .arg_ptr(nul)
        .arg_f32(1.0)
        .arg_ptr(nul)
        .arg_ptr(nul)
        .arg_ptr(nul)
        .arg_f32(1.0)
        .arg_ptr(nul)
        .arg_u32(INTER as u32)
        .arg_u32(H as u32)
        .arg_u32(TOP_K as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_m1 = g.kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")?;
    let k_m2 = g.kernel("moe_fused_batch2", "moe_expert_gate_up_shared_batch2")?;

    let packed_bytes = INTER * H / 2;
    let scale_bytes = INTER * (H / 16);
    let per_expert = (packed_bytes + scale_bytes) * 2;
    println!(
        "pool: {POOL} experts x {:.2} MB (gate+up) = {:.0} MB",
        per_expert as f64 / 1e6,
        (POOL * per_expert) as f64 / 1e6
    );

    let mut rng = Lcg(0xC0FFEE);
    let mut gp = Vec::with_capacity(POOL);
    let mut gs = Vec::with_capacity(POOL);
    let mut upp = Vec::with_capacity(POOL);
    let mut ups = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        let w: Vec<u8> = (0..packed_bytes).map(|_| rng.byte()).collect();
        let s: Vec<u8> = (0..scale_bytes).map(|_| 0x3Cu8).collect();
        gp.push(up_u8(g, &w)?.0);
        gs.push(up_u8(g, &s)?.0);
        let w2: Vec<u8> = (0..packed_bytes).map(|_| rng.byte()).collect();
        upp.push(up_u8(g, &w2)?.0);
        ups.push(up_u8(g, &s)?.0);
    }
    let gp_d = up_u64(g, &gp)?;
    let gs_d = up_u64(g, &gs)?;
    let upp_d = up_u64(g, &upp)?;
    let ups_d = up_u64(g, &ups)?;
    let s2 = up_f32(g, &vec![1.0f32; POOL])?;

    let a: Vec<bf16> = (0..2 * H)
        .map(|i| bf16::from_f32((i % 7) as f32 * 0.01))
        .collect();
    let a_d = up_bf16(g, &a)?;
    let gout = g.alloc(2 * TOP_K * INTER * 2)?;
    let uout = g.alloc(2 * TOP_K * INTER * 2)?;

    // 2026-09-25: The index sets are built before timing, so the timed loop does no host work.
    let mk = |dup: bool, distinct: usize| -> Vec<Vec<u32>> {
        (0..ITERS + WARMUP)
            .map(|it| {
                let base = (it * 37) % POOL;
                if dup {
                    let ten: Vec<u32> =
                        (0..TOP_K).map(|j| ((base + j * 3) % POOL) as u32).collect();
                    let mut v = ten.clone();
                    v.extend(ten);
                    v
                } else {
                    (0..distinct)
                        .map(|j| ((base + j * 3) % POOL) as u32)
                        .collect()
                }
            })
            .collect()
    };
    let idx_m1: Vec<DevicePtr> = mk(false, TOP_K)
        .iter()
        .map(|v| up_u32(g, v).unwrap())
        .collect();
    let idx_dist: Vec<DevicePtr> = mk(false, 2 * TOP_K)
        .iter()
        .map(|v| up_u32(g, v).unwrap())
        .collect();
    let idx_dup: Vec<DevicePtr> = mk(true, 0).iter().map(|v| up_u32(g, v).unwrap()).collect();

    let run = |kern: KernelHandle,
               idxs: &[DevicePtr],
               grid_y: u32,
               label: &str,
               experts: f64|
     -> Result<()> {
        for it in 0..WARMUP {
            launch(
                g, kern, a_d, gp_d, gs_d, s2, gout, upp_d, ups_d, s2, uout, idxs[it], grid_y,
            )?;
        }
        g.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for it in 0..ITERS {
            launch(
                g,
                kern,
                a_d,
                gp_d,
                gs_d,
                s2,
                gout,
                upp_d,
                ups_d,
                s2,
                uout,
                idxs[WARMUP + it],
                grid_y,
            )?;
        }
        g.synchronize(0)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
        let gb = experts * per_expert as f64 / 1e9;
        println!(
            "  {label:<12} {ms:8.4} ms   if {experts:>4.0} experts read: {:6.0} GB/s",
            gb / (ms / 1e3)
        );
        Ok(())
    };

    println!("\n(H=K={H}, inter=N={INTER}, top_k={TOP_K}, {ITERS} iters, rotating expert base)");
    run(k_m1, &idx_m1, TOP_K as u32, "m1", 10.0)?;
    run(k_m2, &idx_dist, (2 * TOP_K) as u32, "m2_distinct", 20.0)?;
    run(k_m2, &idx_dup, (2 * TOP_K) as u32, "m2_dup", 10.0)?;
    println!(
        "\nREAD: m2_dup ~= m1 => L2 catches duplicates, union-aware batching buys ~0.\n      \
         m2_dup ~= m2_distinct => no reuse, union-aware batching wins the overlap (~30%)."
    );
    Ok(())
}
