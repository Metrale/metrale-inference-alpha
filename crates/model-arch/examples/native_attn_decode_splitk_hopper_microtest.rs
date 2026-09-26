// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Oracle for the Hopper paged-decode attention split-K kernels
//! (`paged_decode_attn_splitk_{fp8,bf16}_hopper` and their reduce kernels) at the Qwen3.8-27B
//! attention shape (24 query heads, 4 KV heads, head dim 256).
//!
//! Owner: model-arch examples (hopper attention kernels).
//! Invariants:
//! - The run fails if a split arm is outside the `harness::passes` band against the non-split
//!   kernel, if row 0 differs between n = 1 and a wider batch at the same split count, if the
//!   KNOWN_BAD control passes, or if a buffer guard was overwritten.
//!
//! # What this grades
//!
//! 1. Determinism. A sequence decoded alone and the same sequence decoded beside others must
//!    produce byte-identical output at the same `num_splits`; this is asserted as an equality,
//!    because nothing is reassociated.
//! 2. Reassociation and bandwidth. Different split counts cannot agree bit for bit: splitting
//!    the KV range re-brackets the online-softmax merge. Each split count is graded against the
//!    non-split kernel with `attn_splitk_harness::REL_RMS_TOL` and `COSINE_FLOOR`, and a
//!    KNOWN_BAD control must fail that band. GB/s is printed per arm.
//!
//! # Run (H100)
//!
//! ```text
//! cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example native_attn_decode_splitk_hopper_microtest
//! ```
//!
//! The split-K kernels exist only in
//! `kernels/hopper/common/paged_decode_{fp8,bf16}_splitk_hopper.cu`; on a GB10 build
//! `Kernels::resolve` fails with the missing kernel's name.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use std::time::Instant;

#[path = "common/attn_splitk_harness.rs"]
mod harness;

use harness::{
    BLOCK, BLOCKS_PER_SEQ, HD, LENGTHS, MAX_N, NKV, NQ, ROWS, SPLITS, Score, bf16_pool,
    block_table, fp8_pool, gbs, guards_intact, known_bad, kv_bytes, passes, queries, score,
    upload_guarded,
};

/// 2026-09-25: Timed repetitions per arm, each synchronised; the minimum is reported, after
/// one untimed launch.
const REPS: usize = 5;
const STREAM: u64 = 0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kv {
    /// 2026-09-25: E4M3, one byte per element; the host passes `cache_stride`.
    Fp8,
    /// 2026-09-25: BF16, two bytes per element; the kernel derives the page stride.
    Bf16,
}

impl Kv {
    fn elem(self) -> usize {
        match self {
            Kv::Fp8 => 1,
            Kv::Bf16 => 2,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Kv::Fp8 => "fp8",
            Kv::Bf16 => "bf16",
        }
    }
}

/// 2026-09-25: Every handle the run needs, resolved before any arm runs, so a build without
/// the split-K kernels fails with a kernel name rather than part way through the table.
struct Kernels {
    base_fp8: KernelHandle,
    base_bf16: KernelHandle,
    splitk_fp8: KernelHandle,
    reduce_fp8: KernelHandle,
    splitk_bf16: KernelHandle,
    reduce_bf16: KernelHandle,
}

impl Kernels {
    fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            base_fp8: gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")?,
            base_bf16: gpu.kernel("paged_decode", "paged_decode_attn")?,
            splitk_fp8: gpu.kernel(
                "paged_decode_fp8_splitk_hopper",
                "paged_decode_attn_splitk_fp8_hopper",
            )?,
            reduce_fp8: gpu.kernel(
                "paged_decode_fp8_splitk_hopper",
                "paged_decode_attn_reduce_fp8_hopper",
            )?,
            splitk_bf16: gpu.kernel(
                "paged_decode_bf16_splitk_hopper",
                "paged_decode_attn_splitk_bf16_hopper",
            )?,
            reduce_bf16: gpu.kernel(
                "paged_decode_bf16_splitk_hopper",
                "paged_decode_attn_reduce_bf16_hopper",
            )?,
        })
    }
}

/// 2026-09-25: Device buffers, uploaded once and reused by every arm.
struct Fixture {
    k_fp8: DevicePtr,
    v_fp8: DevicePtr,
    k_bf16: DevicePtr,
    v_bf16: DevicePtr,
    q: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    output: DevicePtr,
    workspace: DevicePtr,
    output_bytes: usize,
}

fn upload_fixture(gpu: &dyn GpuBackend) -> Result<Fixture> {
    let fp8_k = fp8_pool(0x5eed_0001);
    let fp8_v = fp8_pool(0x5eed_0002);
    let bf_k = bf16_pool(0x5eed_0003);
    let bf_v = bf16_pool(0x5eed_0004);
    let q = queries(MAX_N, 0x5eed_0005);
    let table = block_table(MAX_N);
    let table_bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();

    let output_bytes = MAX_N * NQ * HD * 2;
    // 2026-09-25: `(head_dim + 2)` F32 per (seq, head, split), the workspace layout of
    // `paged_decode_splitk_hopper.cuh`, sized for the widest split count.
    let ws_bytes =
        MAX_N * NQ * (*SPLITS.iter().max().expect("SPLITS is not empty") as usize) * (HD + 2) * 4;

    Ok(Fixture {
        k_fp8: upload_guarded(gpu, &fp8_k)?,
        v_fp8: upload_guarded(gpu, &fp8_v)?,
        k_bf16: upload_guarded(gpu, &bf_k)?,
        v_bf16: upload_guarded(gpu, &bf_v)?,
        q: upload_guarded(gpu, &q)?,
        block_tables: upload_guarded(gpu, &table_bytes)?,
        seq_lens: upload_guarded(gpu, &vec![0u8; MAX_N * 4])?,
        output: upload_guarded(gpu, &vec![0u8; output_bytes])?,
        workspace: upload_guarded(gpu, &vec![0u8; ws_bytes])?,
        output_bytes,
    })
}

#[derive(Clone, Copy)]
struct Arm {
    kv: Kv,
    l: usize,
    n: usize,
    splits: u32,
}

/// 2026-09-25: Launch one arm once. `splits == 0` launches the non-split kernel.
fn launch(gpu: &dyn GpuBackend, k: &Kernels, f: &Fixture, arm: Arm) -> Result<()> {
    let (n, hd, nq, nkv) = (arm.n as u32, HD as u32, NQ as u32, NKV as u32);
    let blocks = BLOCKS_PER_SEQ as u32;
    let block = BLOCK as u32;
    let inv_sqrt_d = 1.0f32 / (HD as f32).sqrt();
    let q_stride = (NQ * HD) as u32;
    // 2026-09-25: The FP8 `cache_stride` is the per-block stride in elements.
    let cache_stride = (BLOCK * NKV * HD) as u64;
    match (arm.kv, arm.splits) {
        (Kv::Fp8, 0) => ops::paged_decode_attn_fp8(
            gpu,
            k.base_fp8,
            f.q,
            f.k_fp8,
            f.v_fp8,
            f.output,
            f.block_tables,
            f.seq_lens,
            blocks,
            n,
            nq,
            nkv,
            hd,
            block,
            inv_sqrt_d,
            1.0,
            1.0,
            q_stride,
            cache_stride,
            0,
            STREAM,
        ),
        (Kv::Bf16, 0) => ops::paged_decode_attn_bf16(
            gpu,
            k.base_bf16,
            f.q,
            f.k_bf16,
            f.v_bf16,
            f.output,
            f.block_tables,
            f.seq_lens,
            blocks,
            n,
            nq,
            nkv,
            hd,
            block,
            inv_sqrt_d,
            q_stride,
            0,
            STREAM,
        ),
        (Kv::Fp8, splits) => {
            ops::paged_decode_attn_splitk_fp8(
                gpu,
                k.splitk_fp8,
                f.q,
                f.k_fp8,
                f.v_fp8,
                f.workspace,
                f.block_tables,
                f.seq_lens,
                blocks,
                nq,
                nkv,
                hd,
                block,
                inv_sqrt_d,
                splits,
                1.0,
                1.0,
                q_stride,
                cache_stride,
                n,
                0,
                STREAM,
            )?;
            ops::paged_decode_attn_reduce_fp8(
                gpu,
                k.reduce_fp8,
                f.workspace,
                f.output,
                f.seq_lens,
                nq,
                hd,
                splits,
                n,
                STREAM,
            )
        }
        (Kv::Bf16, splits) => {
            ops::paged_decode_attn_splitk_bf16(
                gpu,
                k.splitk_bf16,
                f.q,
                f.k_bf16,
                f.v_bf16,
                f.workspace,
                f.block_tables,
                f.seq_lens,
                blocks,
                nq,
                nkv,
                hd,
                block,
                inv_sqrt_d,
                splits,
                q_stride,
                n,
                0,
                STREAM,
            )?;
            ops::paged_decode_attn_reduce_fp8(
                gpu,
                k.reduce_bf16,
                f.workspace,
                f.output,
                f.seq_lens,
                nq,
                hd,
                splits,
                n,
                STREAM,
            )
        }
    }
}

/// 2026-09-25: Run an arm `REPS` times; return the output bytes of the live rows and the best
/// time in seconds.
fn run(gpu: &dyn GpuBackend, k: &Kernels, f: &Fixture, arm: Arm) -> Result<(Vec<u8>, f64)> {
    // 2026-09-25: Zero the output every time: a split-K arm that failed to write a row would
    // otherwise be graded on the previous arm's bytes.
    gpu.memset(f.output, 0, f.output_bytes)?;
    gpu.memset(
        f.workspace,
        0,
        MAX_N * NQ * (*SPLITS.iter().max().expect("SPLITS is not empty") as usize) * (HD + 2) * 4,
    )?;
    launch(gpu, k, f, arm)?;
    gpu.synchronize(STREAM)?;
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let t0 = Instant::now();
        launch(gpu, k, f, arm)?;
        gpu.synchronize(STREAM)?;
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let live = arm.n * NQ * HD * 2;
    let mut out = vec![0u8; live];
    gpu.copy_d2h(f.output, &mut out)?;
    Ok((out, best))
}

fn report(arm: Arm, s: &Score, seconds: f64) -> String {
    format!(
        "{kv:<4} L={l:<6} n={n:<3} splits={sp:<2} {ms:8.3} ms  {gbs:8.1} GB/s  \
         rel_rms={rms:.3e} max_abs={max:.3e} cos={cos:.9}",
        kv = arm.kv.label(),
        l = arm.l,
        n = arm.n,
        sp = arm.splits,
        ms = seconds * 1e3,
        gbs = gbs(kv_bytes(arm.n, arm.l, arm.kv.elem()), seconds),
        rms = s.rel_rms,
        max = s.max_abs,
        cos = s.cosine,
    )
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &gpu;
    let k = Kernels::resolve(gpu)?;
    let f = upload_fixture(gpu)?;

    eprintln!(
        "native_attn_decode_splitk_hopper_microtest: nq={NQ} nkv={NKV} hd={HD} block={BLOCK} \
         sm_count={sms} auto_splits={auto}",
        sms = metrale_kernels::TARGET_SM_COUNT,
        auto =
            metrale_kernels::attn_splitk::auto_splits(metrale_kernels::TARGET_SM_COUNT, NQ as u32),
    );

    let mut failures: Vec<String> = Vec::new();
    // 2026-09-25: Row 0's bytes at n = 1, keyed by (kv, L, splits): the reference for the
    // co-batch determinism equality.
    let mut solo: std::collections::HashMap<(usize, usize, u32), Vec<u8>> =
        std::collections::HashMap::new();

    for kv in [Kv::Fp8, Kv::Bf16] {
        for l in LENGTHS {
            for n in ROWS {
                let lens: Vec<u8> = (0..MAX_N)
                    .map(|i| if i < n { l as i32 } else { 0 })
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                gpu.copy_h2d(&lens, f.seq_lens)?;

                let base_arm = Arm {
                    kv,
                    l,
                    n,
                    splits: 0,
                };
                let (reference, base_s) = run(gpu, &k, &f, base_arm)?;
                eprintln!(
                    "{}",
                    report(base_arm, &score(&reference, &reference), base_s)
                );

                for splits in SPLITS {
                    let arm = Arm { kv, l, n, splits };
                    let (observed, seconds) = run(gpu, &k, &f, arm)?;
                    let s = score(&observed, &reference);
                    eprintln!("{}", report(arm, &s, seconds));
                    if !passes(&s) {
                        failures.push(format!(
                            "{} L={l} n={n} splits={splits}: rel_rms={:.3e} cos={:.9} \
                             outside the documented band",
                            kv.label(),
                            s.rel_rms,
                            s.cosine
                        ));
                    }

                    // 2026-09-25: Row 0 sees the same KV, the same Q and the same split
                    // partition whatever else is in the batch, so its bytes must be identical.
                    let row0 = observed[..NQ * HD * 2].to_vec();
                    let key = (kv as usize, l, splits);
                    match solo.get(&key) {
                        None => {
                            solo.insert(key, row0);
                        }
                        Some(alone) if *alone != row0 => failures.push(format!(
                            "{} L={l} splits={splits}: row 0 differs between n=1 and n={n} — \
                             the reduction tree moved with the co-batched count, which is the \
                             nondeterminism the split policy exists to prevent",
                            kv.label()
                        )),
                        Some(_) => {}
                    }
                }

                // 2026-09-25: The control: the band must refuse a wrong answer, or it grades
                // nothing.
                let bad = score(&known_bad(&reference), &reference);
                ensure!(
                    !passes(&bad),
                    "KNOWN_BAD control PASSED at {} L={l} n={n} (rel_rms={:.3e} cos={:.9}) — \
                     the tolerance grades nothing",
                    kv.label(),
                    bad.rel_rms,
                    bad.cosine
                );
            }
        }
    }

    for (ptr, len) in [
        (f.output, f.output_bytes),
        (f.q, MAX_N * NQ * HD * 2),
        (f.block_tables, MAX_N * BLOCKS_PER_SEQ * 4),
    ] {
        guards_intact(gpu, ptr, len)?;
    }

    ensure!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("native_attn_decode_splitk_hopper_microtest: OK");
    Ok(())
}
