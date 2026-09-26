// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU tests of `CapturedStep` over one layer's decode body
//! (`Predictor::project_q`, `score_blocks`, then `TiledAttention` `begin_step`,
//! `step_tile` and `finalize`): graph replay matches the eager output within
//! 1e-2, and a graph of 32 bodies replays at least 1.05x faster than issuing
//! them eagerly.
//!
//! Owner: storage (tests).
//! Invariants: none beyond the types.

use std::ffi::c_void;
use std::time::Instant;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use metrale_storage::cuda_graph::CapturedStep;
use metrale_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use metrale_storage::predictor::{Predictor, PredictorDims};
use metrale_storage::tiled_attention::{TiledAttention, TiledAttentionDims};

const NUM_SEQS: usize = 1;
const NUM_LAYERS: usize = 1;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const R: usize = 32;
const NUM_BLOCKS: usize = 16;

fn random_bf16(n: usize, rng: &mut ChaCha8Rng) -> Vec<bf16> {
    let dist = StandardNormal;
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(rng);
            bf16::from_f32(v * inv)
        })
        .collect()
}

#[allow(dead_code)]
struct Bundle {
    predictor: Predictor,
    attn: TiledAttention,
    q: DeviceBuffer,
    q_proj: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    block_table: DeviceBuffer,
    counts: DeviceBuffer,
    scores: DeviceBuffer,
    output: DeviceBuffer,
    n_q: usize,
    n_k: usize,
    n_out: usize,
}

fn build(ctx: &CudaCtx, seed: u64) -> Bundle {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q_host = random_bf16(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM, &mut rng);
    let k_host = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let v_host = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);

    let predictor = Predictor::new(
        ctx,
        PredictorDims {
            num_layers: NUM_LAYERS,
            num_q_heads: NUM_Q_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            r: R,
            block_size: BLOCK_SIZE,
            max_blocks: NUM_BLOCKS,
        },
        0xCAFE_F00D,
    )
    .unwrap();
    let attn = TiledAttention::new(TiledAttentionDims {
        max_seqs: NUM_SEQS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
        tile_capacity: NUM_BLOCKS,
    })
    .unwrap();

    let q = DeviceBuffer::new(q_host.len() * 2).unwrap();
    let q_proj = DeviceBuffer::new(NUM_Q_HEADS * R * 2).unwrap();
    let k = DeviceBuffer::new(k_host.len() * 2).unwrap();
    let v = DeviceBuffer::new(v_host.len() * 2).unwrap();
    let block_table_host: Vec<i32> = (0..NUM_BLOCKS as i32).collect();
    let counts_host = [NUM_BLOCKS as i32; NUM_SEQS];
    let block_table = DeviceBuffer::new(block_table_host.len() * 4).unwrap();
    let counts = DeviceBuffer::new(counts_host.len() * 4).unwrap();
    let scores = DeviceBuffer::new(NUM_BLOCKS * 4).unwrap();
    let output = DeviceBuffer::new(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM * 2).unwrap();

    copy_h_to_d_async(
        q.ptr,
        q_host.as_ptr() as *const c_void,
        q_host.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        k.ptr,
        k_host.as_ptr() as *const c_void,
        k_host.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        v.ptr,
        v_host.as_ptr() as *const c_void,
        v_host.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        block_table.ptr,
        block_table_host.as_ptr() as *const c_void,
        block_table_host.len() * 4,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        counts.ptr,
        counts_host.as_ptr() as *const c_void,
        counts_host.len() * 4,
        ctx.stream,
    )
    .unwrap();
    // 2026-09-25: Project every block's K, so `score_blocks` scores real data.
    for blk in 0..NUM_BLOCKS {
        let block_offset = blk * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2;
        predictor
            .project_kv_block(ctx, 0, blk, k.ptr + block_offset as u64)
            .unwrap();
    }
    stream_sync(ctx.stream).unwrap();

    Bundle {
        predictor,
        attn,
        q,
        q_proj,
        k,
        v,
        block_table,
        counts,
        scores,
        output,
        n_q: q_host.len(),
        n_k: k_host.len(),
        n_out: NUM_SEQS * NUM_Q_HEADS * HEAD_DIM,
    }
}

/// 2026-09-25: Issue one layer's decode body on `ctx.stream`, eagerly or inside
/// `CapturedStep::capture`.
fn issue_decode_body(ctx: &CudaCtx, b: &Bundle) -> anyhow::Result<()> {
    b.predictor.project_q(ctx, b.q.ptr, b.q_proj.ptr)?;
    b.predictor.score_blocks(
        ctx,
        b.q_proj.ptr,
        b.predictor.a_g_dev_ptr(),
        b.scores.ptr,
        NUM_BLOCKS,
    )?;
    b.attn.begin_step(ctx, NUM_SEQS)?;
    let (s_blk, s_tok, s_kvh) = b.attn.paged_strides();
    b.attn.step_tile(
        ctx,
        b.q.ptr,
        b.k.ptr,
        b.v.ptr,
        b.block_table.ptr,
        b.counts.ptr,
        NUM_SEQS,
        s_blk,
        s_tok,
        s_kvh,
        BLOCK_SIZE as i32,
    )?;
    b.attn.finalize(ctx, b.output.ptr, NUM_SEQS)?;
    Ok(())
}

fn read_output(ctx: &CudaCtx, b: &Bundle) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); b.n_out];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        b.output.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    out
}

#[test]
#[ignore = "requires GPU"]
fn graph_replay_matches_eager() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let bundle = build(&ctx, 0xCAFE);
    issue_decode_body(&ctx, &bundle).unwrap();
    let eager_out = read_output(&ctx, &bundle);

    let captured = CapturedStep::capture(ctx.stream, || issue_decode_body(&ctx, &bundle)).unwrap();
    captured.launch(ctx.stream).unwrap();
    let graph_out = read_output(&ctx, &bundle);

    let mut max_d = 0.0_f32;
    for (a, b) in eager_out.iter().zip(&graph_out) {
        let d = (a.to_f32() - b.to_f32()).abs();
        if d > max_d {
            max_d = d;
        }
    }
    eprintln!("graph vs eager max abs diff = {max_d:.3e}");
    assert!(max_d < 1e-2, "graph replay diverged: {max_d}");
}

#[test]
#[ignore = "requires GPU"]
fn graph_replay_speedup() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let bundle = build(&ctx, 0xBEEF);

    // 2026-09-25: One "step" is the body issued `N_LAYERS` times.
    const N_LAYERS: usize = 32;
    const WARMUP: usize = 4;
    const ITERS: usize = 64;

    for _ in 0..WARMUP {
        for _ in 0..N_LAYERS {
            issue_decode_body(&ctx, &bundle).unwrap();
        }
    }
    stream_sync(ctx.stream).unwrap();
    let t = Instant::now();
    for _ in 0..ITERS {
        for _ in 0..N_LAYERS {
            issue_decode_body(&ctx, &bundle).unwrap();
        }
    }
    stream_sync(ctx.stream).unwrap();
    let eager_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // 2026-09-25: The whole `N_LAYERS`-body step captured as one graph.
    let captured = CapturedStep::capture(ctx.stream, || {
        for _ in 0..N_LAYERS {
            issue_decode_body(&ctx, &bundle)?;
        }
        Ok(())
    })
    .unwrap();
    for _ in 0..WARMUP {
        captured.launch(ctx.stream).unwrap();
    }
    stream_sync(ctx.stream).unwrap();
    let t = Instant::now();
    for _ in 0..ITERS {
        captured.launch(ctx.stream).unwrap();
    }
    stream_sync(ctx.stream).unwrap();
    let graph_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let speedup = eager_us / graph_us;
    eprintln!(
        "{N_LAYERS}-layer step | eager: {eager_us:.0} µs | graph: {graph_us:.0} µs | speedup: {speedup:.2}×"
    );
    // 2026-09-25: The graph runs the same kernels, so the speedup comes only
    // from the launches it replaces.
    assert!(
        speedup >= 1.05,
        "graph speedup {speedup:.2}× < 1.05× — launch amortisation regressed"
    );
}
