// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU test of the predictor's block recall. K is Gaussian noise
//! except one planted key per q head, `NEEDLE_BETA` times that head's query,
//! each in a different block. Over `SEEDS`, the mean share of planted blocks
//! among the predictor's top `NUM_NEEDLES` blocks must reach `RECALL_TARGET`.
//!
//! Owner: storage (tests).
//! Invariants: none beyond the types.

use std::collections::HashSet;
use std::ffi::c_void;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use metrale_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use metrale_storage::predictor::{Predictor, PredictorDims};

const NUM_LAYERS: usize = 1;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const R: usize = 32;
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 256;
const NUM_NEEDLES: usize = NUM_Q_HEADS;
const NEEDLE_BETA: f32 = 32.0;
const RECALL_K_FRAC: f32 = 0.10;
const SEEDS: &[u64] = &[1, 7, 23, 99, 1337];
const RECALL_TARGET: f32 = 0.95;

fn random_bf16(n: usize, rng: &mut ChaCha8Rng, scale: f32) -> Vec<bf16> {
    let dist = StandardNormal;
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(rng);
            bf16::from_f32(v * scale)
        })
        .collect()
}

/// 2026-09-25: Q, K and the set of blocks holding a planted key: one per q head,
/// in distinct random blocks, with iid Gaussian K elsewhere.
fn build_needle_haystack(seed: u64) -> (Vec<bf16>, Vec<bf16>, HashSet<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let q = random_bf16(NUM_Q_HEADS * HEAD_DIM, &mut rng, inv);
    let total_tokens = NUM_BLOCKS * BLOCK_SIZE;
    let mut k = random_bf16(total_tokens * NUM_KV_HEADS * HEAD_DIM, &mut rng, inv);

    let gqa = NUM_Q_HEADS / NUM_KV_HEADS;
    use rand::seq::SliceRandom;
    let mut block_pool: Vec<usize> = (0..NUM_BLOCKS).collect();
    block_pool.shuffle(&mut rng);

    let mut needles: HashSet<usize> = HashSet::new();
    for n in 0..NUM_NEEDLES {
        let qh = n;
        let kh = qh / gqa;
        let blk = block_pool[n];
        needles.insert(blk);
        let off = (n.wrapping_mul(7919)) % BLOCK_SIZE;
        let k_offset = (blk * BLOCK_SIZE * NUM_KV_HEADS + off * NUM_KV_HEADS + kh) * HEAD_DIM;
        for i in 0..HEAD_DIM {
            let qv = q[qh * HEAD_DIM + i].to_f32();
            k[k_offset + i] = bf16::from_f32(NEEDLE_BETA * qv);
        }
    }
    (q, k, needles)
}

fn topk(scores: &[f32], k: usize) -> HashSet<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap());
    idx.into_iter().take(k).collect()
}

#[test]
#[ignore = "requires GPU"]
fn recall_at_10_percent_meets_target() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let dims = PredictorDims {
        num_layers: NUM_LAYERS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        r: R,
        block_size: BLOCK_SIZE,
        max_blocks: NUM_BLOCKS,
    };

    let mut recalls = Vec::new();
    for &seed in SEEDS {
        let pred = Predictor::new(&ctx, dims, seed.wrapping_mul(0xCAFEF00D)).unwrap();
        let (q, k, needle_blocks) = build_needle_haystack(seed);

        let block_floats = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
        let block_bytes = block_floats * 2;
        let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
        for blk in 0..NUM_BLOCKS {
            let off = blk * block_floats;
            copy_h_to_d_async(
                k_block_dev.ptr,
                k[off..off + block_floats].as_ptr() as *const c_void,
                block_bytes,
                ctx.stream,
            )
            .unwrap();
            pred.project_kv_block(&ctx, 0, blk, k_block_dev.ptr)
                .unwrap();
        }
        let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
        let q_proj_dev = DeviceBuffer::new(NUM_Q_HEADS * R * 2).unwrap();
        copy_h_to_d_async(
            q_dev.ptr,
            q.as_ptr() as *const c_void,
            q.len() * 2,
            ctx.stream,
        )
        .unwrap();
        pred.project_q(&ctx, q_dev.ptr, q_proj_dev.ptr).unwrap();

        let scores_dev = DeviceBuffer::new(NUM_BLOCKS * 4).unwrap();
        pred.score_blocks(
            &ctx,
            q_proj_dev.ptr,
            pred.a_g_dev_ptr(),
            scores_dev.ptr,
            NUM_BLOCKS,
        )
        .unwrap();
        let mut scores = vec![0.0_f32; NUM_BLOCKS];
        copy_d_to_h_async(
            scores.as_mut_ptr() as *mut c_void,
            scores_dev.ptr,
            NUM_BLOCKS * 4,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();

        // 2026-09-25: Recall against the planted blocks rather than a softmax
        // top-k, where a noise key that happens to align with Q could outrank a
        // weakly scored planted block.
        let pred_top = topk(&scores, NUM_NEEDLES);
        let hits = pred_top.intersection(&needle_blocks).count();
        let recall = hits as f32 / NUM_NEEDLES as f32;
        eprintln!("seed={seed} needles={NUM_NEEDLES} hits={hits} recall={recall:.3}");
        recalls.push(recall);
    }
    let mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
    eprintln!(
        "mean recall@{:.0}% over {} seeds = {:.3}",
        RECALL_K_FRAC * 100.0,
        SEEDS.len(),
        mean
    );
    assert!(
        mean >= RECALL_TARGET,
        "recall@10% mean = {mean:.3} < target {RECALL_TARGET}; bump R from {R} or rework predictor"
    );
}
