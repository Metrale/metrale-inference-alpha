// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fixtures and graders for `native_attn_decode_splitk_hopper_microtest`.
//!
//! Nothing here launches a kernel: it builds the paged-KV fixture, scores two
//! BF16 outputs and provides the `known_bad` control.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
#![allow(dead_code)]

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: Qwen3.8-27B full-attention heads (`q_heads`, `kv_heads`,
/// `head_dim` in `kernels/hopper/qwen3.8-27b/MODEL.toml`).
pub const NQ: usize = 24;
pub const NKV: usize = 4;
pub const HD: usize = 256;
/// 2026-09-25: Paged KV block size in tokens.
pub const BLOCK: usize = 16;
/// 2026-09-25: Sentinel bytes either side of a buffer from [`upload_guarded`].
pub const GUARD: usize = 256;
pub const SENTINEL: u8 = 0x5a;

/// 2026-09-25: Context lengths under test.
pub const LENGTHS: [usize; 3] = [1335, 4847, 16384];
/// 2026-09-25: Co-batched row counts.
pub const ROWS: [usize; 3] = [1, 4, 16];
/// 2026-09-25: Split counts the example grades against its `splits: 0`
/// reference arm.
pub const SPLITS: [u32; 4] = [1, 2, 4, 6];

pub const MAX_L: usize = 16384;
pub const MAX_N: usize = 16;
pub const BLOCKS_PER_SEQ: usize = MAX_L / BLOCK;
pub const TOTAL_BLOCKS: usize = MAX_N * BLOCKS_PER_SEQ;

/// 2026-09-25: Deterministic LCG. `next_f32` returns values in [-1, 0): the
/// 31-bit draw is divided by 2^31 before the 1 is subtracted.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// 2026-09-25: Upload with `GUARD` sentinel bytes either side and return the
/// pointer to the data, so an overrun in either direction lands in bytes
/// [`guards_intact`] checks.
pub fn upload_guarded(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let mut framed = vec![SENTINEL; bytes.len() + 2 * GUARD];
    framed[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
    let base = gpu.alloc(framed.len())?;
    gpu.copy_h2d(&framed, base)?;
    Ok(DevicePtr(base.0 + GUARD as u64))
}

/// 2026-09-25: Fails unless both guard bands around a buffer uploaded by
/// [`upload_guarded`] still hold `SENTINEL`.
pub fn guards_intact(gpu: &dyn GpuBackend, data: DevicePtr, len: usize) -> Result<()> {
    let mut head = vec![0u8; GUARD];
    let mut tail = vec![0u8; GUARD];
    gpu.copy_d2h(DevicePtr(data.0 - GUARD as u64), &mut head)?;
    gpu.copy_d2h(DevicePtr(data.0 + len as u64), &mut tail)?;
    ensure!(
        head.iter().all(|&b| b == SENTINEL) && tail.iter().all(|&b| b == SENTINEL),
        "a kernel wrote outside its extent"
    );
    Ok(())
}

pub fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect()
}

/// 2026-09-25: How two BF16 outputs compare: relative RMS, worst absolute
/// element difference, cosine and bit equality.
#[derive(Debug, Clone, Copy)]
pub struct Score {
    pub rel_rms: f64,
    pub max_abs: f64,
    pub cosine: f64,
    pub bit_equal: bool,
}

/// 2026-09-25: Grade `observed` against `reference`. Splitting the KV range
/// re-brackets the online-softmax merge, which is not associative in floating
/// point, so the example grades split counts with [`passes`], not equality. It
/// requires bit equality separately: row 0 at the same split count, decoded
/// alone and co-batched.
pub fn score(observed: &[u8], reference: &[u8]) -> Score {
    let a = to_f32(observed);
    let b = to_f32(reference);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = max_abs.max((x - y).abs());
    }
    Score {
        rel_rms: if den > 0.0 {
            (num / den).sqrt()
        } else {
            num.sqrt()
        },
        max_abs,
        cosine: if na > 0.0 && nb > 0.0 {
            dot / (na.sqrt() * nb.sqrt())
        } else {
            0.0
        },
        bit_equal: observed == reference,
    }
}

/// 2026-09-25: The band a re-bracketed softmax merge must land in: relative RMS
/// at most `REL_RMS_TOL` and cosine at least `COSINE_FLOOR` ([`passes`]).
pub const REL_RMS_TOL: f64 = 2e-3;
pub const COSINE_FLOOR: f64 = 0.999_99;

pub fn passes(s: &Score) -> bool {
    s.rel_rms <= REL_RMS_TOL && s.cosine >= COSINE_FLOOR
}

/// 2026-09-25: The negative control: `reference` with head 0 of row 0 zeroed.
/// The example requires [`passes`] to refuse it.
pub fn known_bad(reference: &[u8]) -> Vec<u8> {
    let mut bad = reference.to_vec();
    let head_bytes = HD * 2;
    for b in bad.iter_mut().take(head_bytes) {
        *b = 0;
    }
    bad
}

/// 2026-09-25: Bytes of K and V a launch reads, `2 * n * L * NKV * HD * elem`:
/// the numerator of the example's GB/s figures.
pub fn kv_bytes(n: usize, l: usize, elem: usize) -> f64 {
    (2 * n * l * NKV * HD * elem) as f64
}

pub fn gbs(bytes: f64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        bytes / seconds / 1e9
    } else {
        0.0
    }
}

/// 2026-09-25: `block_tables[seq][logical] = physical`, with every sequence on
/// its own pages, so no two rows read the same KV bytes.
pub fn block_table(n: usize) -> Vec<i32> {
    let mut table = vec![0i32; n * BLOCKS_PER_SEQ];
    for (seq, chunk) in table.chunks_mut(BLOCKS_PER_SEQ).enumerate() {
        for (logical, slot) in chunk.iter_mut().enumerate() {
            *slot = (seq * BLOCKS_PER_SEQ + logical) as i32;
        }
    }
    table
}

/// 2026-09-25: One BF16 query row per sequence: `[n, NQ * HD]`.
pub fn queries(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(n * NQ * HD * 2);
    for _ in 0..(n * NQ * HD) {
        out.extend_from_slice(&bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes());
    }
    out
}

/// 2026-09-25: An FP8 E4M3 KV pool built directly as bytes: the magnitude byte
/// is `0x20 | (mag >> 1)`, 0x20..=0x3C, which decodes to 0.125..=1.5, and the
/// sign bit is set when `next_f32` is negative.
pub fn fp8_pool(seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    (0..TOTAL_BLOCKS * BLOCK * NKV * HD)
        .map(|_| {
            let v = rng.next_f32();
            let sign = if v < 0.0 { 0x80u8 } else { 0 };
            let mag = ((v.abs() * 56.0) as u8).min(0x3f);
            sign | 0x20 | (mag >> 1)
        })
        .collect()
}

/// 2026-09-25: A BF16 KV pool of the same shape as [`fp8_pool`].
pub fn bf16_pool(seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(TOTAL_BLOCKS * BLOCK * NKV * HD * 2);
    for _ in 0..(TOTAL_BLOCKS * BLOCK * NKV * HD) {
        out.extend_from_slice(&bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes());
    }
    out
}
