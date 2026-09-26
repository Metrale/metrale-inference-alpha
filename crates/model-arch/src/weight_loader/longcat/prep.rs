// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LongCat MLA weight preparation: load-time transforms that let the shared MLA
//! runtime serve LongCat.
//!
//! Owner: model-arch weight loader (LongCat).
//! Invariants:
//! - RoPE rows of `q_b_proj` (rows `nope..nope + rope` of each head) and the trailing `rope`
//!   rows of `kv_a_proj_with_mqa` are de-interleaved: source row `2j` becomes row `j`,
//!   source `2j + 1` becomes row `j + rope / 2`.
//! - All of `q_b_proj` is scaled by `scale_q`, and `kv_a_layernorm.weight` by `scale_kv`.
//!   An RMSNorm output is linear in its gain, so scaling the gain scales `k_pass`; `k_rot`
//!   is split off before the norm and stays unscaled.
//! - Heads are padded to `padded_hd` with zeros:
//!
//!   ```text
//!   q_b_proj  per head [nope | rope]  -> [nope | 0 | rope]        width padded_hd
//!   kv_b_proj per head [nope | v]     -> [nope | 0 | v | 0]       padded_nope + padded_hd
//!   o_proj    per head reads v        -> reads padded_hd, pad columns zero
//!   ```
//!
//!   The zero `o_proj` columns cancel whatever lands in a padded V lane, and the zero Q
//!   lanes keep the padded K lanes out of every score. The loader keeps the softmax scale at
//!   `1/sqrt(nope + rope)` with `set_attn_scale_override`.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::weight_map::{DenseWeight, dense};

const BF16: usize = 2;

fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn to_bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| {
            let bits = x.to_bits();
            // 2026-09-25: Round to nearest even.
            let r = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
            r.to_le_bytes()
        })
        .collect()
}

/// 2026-09-25: De-interleave `rows` rope rows in place: source row `2j` becomes
/// row `j`, source `2j+1` becomes row `j + rows/2`.
fn deinterleave_rope_rows(host: &mut [u8], base_row: usize, rows: usize, cols: usize) {
    let half = rows / 2;
    let row_bytes = cols * BF16;
    let start = base_row * row_bytes;
    let src: Vec<u8> = host[start..start + rows * row_bytes].to_vec();
    for j in 0..half {
        let (a, b) = (2 * j, 2 * j + 1);
        host[start + j * row_bytes..start + (j + 1) * row_bytes]
            .copy_from_slice(&src[a * row_bytes..(a + 1) * row_bytes]);
        host[start + (half + j) * row_bytes..start + (half + j + 1) * row_bytes]
            .copy_from_slice(&src[b * row_bytes..(b + 1) * row_bytes]);
    }
}

fn upload(host: &[u8], gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let p = gpu.alloc(host.len())?;
    gpu.copy_h2d(host, p)?;
    Ok(p)
}

/// 2026-09-25: `q_b_proj` `[n_heads*(nope+rope), q_lora]`: de-interleave each
/// head's rope rows, scale by `scale_q`, and pad each head to `padded_hd` as
/// `[nope | zeros | rope]`, rope at `[padded_nope, padded_hd)`.
pub(super) fn prep_q_b(
    store: &WeightStore,
    name: &str,
    n_heads: usize,
    nope: usize,
    rope: usize,
    q_lora: usize,
    scale_q: f32,
    padded_hd: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let hd = nope + rope;
    anyhow::ensure!(
        padded_hd >= hd,
        "longcat prep: padded_hd {padded_hd} < {hd}"
    );
    let bytes = n_heads * hd * q_lora * BF16;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    for head in 0..n_heads {
        deinterleave_rope_rows(&mut host, head * hd + nope, rope, q_lora);
    }
    if (scale_q - 1.0).abs() > f32::EPSILON {
        let mut f = to_f32(&host);
        for v in &mut f {
            *v *= scale_q;
        }
        host = to_bf16(&f);
    }
    // 2026-09-25: Pad: [nope | rope] -> [nope | zeros | rope] at width `padded_hd`.
    let row = q_lora * BF16;
    let padded_nope = padded_hd - rope;
    let mut out = vec![0u8; n_heads * padded_hd * row];
    for head in 0..n_heads {
        let src = head * hd * row;
        let dst = head * padded_hd * row;
        out[dst..dst + nope * row].copy_from_slice(&host[src..src + nope * row]);
        let rsrc = src + nope * row;
        let rdst = dst + padded_nope * row;
        out[rdst..rdst + rope * row].copy_from_slice(&host[rsrc..rsrc + rope * row]);
    }
    Ok(DenseWeight {
        weight: upload(&out, gpu)?,
    })
}

/// 2026-09-25: `kv_b_proj` `[n_heads*(nope+v_dim), kv_lora]`: pad each head's
/// block from `[nope | v]` to `[nope | zeros | v | zeros]`, `padded_nope +
/// padded_hd` rows, so K (`padded_nope + rope`) and V are both `padded_hd` wide.
pub(super) fn prep_kv_b(
    store: &WeightStore,
    name: &str,
    n_heads: usize,
    nope: usize,
    v_dim: usize,
    rope: usize,
    kv_lora: usize,
    padded_hd: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let row = kv_lora * BF16;
    let src_stride = nope + v_dim;
    let mut host = vec![0u8; n_heads * src_stride * row];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    let padded_nope = padded_hd - rope;
    let dst_stride = padded_nope + padded_hd;
    let mut out = vec![0u8; n_heads * dst_stride * row];
    for head in 0..n_heads {
        let src = head * src_stride * row;
        let dst = head * dst_stride * row;
        out[dst..dst + nope * row].copy_from_slice(&host[src..src + nope * row]);
        let vsrc = src + nope * row;
        let vdst = dst + padded_nope * row;
        out[vdst..vdst + v_dim * row].copy_from_slice(&host[vsrc..vsrc + v_dim * row]);
    }
    Ok(DenseWeight {
        weight: upload(&out, gpu)?,
    })
}

/// 2026-09-25: `o_proj` `[hidden, n_heads*v_dim]`: widen to
/// `[hidden, n_heads*padded_hd]` with zero columns in each head's pad lanes, so
/// whatever the attention leaves in a padded V lane is multiplied by zero.
pub(super) fn prep_o_proj(
    store: &WeightStore,
    name: &str,
    hidden: usize,
    n_heads: usize,
    v_dim: usize,
    padded_hd: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let src_cols = n_heads * v_dim;
    let dst_cols = n_heads * padded_hd;
    let mut host = vec![0u8; hidden * src_cols * BF16];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    let mut out = vec![0u8; hidden * dst_cols * BF16];
    for r in 0..hidden {
        for head in 0..n_heads {
            let s = (r * src_cols + head * v_dim) * BF16;
            let d = (r * dst_cols + head * padded_hd) * BF16;
            out[d..d + v_dim * BF16].copy_from_slice(&host[s..s + v_dim * BF16]);
        }
    }
    Ok(DenseWeight {
        weight: upload(&out, gpu)?,
    })
}

/// 2026-09-25: `kv_a_proj_with_mqa` `[kv_lora + rope, hidden]`: de-interleave
/// the trailing rope rows (`k_rot`). Not scaled: `k_rot` bypasses
/// `kv_a_layernorm`.
pub(super) fn prep_kv_a(
    store: &WeightStore,
    name: &str,
    kv_lora: usize,
    rope: usize,
    hidden: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let bytes = (kv_lora + rope) * hidden * BF16;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    deinterleave_rope_rows(&mut host, kv_lora, rope, hidden);
    Ok(DenseWeight {
        weight: upload(&host, gpu)?,
    })
}

/// 2026-09-25: `kv_a_layernorm.weight` `[kv_lora]` scaled by `scale_kv`; the
/// weight is returned as is when `scale_kv` is 1.
pub(super) fn prep_kv_a_norm(
    store: &WeightStore,
    name: &str,
    kv_lora: usize,
    scale_kv: f32,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    if (scale_kv - 1.0).abs() <= f32::EPSILON {
        return Ok(w);
    }
    let mut host = vec![0u8; kv_lora * BF16];
    gpu.copy_d2h(w.weight, &mut host)?;
    let mut f = to_f32(&host);
    for v in &mut f {
        *v *= scale_kv;
    }
    Ok(DenseWeight {
        weight: upload(&to_bf16(&f), gpu)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleave_moves_even_rows_first() {
        // 2026-09-25: 4 rope rows of width 1: [a,b,c,d] -> [a,c,b,d], even
        // rows to the front half, odd rows to the back half.
        let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut host = to_bf16(&vals);
        deinterleave_rope_rows(&mut host, 0, 4, 1);
        assert_eq!(to_f32(&host), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn deinterleave_respects_base_row_and_width() {
        // 2026-09-25: 2 leading rows untouched, then 4 rope rows of width 2.
        let vals: [f32; 12] = [
            -1.0, -1.0, -2.0, -2.0, // 2026-09-25: leading rows
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
        ];
        let mut host = to_bf16(&vals);
        deinterleave_rope_rows(&mut host, 2, 4, 2);
        let got = to_f32(&host);
        assert_eq!(&got[..4], &[-1.0, -1.0, -2.0, -2.0]);
        assert_eq!(&got[4..], &[1.0, 1.5, 3.0, 3.5, 2.0, 2.5, 4.0, 4.5]);
    }
}
