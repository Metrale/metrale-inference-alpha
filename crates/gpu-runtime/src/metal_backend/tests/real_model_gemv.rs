// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `mlx_int8_gemv` parity on real checkpoint weights (layer 3
//! `q_proj` of Qwen3.5-4B-MLX-8bit).
//!
//! Owner: gpu-runtime (metal tests).
//! Invariants: none beyond the types.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
use crate::mlx_int8::MlxInt8Weight;

/// 2026-09-25: Loads `language_model.model.layers.3.self_attn.q_proj`, keeps its
/// first 128 output rows, runs `MlxInt8Weight::gemv` on a synthetic activation of
/// the model's hidden size, and compares with a CPU reference that dequantises
/// the same bytes. Ignored by default: it needs the checkpoint in
/// `METRALE_MLX_MODEL_DIR` (default `~/models/Qwen3.5-4B-MLX-8bit`).
#[test]
#[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
fn metal_mlx_int8_gemv_real_model_q_proj() {
    use safetensors::SafeTensors;

    let model_dir = std::env::var("METRALE_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    if !st_path.exists() {
        eprintln!("skipping: {} not found", st_path.display());
        return;
    }

    let file = std::fs::File::open(&st_path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

    let base = "language_model.model.layers.3.self_attn.q_proj";
    let weight = st.tensor(&format!("{base}.weight")).unwrap();
    let scales = st.tensor(&format!("{base}.scales")).unwrap();
    let biases = st.tensor(&format!("{base}.biases")).unwrap();

    let weight_shape = weight.shape();
    let full_out = weight_shape[0];
    let in_packed_cols = weight_shape[1];
    let in_features = (in_packed_cols * 4) as u32;
    assert_eq!(
        in_features, 2560,
        "expected hidden_size=2560 for Qwen3.5-4B"
    );
    assert_eq!(
        full_out, 8192,
        "expected num_heads*head_dim*2=8192 for layer 3 q_proj (with attn output gate)"
    );

    let n_rows: usize = 128;
    let group_size: u32 = 64;
    let groups_per_row = (in_features / group_size) as usize;

    let weight_data = weight.data();
    let scales_data = scales.data();
    let biases_data = biases.data();

    let row_stride_packed = in_packed_cols * 4;
    let row_stride_scales = groups_per_row * 2;

    let mut packed_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_packed);
    let mut scales_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_scales);
    let mut biases_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_scales);
    for r in 0..n_rows {
        let p_off = r * row_stride_packed;
        packed_slice.extend_from_slice(&weight_data[p_off..p_off + row_stride_packed]);
        let s_off = r * row_stride_scales;
        scales_slice.extend_from_slice(&scales_data[s_off..s_off + row_stride_scales]);
        biases_slice.extend_from_slice(&biases_data[s_off..s_off + row_stride_scales]);
    }

    let x_bf16: Vec<half::bf16> = (0..in_features)
        .map(|i| half::bf16::from_f32(0.05 + 0.001 * (i as f32).sin()))
        .collect();

    // 2026-09-25: `sum_abs_terms[r]` is the row's sum of |w * x|; it sizes the
    // fp32 reordering allowance of the bound below.
    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows];
    let mut sum_abs_terms: Vec<f32> = vec![0.0; n_rows];
    for r in 0..n_rows {
        let mut acc: f32 = 0.0;
        for c in 0..in_features as usize {
            let word_off = r * row_stride_packed + (c / 4) * 4;
            let word = u32::from_le_bytes([
                packed_slice[word_off],
                packed_slice[word_off + 1],
                packed_slice[word_off + 2],
                packed_slice[word_off + 3],
            ]);
            let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
            let g = c / group_size as usize;
            let s_idx = (r * groups_per_row + g) * 2;
            let s =
                half::bf16::from_le_bytes([scales_slice[s_idx], scales_slice[s_idx + 1]]).to_f32();
            let b =
                half::bf16::from_le_bytes([biases_slice[s_idx], biases_slice[s_idx + 1]]).to_f32();
            let w = byte * s + b;
            let term = w * x_bf16[c].to_f32();
            acc += term;
            sum_abs_terms[r] += term.abs();
        }
        expected[r] = half::bf16::from_f32(acc);
    }

    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = n_rows as u32;
    let k: u32 = in_features;

    let packed_ptr = backend.alloc(packed_slice.len()).unwrap();
    let scales_ptr = backend.alloc(scales_slice.len()).unwrap();
    let biases_ptr = backend.alloc(biases_slice.len()).unwrap();
    let x_bytes = bf16_slice_to_bytes(&x_bf16);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let y_ptr = backend.alloc(n_rows * 2).unwrap();
    backend.copy_h2d(&packed_slice, packed_ptr).unwrap();
    backend.copy_h2d(&scales_slice, scales_ptr).unwrap();
    backend.copy_h2d(&biases_slice, biases_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    // 2026-09-25: Launch through `MlxInt8Weight::gemv`, so the test uses the
    // geometry production uses: `ceil(N/4)` threadgroups of 128 threads.
    let weight = MlxInt8Weight {
        packed: packed_ptr,
        scales: scales_ptr,
        biases: biases_ptr,
        out_features: n,
        in_features: k,
        group_size,
    };
    weight
        .gemv(&backend, x_ptr, y_ptr, backend.default_stream())
        .expect("launch real-model gemv");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; n_rows * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    // 2026-09-25: Per row, kernel and reference sum the same fp32 products in
    // different orders (lane-strided plus `simd_sum`, against sequential) and
    // round once to BF16. They may differ by one BF16 ulp of the value plus a
    // reordering allowance: a K-term fp32 sum carries at most
    // (K-1) * eps * sum|terms| of rounding error, so two orders differ by at most
    // twice that, and the +2 terms cover an fma contraction of `byte * s + b`,
    // which the kernels' -ffast-math permits. Both the per-row bound and the
    // cosine / norm-ratio gates must hold.
    let mut max_abs_diff: f32 = 0.0;
    let mut worst_ratio: f32 = 0.0;
    for i in 0..n_rows {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        assert!(
            a.is_finite(),
            "real-model gemv produced non-finite at row {i}: {a}"
        );
        let d = (e - a).abs();
        max_abs_diff = max_abs_diff.max(d);
        let fp32_allowance = (2.0 * k as f32 + 2.0) * f32::EPSILON * sum_abs_terms[i];
        let bound = bf16_ulp(e.abs().max(a.abs())) + fp32_allowance;
        worst_ratio = worst_ratio.max(d / bound);
        assert!(
            d <= bound,
            "real-model gemv row {i}: |kernel - cpu| = {d} exceeds the derived \
             bound {bound} (1 bf16 ulp + fp32 reorder allowance {fp32_allowance}); \
             expected {e}, got {a}"
        );
    }
    let cos = cosine_bf16(&expected, &actual);
    let mag = norm_ratio_bf16(&expected, &actual);
    eprintln!(
        "metal_mlx_int8_gemv_real_model_q_proj: rows={n_rows} K={k} \
         max_abs={max_abs_diff:.3e} worst_d/bound={worst_ratio:.3} \
         cos={cos:.7} norm_ratio={mag:.7}"
    );
    assert!(
        cos >= COSINE_GATE,
        "real-model gemv: cosine {cos} < {COSINE_GATE}"
    );
    assert!(
        mag >= COSINE_GATE,
        "real-model gemv: norm ratio {mag} < {COSINE_GATE}"
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}
