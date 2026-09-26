// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A full-attention block forward on real Qwen3.5-4B-MLX-8bit layer 3
//! weights.
//!
//! Owner: gpu-runtime (metal tests).
//! Invariants: none beyond the types.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// 2026-09-25: Loads every weight of `language_model.model.layers.3` and runs one
/// token through:
///
///   x → rms_norm(input_ln)
///     → q_proj | k_proj | v_proj             (3× mlx_int8_gemv)
///     → the two halves of the q_proj output as Q and attn_gate
///     → q_norm(Q per head) | k_norm(K per head)  (2× rms_norm)
///     → rope_apply(Q) | rope_apply(K)
///     → kv_cache_append at pos 0
///     → attention_decode with seq_len=1
///     → sigmoid_gate(attn_gate, attn_out)
///     → o_proj                                (mlx_int8_gemv)
///     → bf16_add(x, o)                       (residual)
///     → rms_norm(post_attention_ln)
///     → gate_proj | up_proj                  (2× mlx_int8_gemv)
///     → silu_gate
///     → down_proj                            (mlx_int8_gemv)
///     → bf16_add(x_resid, ffn_out)            (residual)
///     → x_final
///
/// It asserts that every element of `x_final` is finite, that the mean |x_final|
/// lies in (1e-3, 50) and that max |x_final| < 1e3. There is no numeric
/// reference; each kernel of the chain has its own parity test in this suite.
/// Ignored by default: it needs the checkpoint in `METRALE_MLX_MODEL_DIR` (default
/// `~/models/Qwen3.5-4B-MLX-8bit`).
#[test]
#[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
fn metal_real_model_full_attention_block_layer3() {
    use crate::mlx_int8::MlxInt8Weight;
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

    let hidden_size: u32 = 2560;
    let num_heads: u32 = 16;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 256;
    let intermediate_size: u32 = 9216;
    let rms_eps: f32 = 1e-6;
    let group_size: u32 = 64;
    let q_total: u32 = num_heads * head_dim * 2;
    let q_only: u32 = num_heads * head_dim;
    let kv_dim: u32 = num_kv_heads * head_dim;

    let layer = "language_model.model.layers.3";
    let Some(backend) = maybe_backend() else {
        return;
    };

    let load_bf16 = |name: &str| -> DevicePtr {
        let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
        let p = backend.alloc(t.data().len()).unwrap();
        backend.copy_h2d(t.data(), p).unwrap();
        p
    };
    let input_ln = load_bf16(&format!("{layer}.input_layernorm.weight"));
    let q_norm = load_bf16(&format!("{layer}.self_attn.q_norm.weight"));
    let k_norm = load_bf16(&format!("{layer}.self_attn.k_norm.weight"));
    let post_ln = load_bf16(&format!("{layer}.post_attention_layernorm.weight"));

    let q_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.q_proj"),
        group_size,
    )
    .expect("load q_proj");
    let k_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.k_proj"),
        group_size,
    )
    .expect("load k_proj");
    let v_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.v_proj"),
        group_size,
    )
    .expect("load v_proj");
    let o_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.o_proj"),
        group_size,
    )
    .expect("load o_proj");
    let gate_p = MlxInt8Weight::load(&backend, &st, &format!("{layer}.mlp.gate_proj"), group_size)
        .expect("load gate_proj");
    let up_p = MlxInt8Weight::load(&backend, &st, &format!("{layer}.mlp.up_proj"), group_size)
        .expect("load up_proj");
    let down_p = MlxInt8Weight::load(&backend, &st, &format!("{layer}.mlp.down_proj"), group_size)
        .expect("load down_proj");

    assert_eq!(q_proj.out_features, q_total);
    assert_eq!(q_proj.in_features, hidden_size);
    assert_eq!(k_proj.out_features, kv_dim);
    assert_eq!(o_proj.in_features, q_only);
    assert_eq!(o_proj.out_features, hidden_size);
    assert_eq!(gate_p.out_features, intermediate_size);
    assert_eq!(down_p.in_features, intermediate_size);

    let x_init: Vec<half::bf16> = (0..hidden_size)
        .map(|i| half::bf16::from_f32(0.4 * (i as f32 * 0.013).sin()))
        .collect();
    let x_bytes = bf16_slice_to_bytes(&x_init);

    let alloc_bf16 = |n: u32| -> DevicePtr { backend.alloc(n as usize * 2).unwrap() };
    let x = alloc_bf16(hidden_size);
    let x_norm = alloc_bf16(hidden_size);
    let q_full = alloc_bf16(q_total);
    let k = alloc_bf16(kv_dim);
    let v = alloc_bf16(kv_dim);
    let attn_out = alloc_bf16(q_only);
    let gated_attn = alloc_bf16(q_only);
    let o = alloc_bf16(hidden_size);
    let x_resid = alloc_bf16(hidden_size);
    let x_norm2 = alloc_bf16(hidden_size);
    let gate_act = alloc_bf16(intermediate_size);
    let up_act = alloc_bf16(intermediate_size);
    let ffn_act = alloc_bf16(intermediate_size);
    let ffn_out = alloc_bf16(hidden_size);
    let x_final = alloc_bf16(hidden_size);
    let max_seq: u32 = 1;
    let k_cache = alloc_bf16(max_seq * kv_dim);
    let v_cache = alloc_bf16(max_seq * kv_dim);
    backend.copy_h2d(&x_bytes, x).unwrap();

    let rope_theta: f32 = 10_000_000.0;
    let rotary_dim: u32 = head_dim / 4;
    let half_dim = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    let inv_freq_bytes: Vec<u8> = inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect();
    let inv_freq_ptr = backend.alloc(inv_freq_bytes.len()).unwrap();
    backend.copy_h2d(&inv_freq_bytes, inv_freq_ptr).unwrap();
    let positions: Vec<u8> = 0u32.to_le_bytes().to_vec();
    let positions_ptr = backend.alloc(positions.len()).unwrap();
    backend.copy_h2d(&positions, positions_ptr).unwrap();

    let stream = backend.default_stream();
    let n_tokens: u32 = 1;

    let rms = backend.kernel("rms_norm", "rms_norm").unwrap();
    let launch_rms = |x_in: DevicePtr, w: DevicePtr, x_out: DevicePtr, n_tok: u32, hid: u32| {
        backend
            .launch_typed(
                rms,
                [n_tok, 1, 1],
                [128, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&hid.to_le_bytes()),
                    KernelArg::Bytes(&rms_eps.to_le_bytes()),
                    KernelArg::Buffer(x_in),
                    KernelArg::Buffer(w),
                    KernelArg::Buffer(x_out),
                ],
            )
            .expect("rms_norm launch");
    };
    launch_rms(x, input_ln, x_norm, n_tokens, hidden_size);

    q_proj.gemv(&backend, x_norm, q_full, stream).unwrap();
    k_proj.gemv(&backend, x_norm, k, stream).unwrap();
    v_proj.gemv(&backend, x_norm, v, stream).unwrap();

    let q_view = q_full;
    let gate_view = q_full.offset(q_only as usize * 2);

    // 2026-09-25: Each head is one `rms_norm` row of length `head_dim`.
    let q_norm_out = alloc_bf16(q_only);
    let k_norm_out = alloc_bf16(kv_dim);
    launch_rms(q_view, q_norm, q_norm_out, num_heads, head_dim);
    launch_rms(k, k_norm, k_norm_out, num_kv_heads, head_dim);
    backend
        .copy_d2d_async(q_norm_out, q_view, q_only as usize * 2, stream)
        .unwrap();
    backend
        .copy_d2d_async(k_norm_out, k, kv_dim as usize * 2, stream)
        .unwrap();

    let rope = backend.kernel("rope_apply", "rope_apply").unwrap();
    let launch_rope = |x_inout: DevicePtr, n_h: u32| {
        backend
            .launch_typed(
                rope,
                [half_dim, n_h, 1],
                [1, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&n_tokens.to_le_bytes()),
                    KernelArg::Bytes(&n_h.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&rotary_dim.to_le_bytes()),
                    KernelArg::Buffer(positions_ptr),
                    KernelArg::Buffer(inv_freq_ptr),
                    KernelArg::Buffer(x_inout),
                ],
            )
            .expect("rope launch");
    };
    launch_rope(q_view, num_heads);
    launch_rope(k, num_kv_heads);

    let cache_pos: u32 = 0;
    let kvap = backend
        .kernel("kv_cache_append", "kv_cache_append")
        .unwrap();
    backend
        .launch_typed(
            kvap,
            [head_dim, num_kv_heads, 1],
            [1, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&cache_pos.to_le_bytes()),
                KernelArg::Buffer(k),
                KernelArg::Buffer(v),
                KernelArg::Buffer(k_cache),
                KernelArg::Buffer(v_cache),
            ],
        )
        .expect("kv_cache_append launch");

    let seq_len: u32 = 1;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let attn = backend
        .kernel("attention_decode", "attention_decode")
        .unwrap();
    backend
        .launch_typed(
            attn,
            [num_heads, 1, 1],
            [32, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(q_view),
                KernelArg::Buffer(k_cache),
                KernelArg::Buffer(v_cache),
                KernelArg::Buffer(attn_out),
            ],
        )
        .expect("attention_decode launch");

    let sg = backend.kernel("sigmoid_gate", "sigmoid_gate").unwrap();
    backend
        .launch_typed(
            sg,
            [q_only.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&q_only.to_le_bytes()),
                KernelArg::Buffer(gate_view),
                KernelArg::Buffer(attn_out),
                KernelArg::Buffer(gated_attn),
            ],
        )
        .expect("sigmoid_gate launch");

    o_proj.gemv(&backend, gated_attn, o, stream).unwrap();

    let add = backend.kernel("bf16_add", "bf16_add").unwrap();
    let launch_add = |a: DevicePtr, b: DevicePtr, out: DevicePtr, n: u32| {
        backend
            .launch_typed(
                add,
                [n.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(a),
                    KernelArg::Buffer(b),
                    KernelArg::Buffer(out),
                ],
            )
            .expect("bf16_add launch");
    };
    launch_add(x, o, x_resid, hidden_size);

    launch_rms(x_resid, post_ln, x_norm2, n_tokens, hidden_size);

    gate_p.gemv(&backend, x_norm2, gate_act, stream).unwrap();
    up_p.gemv(&backend, x_norm2, up_act, stream).unwrap();
    let silu = backend.kernel("silu_gate", "silu_gate").unwrap();
    backend
        .launch_typed(
            silu,
            [intermediate_size.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&intermediate_size.to_le_bytes()),
                KernelArg::Buffer(gate_act),
                KernelArg::Buffer(up_act),
                KernelArg::Buffer(ffn_act),
            ],
        )
        .expect("silu_gate launch");
    down_p.gemv(&backend, ffn_act, ffn_out, stream).unwrap();

    launch_add(x_resid, ffn_out, x_final, hidden_size);

    backend.synchronize(stream).unwrap();

    let mut x_final_raw = vec![0u8; hidden_size as usize * 2];
    backend.copy_d2h(x_final, &mut x_final_raw).unwrap();
    let final_vals = bytes_to_bf16_vec(&x_final_raw);

    let mut nan_or_inf = 0;
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    for v in &final_vals {
        let f = v.to_f32();
        if !f.is_finite() {
            nan_or_inf += 1;
        }
        let a = f.abs();
        sum_abs += a;
        if a > max_abs {
            max_abs = a;
        }
    }
    let mean_abs = sum_abs / final_vals.len() as f32;

    assert_eq!(
        nan_or_inf,
        0,
        "x_final has {nan_or_inf} non-finite values out of {}",
        final_vals.len()
    );
    assert!(
        mean_abs > 1e-3 && mean_abs < 50.0,
        "x_final mean-abs {mean_abs} outside sanity band; max_abs={max_abs}"
    );
    assert!(
        max_abs < 1e3,
        "x_final max_abs {max_abs} suggests activation explosion"
    );

    for ptr in [
        input_ln,
        q_norm,
        k_norm,
        post_ln,
        x,
        x_norm,
        q_full,
        k,
        v,
        q_norm_out,
        k_norm_out,
        attn_out,
        gated_attn,
        o,
        x_resid,
        x_norm2,
        gate_act,
        up_act,
        ffn_act,
        ffn_out,
        x_final,
        k_cache,
        v_cache,
        inv_freq_ptr,
        positions_ptr,
    ] {
        backend.free(ptr).unwrap();
    }
    for w in [&q_proj, &k_proj, &v_proj, &o_proj, &gate_p, &up_p, &down_p] {
        w.release(&backend).unwrap();
    }
}
