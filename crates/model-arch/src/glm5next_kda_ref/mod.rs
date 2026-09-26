// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU reference for GLM-5.3-Flash KDA (Kimi Delta Attention).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA reference).
//! Invariants:
//! - Nothing here runs on a GPU or reads a checkpoint; production code does not call it.
//! - `lower_bound` and `rms_eps` are arguments; the reference has no built-in value for either.
//!
//! It is checked against `kda_golden.json`, which `gen_kda_golden.py` produces by calling the
//! `transformers` `glm5_next` functions (`Glm5NextTextForgetGate`, `Glm5NextTextRMSNormGated`,
//! `l2norm`, `recurrent_kimi_delta_attention`, `chunk_kimi_delta_attention`).
//!
//! KDA differs from the Qwen GDN path (`compute_gdn_gates` in
//! `kernels/gb10/common/ssm_preprocess.cu`) in four ways:
//! 1. The decay is per (head, key channel), `[T, H, head_dim]`; GDN writes one value per head.
//! 2. The decay comes from a low-rank projection, `f_a: hidden -> head_dim`,
//!    `f_b: head_dim -> H*head_dim`; GDN reads it from the fused `BA` projection.
//! 3. The gate is bounded, `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`; GDN computes
//!    `exp(-exp(A_log) * softplus(a + dt_bias))`.
//! 4. The output gate is low-rank `g_a`/`g_b`; GDN takes a full-rank `Z` from its fused `QKVZ`
//!    projection, and a KDA checkpoint has no `Z` tensor.
//!
//! # Layout conventions
//!
//! * `q`/`k`/`v`/`gate`: `[T, H, D]`, row-major, index `((t * H) + h) * D + d`.
//! * `beta`: `[T, H]`, index `t * H + h`.
//! * `dt_bias`: `[H * D]`, per channel.
//! * `a_log`: `[H]`, per head.
//! * `state`: `[H, D_k, D_v]`, index `(h * D_k + kd) * D_v + vd`.
//! * Weights follow the torch `Linear` convention `[out, in]`, row-major.

/// 2026-09-25: KDA geometry. The low-rank width of both `f_a`/`f_b` and `g_a`/`g_b` is
/// `head_dim`.
#[derive(Clone, Copy, Debug)]
pub struct KdaDims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub tokens: usize,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 2026-09-25: `x / sqrt(sum(x^2) + eps)` over the trailing dimension, not `x / max(norm, eps)`;
/// the two differ for small-norm rows. The GPU `l2_norm_bf16` uses the same form.
pub fn l2norm_rows(x: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let inv = 1.0 / (row_in.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        for (o, i) in row_out.iter_mut().zip(row_in) {
            *o = i * inv;
        }
    }
    out
}

/// 2026-09-25: `y = x @ w^T` for `x: [m, k]`, `w: [n, k]` (torch `Linear` weight layout), no bias.
pub fn linear(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// 2026-09-25: The bounded GLM forget gate:
///
/// `gate[t, h, d] = lower_bound * sigmoid(exp(a_log[h]) * (g_lowrank[t, h*D + d] + dt_bias[h*D + d]))`
///
/// `a_log` is per head and `dt_bias` per channel, so they are separate arguments.
pub fn bounded_gate(
    g_lowrank: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    dims: KdaDims,
    lower_bound: f32,
) -> Vec<f32> {
    let (h_n, d) = (dims.heads, dims.head_dim);
    let mut out = vec![0.0f32; dims.tokens * h_n * d];
    for t in 0..dims.tokens {
        for h in 0..h_n {
            let decay = a_log[h].exp();
            for dd in 0..d {
                let ch = h * d + dd;
                let g = g_lowrank[t * h_n * d + ch] + dt_bias[ch];
                out[(t * h_n + h) * d + dd] = lower_bound * sigmoid(decay * g);
            }
        }
    }
    out
}

/// 2026-09-25: The unbounded Qwen GDN gate, used by the tests to show the two laws diverge.
///
/// `compute_gdn_gates` stores `exp(g)`; this returns `g`, as the KDA recurrence takes the
/// exponential itself.
pub fn unbounded_gdn_gate(g_raw: f32, dt_bias: f32, a_log: f32) -> f32 {
    let x = g_raw + dt_bias;
    // 2026-09-25: softplus(x), linear above 20.
    let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    -a_log.exp() * softplus
}

/// 2026-09-25: Decode formulation: one token at a time, carrying `state`.
///
/// ```text
/// S <- S * diag(exp(g_t))          decay along the key axis, per channel
/// delta <- (v_t - S^T k_t) * beta_t
/// S <- S + k_t (x) delta
/// o_t <- S^T q_t
/// ```
///
/// `q`/`k` are L2-normalised and `q` scaled by `1/sqrt(head_dim)` inside this function, as the
/// golden's `use_qk_l2norm_in_kernel=True` run does. Pass raw post-conv q/k/v.
pub fn kda_recurrent(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    dims: KdaDims,
    state: &mut [f32],
) -> Vec<f32> {
    let d = dims.head_dim;
    let qn = l2norm_rows(q, d, 1e-6);
    let kn = l2norm_rows(k, d, 1e-6);
    kda_recurrent_prenorm(&qn, &kn, v, gate, beta, dims, state)
}

/// 2026-09-25: Same recurrence, but `q`/`k` are already L2-normalised, the input the
/// `kda_recurrent` GPU kernel gets from `causal_conv1d_update_l2norm`.
///
/// Passing normalised vectors to [`kda_recurrent`] instead would normalise them again: nearly a
/// no-op in fp32, but on bf16-rounded vectors it undoes the rounding error in the norm, so the
/// reference would no longer match the kernel's input.
pub fn kda_recurrent_prenorm(
    qn: &[f32],
    kn: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    dims: KdaDims,
    state: &mut [f32],
) -> Vec<f32> {
    let (h_n, d, t_n) = (dims.heads, dims.head_dim, dims.tokens);
    let scale = 1.0 / (d as f32).sqrt();

    let mut out = vec![0.0f32; t_n * h_n * d];
    let mut delta = vec![0.0f32; d];
    for t in 0..t_n {
        for h in 0..h_n {
            let base = (t * h_n + h) * d;
            let s = &mut state[h * d * d..(h + 1) * d * d];

            for kd in 0..d {
                let decay = gate[base + kd].exp();
                for vd in 0..d {
                    s[kd * d + vd] *= decay;
                }
            }
            let b = beta[t * h_n + h];
            for vd in 0..d {
                let mut kv = 0.0f32;
                for kd in 0..d {
                    kv += s[kd * d + vd] * kn[base + kd];
                }
                delta[vd] = (v[base + vd] - kv) * b;
            }
            for kd in 0..d {
                let kk = kn[base + kd];
                for vd in 0..d {
                    s[kd * d + vd] += kk * delta[vd];
                }
            }
            for vd in 0..d {
                let mut acc = 0.0f32;
                for kd in 0..d {
                    acc += s[kd * d + vd] * qn[base + kd] * scale;
                }
                out[base + vd] = acc;
            }
        }
    }
    out
}

/// 2026-09-25: Prefill formulation: the chunked (WY-style) delta rule with a per-channel decay
/// mask, checked against the golden's `chunk_kimi_delta_attention` output. `tokens % chunk != 0`
/// is handled by zero-padding.
pub fn kda_chunked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    dims: KdaDims,
    chunk: usize,
    state: &mut [f32],
) -> Vec<f32> {
    let d = dims.head_dim;
    // 2026-09-25: L2 runs on the real tokens; padding happens after, in the prenorm call.
    let qn = l2norm_rows(q, d, 1e-6);
    let kn = l2norm_rows(k, d, 1e-6);
    kda_chunked_prenorm(&qn, &kn, v, gate, beta, dims, chunk, state)
}

/// 2026-09-25: Same chunked formulation, but `q`/`k` are already L2-normalised, as the GPU
/// prefill's `l2_norm_bf16` leaves them (in bf16). Separate from [`kda_chunked`] for the reason
/// given on [`kda_recurrent_prenorm`].
#[allow(clippy::too_many_arguments)]
pub fn kda_chunked_prenorm(
    qn: &[f32],
    kn: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    dims: KdaDims,
    chunk: usize,
    state: &mut [f32],
) -> Vec<f32> {
    let (h_n, d, t_n) = (dims.heads, dims.head_dim, dims.tokens);
    let scale = 1.0 / (d as f32).sqrt();
    let pad = (chunk - t_n % chunk) % chunk;
    let tt = t_n + pad;
    let n_chunks = tt / chunk;

    let mut out = vec![0.0f32; t_n * h_n * d];
    for h in 0..h_n {
        // 2026-09-25: Per-head padded views. Pad rows are zero, so their k/v contributions vanish
        // and their decay adds 0 to the cumulative sum.
        let mut qh = vec![0.0f32; tt * d];
        let mut kh = vec![0.0f32; tt * d];
        let mut vh = vec![0.0f32; tt * d];
        let mut gh = vec![0.0f32; tt * d];
        let mut bh = vec![0.0f32; tt];
        for t in 0..t_n {
            let src = (t * h_n + h) * d;
            for dd in 0..d {
                qh[t * d + dd] = qn[src + dd] * scale;
                kh[t * d + dd] = kn[src + dd];
                vh[t * d + dd] = v[src + dd];
                gh[t * d + dd] = gate[src + dd];
            }
            bh[t] = beta[t * h_n + h];
        }

        let s = &mut state[h * d * d..(h + 1) * d * d];
        for c in 0..n_chunks {
            let off = c * chunk;
            // 2026-09-25: Cumulative decay within the chunk, per channel.
            let mut gc = vec![0.0f32; chunk * d];
            for i in 0..chunk {
                for dd in 0..d {
                    let prev = if i == 0 { 0.0 } else { gc[(i - 1) * d + dd] };
                    gc[i * d + dd] = prev + gh[(off + i) * d + dd];
                }
            }
            let dmask = |i: usize, j: usize, dd: usize| (gc[i * d + dd] - gc[j * d + dd]).exp();

            // 2026-09-25: attn[i][j] = -sum_dd k_beta[i][dd] * k[j][dd] * exp(gc[i][dd] - gc[j][dd]),
            // j < i.
            let mut attn = vec![0.0f32; chunk * chunk];
            for i in 0..chunk {
                for j in 0..i {
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += kh[(off + i) * d + dd]
                            * bh[off + i]
                            * kh[(off + j) * d + dd]
                            * dmask(i, j, dd);
                    }
                    attn[i * chunk + j] = -acc;
                }
            }
            // 2026-09-25: Forward substitution: attn[i, :i] += attn[i, :i] @ attn[:i, :i].
            for i in 1..chunk {
                let row: Vec<f32> = (0..i).map(|j| attn[i * chunk + j]).collect();
                for j in 0..i {
                    let mut acc = 0.0f32;
                    for (m, r) in row.iter().enumerate() {
                        acc += r * attn[m * chunk + j];
                    }
                    attn[i * chunk + j] = row[j] + acc;
                }
            }
            for i in 0..chunk {
                attn[i * chunk + i] = 1.0;
            }

            // 2026-09-25: value = attn @ v_beta; k_cumdecay = attn @ (k_beta * exp(gc)).
            let mut value = vec![0.0f32; chunk * d];
            let mut k_cumdecay = vec![0.0f32; chunk * d];
            for i in 0..chunk {
                for dd in 0..d {
                    let (mut av, mut ak) = (0.0f32, 0.0f32);
                    for j in 0..chunk {
                        let a = attn[i * chunk + j];
                        av += a * vh[(off + j) * d + dd] * bh[off + j];
                        ak += a * kh[(off + j) * d + dd] * bh[off + j] * gc[j * d + dd].exp();
                    }
                    value[i * d + dd] = av;
                    k_cumdecay[i * d + dd] = ak;
                }
            }

            // 2026-09-25: v_new = value - k_cumdecay @ S.
            let mut v_new = vec![0.0f32; chunk * d];
            for i in 0..chunk {
                for vd in 0..d {
                    let mut vp = 0.0f32;
                    for kd in 0..d {
                        vp += k_cumdecay[i * d + kd] * s[kd * d + vd];
                    }
                    v_new[i * d + vd] = value[i * d + vd] - vp;
                }
            }

            // 2026-09-25: out = (q * exp(gc)) @ S + attn_intra @ v_new.
            for i in 0..chunk {
                if off + i >= t_n {
                    continue;
                }
                let dst = ((off + i) * h_n + h) * d;
                for vd in 0..d {
                    let mut acc = 0.0f32;
                    for kd in 0..d {
                        acc += qh[(off + i) * d + kd] * gc[i * d + kd].exp() * s[kd * d + vd];
                    }
                    out[dst + vd] = acc;
                }
                for j in 0..=i {
                    let mut intra = 0.0f32;
                    for dd in 0..d {
                        intra += qh[(off + i) * d + dd] * kh[(off + j) * d + dd] * dmask(i, j, dd);
                    }
                    for vd in 0..d {
                        out[dst + vd] += intra * v_new[j * d + vd];
                    }
                }
            }

            // 2026-09-25: S <- S * exp(gc_last) + sum_i k[i] * exp(gc_last - gc[i]) (x) v_new[i].
            let last = (chunk - 1) * d;
            for kd in 0..d {
                let gl = gc[last + kd];
                for vd in 0..d {
                    s[kd * d + vd] *= gl.exp();
                }
                for i in 0..chunk {
                    let w = kh[(off + i) * d + kd] * (gl - gc[i * d + kd]).exp();
                    for vd in 0..d {
                        s[kd * d + vd] += w * v_new[i * d + vd];
                    }
                }
            }
        }
    }
    out
}

/// 2026-09-25: Gated RMSNorm over the trailing `d`, FP32, with a sigmoid gate. `eps` is the
/// caller's `rms_norm_eps`.
pub fn rms_norm_gated(x: &[f32], weight: &[f32], gate: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for i in (0..x.len()).step_by(d) {
        let var = x[i..i + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for dd in 0..d {
            out[i + dd] = x[i + dd] * inv * weight[dd] * sigmoid(gate[i + dd]);
        }
    }
    out
}

/// 2026-09-25: Weights for one KDA layer, reference only. Torch `Linear` layout `[out, in]`
/// throughout. There is no `Z` tensor: the output gate is `g_a`/`g_b` and the decay source is
/// `f_a`/`f_b`.
pub struct KdaWeights<'a> {
    pub w_f_a: &'a [f32],
    pub w_f_b: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub w_b: &'a [f32],
    pub w_g_a: &'a [f32],
    pub w_g_b: &'a [f32],
    pub o_norm_w: &'a [f32],
    pub w_o: &'a [f32],
}

/// 2026-09-25: Reference for one KDA layer from post-conv q/k/v to the `o_proj` output. The short
/// conv is not included; the golden's inputs are post-conv.
#[allow(clippy::too_many_arguments)]
pub fn kda_reference_layer(
    hidden: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    w: &KdaWeights<'_>,
    dims: KdaDims,
    lower_bound: f32,
    rms_eps: f32,
    state: &mut [f32],
) -> Vec<f32> {
    let (t_n, h_n, d, hid) = (dims.tokens, dims.heads, dims.head_dim, dims.hidden);

    let f_a = linear(hidden, t_n, hid, w.w_f_a, d);
    let g_lowrank = linear(&f_a, t_n, d, w.w_f_b, h_n * d);
    let gate = bounded_gate(&g_lowrank, w.dt_bias, w.a_log, dims, lower_bound);

    let beta: Vec<f32> = linear(hidden, t_n, hid, w.w_b, h_n)
        .iter()
        .map(|x| sigmoid(*x))
        .collect();

    let core = kda_recurrent(q, k, v, &gate, &beta, dims, state);

    let g_a = linear(hidden, t_n, hid, w.w_g_a, d);
    let out_gate = linear(&g_a, t_n, d, w.w_g_b, h_n * d);
    let normed = rms_norm_gated(&core, w.o_norm_w, &out_gate, d, rms_eps);

    linear(&normed, t_n, h_n * d, w.w_o, hid)
}

#[cfg(test)]
mod tests;
