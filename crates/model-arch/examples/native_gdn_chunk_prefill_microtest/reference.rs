// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The f64 CPU reference of the GDN chunked-prefill oracle, and the
//! gather that cuts the reference heads out of a full kernel output.
//! `main.rs` scores every arm against it.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants: none beyond the types.

use crate::{C, Case, KD, NK, NV, REF_HEADS, VD};

/// 2026-09-25: f64 reference for heads [0, REF_HEADS): the same recurrence
/// the kernels run, laid out for the reference heads only in the kernels' own
/// index order.
pub(super) struct Ref {
    pub(super) sc: Vec<f64>,
    pub(super) uc: Vec<f64>,
    pub(super) hf: Vec<f64>,
}

pub(super) fn ref_chunk_delta_h(c: &Case, w: &[f32], u: &[f32], gc: &[f32], h0: &[f32]) -> Ref {
    let hr = NV / NK;
    let (n1, n2) = (c.nt * REF_HEADS * KD * VD, c.nt * REF_HEADS * C * VD);
    let mut r = Ref {
        sc: vec![0.0; n1],
        uc: vec![0.0; n2],
        hf: vec![0.0; REF_HEADS * KD * VD],
    };
    for vh in 0..REF_HEADS {
        let kh = vh / hr;
        let mut s = vec![0.0f64; KD * VD];
        for (i, v) in s.iter_mut().enumerate() {
            *v = h0[vh * KD * VD + i] as f64;
        }
        for ch in 0..c.nt {
            let cs = ch * C;
            let ce = (c.t - cs).min(C);
            let base = ch * NV + vh;
            let rbase = ch * REF_HEADS + vh;
            r.sc[rbase * KD * VD..(rbase + 1) * KD * VD].copy_from_slice(&s);
            let gl = gc[base * C + ce - 1] as f64;
            let mut duc = vec![0.0f64; C * VD];
            for i in 0..ce {
                let dc = (gl - gc[base * C + i] as f64).exp();
                for v in 0..VD {
                    let mut ws = 0.0f64;
                    for k in 0..KD {
                        ws += w[base * C * KD + i * KD + k] as f64 * s[k * VD + v];
                    }
                    let uci = u[base * C * VD + i * VD + v] as f64 - ws;
                    r.uc[rbase * C * VD + i * VD + v] = uci;
                    duc[i * VD + v] = dc * uci;
                }
            }
            let edl = gl.exp();
            for k in 0..KD {
                for v in 0..VD {
                    let mut acc = edl * s[k * VD + v];
                    for i in 0..ce {
                        acc += duc[i * VD + v] * c.key[(cs + i) * NK * KD + kh * KD + k].to_f64();
                    }
                    s[k * VD + v] = acc;
                }
            }
        }
        r.hf[vh * KD * VD..(vh + 1) * KD * VD].copy_from_slice(&s);
    }
    r
}

/// 2026-09-25: Gather the reference heads' slice out of a full-NV kernel
/// output.
pub(super) fn take_heads(full: &[f32], nt: usize, per_head: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nt * REF_HEADS * per_head);
    for ch in 0..nt {
        for vh in 0..REF_HEADS {
            let b = (ch * NV + vh) * per_head;
            out.extend_from_slice(&full[b..b + per_head]);
        }
    }
    out
}
