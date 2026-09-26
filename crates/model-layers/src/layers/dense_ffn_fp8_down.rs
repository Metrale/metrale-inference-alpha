// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Chooses the arm that runs the native-FP8 decode FFN for one token
//! (`Fp8DownArm`), as a pure function `DenseFfnLayer::forward` calls.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `fp8_down_arm` returns `PerProjection` for a non-SiLU layer or a missing
//!   `w8a16_gemv_dual`, and returns `SplitSilu` or `FusedSilu` only when every kernel
//!   that arm launches resolved.
//!
//! Why `SplitSilu` is the preferred arm: `w8a16_gemv_silu_input` gives each of the 4
//! outputs of a 256-thread CTA a 64-lane team that walks the whole K and evaluates
//! `(g / (1 + __expf(-g))) * u` for every element, so one launch computes the SwiGLU once
//! per output rather than once per element. `SplitSilu` computes it once with
//! `moe_silu_mul` and then runs the plain `w8a16_gemv`. Measured 2026-09-11 on 1xH100,
//! Qwen/Qwen3.8-27B-FP8, C=1: the fused down projection took 6.65 ms (30.4%) of a
//! 21.891 ms decode step at 858 GB/s, against 1,979 GB/s for `w8a16_gemv_dual`.

/// 2026-09-25: Which kernels the native-FP8 decode FFN runs for one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fp8DownArm {
    /// 2026-09-25: `w8a16_gemv_dual`, then `moe_silu_mul` writes `silu(gate)*up` over
    /// `gate_out`, then the plain `w8a16_gemv` for down. The default:
    /// `ModelLevers::decode_split_silu` comes from `[defaults] decode_split_silu`, true in
    /// every `kernels/*/HARDWARE.toml` that declares it, and `METRALE_NO_DECODE_SPLIT_SILU`
    /// turns it off.
    ///
    /// Not bit-identical to `FusedSilu`: `moe_silu_mul` rounds `g * (1 / (1 + e^-g)) * u`
    /// to BF16 before the GEMV reads it, while the fused kernel keeps
    /// `(g / (1 + e^-g)) * u` in FP32 inside the dot product. The non-fused FP8 prefill
    /// path runs the same `moe_silu_mul`.
    SplitSilu,
    /// 2026-09-25: `w8a16_gemv_dual`, then the fused `w8a16_gemv_silu_input`. Taken
    /// when the split-SiLU lever is off or `act_mul` / `w8a16_gemv` did not resolve.
    FusedSilu,
    /// 2026-09-25: Neither fused path is usable: four launches, `w8a16_gemv` for gate and
    /// up, `act_mul`, then `w8a16_gemv` for down.
    PerProjection,
}

/// 2026-09-25: The arm rule `DenseFfnLayer::forward` uses, pure so the CPU tests can
/// cover every combination without a GPU.
///
/// `is_silu` is the layer's activation; `split_silu_lever` is
/// `ModelLevers::decode_split_silu`; every other `bool` is "this handle is nonzero".
pub(crate) fn fp8_down_arm(
    is_silu: bool,
    dual: bool,
    fused_silu: bool,
    act_mul: bool,
    plain_gemv: bool,
    split_silu_lever: bool,
) -> Fp8DownArm {
    // 2026-09-25: Both fused arms read the gate/up pair that `w8a16_gemv_dual` writes
    // to `gate_out`/`up_out`, so without it neither can run.
    if !is_silu || !dual {
        return Fp8DownArm::PerProjection;
    }
    if split_silu_lever && act_mul && plain_gemv {
        Fp8DownArm::SplitSilu
    } else if fused_silu {
        Fp8DownArm::FusedSilu
    } else {
        Fp8DownArm::PerProjection
    }
}
