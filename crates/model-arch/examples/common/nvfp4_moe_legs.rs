// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The configurations of `nvfp4_moe_grouped_microtest` (`Leg`), their row envelopes,
//! and the tensor-core leg's SiLU-row reader.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use super::fixture::bf16_to_f64;

/// 2026-09-27: The configurations `forward_nvfp4_grouped_decode` launches. 2026-10-02:
/// `AllNvfp4Tc` is `AllNvfp4` on the tensor-core expert kernels (`moe_nvfp4_grouped_tc.cu`),
/// the declared NVFP4 checkpoint's path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Leg {
    AllNvfp4,
    Nvfp4,
    Nvfp4GateUp,
    AllNvfp4Tc,
}

impl Leg {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::AllNvfp4 => "all-nvfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Nvfp4GateUp => "nvfp4-gate-up",
            Self::AllNvfp4Tc => "all-nvfp4-tc",
        }
    }
    pub(crate) fn fp8_shared(self) -> bool {
        matches!(self, Self::Nvfp4 | Self::Nvfp4GateUp)
    }
    pub(crate) fn fp8_down(self) -> bool {
        self == Self::Nvfp4GateUp
    }
    pub(crate) fn tc(self) -> bool {
        self == Self::AllNvfp4Tc
    }
    /// 2026-10-02: The widest row count production admits on this leg's kernels.
    pub(crate) fn max_m(self) -> usize {
        if self.tc() {
            ops_rows::TC_MAX
        } else {
            ops_rows::SCALAR_MAX
        }
    }
}

/// 2026-10-02: The row envelopes of `forward_nvfp4_grouped_decode`.
pub(crate) mod ops_rows {
    pub(crate) const SCALAR_MAX: usize =
        metrale_model_layers::layers::moe::NVFP4_GROUPED_DECODE_MAX_ROWS;
    pub(crate) const TC_MAX: usize =
        metrale_model_layers::layers::moe::NVFP4_GROUPED_DECODE_TC_MAX_ROWS;
}

/// 2026-10-02: The tensor-core kernels' SiLU rows: per row, `n` BF16 hi terms then `n` BF16 lo
/// terms in the bytes of `n` FP32 values; each value is hi + lo.
pub(crate) fn hi_lo_rows(b: &[u8], n: usize) -> Vec<f64> {
    let v = bf16_to_f64(b);
    v.chunks_exact(2 * n)
        .flat_map(|r| (0..n).map(move |i| r[i] + r[n + i]))
        .collect()
}
