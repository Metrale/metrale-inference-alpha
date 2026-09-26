// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel target descriptor: the `(arch, model, quant)` key of a compiled kernel set.
//!
//! Owner: metrale-core.
//! Invariants: none beyond the types.

/// 2026-09-25: The `(arch, model, quant)` key of one compiled kernel set; each
/// `metrale_kernels::TargetPtxSet` carries one as its `target`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelTarget {
    /// 2026-09-25: Architecture identifier. For CUDA it is the base SM
    /// (`sm_121`, `sm_90`): the build strips a HARDWARE.toml feature suffix
    /// (`sm_90a` -> `sm_90`). Other backends' names are kept verbatim.
    pub arch: &'static str,
    /// 2026-09-25: Model directory name under `kernels/<hardware>/` (e.g. `qwen3-next-80b-a3b`).
    pub model: &'static str,
    /// 2026-09-25: Quantization directory name under the model (e.g. `nvfp4`, `fp8`, `bf16`).
    pub quant: &'static str,
}

impl KernelTarget {
    pub const GB10_QWEN3_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3-next-80b-a3b",
        quant: "nvfp4",
    };

    pub const GB10_QWEN35_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-35b-a3b",
        quant: "nvfp4",
    };

    pub const GB10_QWEN35_122B_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-122b-a10b",
        quant: "nvfp4",
    };

    pub fn model_contains(&self, substring: &str) -> bool {
        self.model.contains(substring)
    }
}

impl std::fmt::Display for KernelTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.arch, self.model, self.quant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_target_display() {
        let t = KernelTarget::GB10_QWEN3_NVFP4;
        assert_eq!(t.to_string(), "(sm_121, qwen3-next-80b-a3b, nvfp4)");
    }

    #[test]
    fn target_equality_observes_each_dispatch_dimension() {
        let a = KernelTarget::GB10_QWEN3_NVFP4;
        assert_eq!(a, KernelTarget::GB10_QWEN3_NVFP4);
        assert_ne!(
            a,
            KernelTarget {
                arch: "sm_100a",
                ..a
            }
        );
        assert_ne!(
            a,
            KernelTarget {
                model: "llama-70b",
                ..a
            }
        );
        assert_ne!(a, KernelTarget { quant: "fp8", ..a });
    }
}
