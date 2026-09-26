// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which cross-architecture kernel families a model's config says
//! exist, and the lookup helpers `Qwen3AttentionLayer`'s constructor gates on
//! them.
//!
//! The config states whether the model has MLA (`kv_lora_rank`),
//! hyper-connections (`hc_mult`), compressed attention (`compress_ratios`) or
//! heads wider than 256 (`head_dim`). The constructor looks up a family's
//! kernels only when the model has it, so a model's boot audit holds no failed
//! lookup for a family it cannot use.
//!
//! Owner: model-layers (attention).
//! Invariants: `gated(false, ..)` issues no kernel lookup.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use metrale_config::ModelConfig;

/// 2026-09-25: Per-family presence flags, derived once per layer construction.
#[derive(Clone, Copy, Debug)]
pub(super) struct ArchProbes {
    /// 2026-09-25: Multi-head latent attention: `kv_lora_rank > 0`, the width
    /// of the compressed-KV latent.
    pub mla: bool,
    /// 2026-09-25: Hyper-connections: `hc_mult > 0`.
    pub hyper_connection: bool,
    /// 2026-09-25: Compressed attention: some entry of the per-layer
    /// `compress_ratios` is nonzero. An empty or all-zero list is full
    /// attention on every layer.
    pub compressed_attn: bool,
    /// 2026-09-25: The `*_512` kernel arms: `head_dim > 256`, or an MLA model.
    /// Gemma-4's parser stores `global_head_dim` in `config.head_dim` when the
    /// checkpoint has one. The MLA prefill paths (`prefill/paged_mla.rs`,
    /// `paged_v4.rs`, `cache_skip_v4.rs`) read these handles too, hence the
    /// `kv_lora_rank` term.
    pub wide_head_dim: bool,
}

impl ArchProbes {
    pub(super) fn from_config(config: &ModelConfig) -> Self {
        Self {
            mla: config.kv_lora_rank > 0,
            hyper_connection: config.hc_mult > 0,
            compressed_attn: config.compress_ratios.iter().any(|&r| r > 0),
            wide_head_dim: config.head_dim > 256 || config.kv_lora_rank > 0,
        }
    }
}

/// 2026-09-26: `try_kernel` when `enabled`, otherwise `KernelHandle(0)` with no
/// lookup. A lookup that is never issued leaves no failed row in the boot
/// audit.
///
/// `#[track_caller]` so the audit names the dispatch site in the calling
/// `init_{proj,decode,prefill}_kernels.rs` resolver, not this function.
#[track_caller]
pub(super) fn gated(enabled: bool, gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    if enabled {
        crate::layers::try_kernel(gpu, module, func)
    } else {
        KernelHandle(0)
    }
}

/// 2026-09-25: A resolved handle as an `Option`: `None` for a zero handle.
///
/// Used for kernels only some targets build, where a zero handle means "this
/// target does not carry the kernel" and the caller keeps another arm. The
/// `Option` type makes every reader handle that case.
pub(super) fn present(handle: KernelHandle) -> Option<KernelHandle> {
    (handle.0 != 0).then_some(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: A plain GQA config with 128-wide heads claims none of the
    /// cross-architecture families.
    #[test]
    fn a_plain_gqa_model_probes_for_nothing() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.head_dim = 128;
        let p = ArchProbes::from_config(&cfg);
        assert!(!p.mla);
        assert!(!p.hyper_connection);
        assert!(!p.compressed_attn);
        assert!(!p.wide_head_dim);
    }

    #[test]
    fn each_architecture_signal_enables_only_its_kernel_families() {
        for (name, configure, expected) in [
            (
                "MLA",
                (512usize, 0usize, Vec::new(), 128usize),
                [true, false, false, true],
            ),
            (
                "hyper connection",
                (0, 4, Vec::new(), 128),
                [false, true, false, false],
            ),
            (
                "compressed attention",
                (0, 0, vec![0, 8, 8], 128),
                [false, false, true, false],
            ),
            (
                "wide head",
                (0, 0, Vec::new(), 512),
                [false, false, false, true],
            ),
        ] {
            let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
            (
                cfg.kv_lora_rank,
                cfg.hc_mult,
                cfg.compress_ratios,
                cfg.head_dim,
            ) = configure;
            let p = ArchProbes::from_config(&cfg);
            assert_eq!(
                [
                    p.mla,
                    p.hyper_connection,
                    p.compressed_attn,
                    p.wide_head_dim
                ],
                expected,
                "{name}"
            );
        }
    }

    /// 2026-09-25: A config `head_dim` of 512, as Gemma-4's parser stores its
    /// global head dim, enables the wide-head family.
    #[test]
    fn a_heterogeneous_model_gates_on_its_max_head_dim() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.head_dim = 512;
        assert!(ArchProbes::from_config(&cfg).wide_head_dim);
    }

    /// 2026-09-25: An all-zero `compress_ratios` (every layer full attention)
    /// is not compressed attention.
    #[test]
    fn all_zero_compress_ratios_is_not_compressed_attention() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.compress_ratios = vec![0, 0, 0];
        assert!(!ArchProbes::from_config(&cfg).compressed_attn);
    }
}
