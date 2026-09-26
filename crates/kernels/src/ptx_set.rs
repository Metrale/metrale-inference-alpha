// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-target declarations the generated `target_ptx.rs`
//! constructs: a target's kernel blobs and what its MODEL.toml says about it.
//! Selection among targets lives in [`super::query`] and [`super::resolve`].
//!
//! Owner: kernels crate.
//! Invariants: none beyond the types.

use metrale_core::target::KernelTarget;

use super::{ModelBehavior, SamplingPresets};

/// 2026-09-25: One `(model_type, hidden_size)` pair a kernel target supports,
/// from MODEL.toml `[[model_types]]`.
pub struct ModelTypeMatch {
    pub model_type: &'static str,
    /// 2026-09-25: `None` is a wildcard; [`crate::resolve::resolve_target`]
    /// tries it only when no target declares the exact hidden size.
    pub hidden_size: Option<usize>,
}

/// 2026-09-25: A target's DFlash drafter pairing, from MODEL.toml `[dflash]`
/// (`parse_dflash` in `build_parse.rs`). The section requires `draft_model`;
/// the other keys default as noted.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    /// 2026-09-25: Drafter checkpoint id or path. metrale-server uses it for
    /// `--dflash` when no `--draft-model` is given.
    pub draft_model: &'static str,
    /// 2026-09-25: Block size γ. Default 16.
    pub gamma: usize,
    /// 2026-09-25: Drafter sliding-window size in tokens. Default 0.
    pub window_size: usize,
    /// 2026-09-25: `[dflash].mask_token_id`. Default 0.
    pub mask_token_id: u32,
    /// 2026-09-25: `[dflash].target_layer_ids`. Default empty.
    pub target_layer_ids: &'static [usize],
}

/// 2026-09-25: One compiled kernel target: its [`KernelTarget`] identity,
/// kernel blobs and MODEL.toml declarations.
///
/// Each blob is one compiled module, emitted as `&'static [u8]` by `build.rs`
/// (`include_bytes!`): PTX text for NVIDIA, binary objects for SCALE, HIP and
/// Metal.
pub struct TargetPtxSet {
    pub target: KernelTarget,
    /// 2026-09-25: The `kernels/<hw>/HARDWARE.toml` `[hardware].arch` this
    /// target was compiled with, verbatim (`sm_90a`, `sm_121f`, `gfx1151`).
    /// [`KernelTarget::arch`] holds the same string with an `a`/`f` suffix
    /// stripped (`build_arch::kernel_target_arch`). The GPU arch preflight
    /// judges compatibility on this field, because the suffix changes which
    /// devices the PTX runs on; it skips an empty value.
    pub ptx_arch: &'static str,
    pub modules: Vec<(&'static str, &'static [u8])>,
    pub sampling: SamplingPresets,
    pub behavior: ModelBehavior,
    pub model_type_matches: Vec<ModelTypeMatch>,
    /// 2026-09-25: MODEL.toml `[model] match_names`: case-insensitive
    /// substrings of a checkpoint reference (HF id, `--model-name`, resolved
    /// model dir) that identify checkpoints this target serves. Consulted only
    /// to break a tie between targets that declare the same
    /// `(model_type, hidden_size)`; see [`crate::resolve::resolve_target`].
    /// `build.rs` (`validate_collision_match_names`) panics if a colliding
    /// target leaves it empty.
    pub match_names: &'static [&'static str],
    /// 2026-09-25: DFlash drafter pairing; `None` when MODEL.toml has no
    /// `[dflash]` section with a `draft_model`.
    pub dflash: Option<DflashConfig>,
    /// 2026-09-25: `(module, kernel)` pairs this target's leaf-role files
    /// drop relative to their `common/` namesakes: the `common/` file defines
    /// the kernel, the model's file of the same name does not, and a model
    /// file replaces its namesake whole. The boot kernel audit reports them
    /// per module.
    pub shadowed_dropped: &'static [(&'static str, &'static str)],
    /// 2026-09-25: `(module, kernel)` lookups this model's dispatch may issue
    /// and fail to resolve without that being an error, from MODEL.toml
    /// `[expected_absent]`; the build panics on an entry without a reason.
    /// The boot kernel gate refuses to serve on any other unresolved lookup
    /// unless `--dangerously-allow-unresolved-kernel-lookups` is set.
    pub expected_absent: &'static [(&'static str, &'static str)],
}
