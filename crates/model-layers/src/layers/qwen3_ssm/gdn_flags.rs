// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GDN / SSM decode-path flags, resolved once per process from the serve command line or, failing that, the environment.
//!
//! Owner: model-layers (qwen3 SSM).
//! Invariants:
//! - The flags are one process-wide cell: the first of `set_from_cli` or
//!   `flags()` fixes them, and later calls see the same value. A model loaded
//!   later in the same process keeps that resolution.
//! - The server publishes `h_f16_pool` only together with `h_f16` (both come
//!   from `ssm_h_dtype_bits`), and the environment fallback never sets it.
//!
//! The flags select kernels and are coupled: the FP16 h-state twins exist
//! only on the fused-norm arm, so `h_f16` without `fused_norm` would reach an
//! FP32-only kernel. The server's argument validation rejects that pair
//! before load. When the command line gives any GDN flag, the server
//! publishes the whole cell through `set_from_cli`; otherwise `flags()`
//! reads the environment on first use.

/// 2026-09-25: The resolved flags; empty until `set_from_cli` or the first
/// `flags()`.
static FLAGS: std::sync::OnceLock<GdnFlags> = std::sync::OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnFlags {
    /// 2026-09-25: `--ssm-h-dtype f16` (or `f16-pool`): store the GDN decode
    /// h-state as FP16.
    pub h_f16: bool,
    /// 2026-09-25: `--ssm-h-dtype f16-pool`: also size the h pools at 2 bytes
    /// per element. Always set together with `h_f16` (a narrow pool holding
    /// FP32 would be an out-of-bounds write).
    pub h_f16_pool: bool,
    /// 2026-09-25: `--gdn-fused-norm`: fused GDN output-norm decode kernel.
    pub fused_norm: bool,
    /// 2026-09-25: `--ssm-batched-recurrent`: one strided recurrent launch per
    /// batch.
    pub batched_recurrent: bool,
    /// 2026-09-25: `--exact-verify`: MTP verify runs the sequential-decode-exact
    /// per-token GDN/SSM chain instead of the WY-chunkwise and fused BF16-conv
    /// arms. Off by default. It makes only the GDN/SSM verify chain exact; the
    /// FFN and attention projections still pick kernels by row count.
    pub exact_verify: bool,
}

impl GdnFlags {
    /// 2026-09-25: Whether the MTP-verify pass runs the exact chain:
    /// `exact_verify` and not `h_f16`. Pure, so tests need not touch the
    /// process-wide cell. An FP16 h-state forces the non-exact arms even when
    /// exact was requested; the server's argument validation also rejects
    /// that pair.
    pub fn verify_exact_active(self) -> bool {
        self.exact_verify && !self.h_f16
    }
    /// 2026-09-25: The reading used when the command line published nothing.
    ///
    /// `batched_recurrent` is the resolved `ssm_batched_recurrent` setting: the
    /// target's `HARDWARE.toml` `[defaults]` row (on for `hopper`, off for
    /// `gb10`, `b200` and `b300`), overridden by `METRALE_SSM_BATCHED_RECURRENT`,
    /// where `0`/`false`/`off`/`no` turn it off and any other value on.
    ///
    /// `METRALE_SSM_H_FP16` is presence-checked: any value, `0` included, turns
    /// it on. `METRALE_GDN_FUSED_NORM` must be exactly `1`.
    fn from_env() -> Self {
        Self {
            h_f16: std::env::var("METRALE_SSM_H_FP16").is_ok(),
            // 2026-09-25: No environment variable feeds the pool width; only
            // `--ssm-h-dtype f16-pool` sets it.
            h_f16_pool: false,
            fused_norm: std::env::var("METRALE_GDN_FUSED_NORM").as_deref() == Ok("1"),
            batched_recurrent: crate::layers::ops::target_defaults::resolved()
                .ssm_batched_recurrent
                .value,
            // 2026-09-25: No environment variable feeds exact verify; only
            // `--exact-verify` sets it.
            exact_verify: false,
        }
    }
}

/// 2026-09-25: Publish the command line's resolution. Call once, before the model
/// builds.
///
/// Returns the value in force: the argument, unless something already read the
/// flags, in which case that earlier reading stays and the caller should say
/// so.
pub fn set_from_cli(flags: GdnFlags) -> GdnFlags {
    let _ = FLAGS.set(flags);
    *FLAGS.get().expect("just set")
}

/// 2026-09-25: The resolved flags, falling back to the environment on first use.
pub fn flags() -> GdnFlags {
    *FLAGS.get_or_init(GdnFlags::from_env)
}

/// 2026-09-25: The widest chain-verify K with an FP16 h-state twin
/// (`gated_delta_rule_wy{5..16}_f16`).
///
/// K=17, the DFlash arm at gamma 16, dispatches the FP32 `gated_delta_rule_wy17`,
/// which has no twin, and an FP32 kernel over an FP16 h-state produces wrong
/// output rather than an error. The DFlash verify width is gamma + 1, so the
/// argument validator and the serve preflight gate on
/// [`MAX_F16_TWIN_DFLASH_GAMMA`], derived from this.
pub const MAX_F16_TWIN_K: usize = 16;

/// 2026-09-25: The largest `--dflash-gamma` whose verify width still has an FP16
/// twin.
pub const MAX_F16_TWIN_DFLASH_GAMMA: usize = MAX_F16_TWIN_K - 1;

/// 2026-09-25: The DFlash gamma for a drafter of this trained block size, when no
/// `--dflash-gamma` was given.
///
/// The one rule for it: the drafter head resolves its gamma through this, and
/// so do the preflight paths that size pools and reserves from a peeked
/// `dflash_config.block_size`.
///
/// `block + 2`, because these drafters chain past their trained block.
/// Measured 2026-08-29 on GB10 with Qwen3.8-27B DFlash2 (block 8): gamma 10
/// was the fastest serve, 63.0 against 56.2 tok/s at gamma 8.
///
/// Clamped to `MAX_F16_TWIN_DFLASH_GAMMA`, so a block-16 drafter lands on 15
/// (verify width 16, which has a kernel for both h-state dtypes) rather than
/// 18.
pub const fn default_dflash_gamma(trained_block_size: usize) -> usize {
    let bumped = trained_block_size + 2;
    if bumped > MAX_F16_TWIN_DFLASH_GAMMA {
        MAX_F16_TWIN_DFLASH_GAMMA
    } else {
        bumped
    }
}

/// 2026-09-25: `--ssm-h-dtype f16` or `f16-pool` (environment fallback:
/// `METRALE_SSM_H_FP16`).
pub fn ssm_h_fp16_enabled() -> bool {
    flags().h_f16
}

/// 2026-09-25: `--ssm-h-dtype f16-pool`: h pools sized at 2 bytes per element.
/// Implies [`ssm_h_fp16_enabled`]; [`ssm_h_dtype_bits`] guarantees that where
/// the flag is decoded.
pub fn ssm_h_f16_pool_enabled() -> bool {
    flags().h_f16_pool
}

/// 2026-09-25: Decode `--ssm-h-dtype` into the two h-state bits it publishes:
/// `(h_f16, h_f16_pool)`.
///
/// The argument validator and the server's kernel-flag plan both call this, so
/// they cannot read the flag differently. Anything other than exactly `f16` or
/// `f16-pool`, including `f32` and an absent flag, is FP32.
pub fn ssm_h_dtype_bits(dtype: Option<&str>) -> (bool, bool) {
    match dtype {
        Some("f16") => (true, false),
        // 2026-09-25: f16-pool is f16 plus the narrow pool, never one without the
        // other.
        Some("f16-pool") => (true, true),
        _ => (false, false),
    }
}

/// 2026-09-25: `--gdn-fused-norm` (environment fallback:
/// `METRALE_GDN_FUSED_NORM=1`).
pub fn gdn_fused_norm_enabled() -> bool {
    flags().fused_norm
}

/// 2026-09-25: `--ssm-batched-recurrent` (environment fallback:
/// `METRALE_SSM_BATCHED_RECURRENT`).
pub fn ssm_batched_recurrent_enabled() -> bool {
    flags().batched_recurrent
}

/// 2026-09-25: `--exact-verify` given and the h-state FP32: the MTP-verify pass
/// runs the sequential-decode-exact chain. See [`GdnFlags::verify_exact_active`].
pub fn verify_exact_enabled() -> bool {
    flags().verify_exact_active()
}

/// 2026-09-25: Batch width at which the multi-sequence decode projections switch
/// to the 128-row M-tile. `None` when `METRALE_NO_SSM_M128` is set (any value,
/// `0` included), which keeps the 64-row tile at every width.
///
/// 65 is derived, not tuned: `ceil(m/64) > ceil(m/128)` first holds at m=65, so
/// below it the wider tile saves no weight reads and only pads MMA rows.
pub(crate) fn ssm_m128_min_m() -> Option<u32> {
    static M: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        if std::env::var("METRALE_NO_SSM_M128").is_ok() {
            None
        } else {
            Some(65)
        }
    })
}

/// 2026-09-25: Write-on-accept K=4 batched verify. Off unless `METRALE_GDN_WOA=1`.
/// With it off the parent wy4 kernel runs, the fold in `woa.rs` returns
/// `Ok(false)`, and the accept path (`async_chkpt.rs`) restores `h` for every K.
///
/// Off by default on a measurement: on GB10 on 2026-09-18, concurrency sweeps
/// with it on had the semantic-loop watchdog fire at C=4 (4 fires) where runs
/// with it off had none, and failed the gate; C=16..128 were unaffected. The
/// cause inside the arm is not identified.
///
/// Public so the serve can tell the gamma resolver whether the K=4 rung has
/// its kernel.
pub fn gdn_woa_enabled() -> bool {
    std::env::var("METRALE_GDN_WOA").ok().as_deref() == Some("1")
}

#[cfg(test)]
mod tests {
    use super::{GdnFlags, ssm_h_dtype_bits};

    const BASE: GdnFlags = GdnFlags {
        h_f16: false,
        h_f16_pool: false,
        fused_norm: false,
        batched_recurrent: false,
        exact_verify: false,
    };

    /// 2026-09-25: With no flags the verify pass runs the WY/chunkwise arms, not
    /// the exact chain.
    #[test]
    fn legacy_wy_verify_is_the_default() {
        assert!(
            !BASE.verify_exact_active(),
            "default must be the legacy WY arms — exact verify is opt-in"
        );
        // 2026-09-25: The other flags do not turn exact mode on.
        assert!(
            !GdnFlags {
                fused_norm: true,
                batched_recurrent: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }

    /// 2026-09-25: `--exact-verify` selects the exact chain, alone and beside
    /// the other GDN flags.
    #[test]
    fn exact_verify_flag_selects_the_exact_chain() {
        assert!(
            GdnFlags {
                exact_verify: true,
                ..BASE
            }
            .verify_exact_active()
        );
        assert!(
            GdnFlags {
                exact_verify: true,
                fused_norm: true,
                batched_recurrent: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }

    /// 2026-09-25: The environment fallback never turns exact verify on: no
    /// variable feeds it. Deterministic despite reading the process
    /// environment, because only fields no variable feeds are asserted.
    #[test]
    fn env_fallback_never_enables_exact_verify() {
        assert!(!GdnFlags::from_env().exact_verify);
        // 2026-09-25: The same holds for the pool width: `METRALE_SSM_H_FP16`
        // alone keeps the FP32-sized pool.
        assert!(!GdnFlags::from_env().h_f16_pool);
    }

    /// 2026-09-25: A narrow pool holding FP32 is an out-of-bounds write, so
    /// `h_f16_pool` without `h_f16` must not come from any spelling. The
    /// argument validator and the kernel-flag plan both decode through this
    /// function.
    #[test]
    fn the_pool_bit_is_never_set_without_the_dtype_bit() {
        for (spelling, expected) in [
            (None, (false, false)),
            (Some("f32"), (false, false)),
            (Some("f16"), (true, false)),
            (Some("f16-pool"), (true, true)),
            (Some(""), (false, false)),
            (Some("F16-POOL"), (false, false)),
            (Some("f16 "), (false, false)),
        ] {
            assert_eq!(ssm_h_dtype_bits(spelling), expected, "{spelling:?}");
        }
    }

    /// 2026-09-25: An FP16 h-state forces the non-exact arms even when exact was
    /// requested. The argument validator rejects the explicit pair; this is the
    /// layer beneath it.
    #[test]
    fn h_f16_forces_non_exact_even_when_requested() {
        assert!(
            !GdnFlags {
                exact_verify: true,
                h_f16: true,
                ..BASE
            }
            .verify_exact_active()
        );
        assert!(
            !GdnFlags {
                h_f16: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }
}
