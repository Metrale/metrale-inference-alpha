// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The process-wide SSM verify-rollback mode and the replay-ring
//! sizing it implies.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - The published mode is set at most once per process; later
//!   `set_ssm_rollback_mode` calls return the mode already in force.

/// 2026-09-25: SSM verify-rollback mode (`--ssm-rollback-mode`, experimental).
///
/// * `Snapshot` (the CLI default): verify writes per-token h/conv state
///   intermediates, and a partial accept restores from them. The only mode
///   with a device path.
/// * `Replay`: keeps only the pre-verify checkpoint per verify slot plus a
///   ring sized for the verify window's per-token GDN inputs. Capture and
///   replay are not implemented: a serve in this mode boots, and every
///   speculative verify entry refuses
///   (`SsmStatePool::require_verify_rollback_supported`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsmRollbackMode {
    Snapshot,
    Replay,
}

impl std::str::FromStr for SsmRollbackMode {
    type Err = String;
    /// 2026-09-25: The parse for the `--ssm-rollback-mode` value; CLI validation
    /// and the serve's publication both call it.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "snapshot" => Ok(Self::Snapshot),
            "replay" => Ok(Self::Replay),
            other => Err(format!(
                "unknown ssm-rollback-mode '{other}' (valid: snapshot, replay)"
            )),
        }
    }
}

/// 2026-09-25: The published rollback mode, written from the serve command line
/// and read by pool construction and preflight. The first write wins, as in
/// `gdn_flags::set_from_cli`.
static ROLLBACK_MODE: std::sync::OnceLock<SsmRollbackMode> = std::sync::OnceLock::new();

/// 2026-09-25: Publish the command line's mode. Returns the mode in force (first
/// write wins).
pub fn set_ssm_rollback_mode(mode: SsmRollbackMode) -> SsmRollbackMode {
    let _ = ROLLBACK_MODE.set(mode);
    *ROLLBACK_MODE.get().expect("just set")
}

/// 2026-09-25: The mode in force. `Snapshot` when nothing was published (tests,
/// examples); reading it that way also fixes it for the process.
pub fn ssm_rollback_mode() -> SsmRollbackMode {
    *ROLLBACK_MODE.get_or_init(|| SsmRollbackMode::Snapshot)
}

/// 2026-09-25: Bytes of one cached verify row of GDN inputs, per SSM layer: the
/// deinterleaved qkvz row (`qkvz_elems` BF16, the rows of
/// `ConvGdnArgs::deinterleaved`) plus the gate/beta row (`nv * 2` FP32, the
/// rows of `ConvGdnArgs::gates_buf`).
pub fn ssm_replay_row_bytes(qkvz_elems: usize, nv: usize) -> usize {
    qkvz_elems * 2 + nv * 2 * 4
}

/// 2026-09-25: Replay-mode verify-window input ring:
/// `mtp_state_slots * (k_ceiling - 1) * num_ssm_layers * row_bytes`, since a
/// partial accept replays at most K-1 tokens. Preflight reserves and
/// `SsmStatePool::new` allocates through this function, with different
/// arguments: preflight passes `num_drafts + 1` and `mtp_state_slots`, the
/// pool its `num_intermediates` and `mtp_slots + 1` (the dummy slot).
pub fn ssm_replay_ring_bytes(
    num_ssm_layers: usize,
    row_bytes: usize,
    k_ceiling: usize,
    mtp_state_slots: usize,
) -> usize {
    mtp_state_slots * k_ceiling.saturating_sub(1) * num_ssm_layers * row_bytes
}
