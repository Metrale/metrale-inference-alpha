// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MTP drafter knobs read from the environment: the draft
//! confidence floor, shadow top-k, multi-sequence mode, the catch-up and
//! refeed switches, debug taps, and the EP propose command.
//!
//! Owner: model-layers (speculative).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: `METRALE_MTP_DRAFT_CONF`: the chain-confidence floor below
/// which a propose's drafts are discarded and the step decodes serially. 0.0,
/// the value when unset, disables it.
pub fn draft_conf_tau() -> f32 {
    parse_draft_conf_tau(std::env::var("METRALE_MTP_DRAFT_CONF").ok().as_deref())
}

/// 2026-09-25: Parse a `METRALE_MTP_DRAFT_CONF` value: clamped to
/// `[0.0, 0.99]`, so a floor above any reachable confidence cannot discard
/// every draft; 0.0 when absent or unparseable. Pure, so tests need not set
/// the process environment.
pub fn parse_draft_conf_tau(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.parse::<f32>().ok())
        .map(|t| t.clamp(0.0, 0.99))
        .unwrap_or(0.0)
}

/// 2026-09-25: `METRALE_MTP_SHADOW_TOPK=k`: log the drafter's top-k
/// candidates at each draft position, and the verify steps' target tokens,
/// without changing token selection. 0 (unset or unparseable) is off; k is
/// capped at 8. `ModelLevers::shadow_topk` and `SchedLevers::shadow_topk`
/// both read it here, once per run.
pub fn shadow_topk() -> usize {
    std::env::var("METRALE_MTP_SHADOW_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(8)
}

/// 2026-09-25: True when the MTP dispatch cap is above one sequence. The
/// catch-up ring, the refeed labels and the carry slot each hold one
/// sequence's data, so [`mtp_catchup_enabled`], [`mtp_refeed_accepted_enabled`]
/// and `mtp_carry::carry_armed` are false in this mode.
pub fn mtp_multi_seq_mode() -> bool {
    mtp_max_seqs() > 1
}

/// 2026-09-25: `METRALE_MTP_ACCEPT_DEBUG`, on when set to any value. It turns
/// on MTP acceptance logs, among them the periodic per-batch-width lines of
/// the scheduler's `mtp_accept_debug` module, verify-graph outcome counts and
/// a drafter coverage line. Read once per process.
pub fn mtp_accept_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MTP_ACCEPT_DEBUG").is_ok())
}

/// 2026-09-25: `METRALE_MTP_CATCHUP=1`: the model keeps a ring of target
/// hiddens, and before a propose it feeds the drafter the pair keys missing
/// since its newest row. Off otherwise, and off in [`mtp_multi_seq_mode`].
pub fn mtp_catchup_enabled() -> bool {
    std::env::var("METRALE_MTP_CATCHUP").ok().as_deref() == Some("1") && !mtp_multi_seq_mode()
}

/// 2026-09-25: `METRALE_MTP_REFEED_ACCEPTED=1`: after a verify, ring the
/// target's hiddens for the accepted positions, so the catch-up feed can
/// rewrite accepted drafter rows with the target's hidden instead of the
/// drafter's own. It writes into the catch-up ring, which the model allocates
/// only when [`mtp_catchup_enabled`]. Off otherwise, and off in
/// [`mtp_multi_seq_mode`]. Measured 2026-07-21 on dgx2 at 2 drafts, about 10k
/// verify steps per arm: p2_uncond 0.4182 off, 0.4452 on.
pub fn mtp_refeed_accepted_enabled() -> bool {
    std::env::var("METRALE_MTP_REFEED_ACCEPTED").ok().as_deref() == Some("1")
        && !mtp_multi_seq_mode()
}

/// 2026-09-25: `METRALE_MTP_REFEED_SHIFT`: an offset added to every re-fed
/// ring label, clamped to `[-4, 4]`; 0 (unset or unparseable) is the derived
/// mapping. It exists to test that mapping: a run at +1 or -1 is compared
/// with 0 by acceptance.
pub fn mtp_refeed_shift() -> isize {
    std::env::var("METRALE_MTP_REFEED_SHIFT")
        .ok()
        .and_then(|v| v.parse::<isize>().ok())
        .unwrap_or(0)
        .clamp(-4, 4)
}

/// 2026-09-25: `METRALE_MTP_REFEED_DEBUG=1`: log a [`hidden_fingerprint`] of
/// each hidden written to or fed from the catch-up ring. Each one copies a
/// row to the host.
pub fn mtp_refeed_debug() -> bool {
    std::env::var("METRALE_MTP_REFEED_DEBUG").ok().as_deref() == Some("1")
}

/// 2026-09-25: FNV-1a over the `h` BF16 values at `p`; 0 when the copy to the
/// host fails.
pub fn hidden_fingerprint(gpu: &dyn GpuBackend, p: DevicePtr, h: usize) -> u64 {
    let mut b = vec![0u8; h * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return 0;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in &b {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// 2026-09-25: EP worker command: run one MTP propose alongside rank 0. The
/// code is followed by `last_token`, `position` and `num_drafts`, each a u32.
pub const EP_CMD_MTP_PROPOSE: u32 = 0xFFFF_FFF5;

/// 2026-09-25: Run the sharded drafter on every rank with the communicator.
/// On unless `METRALE_NO_MTP_EP_PROPOSE=1`; read once per process. It is the
/// GLM-5.3 MTP head's [`DraftProposer::needs_comm`], and a proposer that needs
/// the communicator also has the worker run each propose
/// ([`EP_CMD_MTP_PROPOSE`]).
pub fn mtp_ep_propose_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_MTP_EP_PROPOSE").ok().as_deref() != Some("1"))
}
