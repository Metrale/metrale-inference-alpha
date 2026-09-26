// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-sequence owned-state term of the preflight reserve.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.
//!
//! The term has two owners, reported apart so the boot log names the owner
//! of each byte: the DSA indexer caches of the target's layers, and the state
//! of the draft proposer, which is not in the layer list.

use metrale_config::ModelConfig;
use metrale_model_arch::seq_state_reserve::per_sequence_state_bytes;

use crate::cli;

/// 2026-09-26: Per-sequence owned device state times `--max-batch-size` (0
/// counts as 1), the charge preflight adds to `inference_reserve`. Logs the
/// split by owner when the charge is nonzero. An error from
/// `per_sequence_state_bytes` counts as no state.
///
/// The batch multiplier is applied here, in the only non-test call to
/// `for_batch`.
pub(super) fn per_sequence_reserve(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    let spec_on = args.speculative || args.self_speculative || args.dflash;
    let per_seq = per_sequence_state_bytes(config, args.max_seq_len, spec_on).unwrap_or_default();
    let charge = per_seq.for_batch(args.max_batch_size);
    if charge > 0 {
        tracing::info!(
            "Per-sequence state reserve: {} MB = {} seq x ({} MB target DSA layers + {} MB \
             proposer). Owned per sequence, replicated per rank (EP does not shard the \
             indexer); previously covered only by cuda_headroom.",
            charge / (1024 * 1024),
            args.max_batch_size.max(1),
            per_seq.target_layers / (1024 * 1024),
            per_seq.proposer / (1024 * 1024),
        );
    }
    charge
}

#[cfg(test)]
#[path = "per_sequence_state_tests.rs"]
mod tests;
