// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `Model` trait: the interface between the scheduler and a model.
//!
//! Per request the scheduler calls `prefill` or `prefill_chunk` (logits at the last prompt
//! position), then `decode` or `decode_batch` once per emitted token. With speculative decoding
//! a verify call (`decode_verify_graphed`, `_k3`, `_k4`, `_kgamma`, `decode_verify_batched`)
//! takes `[last_token, draft0, ..]` and returns per-position argmax ids, and the scheduler rolls
//! the state back past a rejected draft. `mixed_forward` and `mixed_forward_batch` run decode
//! rows and prefill chunks in one pass.
//!
//! `Model: Send + Sync` so a model can move to the scheduler thread as a `Box<dyn Model>`.
//! `TransformerModel` relies on every call then coming from that one thread (its
//! `unsafe impl Sync` in `model/types.rs`).
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

pub use metrale_scheduler::{FeedSource, RowMask};

/// 2026-09-26: Per-call options for [`ModelVerify::decode_verify_batched`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifyBatchedOpts {
    /// 2026-09-26: Ask the GDN layers for the write-on-accept verify. The caller must then run
    /// [`ModelVerify::gdn_fold_accepted`] with every verdict before its `commit_accepted_prefix`
    /// calls. `false` in `Default`.
    pub write_on_accept: bool,
}

/// 2026-09-26: One beam-search request for a translation model (NLLB): the per-request
/// parameters the scheduler copies from the sequence. [`ModelForward::generate_beam_batch`] runs
/// the search to completion.
#[derive(Debug, Clone)]
pub struct BeamReq {
    /// 2026-09-25: Source subword ids; the model adds the source-language id and EOS itself.
    pub prompt_tokens: Vec<u32>,
    /// 2026-09-25: Source and target language token ids; `0` uses the deployment default.
    pub src_lang_id: u32,
    pub tgt_lang_id: u32,
    /// 2026-09-25: LoRA slot: `>= 0` applies the adapter, `-1` runs the base model.
    pub adapter_slot: i32,
    pub num_beams: usize,
    pub max_new: usize,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

/// 2026-09-25: The padded batch size for `n` live sequences: the smallest rung of the ladder
/// that is at least `n`, or `n` itself above 128. Batched decode pads to a rung so its CUDA
/// graphs are keyed by a few batch shapes instead of one per `n`.
#[inline]
pub fn padded_batch_n(n: usize) -> usize {
    [2usize, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        .iter()
        .copied()
        .find(|&s| s >= n)
        .unwrap_or(n)
}

mod adapters;
mod device_feed;
mod draft;
mod ep;
mod forward;
mod lifecycle;
mod logits;
mod ssm_state;
mod streams;
mod verify;
mod vision;

pub use adapters::ModelAdapters;
pub use device_feed::ModelDeviceFeed;
pub use draft::ModelDraft;
pub use ep::ModelEp;
pub use forward::ModelForward;
pub use lifecycle::ModelLifecycle;
pub use logits::ModelLogits;
pub use ssm_state::ModelSsmState;
pub use streams::ModelStreams;
pub use verify::ModelVerify;
pub use vision::ModelVision;

/// 2026-09-26: The whole scheduler-facing model interface: `Send + Sync` and every supertrait
/// below. An implementor writes one `impl` per supertrait and an empty `impl Model`.
pub trait Model:
    Send
    + Sync
    + ModelLifecycle
    + ModelForward
    + ModelLogits
    + ModelAdapters
    + ModelSsmState
    + ModelVerify
    + ModelDraft
    + ModelVision
    + ModelEp
    + ModelStreams
    + ModelDeviceFeed
{
}

#[cfg(test)]
mod padded_batch_n_tests {
    use super::padded_batch_n;

    /// 2026-09-25: The rung for each `n` up to 128, and `n` itself above.
    #[test]
    fn ladder_rungs() {
        for (n, want) in [
            (1usize, 2usize),
            (2, 2),
            (3, 4),
            (5, 8),
            (9, 12),
            (13, 16),
            (16, 16),
            (17, 24),
            (25, 32),
            (32, 32),
        ] {
            assert_eq!(padded_batch_n(n), want, "n={n}");
        }
        for (n, want) in [
            (33usize, 48usize),
            (48, 48),
            (49, 64),
            (64, 64),
            (65, 96),
            (96, 96),
            (97, 128),
            (128, 128),
        ] {
            assert_eq!(padded_batch_n(n), want, "n={n}");
        }
        assert_eq!(padded_batch_n(129), 129);
    }
}

/// 2026-09-25: An EP worker command that was received and then failed to execute. The worker
/// loop (`serve_phases/build.rs`) logs it and keeps running, since the head raises the same
/// error for the same command and answers the client with it. Any other error from
/// `Model::ep_worker_step` came from the receive, and the worker exits.
#[derive(Debug)]
pub struct EpCommandFailed(pub anyhow::Error);

impl std::fmt::Display for EpCommandFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for EpCommandFailed {}

#[cfg(test)]
mod ep_command_failed_tests {
    use super::EpCommandFailed;

    /// 2026-09-25: The worker loop classifies by downcast, so the tag and the original message
    /// must both survive being boxed into an `anyhow::Error`.
    #[test]
    fn the_tag_and_its_message_survive_anyhow() {
        let inner = anyhow::anyhow!("Prefill chunk layer 3 failed: DSA indexer cache: 16385");
        let tagged = anyhow::Error::new(EpCommandFailed(inner));
        assert!(
            tagged.downcast_ref::<EpCommandFailed>().is_some(),
            "the worker loop cannot tell a command failure from a dead link without this"
        );
        assert!(format!("{tagged:#}").contains("DSA indexer cache: 16385"));
    }

    /// 2026-09-25: An untagged error, as from a failed receive, is not a command failure.
    #[test]
    fn an_untagged_error_stays_fatal() {
        let recv = anyhow::anyhow!("ep_recv_seq_and_cmd: peer closed");
        assert!(recv.downcast_ref::<EpCommandFailed>().is_none());
    }
}
