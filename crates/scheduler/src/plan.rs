// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: What the core asks the device to run ([`StepPlan`]) and what comes back
//! ([`StepResult`]). Generic over the device's logits handle `P`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: A short label for traces.
pub trait Label {
    fn label(&self) -> String;
}

/// 2026-09-25: Where a fed decode row's input token comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedSource {
    /// 2026-09-25: The previous step's argmax for row `from_row`, read from the device
    /// feed cell that step wrote; the host does not know the id at launch.
    Feed { from_row: u32 },
    /// 2026-09-25: A token id the host supplies.
    Host(u32),
}

/// 2026-09-25: The two token ids the host would re-pick around for a row on the
/// argmax fast path (`u32::MAX` = none): the closed-thinking mask.
pub type RowMask = [u32; 2];

/// 2026-09-25: How the logits of a decode step come back to the host.
pub enum Readback<'a> {
    /// 2026-09-25: One argmax id per row, computed on the device.
    Argmax,
    /// 2026-09-25: The device argmax with the per-row two-id mask applied exactly as
    /// the host re-pick would (see `argmax_feed.cu`); the ids also land
    /// in the device feed cells for a following fed step.
    ArgmaxMasked { masks: Vec<RowMask> },
    /// 2026-09-25: The whole logits block, copied into `into` (resized by the router to
    /// `rows * vocab * elem_bytes`; the core samples over it).
    HostLogits { into: &'a mut Vec<u8> },
}

impl std::fmt::Debug for Readback<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Argmax => f.write_str("Argmax"),
            Self::ArgmaxMasked { masks } => write!(f, "ArgmaxMasked{{n={}}}", masks.len()),
            Self::HostLogits { .. } => f.write_str("HostLogits"),
        }
    }
}

/// 2026-09-25: The DFlash context commit a decode step performs between its forward and
/// its readback (the drafter's view of the serial tokens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCtxCommit {
    None,
    /// 2026-09-25: `commit_ctx(row, 1, seq_len - 1, row)` for every row.
    Unified,
    /// 2026-09-25: `dflash_serial_ctx_append(row 0)` (single-row batches only).
    SerialAppend,
}

/// 2026-09-25: One device step. The sequence rows it advances travel beside it as
/// `&mut S`.
#[derive(Debug)]
pub enum StepPlan<'a> {
    /// 2026-09-25: Batched decode of one token per row, the context commit, then the
    /// readback.
    Decode {
        /// 2026-09-25: The pending decode input of each row, in row order.
        tokens: Vec<u32>,
        ctx: DecodeCtxCommit,
        readback: Readback<'a>,
    },
    /// 2026-09-25: A decode step launched while the previous one is still in flight:
    /// each row's input is resolved on the device from `sources`, the
    /// readback is the masked feed argmax, and the host bookkeeping the
    /// plain step performs at launch (`tokens.push`, `seq_len += 1`) is
    /// deferred to the core, which applies it once the previous step is
    /// committed. The context commit runs between the forward and the
    /// readback, as it does for the plain step.
    DecodeFed {
        sources: Vec<FeedSource>,
        masks: Vec<RowMask>,
        ctx: DecodeCtxCommit,
    },
}

impl Label for StepPlan<'_> {
    fn label(&self) -> String {
        match self {
            Self::Decode {
                tokens,
                ctx,
                readback,
            } => format!(
                "Decode{{n={}, ctx={ctx:?}, readback={readback:?}}}",
                tokens.len()
            ),
            Self::DecodeFed { sources, ctx, .. } => {
                let fed = sources
                    .iter()
                    .filter(|s| matches!(s, FeedSource::Feed { .. }))
                    .count();
                format!("DecodeFed{{n={}, fed={fed}, ctx={ctx:?}}}", sources.len())
            }
        }
    }
}

/// 2026-09-25: The rows of a decode step after the readback.
#[derive(Debug)]
pub enum DecodeRows {
    /// 2026-09-25: The device argmax per row.
    Tokens(Vec<u32>),
    /// 2026-09-25: The logits block was copied into the plan's buffer; `elem_bytes` is
    /// the element width the model writes (2 = BF16, 4 = FP32).
    HostLogits { elem_bytes: usize },
}

#[derive(Debug)]
pub enum StepOutcome<P> {
    Decode {
        /// 2026-09-25: The device logits the step produced, for a later
        /// [`crate::Effect::ReadLogits`].
        logits: P,
        rows: DecodeRows,
    },
}

#[derive(Debug)]
pub struct StepResult<P> {
    pub ticket: crate::Ticket,
    pub outcome: StepOutcome<P>,
}
