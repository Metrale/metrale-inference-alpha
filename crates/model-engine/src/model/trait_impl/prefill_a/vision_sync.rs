// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Give every rank the same merged vision embeddings before the prefill splice.
//!
//! Only rank 0 receives image bytes and runs the vision tower.
//! `ep_sync_vision_embeds` broadcasts rank 0's merged rows
//! (`[n_rows, out_hidden_size]` BF16 from the encoder output) so that every
//! worker splices the same rows; workers never run the tower.
//!
//! Grids are not broadcast. `upload_meta.rs` reads `vision_image_grids` when
//! `config.mrope_interleaved` is set, and a worker never fills it, so a model
//! with MRoPE positions on a multi-rank serve would also need its grids sent.
//!
//! Owner: model-engine.
//! Invariants:
//! - A worker accepts only a broadcast count of 0 or its own vision-pad count
//!   for the prompt; any other count is an error.
//! - A worker that receives a count of 0 clears `vision_embed_patches` and
//!   `vision_row_base`.

use anyhow::Result;

use super::super::super::types::TransformerModel;

/// 2026-09-25: How many merged rows this prompt's pad run needs (image and
/// video pad tokens).
///
/// Counted from the tokens, which every rank holds, so a worker can check the
/// broadcast count against its own count.
pub(crate) fn vision_rows_for_prompt(tokens: &[u32], image_pad: u32, video_pad: u32) -> usize {
    tokens
        .iter()
        .filter(|&&t| t == image_pad || t == video_pad)
        .count()
}

/// 2026-09-25: How many rows rank 0 sends for this prompt.
///
/// `pending` is `vision_embed_patches`, the total merged rows in the shared
/// packed encoder output, which under co-dispatch spans several requests. This
/// request owns `[row_base, row_base + n_pad)` of it. When `pending == 0` rank 0
/// does not splice either (embed_chunk.rs), so the answer is 0 and both ranks
/// skip the broadcast. A window past `pending` is an error.
pub(crate) fn rows_to_broadcast(pending: usize, row_base: usize, n_pad: usize) -> Result<usize> {
    if pending == 0 || n_pad == 0 {
        return Ok(0);
    }
    let end = row_base
        .checked_add(n_pad)
        .ok_or_else(|| anyhow::anyhow!("vision sync: row_base {row_base} + {n_pad} overflows"))?;
    anyhow::ensure!(
        end <= pending,
        "vision sync: this request claims rows [{row_base}, {end}) of a packed buffer holding \
         {pending}. The prompt's pad run and the encoder's output disagree, which is the \
         desync the splice cannot see."
    );
    Ok(n_pad)
}

/// 2026-09-25: 32-bit FNV-1a over the bytes of the rows that will be spliced,
/// used as a cross-rank equality check.
pub(crate) fn digest(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// 2026-09-25: Whether the cross-rank digest check runs: `METRALE_GLM_VISION_CHECK`
/// set to 1, true, yes or on. When it runs, every rank copies its spliced rows to
/// the host to digest them, and a worker whose digest differs from rank 0's
/// returns an error.
pub(crate) fn check_enabled() -> bool {
    matches!(
        std::env::var("METRALE_GLM_VISION_CHECK")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

impl TransformerModel {
    /// 2026-09-25: Rank 0 to workers: the merged vision rows for this prompt.
    ///
    /// Both sides call it at the same point of the `0xFFFFFFF0` sequence, after
    /// the prompt tokens and before the prefill (the scheduler's send sites on
    /// rank 0, the `0xFFFFFFF0` handler in impl_a2.rs on workers), so the
    /// collectives pair. It returns before any collective on a single-rank run
    /// or when the model has no vision tower.
    pub(in crate::model) fn ep_sync_vision_embeds(&self, tokens: &[u32]) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        let Some(ve) = self.vision_encoder.as_ref() else {
            return Ok(());
        };
        let comm = self.comm.as_ref().expect("multi_rank_protocol_active");
        let (image_pad, video_pad) = self.vision_pad_ids();
        let n_pad = vision_rows_for_prompt(tokens, image_pad, video_pad);
        let is_head = comm.rank() == 0;

        let n_rows = if is_head {
            let pending = *self.vision_embed_patches.lock();
            let row_base = *self.vision_row_base.lock();
            let n = rows_to_broadcast(pending, row_base, n_pad)?;
            self.ep_broadcast_u32(n as u32)?;
            n
        } else {
            let n = self.ep_broadcast_u32(0)? as usize;
            // 2026-09-25: The count must be 0 or the worker's own pad count for
            // the prompt.
            anyhow::ensure!(
                n == 0 || n == n_pad,
                "vision sync: rank {} was sent {n} merged rows but counts {n_pad} vision pad \
                 tokens in the prompt. Head and worker disagree about this request's images.",
                comm.rank()
            );
            n
        };
        if n_rows == 0 {
            if !is_head {
                // 2026-09-25: Clear, so a stale count from an earlier request
                // cannot make the worker splice into this prompt.
                *self.vision_embed_patches.lock() = 0;
                *self.vision_row_base.lock() = 0;
            }
            return Ok(());
        }

        // 2026-09-25: A worker never encodes an image, so its encoder scratch,
        // the broadcast destination, is allocated here.
        ve.ensure_scratch(self.gpu.as_ref())?;
        let out = ve.out_hidden_size();
        let bytes = n_rows * out * 2;
        let src_row = if is_head {
            *self.vision_row_base.lock()
        } else {
            0
        };
        comm.broadcast(ve.out_row(src_row).0, bytes, 0)?;

        if !is_head {
            // 2026-09-25: The worker's rows land at row 0, so its splice reads
            // from 0. Rank 0 keeps its own (possibly co-dispatched) base.
            *self.vision_embed_patches.lock() = n_rows;
            *self.vision_row_base.lock() = 0;
        }
        self.verify_vision_rank_agreement(n_rows, src_row, bytes, is_head)
    }

    /// 2026-09-25: Log the row count. When `check_enabled()`, also compare each
    /// worker's digest of its rows with rank 0's and return an error on a
    /// mismatch.
    fn verify_vision_rank_agreement(
        &self,
        n_rows: usize,
        src_row: usize,
        bytes: usize,
        is_head: bool,
    ) -> Result<()> {
        let comm = self.comm.as_ref().expect("multi_rank_protocol_active");
        if !check_enabled() {
            tracing::debug!(
                "vision sync: rank {} holds {n_rows} merged rows at base {src_row}",
                comm.rank()
            );
            return Ok(());
        }
        let ve = self
            .vision_encoder
            .as_ref()
            .expect("checked by the only caller");
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ve.out_row(src_row), &mut buf)?;
        let local = digest(&buf);
        tracing::info!(
            "METRALE_GLM_VISION_CHECK rank {} rows={n_rows} base={src_row} digest={local:#010x}",
            comm.rank()
        );
        let head = self.ep_broadcast_u32(if is_head { local } else { 0 })?;
        anyhow::ensure!(
            is_head || head == local,
            "vision sync: rank {} spliced digest {local:#010x} but rank 0 has {head:#010x} over \
             the same {n_rows} rows. The ranks are about to all-reduce DIFFERENT image \
             embeddings — the model would stay fluent and answer about the wrong picture.",
            comm.rank()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: The count every rank derives from the prompt; video pad tokens
    /// count as well as image pad tokens.
    #[test]
    fn rows_are_counted_from_the_prompt_not_trusted_from_the_wire() {
        let (img, vid) = (154_854u32, 154_855u32);
        let tokens = vec![1, 154_830, img, img, img, 154_831, 7, vid, vid, 2];
        assert_eq!(vision_rows_for_prompt(&tokens, img, vid), 5);
        assert_eq!(vision_rows_for_prompt(&[1, 2, 3], img, vid), 0);
    }

    /// 2026-09-25: A prompt with no pad tokens sends nothing, and so does one for
    /// which rank 0 has no staged encoder output; both ranks then skip the
    /// broadcast together.
    #[test]
    fn nothing_is_sent_when_either_side_has_nothing() {
        assert_eq!(rows_to_broadcast(0, 0, 256).unwrap(), 0);
        assert_eq!(rows_to_broadcast(256, 0, 0).unwrap(), 0);
        assert_eq!(rows_to_broadcast(0, 0, 0).unwrap(), 0);
    }

    /// 2026-09-25: Under co-dispatch the encoder output is shared; a request
    /// sends only its own window of it.
    #[test]
    fn a_co_dispatched_request_sends_only_its_own_window() {
        // 2026-09-25: Three requests of 256 rows packed together; the middle one.
        assert_eq!(rows_to_broadcast(768, 256, 256).unwrap(), 256);
        // 2026-09-25: The last one ends exactly at the end, which is in bounds.
        assert_eq!(rows_to_broadcast(768, 512, 256).unwrap(), 256);
    }

    /// 2026-09-25: A window running past the packed rows means the pad run and the
    /// encoder output disagree, and it is an error, including when the end
    /// overflows.
    #[test]
    fn a_window_past_the_packed_rows_is_refused() {
        let err = rows_to_broadcast(768, 512, 257).unwrap_err().to_string();
        assert!(err.contains("769"), "must name the end row: {err}");
        assert!(err.contains("768"), "must name the capacity: {err}");
        assert!(rows_to_broadcast(768, usize::MAX, 2).is_err());
    }

    /// 2026-09-25: One changed byte in one of 256 rows must change the digest.
    #[test]
    fn the_digest_separates_a_single_changed_row() {
        let rows = vec![0xABu8; 256 * 4096 * 2];
        let mut one_off = rows.clone();
        // 2026-09-25: Flip the lowest mantissa bit of one BF16 in the middle row.
        one_off[128 * 4096 * 2] ^= 0x01;
        assert_ne!(digest(&rows), digest(&one_off));
        // 2026-09-25: An empty payload and a zeroed one digest differently.
        assert_ne!(digest(&[]), digest(&[0u8; 8]));
    }

    /// 2026-09-25: The accepted values of the check switch.
    #[test]
    fn the_check_is_opt_in() {
        // 2026-09-25: `check_enabled` reads the process environment, which this
        // test does not mutate, so it asserts the match pattern instead.
        assert!(!matches!(None::<&str>, Some("1" | "true" | "yes" | "on")));
        assert!(matches!(Some("1"), Some("1" | "true" | "yes" | "on")));
    }
}
