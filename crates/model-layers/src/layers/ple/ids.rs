// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: PLE n-gram row ids: the EOS-aware right shift and the multiply-XOR hash.
//!
//! Owner: model-layers (PLE).
//! Invariants:
//! - Pure token-id arithmetic: no device and no weights, so `tests.rs` checks
//!   it bit-exactly against ids recorded from the reference forward.
//! - `ple_ngram_ids` returns one row of `ngram_heads()` ids per input token.
//!
//! The LongCat n-gram hash in `ngram_embed/ids.rs` sums `shift * m` and takes
//! one modulus. This one multiplies each shifted token by an odd multiplier,
//! XORs the products, then takes a per-head modulus and adds a per-head offset
//! into one concatenated table. Both give valid rows for the same tokens, so
//! using the wrong one fails silently.
//!
//! Reference: `Qwen4ExpTextNGramEmbedding.forward` and
//! `_shift_right_ignore_eos` in `bench/qwen4_exp/ref/modeling_qwen4_exp.py`.

/// 2026-09-25: Geometry for one PLE site, read from the checkpoint rather than derived.
///
/// `multipliers`, `head_vocab_sizes` and `head_offsets` are the checkpoint's
/// `ple_embedding.layer_multipliers`, `.ngram_heads_vocab_sizes` and
/// `.ngram_heads_offsets`. The reference can also derive them (SplitMix64
/// and a prime search); reading them keeps the ids tied to the checkpoint.
#[derive(Clone, Debug)]
pub struct PleIdDims {
    /// 2026-09-25: `ngram_size`. The loader also uses it as the PLE conv dilation.
    pub ngram_size: usize,
    /// 2026-09-25: `heads_per_ngram`. Heads are grouped by n-gram order:
    /// `[0, heads_per_ngram)` uses order 2, the next block order 3, and so on.
    pub heads_per_ngram: usize,
    /// 2026-09-25: `layer_multipliers[ngram_size]`; `validate` refuses an even one.
    pub multipliers: Vec<u64>,
    /// 2026-09-25: `ngram_heads_vocab_sizes[ngram_heads]`: each head's modulus.
    pub head_vocab_sizes: Vec<u64>,
    /// 2026-09-25: `ngram_heads_offsets[ngram_heads]`: where each head's range starts
    /// in the single concatenated table.
    pub head_offsets: Vec<u64>,
    pub eos_token_id: u32,
}

impl PleIdDims {
    /// 2026-09-25: `(ngram_size - 1) * heads_per_ngram`. Times `head_dim` this is
    /// the PLE embedding width, because the head slices are concatenated
    /// (`PleLayer::new` checks the product).
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// 2026-09-25: How many previous tokens a decode step must carry to reproduce
    /// prefill's ids.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// 2026-09-25: Check the table lengths, that every multiplier is odd and every
    /// head modulus is non-zero. The loader and `PleLayer::new` call it; an
    /// error means the checkpoint does not have the reference's geometry.
    pub fn validate(&self) -> anyhow::Result<()> {
        let heads = self.ngram_heads();
        anyhow::ensure!(
            self.multipliers.len() == self.ngram_size,
            "PLE: layer_multipliers has {} entries, expected ngram_size={}",
            self.multipliers.len(),
            self.ngram_size
        );
        anyhow::ensure!(
            self.head_vocab_sizes.len() == heads && self.head_offsets.len() == heads,
            "PLE: head vocab/offsets are {}/{}, expected ngram_heads={heads}",
            self.head_vocab_sizes.len(),
            self.head_offsets.len()
        );
        // 2026-09-25: The reference builds these as `2 * (splitmix64(..) % half) + 1`.
        for (i, m) in self.multipliers.iter().enumerate() {
            anyhow::ensure!(
                m % 2 == 1,
                "PLE: layer_multipliers[{i}] = {m} is even; the reference \
                 derives `2*x + 1`, so this checkpoint is not what we think"
            );
        }
        anyhow::ensure!(
            self.head_vocab_sizes.iter().all(|v| *v > 0),
            "PLE: a head vocab size is 0 — modulus would divide by zero"
        );
        Ok(())
    }
}

/// 2026-09-25: Right-shift by `shift`, refusing to read across an EOS boundary.
///
/// Positions whose source would fall before the current EOS-delimited
/// segment get EOS instead. Transcribed from `_shift_right_ignore_eos`:
/// `previous_eos` is the last EOS position strictly before each index, so a
/// token sitting on an EOS still belongs to the segment the previous EOS started.
fn shift_right_ignore_eos(tokens: &[u32], shift: usize, eos: u32) -> Vec<u32> {
    if shift == 0 {
        return tokens.to_vec();
    }
    let mut out = vec![eos; tokens.len()];
    // 2026-09-25: `prev_eos` is the reference's `previous_eos`: the last EOS index
    // strictly less than `pos`, or -1.
    let mut prev_eos: i64 = -1;
    let mut seen_eos: i64 = -1;
    for pos in 0..tokens.len() {
        let segment_start = prev_eos + 1;
        let position_in_segment = pos as i64 - segment_start;
        let source = pos as i64 - shift as i64;
        if position_in_segment >= shift as i64 && source >= 0 {
            out[pos] = tokens[source as usize];
        }
        if tokens[pos] == eos {
            seen_eos = pos as i64;
        }
        prev_eos = seen_eos;
    }
    out
}

/// 2026-09-25: Row ids for every head, one row per token in `tokens`.
///
/// `tokens` must already be `context ++ new`, where `context` is the
/// `context_len` preceding tokens (EOS-filled at the start of a sequence).
/// Returns `[tokens.len()][ngram_heads]`; callers slice off the last
/// `new.len()` rows, exactly as the reference's
/// `torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]` does.
pub fn ple_ngram_ids(dims: &PleIdDims, tokens: &[u32]) -> Vec<Vec<u64>> {
    let heads = dims.ngram_heads();
    let shifted: Vec<Vec<u32>> = (0..dims.ngram_size)
        .map(|s| shift_right_ignore_eos(tokens, s, dims.eos_token_id))
        .collect();

    let mut out = vec![vec![0u64; heads]; tokens.len()];
    for ngram in 2..=dims.ngram_size {
        let start = (ngram - 2) * dims.heads_per_ngram;
        for (pos, row) in out.iter_mut().enumerate() {
            // 2026-09-25: `mixed = shifted[0]*m[0]`, then XOR `shifted[p]*m[p]` for
            // p in 1..ngram. The reference bounds each multiplier by
            // `(2^63 - 1) / vocab_size`, so the product of a token id below
            // the vocabulary size and a multiplier does not overflow.
            let mut mixed = (shifted[0][pos] as u64).wrapping_mul(dims.multipliers[0]);
            for p in 1..ngram {
                mixed ^= (shifted[p][pos] as u64).wrapping_mul(dims.multipliers[p]);
            }
            for h in start..start + dims.heads_per_ngram {
                row[h] = mixed % dims.head_vocab_sizes[h] + dims.head_offsets[h];
            }
        }
    }
    out
}
