// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: N-gram embedding (arXiv 2601.21204): the dimensions, the
//! host-side polynomial-rolling-hash ids that select rows in the
//! `K * (N-1)` lookup tables (`ids`), the tables (`table`) and the GPU
//! embedding (`embed`). `ids_match_python_reference` checks the ids against
//! `bench/ngram_ref/ngram_id_fixtures.json`, written by
//! `bench/ngram_ref/make_fixtures.py`.
//!
//! Table `index = (i-2)*K + j` for n-gram size `i` and split `j`:
//!
//! ```text
//!   T      = ratio * vocab_size + 2*index + 1        (table row count)
//!   mods   = [V^1 mod T, V^2 mod T, ..., V^(i-1) mod T]
//!   id_t   = ( x_t + Σ_{d=1..i-1} shift_d(x)_t * mods[d-1] ) mod T
//! ```
//!
//! where `shift_d` is a right-shift by `d` that resets at document
//! boundaries: a position within `d` tokens of a segment start (a segment
//! ends at an EOS token, inclusive) contributes token id 0. The ids depend
//! only on token ids.
//!
//! Owner: model-layers (n-gram embedding).
//! Invariants:
//! - `ngram_ids` sums in u64 and reduces once per id, so the ids are exact
//!   only while `neighbor_num * vocab_size * T < 2^64`; the code does not
//!   check this.

/// 2026-09-25: The n-gram dimensions, read from `ModelConfig`.
#[derive(Debug, Clone, Copy)]
pub struct NgramDims {
    pub vocab_size: u64,
    pub ratio: u64,
    /// 2026-09-25: Largest n-gram size N.
    pub neighbor_num: usize,
    /// 2026-09-25: Hash splits K per n-gram size.
    pub split_num: usize,
    pub eos_token_id: u32,
    pub hidden_size: usize,
}

impl NgramDims {
    pub fn from_config(c: &metrale_config::ModelConfig) -> Option<Self> {
        if c.ngram_vocab_size_ratio == 0 {
            return None;
        }
        Some(Self {
            vocab_size: c.vocab_size as u64,
            ratio: c.ngram_vocab_size_ratio as u64,
            neighbor_num: c.emb_neighbor_num,
            split_num: c.emb_split_num,
            eos_token_id: c.eos_token_id,
            hidden_size: c.hidden_size,
        })
    }

    pub fn num_tables(&self) -> usize {
        self.split_num * (self.neighbor_num - 1)
    }

    /// 2026-09-25: Per-table embedding dim, `hidden_size / num_tables()`
    /// (integer division).
    pub fn table_dim(&self) -> usize {
        self.hidden_size / self.num_tables()
    }

    /// 2026-09-25: Row count of table `index`: `ratio*vocab + 2*index + 1`,
    /// so the K tables of one n-gram size have distinct sizes.
    pub fn table_rows(&self, index: usize) -> u64 {
        self.ratio * self.vocab_size + 2 * index as u64 + 1
    }

    /// 2026-09-25: `V^1 mod T, ..., V^(i-1) mod T` for table (i, j).
    pub fn vocab_mods(&self, ngram: usize, split: usize) -> Vec<u64> {
        let index = (ngram - 2) * self.split_num + split;
        let t = self.table_rows(index);
        let mut mods = Vec::with_capacity(ngram - 1);
        let mut power: u64 = 1;
        for _ in 0..ngram - 1 {
            power = (power * self.vocab_size) % t;
            mods.push(power);
        }
        mods
    }
}

mod embed;
mod ids;
mod table;

pub use embed::NgramEmbedding;
pub use ids::ngram_ids;
pub use table::NgramTable;

#[cfg(test)]
mod tests;
