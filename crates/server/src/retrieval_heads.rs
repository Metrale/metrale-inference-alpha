// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A table of retrieval-critical attention heads by (layer, head), with lookups.
//!
//! Owner: server.
//! Invariants: `count_in_layer` counts only heads whose layer is below the
//! `num_layers` given to `new`; `is_retrieval_head` and `total` see every head.
//!
//! Nothing outside this module constructs or reads the table: no calibration
//! produces one and no KV path consults it.

use std::collections::HashSet;

/// 2026-09-26: Identifies a specific (layer, head) pair in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeadId {
    pub layer: u16,
    pub head: u16,
}

/// 2026-09-26: Set of heads marked retrieval-critical. Lookup is a `HashSet` probe.
#[derive(Debug, Clone, Default)]
pub struct RetrievalHeadSet {
    heads: HashSet<HeadId>,
    /// 2026-09-26: Per-layer head count, for `count_in_layer`.
    layer_counts: Vec<u16>,
}

impl RetrievalHeadSet {
    /// 2026-09-26: Empty set: no head is a retrieval head.
    pub fn empty() -> Self {
        Self::default()
    }

    /// 2026-09-26: Build from a list of heads. `num_layers` sizes the per-layer count
    /// array; a head at a layer at or above it is kept but not counted per layer.
    pub fn new(heads: impl IntoIterator<Item = HeadId>, num_layers: usize) -> Self {
        let heads: HashSet<HeadId> = heads.into_iter().collect();
        let mut layer_counts = vec![0u16; num_layers];
        for h in &heads {
            if (h.layer as usize) < num_layers {
                layer_counts[h.layer as usize] += 1;
            }
        }
        Self {
            heads,
            layer_counts,
        }
    }

    /// 2026-09-26: True iff (layer, head) is in the set.
    pub fn is_retrieval_head(&self, layer: u16, head: u16) -> bool {
        self.heads.contains(&HeadId { layer, head })
    }

    /// 2026-09-26: Number of retrieval heads in `layer`; 0 outside the count array.
    pub fn count_in_layer(&self, layer: u16) -> u16 {
        self.layer_counts.get(layer as usize).copied().unwrap_or(0)
    }

    /// 2026-09-26: Total retrieval heads across all layers.
    pub fn total(&self) -> usize {
        self.heads.len()
    }

    /// 2026-09-26: True iff the set is non-empty.
    pub fn is_calibrated(&self) -> bool {
        !self.heads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_set_is_not_calibrated() {
        let s = RetrievalHeadSet::empty();
        assert!(!s.is_calibrated());
        assert_eq!(s.total(), 0);
        assert!(!s.is_retrieval_head(0, 0));
    }

    #[test]
    fn lookup_after_construction() {
        let heads = vec![
            HeadId { layer: 5, head: 3 },
            HeadId { layer: 10, head: 7 },
            HeadId {
                layer: 10,
                head: 11,
            },
        ];
        let s = RetrievalHeadSet::new(heads, 40);
        assert!(s.is_calibrated());
        assert!(s.is_retrieval_head(5, 3));
        assert!(s.is_retrieval_head(10, 7));
        assert!(!s.is_retrieval_head(5, 4));
        assert_eq!(s.count_in_layer(10), 2);
        assert_eq!(s.count_in_layer(5), 1);
        assert_eq!(s.count_in_layer(0), 0);
        assert_eq!(s.total(), 3);
    }

    #[test]
    fn out_of_range_layer_returns_zero() {
        let heads = vec![HeadId { layer: 5, head: 3 }];
        let s = RetrievalHeadSet::new(heads, 40);
        assert_eq!(s.count_in_layer(100), 0, "out-of-range layer is safe");
    }
}
