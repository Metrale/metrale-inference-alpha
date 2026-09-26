// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The BFCL samples whose answer depended on what ran before them.
//! Measured 2026-09-08: the golden n=995 draw run whole and the same draw run
//! as four shards disagreed on exactly these `sample_id`s.
//! The mechanism is cross-request SSM snapshot reuse, described in
//! [`crate::gate::group`]. The driver warns on each of these it runs and
//! reports the count as the `known_partition_sensitive` metric; membership
//! does not change how a sample is scored.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: [`KNOWN_PARTITION_SENSITIVE`] is sorted and free of duplicates.

/// 2026-09-26: Sorted, because [`is_known`] binary-searches it.
pub const KNOWN_PARTITION_SENSITIVE: &[&str] = &[
    "live_irrelevance_14-2-2",
    "live_irrelevance_15-2-3",
    "live_irrelevance_16-2-4",
    "live_irrelevance_2-0-2",
    "live_irrelevance_40-2-28",
    "live_irrelevance_47-2-35",
    "live_irrelevance_55-2-43",
    "live_irrelevance_71-2-59",
    "live_irrelevance_79-2-67",
    "live_irrelevance_8-0-8",
    "live_multiple_57-22-4",
    "live_parallel_multiple_3-2-1",
];

pub fn is_known(sample_id: &str) -> bool {
    KNOWN_PARTITION_SENSITIVE.binary_search(&sample_id).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_is_sorted_and_deduplicated_so_binary_search_is_valid() {
        let mut sorted = KNOWN_PARTITION_SENSITIVE.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, KNOWN_PARTITION_SENSITIVE, "keep the list sorted");
        assert_eq!(KNOWN_PARTITION_SENSITIVE.len(), 12, "#936 records twelve");
    }

    #[test]
    fn every_id_names_a_single_turn_subset() {
        for id in KNOWN_PARTITION_SENSITIVE {
            let subset = super::super::draw::SINGLE_TURN_SUBSETS
                .iter()
                .filter(|s| id.starts_with(&format!("{s}_")))
                .max_by_key(|s| s.len());
            assert!(subset.is_some(), "{id} does not belong to a scored subset");
        }
    }

    #[test]
    fn membership_is_exact() {
        assert!(is_known("live_irrelevance_2-0-2"));
        assert!(is_known("live_parallel_multiple_3-2-1"));
        assert!(!is_known("live_irrelevance_2-0-3"));
        assert!(!is_known("live_irrelevance_2-0"));
        assert!(!is_known("simple_python_0"));
    }
}
