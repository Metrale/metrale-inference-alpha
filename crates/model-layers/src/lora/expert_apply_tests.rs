// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `expert_delta_workitems`: `expert_offsets` and the
//! adapted experts to (expert, row_off, rows) work items. GPU-free.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use crate::lora::*;

#[test]
fn workitems_map_offsets_to_row_ranges() {
    let offsets = [0u32, 3, 3, 10, 12];
    let work = expert_delta_workitems(&offsets, &[0, 2, 3]);
    assert_eq!(
        work,
        vec![
            ExpertWork {
                expert: 0,
                row_off: 0,
                rows: 3
            },
            ExpertWork {
                expert: 2,
                row_off: 3,
                rows: 7
            },
            ExpertWork {
                expert: 3,
                row_off: 10,
                rows: 2
            },
        ]
    );
}

#[test]
fn workitems_skip_empty_and_out_of_range() {
    let offsets = [0u32, 5, 5];
    assert!(expert_delta_workitems(&offsets, &[1, 9]).is_empty());
    assert_eq!(
        expert_delta_workitems(&offsets, &[0]),
        vec![ExpertWork {
            expert: 0,
            row_off: 0,
            rows: 5
        }]
    );
    // 2026-09-25: A decreasing pair is skipped rather than underflowing.
    assert!(expert_delta_workitems(&[0, 7, 3], &[1]).is_empty());
    // 2026-09-25: A table of fewer than two offsets describes no experts.
    assert!(expert_delta_workitems(&[], &[0]).is_empty());
    assert!(expert_delta_workitems(&[0], &[0]).is_empty());
}

#[test]
fn workitems_only_adapted_experts_launch() {
    let offsets = [0u32, 4, 8, 12];
    let work = expert_delta_workitems(&offsets, &[1]);
    assert_eq!(
        work,
        vec![ExpertWork {
            expert: 1,
            row_off: 4,
            rows: 4
        }]
    );
}
