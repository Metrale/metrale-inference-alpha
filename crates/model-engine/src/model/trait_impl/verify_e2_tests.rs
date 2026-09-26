// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the batched-verify staging helpers in `verify_e2.rs`.
//!
//! * [`verify_wy_cache_key`] must be injective over everything the staged WY
//!   pointer tables depend on. A collision would let a verify step reuse
//!   tables staged for a different batch, so its GDN layers would read
//!   another sequence's recurrent state.
//! * [`value_switch_armed`] must accept only the exact value `1`. A looser
//!   predicate would arm `METRALE_K4_DIAG`, which turns off the batched-verify
//!   graphs, for anyone who sets it to `0`.
//!
//! Owner: model-engine (speculative verify).
//! Invariants: none beyond the types.

use super::{value_switch_armed, verify_wy_cache_key};

/// 2026-09-25: The full input set, so a test can vary one axis at a time.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Inputs {
    slots: Vec<u32>,
    k: usize,
    ghosts: Vec<(u32, u32)>,
}

impl Inputs {
    fn key(&self) -> Vec<u64> {
        verify_wy_cache_key(&self.slots, self.k, &self.ghosts)
    }
}

fn base() -> Inputs {
    Inputs {
        slots: vec![3, 7],
        k: 4,
        ghosts: vec![],
    }
}

/// 2026-09-25: Equal inputs built separately (not cloned) give the same key;
/// otherwise the cache would miss on every step.
#[test]
fn key_is_reused_when_no_input_changes() {
    let a = Inputs {
        slots: vec![3, 7],
        k: 4,
        ghosts: vec![(11, 2)],
    };
    let b = Inputs {
        slots: vec![3, 7],
        k: 4,
        ghosts: vec![(11, 2)],
    };
    assert_eq!(a.key(), b.key());
}

/// 2026-09-25: Changing any one input the staged tables depend on must change
/// the key: `k`, the sequence count, a slot value, the slot order, ghost
/// presence, a ghost slot, a ghost depth, the ghost order and the ghost count
/// (`verify_wy_cache_key` lists the inputs).
#[test]
fn key_changes_when_any_input_changes() {
    let b = base();
    let variants: Vec<(&str, Inputs)> = vec![
        ("k", Inputs { k: 3, ..b.clone() }),
        (
            "sequence count (fewer)",
            Inputs {
                slots: vec![3],
                ..b.clone()
            },
        ),
        (
            "sequence count (more)",
            Inputs {
                slots: vec![3, 7, 9],
                ..b.clone()
            },
        ),
        (
            "a slot value",
            Inputs {
                slots: vec![3, 8],
                ..b.clone()
            },
        ),
        (
            "slot order (batch order is table order)",
            Inputs {
                slots: vec![7, 3],
                ..b.clone()
            },
        ),
        (
            "ghost presence",
            Inputs {
                ghosts: vec![(11, 2)],
                ..b.clone()
            },
        ),
    ];
    for (what, v) in variants {
        assert_ne!(b.key(), v.key(), "key must change when {what} changes");
    }

    // 2026-09-25: Ghost axes, against a baseline with ghosts.
    let g = Inputs {
        ghosts: vec![(11, 2), (13, 3)],
        ..base()
    };
    let ghost_variants: Vec<(&str, Inputs)> = vec![
        (
            "a ghost slot",
            Inputs {
                ghosts: vec![(12, 2), (13, 3)],
                ..base()
            },
        ),
        (
            "a ghost depth",
            Inputs {
                ghosts: vec![(11, 4), (13, 3)],
                ..base()
            },
        ),
        (
            "ghost order",
            Inputs {
                ghosts: vec![(13, 3), (11, 2)],
                ..base()
            },
        ),
        (
            "ghost count",
            Inputs {
                ghosts: vec![(11, 2)],
                ..base()
            },
        ),
    ];
    for (what, v) in ghost_variants {
        assert_ne!(g.key(), v.key(), "key must change when {what} changes");
    }
}

/// 2026-09-25: No two inputs in a small domain share a key. Slots and ghost
/// pairs are concatenated into one `Vec<u64>`, and `n` in the key keeps them
/// apart: without it, `slots=[1,2,3], ghosts=[]` and
/// `slots=[1], ghosts=[(2,3)]` would both encode as `[k,1,2,3]`.
#[test]
fn key_is_injective_over_the_reachable_domain() {
    let mut seen: std::collections::HashMap<Vec<u64>, Inputs> = std::collections::HashMap::new();
    let slot_sets: Vec<Vec<u32>> = vec![
        vec![0],
        vec![1],
        vec![0, 1],
        vec![1, 0],
        vec![0, 2],
        vec![0, 1, 2],
    ];
    let ghost_sets: Vec<Vec<(u32, u32)>> = vec![
        vec![],
        vec![(1, 2)],
        vec![(2, 1)],
        vec![(1, 2), (2, 3)],
        vec![(2, 3), (1, 2)],
    ];
    for slots in &slot_sets {
        for k in 2..=4usize {
            for ghosts in &ghost_sets {
                let inp = Inputs {
                    slots: slots.clone(),
                    k,
                    ghosts: ghosts.clone(),
                };
                if let Some(prev) = seen.insert(inp.key(), inp.clone()) {
                    panic!("WY cache key collision: {prev:?} and {inp:?} encode alike");
                }
            }
        }
    }
}

/// 2026-09-25: Only the exact value `1` arms a value switch; absence, an empty
/// value, `0`, `2`, `true` and ` 1` do not.
#[test]
fn value_switch_is_armed_only_by_a_literal_one() {
    assert!(value_switch_armed(Some("1")));
    for raw in [
        None,
        Some(""),
        Some("0"),
        Some("2"),
        Some("true"),
        Some(" 1"),
    ] {
        assert!(
            !value_switch_armed(raw),
            "{raw:?} must not arm a VALUE switch"
        );
    }
}
