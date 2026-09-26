// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: tests for `scheduling_policy`, loaded as its child module
//! with `#[path]`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
use super::*;

#[test]
fn fifo_always_prefills() {
    let policy = FifoPolicy;
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    assert!(policy.should_prefill(Instant::now(), &timings));
    assert!(policy.should_prefill(Instant::now(), &[]));
}

#[test]
fn fifo_selects_first_n() {
    let policy = FifoPolicy;
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 100,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 10,
            index: 1,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 2,
        },
        PendingRequestInfo {
            prompt_len: 200,
            index: 3,
        },
    ];
    assert_eq!(policy.select_prefills(&requests, 2), vec![0, 1]);
    assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1, 2, 3]);
}

#[test]
fn slai_prefills_when_no_active() {
    let policy = SlaiPolicy::new(100);
    assert!(policy.should_prefill(Instant::now(), &[]));
}

#[test]
fn slai_prefills_when_fresh() {
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    assert!(policy.should_prefill(Instant::now(), &timings));
}

#[test]
fn slai_skips_prefill_near_deadline() {
    let policy = SlaiPolicy::new(100);
    let old_time = Instant::now() - Duration::from_millis(85);
    let timings = vec![ActiveSeqTiming {
        last_token_time: old_time,
    }];
    assert!(!policy.should_prefill(Instant::now(), &timings));
}

#[test]
fn slai_prefills_within_margin() {
    let policy = SlaiPolicy::new(100);
    let recent = Instant::now() - Duration::from_millis(50);
    let timings = vec![ActiveSeqTiming {
        last_token_time: recent,
    }];
    assert!(policy.should_prefill(Instant::now(), &timings));
}

#[test]
fn slai_one_urgent_blocks_prefill() {
    let policy = SlaiPolicy::new(100);
    let now = Instant::now();
    let timings = vec![
        ActiveSeqTiming {
            last_token_time: now,
        },
        ActiveSeqTiming {
            last_token_time: now - Duration::from_millis(90),
        },
    ];
    assert!(!policy.should_prefill(Instant::now(), &timings));
}

#[test]
fn slai_selects_shortest_from_all() {
    let policy = SlaiPolicy::new(100);
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 500,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 10,
            index: 1,
        },
        PendingRequestInfo {
            prompt_len: 200,
            index: 2,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 3,
        },
        PendingRequestInfo {
            prompt_len: 300,
            index: 4,
        },
    ];
    // 2026-09-25: capacity 3: seat 0 goes to the queue front (index 0, the
    // 500-token prompt), then the shortest two of the rest: 1(10), 3(50).
    assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 3]);
    // 2026-09-25: one more seat takes 2(200), not 4(300).
    assert_eq!(policy.select_prefills(&requests, 4), vec![0, 1, 3, 2]);
    assert_eq!(policy.select_prefills(&requests, 1), vec![0]);
    // 2026-09-25: capacity 0 returns early, before `capacity - 1`.
    assert!(policy.select_prefills(&requests, 0).is_empty());
}

#[test]
fn slai_selects_all_when_capacity_exceeds() {
    let policy = SlaiPolicy::new(100);
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 100,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 10,
            index: 1,
        },
    ];
    // 2026-09-25: both are selected; the front seat puts index 0 first
    // although index 1 is shorter.
    assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1]);
}

#[test]
fn slai_stable_order_for_equal_lengths() {
    let policy = SlaiPolicy::new(100);
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 50,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 1,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 2,
        },
    ];
    assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 2]);
}

/// 2026-09-25: simulate ticks: `capacity` (at least 1) short requests
/// arrive at the back, the policy selects, and the selected requests leave
/// the queue. Returns the tick on which request `watch` (tracked by id, the
/// long prompt at the queue front) was selected, or `None`.
fn ticks_until_selected(
    policy: &dyn SchedulingPolicy,
    watch: usize,
    capacity: usize,
    ticks: usize,
) -> Option<usize> {
    let mut queue: Vec<usize> = vec![5000];
    let mut ids: Vec<usize> = vec![watch];
    for tick in 0..ticks {
        // 2026-09-25: arrivals come before the selection, so every tick
        // offers at least `capacity` shorter requests.
        for a in 0..capacity.max(1) {
            queue.push(10);
            ids.push(watch + 1 + tick * capacity.max(1) + a);
        }
        let infos: Vec<PendingRequestInfo> = queue
            .iter()
            .enumerate()
            .map(|(i, &prompt_len)| PendingRequestInfo {
                prompt_len,
                index: i,
            })
            .collect();
        let sel = policy.select_prefills(&infos, capacity);
        if sel.iter().any(|&i| ids[i] == watch) {
            return Some(tick);
        }
        let mut rm = sel.clone();
        rm.sort_unstable_by(|a, b| b.cmp(a));
        for i in rm {
            queue.remove(i);
            ids.remove(i);
        }
    }
    None
}

// 2026-09-25: a long prompt at the queue front is selected on the first
// tick, although shorter requests keep arriving.
#[test]
fn slai_does_not_starve_a_long_prompt_behind_short_arrivals() {
    let policy = SlaiPolicy::new(100);
    assert_eq!(
        ticks_until_selected(&policy, 0, 1, 500),
        Some(0),
        "the oldest pending request must be selected immediately"
    );
}

// 2026-09-25: the same at capacities 1 to 8.
#[test]
fn slai_head_reservation_holds_at_every_capacity() {
    let policy = SlaiPolicy::new(100);
    for capacity in 1..=8usize {
        assert_eq!(
            ticks_until_selected(&policy, 0, capacity, 200),
            Some(0),
            "starved at capacity {capacity}"
        );
    }
}

#[test]
fn fifo_never_starves_the_head() {
    assert_eq!(ticks_until_selected(&FifoPolicy, 0, 1, 10), Some(0));
}

#[test]
fn select_prefills_empty() {
    assert!(FifoPolicy.select_prefills(&[], 5).is_empty());
    assert!(SlaiPolicy::new(100).select_prefills(&[], 5).is_empty());
}

#[test]
fn fifo_slice_budget_is_full_chunk() {
    // 2026-09-25: FIFO uses the trait's default: always `full_chunk`.
    let policy = FifoPolicy;
    assert_eq!(policy.prefill_slice_budget(Instant::now(), &[], 4080), 4080);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    assert_eq!(
        policy.prefill_slice_budget(Instant::now(), &timings, 4080),
        4080
    );
}

#[test]
fn slai_slice_budget_full_when_no_active() {
    let policy = SlaiPolicy::new(100);
    assert_eq!(policy.prefill_slice_budget(Instant::now(), &[], 4080), 4080);
}

#[test]
fn slai_slice_budget_zero_past_deadline() {
    // 2026-09-25: 120 ms since the last token is past the 100 ms deadline.
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now() - Duration::from_millis(120),
    }];
    assert_eq!(
        policy.prefill_slice_budget(Instant::now(), &timings, 4080),
        0
    );
}

#[test]
fn slai_slice_budget_bounded_and_wy4_aligned() {
    // 2026-09-25: under the deadline the budget is `full_chunk` unchanged.
    // The `% 4` check holds because 4080 is a multiple of 4; the policy does
    // not align.
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    let b = policy.prefill_slice_budget(Instant::now(), &timings, 4080);
    assert!(b > 0, "non-deadline budget must be > 0");
    assert!(b <= 4080, "must never exceed full_chunk");
    assert_eq!(b % 4, 0, "must be WY4-aligned");
}

#[test]
fn slai_slice_budget_never_exceeds_small_full_chunk() {
    // 2026-09-25: a small `full_chunk` is returned unchanged.
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    let b = policy.prefill_slice_budget(Instant::now(), &timings, 64);
    assert!(b <= 64);
    assert_eq!(b % 4, 0);
}
