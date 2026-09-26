// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pure wave planning for the batched-prefill step.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every stream is in exactly one wave, and each wave lists its streams in
//!   index order.
//! - With `varlen`, the streams of a wave share `chunk_start` and `is_last`,
//!   and a wave of two or more streams holds at most `wave_token_cap`
//!   tokens.
//!
//! `plan_prefill_waves` splits one tick's prefilling streams into waves,
//! one `model.prefill_batch_chunk` call each. `check_kernel_batched_eligible`
//! (model-engine) admits a batch only when its streams share `chunk_start`
//! and `is_last`; varlen (`--prefill-varlen-batch`) lets their `chunk_len`
//! differ. Without varlen this returns one wave holding every stream.

/// 2026-09-25: Per-stream chunk geometry the planner partitions on. A
/// projection of `PrefillSlice` — kept as plain data so the planner is
/// unit-testable without a `SequenceState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WaveGeom {
    pub chunk_start: usize,
    pub chunk_len: usize,
    pub is_last: bool,
}

/// 2026-09-25: Partition stream indices `0..geoms.len()` into dispatch
/// waves.
///
/// Without `varlen`, or with one stream, a single wave holds every stream.
/// Otherwise first-fit in index order: each stream joins the earliest wave
/// whose head shares its `(chunk_start, is_last)` and whose token total stays
/// within `wave_token_cap`, or opens a new wave. A stream whose own chunk
/// exceeds the cap gets a wave to itself, so it still advances.
pub(super) fn plan_prefill_waves(
    geoms: &[WaveGeom],
    varlen: bool,
    wave_token_cap: usize,
) -> Vec<Vec<usize>> {
    if geoms.is_empty() {
        return Vec::new();
    }
    if !varlen || geoms.len() == 1 {
        return vec![(0..geoms.len()).collect()];
    }
    debug_assert!(
        wave_token_cap > 0,
        "wave_token_cap must be explicit-nonzero"
    );
    // 2026-09-25: (head geometry, Σ chunk_len, member indices)
    let mut waves: Vec<(WaveGeom, usize, Vec<usize>)> = Vec::new();
    for (i, g) in geoms.iter().enumerate() {
        let placed = waves.iter_mut().find(|(head, total, _)| {
            head.chunk_start == g.chunk_start
                && head.is_last == g.is_last
                && total + g.chunk_len <= wave_token_cap
        });
        match placed {
            Some((_, total, members)) => {
                *total += g.chunk_len;
                members.push(i);
            }
            None => waves.push((*g, g.chunk_len, vec![i])),
        }
    }
    waves.into_iter().map(|(_, _, members)| members).collect()
}

#[cfg(test)]
mod tests {
    use super::{WaveGeom, plan_prefill_waves};

    fn g(chunk_start: usize, chunk_len: usize, is_last: bool) -> WaveGeom {
        WaveGeom {
            chunk_start,
            chunk_len,
            is_last,
        }
    }

    #[test]
    fn flag_off_is_one_wave_with_every_stream_in_order() {
        // 2026-09-25: With the flag off, one prefill_batch_chunk call takes
        // all streams, whatever their geometry or total token count.
        let geoms = [g(0, 200, true), g(2048, 512, false), g(0, 4096, true)];
        assert_eq!(plan_prefill_waves(&geoms, false, 2048), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn empty_streams_no_waves() {
        assert!(plan_prefill_waves(&[], true, 2048).is_empty());
        assert!(plan_prefill_waves(&[], false, 2048).is_empty());
    }

    #[test]
    fn ragged_chunk0_wave_packs_up_to_the_budget() {
        // 2026-09-25: Eleven 200-token fresh prompts against a 2048-token
        // cap: the first ten fit (Σ = 2000), the eleventh opens wave 2.
        let geoms: Vec<WaveGeom> = (0..11).map(|_| g(0, 200, true)).collect();
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0], (0..10).collect::<Vec<_>>());
        assert_eq!(waves[1], vec![10]);
    }

    #[test]
    fn budget_cap_is_exact_not_off_by_one() {
        // 2026-09-25: 1024 + 1024 == cap exactly ⇒ same wave; +1 more opens
        // a new one.
        let geoms = [g(0, 1024, true), g(0, 1024, true), g(0, 1, true)];
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn mixed_geometry_splits_into_compatible_waves() {
        // 2026-09-25: `check_kernel_batched_eligible` refuses a batch whose
        // streams differ in `chunk_start` or `is_last`, so no wave may mix
        // them.
        let geoms = [
            g(0, 200, true),
            g(0, 2048, false),
            g(0, 300, true),
            g(2048, 512, false),
            g(0, 250, true),
        ];
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves, vec![vec![0, 2, 4], vec![1], vec![3]]);
        // 2026-09-25: Uniform (chunk_start, is_last) per wave, and Σ ≤ cap for
        // every multi-member wave.
        for wave in &waves {
            let head = geoms[wave[0]];
            let total: usize = wave.iter().map(|&i| geoms[i].chunk_len).sum();
            assert!(wave.len() == 1 || total <= 2048);
            for &i in wave {
                assert_eq!(geoms[i].chunk_start, head.chunk_start);
                assert_eq!(geoms[i].is_last, head.is_last);
            }
        }
    }

    #[test]
    fn oversized_stream_gets_a_singleton_wave() {
        // 2026-09-25: A chunk over the cap still gets a wave, and no sibling
        // joins it.
        let geoms = [g(0, 4096, true), g(0, 100, true)];
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves, vec![vec![0], vec![1]]);
    }

    #[test]
    fn every_stream_is_assigned_exactly_once() {
        let geoms: Vec<WaveGeom> = (0..37)
            .map(|i| g((i % 3) * 1024, 100 + i * 7, i % 2 == 0))
            .collect();
        let waves = plan_prefill_waves(&geoms, true, 1024);
        let mut seen = vec![0usize; geoms.len()];
        for wave in &waves {
            assert!(!wave.is_empty());
            for &i in wave {
                seen[i] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each stream in exactly one wave"
        );
        // 2026-09-25: Index order within each wave.
        for wave in &waves {
            assert!(wave.windows(2).all(|w| w[0] < w[1]));
        }
    }
}
