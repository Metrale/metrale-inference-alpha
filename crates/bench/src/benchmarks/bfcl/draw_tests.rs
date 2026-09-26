// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that the draw rules and the parameter defaults produce the
//! pinned draw sizes from the BFCL v4 subset counts.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: The BFCL v4 single-turn per-subset counts
/// (`reference_subset_totals`), from which the golden config gives n = 995
/// only because the category selection excludes `live_relevance` (16).
fn real_totals() -> BTreeMap<String, usize> {
    super::reference_subset_totals()
}

#[test]
fn the_golden_draw_is_exactly_995() {
    let p = plan(&DrawSpec::golden(), &real_totals());
    assert_eq!(
        total(&p),
        995,
        "golden draw must be the MLPerf n=995: {p:?}"
    );
}

/// 2026-09-26: The echolp draw gives n = 1004 from the same totals; its
/// committed baseline is for that draw.
#[test]
fn the_echolp_draw_is_exactly_1004() {
    let p = plan(&DrawSpec::echolp(), &real_totals());
    assert_eq!(
        total(&p),
        1004,
        "echolp draw must be n=1004 or its baseline does not apply: {p:?}"
    );
}

/// 2026-09-26: The two draws differ in composition, not merely in size, which
/// is why each has its own baseline.
#[test]
fn golden_and_echolp_are_different_draws() {
    let g: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    let e: BTreeMap<String, usize> = plan(&DrawSpec::echolp(), &real_totals())
        .into_iter()
        .collect();
    assert_ne!(g, e, "the draws must not collapse onto each other");
    // 2026-09-26: live is weighted 23% vs 10%, so echolp takes more
    // live_multiple.
    assert!(
        e["live_multiple"] > g["live_multiple"],
        "echolp must weight `live` more heavily: {} vs {}",
        e["live_multiple"],
        g["live_multiple"]
    );
}

#[test]
fn the_golden_per_subset_counts_match_the_reference_rule() {
    let p: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    let expected: BTreeMap<String, usize> = [
        // 2026-09-26: hallucination @10%: int(240*.10)=24, int(884*.10)=88.
        ("irrelevance", 24),
        ("live_irrelevance", 88),
        // 2026-09-26: live @10%: int(1053*.10)=105, int(258*.10)=25.
        ("live_multiple", 105),
        // 2026-09-26: The floor of 25 takes these two whole rather than 1 and 2.
        ("live_parallel", 16),
        ("live_parallel_multiple", 24),
        ("live_simple", 25),
        // 2026-09-26: non_live @62%.
        ("multiple", 124),
        ("parallel", 124),
        ("parallel_multiple", 124),
        ("simple_java", 62),
        ("simple_javascript", 31),
        ("simple_python", 248),
    ]
    .into_iter()
    .map(|(subset, count)| (subset.to_string(), count))
    .collect();
    assert_eq!(p, expected);
}

#[test]
fn live_relevance_is_excluded_by_the_category_selection() {
    let p: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    assert!(
        !p.contains_key("live_relevance"),
        "live_relevance belongs to no scored category; including it makes n=1011, not 995"
    );
    assert_eq!(category_of("live_relevance"), None);
}

#[test]
fn the_full_draw_keeps_the_golden_composition() {
    let p = plan(&DrawSpec::full(), &real_totals());
    // 2026-09-26: Every sample of the three scored categories: 3641 total
    // minus the 16 uncategorised live_relevance rows.
    assert_eq!(total(&p), 3625);
    assert!(!p.iter().any(|(s, _)| s == "live_relevance"));
}

#[test]
fn an_empty_category_selection_takes_everything_including_live_relevance() {
    let spec = DrawSpec {
        categories: Vec::new(),
        category_pct: BTreeMap::new(),
        subset_floor: None,
    };
    assert_eq!(total(&plan(&spec, &real_totals())), 3641);
}

#[test]
fn a_subset_never_collapses_to_zero() {
    let spec = DrawSpec {
        categories: vec!["non_live".into()],
        category_pct: [("non_live".to_string(), 0.5)].into_iter().collect(),
        subset_floor: None,
    };
    // 2026-09-26: int(50 * 0.005) = 0, raised to 1 by `max(1)`.
    assert_eq!(spec.take_count("simple_javascript", 50), 1);
}

#[test]
fn the_floor_beats_the_percentage() {
    let spec = DrawSpec::golden();
    assert_eq!(spec.take_count("live_parallel", 16), 16);
    // 2026-09-26: Over the floor, the percentage applies again.
    assert_eq!(spec.take_count("live_parallel", 26), 2);
}

#[test]
fn every_subset_maps_to_a_category_except_live_relevance() {
    let uncategorised: Vec<&str> = SINGLE_TURN_SUBSETS
        .iter()
        .copied()
        .filter(|s| category_of(s).is_none())
        .collect();
    assert_eq!(uncategorised, vec!["live_relevance"]);
}

/// 2026-09-26: The parameter defaults reproduce each pinned draw, not just the
/// `DrawSpec` constants: `configure` rebuilds the spec from the parameters. A
/// `subset_floor` default of 0 on echolp, for example, would take
/// `live_parallel` (16 rows) and `live_parallel_multiple` (24) by percentage
/// and draw n=972, not 1004.
#[test]
fn the_parameter_defaults_reproduce_each_pinned_draw() {
    use crate::benchmark::Benchmark as _;
    use crate::benchmarks::bfcl::{Bfcl, Variant};

    for variant in [Variant::Subset, Variant::SubsetEcholp] {
        // 2026-09-26: Read from the variant, not restated: `expected_samples`
        // is what the run warns against and what the committed baselines are
        // checked against.
        let want = variant
            .expected_samples()
            .expect("a gated variant is pinned");
        let mut b = Bfcl::new(variant);
        let defaults = crate::params::ParamValues::defaults(&b.parameters());
        b.configure(&defaults).expect("defaults must validate");
        let n = total(&plan(&b.spec, &real_totals()));
        assert_eq!(
            n, want,
            "{variant:?}: a DEFAULT run draws n={n}, but this draw is pinned at {want}. \
             Its baseline does not apply to n={n}."
        );
    }
}
