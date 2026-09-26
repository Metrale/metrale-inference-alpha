// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for three gate rules: a record counts only for the gate it names,
//! every gate source file is classified as boundary or machinery, and a BENCH.toml
//! `noise` allowance is bounded.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::tests::{tempdir, *};
use super::*;

/// 2026-09-26: A warm TTFT record copied into the cold gate's directory does not count:
/// `check_one` skips a record whose `benchmark_id` names another gate, and says so.
#[test]
fn a_record_from_another_gate_does_not_satisfy_this_one() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }

    let src = gate_dir(root, "ttft-warm-gate");
    let dst = gate_dir(root, "ttft-cold-gate");
    plant(root, "ttft-warm-gate", "abc1234567", 1_785_891_382, "PASS");
    let planted = std::fs::read_dir(&src)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::copy(&planted, dst.join(planted.file_name().unwrap())).unwrap();

    assert_eq!(std::fs::read_dir(&dst).unwrap().count(), 1);

    let status = check_gates(root, "abc1234567");
    match &status["ttft-cold-gate"] {
        GateStatus::Missing(reason) => assert_eq!(
            reason,
            "latest record belongs to ttft-warm-gate, not ttft-cold-gate \
             (2026-08-05-abc1234567.json)"
        ),
        other => panic!("a warm record must not satisfy the cold gate: {other:?}"),
    }
}

/// 2026-09-26: `GATE_MACHINERY` excludes `crates/bench/src/gate`, so these verdict
/// files re-open every gate only because they are in `BOUNDARY_FILES`.
#[test]
fn verdict_logic_is_inside_the_boundary() {
    for f in [
        "crates/bench/src/gate/coverage.rs",
        "crates/bench/src/gate/check.rs",
        "crates/bench/src/gate/scoring.rs",
        "crates/bench/src/gate/closure.rs",
        "crates/bench/src/gate/taxon.rs",
        "crates/bench/src/gate/bench.rs",
    ] {
        let hit = super::coverage::invalidated_by([f]);
        assert_eq!(
            hit.len(),
            super::coverage::REQUIRED.len(),
            "{f} decides a verdict and must re-open every gate, got {hit:?}"
        );
    }
}

/// 2026-09-26: Every non-test source under `src/gate` is in exactly one of
/// `BOUNDARY_FILES` and `GATE_MACHINERY_FILES`, so a new file fails here until it is
/// classified, and a classified file that is gone fails too.
#[test]
fn gate_sources_are_all_classified() {
    let gate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gate");
    let mut on_disk = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&gate_dir).expect("gate dir listable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        // 2026-09-26: Test sources are skipped by name, as in
        // `every_verdict_symbol_is_defined_inside_the_boundary`.
        if name.ends_with("_tests.rs") || name == "tests.rs" {
            continue;
        }
        on_disk.insert(format!("crates/bench/src/gate/{name}"));
    }

    let boundary: std::collections::BTreeSet<&str> = super::coverage::BOUNDARY_FILES
        .iter()
        .copied()
        .filter(|p| p.starts_with("crates/bench/src/gate/"))
        .collect();
    let machinery: std::collections::BTreeSet<&str> = super::coverage::gate_machinery_files()
        .iter()
        .copied()
        .collect();

    let both: Vec<_> = boundary.intersection(&machinery).collect();
    assert!(
        both.is_empty(),
        "classified as BOTH boundary and machinery: {both:?} — a file decides a \
         verdict or it does not"
    );

    let classified: std::collections::BTreeSet<String> = boundary
        .iter()
        .chain(machinery.iter())
        .map(|s| s.to_string())
        .collect();

    let unclassified: Vec<_> = on_disk.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "unclassified gate source(s): {unclassified:?}\n\
         Every file under src/gate must be in BOUNDARY_FILES (it decides a \
         verdict, so editing it re-opens every gate) or in \
         GATE_MACHINERY_FILES (it does not, and someone is on record saying \
         so). `agreement.rs` reached main unclassified and could have judged \
         its own PR by its own new rule."
    );

    let stale: Vec<_> = classified.difference(&on_disk).collect();
    assert!(
        stale.is_empty(),
        "classified but not on disk: {stale:?} — a rename or delete left the \
         classification pointing at nothing, which silently shrinks the boundary"
    );
}

/// 2026-09-26: The file defining each listed verdict function re-opens every gate, so
/// defining one in a file outside `BOUNDARY_FILES` fails here.
#[test]
fn every_verdict_symbol_is_defined_inside_the_boundary() {
    let gate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gate");
    let verdict_fns = [
        "fn record_covers",
        "fn invalidating_paths",
        "fn check_record",
        "fn compare",
        "fn excuses",
        "fn changed_targets",
        "fn baseline_for",
    ];
    for symbol in verdict_fns {
        let mut defined_in = Vec::new();
        for entry in std::fs::read_dir(&gate_dir).expect("gate dir listable") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("gate source readable");
            let declares_symbol = src.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with(symbol)
                    || line.starts_with(&format!("pub {symbol}"))
                    || line.starts_with(&format!("pub(crate) {symbol}"))
            });
            if declares_symbol {
                let rel = format!(
                    "crates/bench/src/gate/{}",
                    path.file_name().unwrap().to_str().unwrap()
                );
                if !rel.ends_with("_tests.rs") && !rel.ends_with("/tests.rs") {
                    defined_in.push(rel);
                }
            }
        }
        assert!(
            !defined_in.is_empty(),
            "{symbol} not found anywhere under src/gate — if it was renamed, \
             rename it here too; the boundary must keep tracking it"
        );
        for rel in defined_in {
            let hit = super::coverage::invalidated_by([rel.as_str()]);
            assert_eq!(
                hit.len(),
                super::coverage::REQUIRED.len(),
                "{rel} contains `{symbol}` but is not in BOUNDARY_FILES — a \
                 verdict function moved out of the boundary (the PR #420 hole, \
                 reintroduced by a file split)"
            );
        }
    }
}

fn bench_toml(root: &std::path::Path, metrics: &str) -> std::path::PathBuf {
    let dir = root.join("kernels/gb10/qwen3.6-27b/nvfp4");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\narch = \"sm_121f\"\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(root.join("kernels/gb10/qwen3.6-27b/MODEL.toml"), "").unwrap();
    let p = root.join("kernels/gb10/qwen3.6-27b/BENCH.toml");
    std::fs::write(
        &p,
        format!(
            "[[benchmarks]]\ngate = \"bfcl-subset\"\nquant = \"nvfp4\"\n\
             checkpoint = \"x/y\"\nrecipe = \"a/b\"\nstatus = \"measured\"\n\
             default = true\n{metrics}"
        ),
    )
    .unwrap();
    p
}

/// 2026-09-26: A `BENCH.toml` edit invalidates no gate, so `load_all` refuses a `noise`
/// above 5% of the bound: it would move the pass line with no visible bound change.
#[test]
fn an_absurd_noise_allowance_is_refused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = 1000.0\n",
    );
    let err = super::bench::load_all(root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / x/y metric overall_accuracy: noise 1000 exceeds 5% of the bound \
             (87.44) — that is a threshold change wearing a measurement-noise label. Move the \
             bound instead, so the ratchet is visible in review.",
            root.join("kernels/gb10/qwen3.6-27b/BENCH.toml").display()
        )
    );
}

#[test]
fn the_real_noise_values_still_load() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = 0.4\n",
    );
    let loaded =
        super::bench::load_all(root).expect("0.4 on an 87.44 floor is real measurement noise");
    assert_eq!(loaded.len(), 1);
    assert_eq!(
        loaded[0].1.metrics.as_ref().unwrap()["overall_accuracy"].noise,
        Some(0.4)
    );
}

/// 2026-09-26: `compare` applies noise to both sides of a two-sided bound, so noise on
/// an exact pin (`min == max`) would let other values pass.
#[test]
fn noise_on_an_exact_pin_is_refused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.samples]\nmin = 995.0\nmax = 995.0\nnoise = 1.0\n",
    );
    let err = super::bench::load_all(root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / x/y metric samples is an EXACT pin (min == max == Some(995.0)) \
             and carries noise 1. Noise on a pin disables it — and a pin is used for things \
             like the BFCL draw size, where a changed draw is undetectable after the fact.",
            root.join("kernels/gb10/qwen3.6-27b/BENCH.toml").display()
        )
    );
}

#[test]
fn negative_and_non_finite_noise_are_refused() {
    for (literal, rendered) in [("-5.0", "-5"), ("nan", "NaN"), ("inf", "inf")] {
        let dir = tempdir::Dir::new();
        let root = dir.path();
        bench_toml(
            root,
            &format!("[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = {literal}\n"),
        );
        let err = super::bench::load_all(root).unwrap_err().to_string();
        assert_eq!(
            err,
            format!(
                "{}: bfcl-subset / x/y metric overall_accuracy: noise must be finite and \
                 non-negative, got {rendered}",
                root.join("kernels/gb10/qwen3.6-27b/BENCH.toml").display()
            )
        );
    }
}
