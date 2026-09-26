// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the `coverage` policy: which paths are on the boundary,
//! what exclusions may say, and which gates a changed path invalidates.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::REQUIRED_GATES;
use super::coverage::{
    self, BOUNDARY_FILES, Exclusion, GateCoverage, NOT_REQUIRED, PERF_PATHS, REQUIRED,
};

/// 2026-09-26: The repository root, two levels above this crate's manifest.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf()
}

/// 2026-09-26: `invalidated_by` keeps a gate when any one path invalidates it, so
/// adding a path never removes a gate.
#[test]
fn adding_a_changed_file_never_removes_a_required_gate() {
    let base = ["crates/bench/src/benchmarks/bfcl/report.rs"];
    let additions = [
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/model-layers/src/layers/ops/fp8_moe.rs",
        "Cargo.lock",
        "crates/bench/src/gate/check.rs",
        "some/unclassified/new/subsystem.rs",
    ];
    let before: Vec<_> = coverage::invalidated_by(base);
    assert!(!before.is_empty(), "the monotonicity oracle must execute");
    for extra in additions {
        let mut with = base.to_vec();
        with.push(extra);
        let after = coverage::invalidated_by(with.iter().copied());
        for gate in &before {
            assert!(
                after.contains(gate),
                "adding {extra} removed {gate} from the required set"
            );
        }
    }
}

#[test]
fn an_unclassified_path_on_the_boundary_invalidates_everything() {
    for path in [
        "crates/model-layers/src/some_new_module.rs",
        "kernels/gb10/brand_new_kernel.cu",
        "vendor/whatever",
        "rust-toolchain.toml",
    ] {
        let hit = coverage::invalidated_by([path]);
        assert_eq!(hit, REQUIRED_GATES, "{path} under-invalidated: {hit:?}");
    }
}

/// 2026-09-26: Boundary entries match by path component, so `Cargo.toml.orig` and
/// `crates2/x` are not under `Cargo.toml` and `crates`.
#[test]
fn lookalike_paths_do_not_match_the_boundary() {
    for path in [
        "Cargo.toml.orig",
        "Cargo.lockfile",
        "crates2/src/lib.rs",
        "kernels_old/x.cu",
        "vendored/thing.rs",
        "jinja-templates-old/x.jinja",
    ] {
        assert!(
            !coverage::on_boundary(path),
            "{path} must not count as a boundary path"
        );
        assert!(
            coverage::invalidated_by([path]).is_empty(),
            "{path} must invalidate nothing"
        );
    }
}

#[test]
fn real_boundary_paths_match() {
    for path in [
        "crates",
        "crates/model-layers/src/lib.rs",
        "Cargo.toml",
        "Cargo.lock",
        "kernels/gb10/common/x.cu",
        "jinja-templates/qwen3_5_moe.jinja",
        "rust-toolchain.toml",
    ] {
        assert!(coverage::on_boundary(path), "{path} should be on-boundary");
    }
}

/// 2026-09-26: Asserted through `invalidates` rather than on the tables:
/// `GATE_MACHINERY` excludes `crates/bench/src/gate`, which holds boundary files,
/// and `invalidates` checks `BOUNDARY_FILES` before any exclusion.
#[test]
fn no_gate_can_exempt_the_file_that_defines_the_rules() {
    for gate in REQUIRED.iter() {
        for boundary in BOUNDARY_FILES {
            assert!(
                coverage::invalidates(gate, boundary),
                "{} does not re-open when {boundary} changes",
                gate.id
            );
        }
    }
}

#[test]
fn editing_the_coverage_map_invalidates_every_gate() {
    for boundary in BOUNDARY_FILES {
        let hit = coverage::invalidated_by([boundary]);
        assert_eq!(
            hit, REQUIRED_GATES,
            "changing {boundary} must re-open everything, got {hit:?}"
        );
    }
}

#[test]
fn every_exclusion_states_why() {
    for gate in REQUIRED.iter() {
        for ex in gate.excludes {
            assert!(!ex.prefix.is_empty(), "{}: empty prefix", gate.id);
            assert!(
                ex.rationale.trim().len() > 20,
                "{} excludes {} with no real rationale: {:?}",
                gate.id,
                ex.prefix,
                ex.rationale
            );
        }
    }
}

#[test]
fn no_exclusion_is_a_dead_glob() {
    let root = repo_root();
    for gate in REQUIRED.iter() {
        for ex in gate.excludes {
            assert!(
                root.join(ex.prefix).exists(),
                "{} excludes {}, which does not exist in this repo",
                gate.id,
                ex.prefix
            );
        }
    }
}

#[test]
fn every_exclusion_is_actually_on_the_boundary() {
    for gate in REQUIRED.iter() {
        for ex in gate.excludes {
            assert!(
                coverage::on_boundary(ex.prefix),
                "{} excludes {}, which is not on the boundary — the rule does nothing",
                gate.id,
                ex.prefix
            );
        }
    }
}

#[path = "coverage_driver_tests.rs"]
mod coverage_driver_tests;

#[test]
fn required_gates_is_derived_from_the_coverage_table() {
    let ids: Vec<&str> = REQUIRED.iter().map(|g| g.id).collect();
    assert_eq!(REQUIRED_GATES.to_vec(), ids);
}

#[test]
fn every_registered_benchmark_is_either_required_or_explicitly_excused() {
    for descriptor in crate::registry::all() {
        let gated = REQUIRED.iter().any(|g| g.id == descriptor.id);
        let excused = NOT_REQUIRED.iter().any(|(id, _)| *id == descriptor.id);
        assert!(
            gated ^ excused,
            "{} is {}",
            descriptor.id,
            if gated {
                "both gated and excused"
            } else {
                "neither gated nor listed in NOT_REQUIRED with a reason"
            }
        );
    }
}

#[test]
fn every_excusal_names_a_real_benchmark_and_a_reason() {
    let mut seen = std::collections::BTreeSet::new();
    for (id, why) in NOT_REQUIRED {
        assert!(seen.insert(id), "{id} is excused more than once");
        assert!(
            crate::registry::find(id).is_some(),
            "{id} is excused but not registered"
        );
        assert!(
            why.trim().len() > 20,
            "{id} is excused without a real reason"
        );
    }
}

#[test]
fn a_driver_change_invalidates_only_its_own_gate() {
    let hit = coverage::invalidated_by(["crates/bench/src/benchmarks/bfcl/report.rs"]);
    // 2026-09-26: `KAT_EQUALITY_EXCLUDES` does not list the BFCL driver directory,
    // because the equality driver imports the BFCL draw (`kat_equality/driver.rs`).
    assert_eq!(
        hit,
        ["bfcl-subset", "bfcl-subset-echolp", "kat-equality-gate"]
    );
}

/// 2026-09-26: The concurrency driver is flat files, so its exclusions name exact
/// files; shared benchmark code and a lookalike neighbour still invalidate every gate.
#[test]
fn the_flat_concurrency_driver_invalidates_only_its_own_gate() {
    for path in [
        "crates/bench/src/benchmarks/concurrency.rs",
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "crates/bench/src/benchmarks/concurrency_report.rs",
    ] {
        assert_eq!(
            coverage::invalidated_by([path]),
            [
                "concurrency-sweep",
                "concurrency-sweep-dflash2",
                "concurrency-sweep-moe"
            ],
            "{path} must re-open exactly the instruments it implements — all THREE \
             concurrency gates run this driver, and a driver change that \
             re-opened only the no-drafter dense one would leave the speculative \
             and MoE ladders certified by records they never measured"
        );
    }

    for path in [
        "crates/bench/src/http.rs",
        "crates/bench/src/benchmarks/stats.rs",
        "crates/bench/src/benchmarks/transcript.rs",
        "crates/bench/src/benchmarks/concurrency.rs.bak",
        "crates/server/src/scheduler/mod.rs",
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/bench/src/gate/coverage.rs",
    ] {
        assert_eq!(
            coverage::invalidated_by([path]),
            REQUIRED_GATES,
            "nearby or shared path {path} escaped the fail-closed boundary"
        );
    }
}

/// 2026-09-26: These files fall under `GATE_MACHINERY`; `check.rs` is not listed
/// because it is one of the `BOUNDARY_FILES`.
#[test]
fn gate_bookkeeping_changes_cost_no_gpu_hours() {
    let hit = coverage::invalidated_by([
        "crates/bench/src/gate/record.rs",
        "crates/bench/src/gate/telemetry.rs",
        "crates/bench/src/gate/codeowners.rs",
    ]);
    assert!(
        hit.is_empty(),
        "record IO, telemetry rendering and CODEOWNERS parsing cannot move a \
         measurement; they should not re-open any gate, got {hit:?}"
    );
}

#[test]
fn documentation_only_changes_require_no_gate() {
    let hit = coverage::invalidated_by([
        "docs/adr/0011-ep-batched-decode-optimization.md",
        "README.md",
        "CONTRIBUTING.md",
    ]);
    assert!(hit.is_empty(), "{hit:?}");
}

#[test]
fn a_gate_with_no_exclusions_is_invalidated_by_any_boundary_path() {
    let bare = GateCoverage {
        id: "brand-new",
        excludes: &[],
    };
    assert!(coverage::invalidates(&bare, "crates/anything.rs"));
    assert!(!coverage::invalidates(&bare, "docs/anything.md"));
}

#[test]
fn an_exclusion_cannot_override_a_boundary_file() {
    let sneaky = GateCoverage {
        id: "sneaky",
        excludes: &[Exclusion {
            prefix: "crates",
            rationale: "a maximally broad exclusion, as an attacker would write it",
        }],
    };
    for boundary in BOUNDARY_FILES {
        assert!(
            coverage::invalidates(&sneaky, boundary),
            "{boundary} escaped via a blanket exclusion"
        );
    }
}

#[test]
fn the_boundary_has_no_duplicate_entries() {
    let mut seen = PERF_PATHS.to_vec();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!(before, seen.len(), "PERF_PATHS contains a duplicate");
}

/// 2026-09-26: `BENCH.toml` under `kernels/` holds thresholds and is compiled by
/// nothing, so `NON_COMPILED_KERNEL_FILES` keeps an edit to it from invalidating a gate.
#[test]
fn a_bench_toml_edit_invalidates_nothing() {
    for gate in REQUIRED.iter() {
        assert!(
            !coverage::invalidates(gate, "kernels/gb10/qwen3.6-27b/BENCH.toml"),
            "{}: a threshold edit re-opened the gate",
            gate.id
        );
    }
}

#[test]
fn the_bench_toml_exemption_does_not_leak_to_neighbours() {
    for gate in REQUIRED.iter() {
        for path in [
            "kernels/gb10/qwen3.6-27b/MODEL.toml",
            "kernels/gb10/qwen3.6-27b/nvfp4/KERNEL.toml",
            "kernels/gb10/qwen3.6-27b/nvfp4/BENCH.toml.cu",
            "kernels/gb10/qwen3.6-27b/BENCH.toml/inner.cu",
            "kernels/gb10/common/NOT-BENCH.toml",
        ] {
            assert!(
                coverage::invalidates(gate, path),
                "{}: {path} was exempted but is not a BENCH.toml",
                gate.id
            );
        }
    }
}

#[test]
fn the_bench_toml_exemption_is_scoped_to_the_kernel_tree() {
    let gate = REQUIRED
        .iter()
        .find(|g| {
            !g.excludes
                .iter()
                .any(|e| e.prefix.starts_with("crates/model-layers"))
        })
        .expect("a gate that does not exclude metrale-model-layers");
    assert!(
        coverage::invalidates(gate, "crates/model-layers/BENCH.toml"),
        "the exemption must not apply outside kernels/"
    );
}

/// 2026-09-26: Every other gate excludes the equality driver directory, and the
/// equality gate itself still re-opens.
#[test]
fn a_change_to_the_equality_driver_reopens_that_gate_and_no_other() {
    let hit = coverage::invalidated_by(["crates/bench/src/benchmarks/kat_equality/compare.rs"]);
    assert_eq!(
        hit,
        ["kat-equality-gate"],
        "editing the equality detector must cost its own gate and nothing else"
    );
}
