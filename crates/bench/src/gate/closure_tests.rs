// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the closure excuse. It exists to skip re-runs, so most tests cover the
//! cases where it must refuse.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "metrale-gate-closure-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// 2026-09-26: Two models on one hardware. `modelA` shadows `common/shared.cu` with its own copy
/// (declared in its `KERNEL.toml` `[shadow]` table, which the resolver requires); `modelB`
/// inherits the common one.
fn fixture(name: &str) -> std::path::PathBuf {
    let root = tmp(name);
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("common")).unwrap();
    std::fs::create_dir_all(hw.join("modelA/nvfp4")).unwrap();
    std::fs::create_dir_all(hw.join("modelB/nvfp4")).unwrap();
    std::fs::write(
        hw.join("HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("modelA/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("modelB/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(
        hw.join("common/shared.cu"),
        "__global__ void s() { int t = 64; }\n",
    )
    .unwrap();
    std::fs::write(
        hw.join("modelA/nvfp4/shared.cu"),
        "__global__ void s() { /*A*/ }\n",
    )
    .unwrap();
    std::fs::write(
        hw.join("modelA/nvfp4/KERNEL.toml"),
        "[shadow]\nshared = \"modelA's own shared kernel\"\n",
    )
    .unwrap();
    root
}

fn attest_all(root: &std::path::Path) -> Attestation {
    attest(root, "sm_121a", "nvcc 13.0.2", &BTreeMap::new())
}

const SHARED: &str = "kernels/gb10/common/shared.cu";

#[test]
fn an_untouched_tree_excuses_a_kernel_path() {
    let root = fixture("untouched");
    let a = attest_all(&root);
    assert!(excuses(&root, &[SHARED.to_string()], &a));
}

/// 2026-09-26: `modelA`'s shadow does not include the common file, so editing the common copy
/// changes only `modelB`.
#[test]
fn a_shared_edit_re_opens_only_the_targets_that_compile_it() {
    let root = fixture("shared-edit");
    let a = attest_all(&root);
    std::fs::write(root.join(SHARED), "__global__ void s() { int t = 128; }\n").unwrap();

    let changed = changed_targets(&root, &[SHARED.to_string()], &a);
    assert_eq!(changed, vec!["gb10/modelB/nvfp4"], "only the inheritor");
    assert!(
        !excuses(&root, &[SHARED.to_string()], &a),
        "one affected target changed, so the gate re-opens"
    );
}

/// 2026-09-26: A shadow that includes the common file changes with it.
#[test]
fn a_shadow_that_includes_the_common_file_is_not_excused() {
    let root = fixture("include-shadow");
    std::fs::write(
        root.join("kernels/gb10/modelA/nvfp4/shared.cu"),
        "#include \"../../common/shared.cu\"\n",
    )
    .unwrap();
    let a = attest_all(&root);
    std::fs::write(root.join(SHARED), "__global__ void s() { int t = 128; }\n").unwrap();

    let mut changed = changed_targets(&root, &[SHARED.to_string()], &a);
    changed.sort();
    assert_eq!(
        changed,
        vec!["gb10/modelA/nvfp4", "gb10/modelB/nvfp4"],
        "the including shadow must re-open as well"
    );
}

/// 2026-09-26: A header is reached only through the include walk.
#[test]
fn editing_an_included_header_is_not_excused() {
    let root = fixture("header");
    std::fs::write(root.join("kernels/gb10/common/tune.cuh"), "#define BR 64\n").unwrap();
    std::fs::write(
        root.join(SHARED),
        "#include \"tune.cuh\"\n__global__ void s() {}\n",
    )
    .unwrap();
    let a = attest_all(&root);

    std::fs::write(
        root.join("kernels/gb10/common/tune.cuh"),
        "#define BR 128\n",
    )
    .unwrap();
    let header = "kernels/gb10/common/tune.cuh".to_string();
    assert_eq!(
        changed_targets(&root, std::slice::from_ref(&header), &a),
        ["gb10/modelB/nvfp4"],
        "the inheriting target compiles the header through shared.cu"
    );
    assert!(
        !excuses(&root, &[header], &a),
        "a header edit must re-open the targets that include it"
    );
}

/// 2026-09-26: `MODEL.toml` is a closure input of its own model only.
#[test]
fn a_model_toml_edit_re_opens_only_that_model() {
    let root = fixture("model-toml");
    let a = attest_all(&root);
    std::fs::write(
        root.join("kernels/gb10/modelA/MODEL.toml"),
        "[behavior]\nthinking_default = true\n",
    )
    .unwrap();
    let path = "kernels/gb10/modelA/MODEL.toml".to_string();
    assert_eq!(
        changed_targets(&root, std::slice::from_ref(&path), &a),
        vec!["gb10/modelA/nvfp4"]
    );
    assert!(!excuses(&root, &[path], &a));
}

#[test]
fn a_record_with_no_attestation_excuses_nothing() {
    let root = fixture("no-attestation");
    assert!(!excuses(&root, &[SHARED.to_string()], &Attestation::new()));
}

/// 2026-09-26: An affected target the record does not attest (such as a model added later) is
/// not excused.
#[test]
fn a_target_missing_from_the_attestation_is_not_excused() {
    let root = fixture("new-model");
    let mut a = attest_all(&root);
    a.remove("gb10/modelB/nvfp4");
    assert!(
        !excuses(&root, &[SHARED.to_string()], &a),
        "an unmentioned affected target must not be excused"
    );
}

/// 2026-09-26: One path outside `kernels/` vetoes the whole set.
#[test]
fn a_non_kernel_path_is_never_excused() {
    let root = fixture("host-path");
    let a = attest_all(&root);
    assert!(!excuses(
        &root,
        &["crates/model-layers/src/lib.rs".to_string()],
        &a
    ));
    assert!(
        !excuses(&root, &[SHARED.to_string(), "Cargo.lock".to_string()], &a),
        "one out-of-scope path must veto the whole set"
    );
}

#[test]
fn a_kernel_path_mapping_to_no_target_is_not_excused() {
    let root = fixture("unknown-target");
    let a = attest_all(&root);
    assert!(!excuses(
        &root,
        &["kernels/newhw/common/x.cu".to_string()],
        &a
    ));
}

/// 2026-09-26: `metrale_closure` hashes an unresolvable include rather than failing, so deleting
/// an included header still changes the digest.
#[test]
fn a_vanished_header_re_opens_the_gate() {
    let root = fixture("vanished-header");
    std::fs::write(root.join("kernels/gb10/common/tune.cuh"), "#define BR 64\n").unwrap();
    std::fs::write(
        root.join(SHARED),
        "#include \"tune.cuh\"\n__global__ void s() {}\n",
    )
    .unwrap();
    let a = attest_all(&root);

    std::fs::remove_file(root.join("kernels/gb10/common/tune.cuh")).unwrap();
    assert!(!excuses(&root, &[SHARED.to_string()], &a));
    assert_eq!(
        changed_targets(&root, &[SHARED.to_string()], &a),
        ["gb10/modelB/nvfp4"],
        "the target that included it is REPORTED, not silently dropped"
    );
}

/// 2026-09-26: After the hardware's vendor changes to one the resolver does not know, nothing
/// is excused.
#[test]
fn a_target_whose_sources_do_not_resolve_is_not_excused() {
    let root = fixture("no-sources");
    let a = attest_all(&root);
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"quantum-abacus\"\n",
    )
    .unwrap();
    assert!(!excuses(&root, &[SHARED.to_string()], &a));
}

/// 2026-09-26: An empty path list excuses nothing, so an empty call cannot read as a pass.
#[test]
fn an_empty_path_list_excuses_nothing() {
    let root = fixture("empty-paths");
    assert!(!excuses(&root, &[], &attest_all(&root)));
}

/// 2026-09-26: The check recomputes under the recorded arch and compiler, and those inputs are
/// inside the digest.
#[test]
fn the_check_reuses_the_recorded_inputs_not_the_current_environment() {
    let root = fixture("inputs");
    let exotic = attest(&root, "sm_999z", "nvcc from the future", &BTreeMap::new());
    assert!(excuses(&root, &[SHARED.to_string()], &exotic));

    let native = attest(&root, "sm_121a", "nvcc 13.0.2", &BTreeMap::new());
    assert_ne!(
        exotic["gb10/modelB/nvfp4"].hash, native["gb10/modelB/nvfp4"].hash,
        "arch and compiler must be inside the digest"
    );
}

/// 2026-09-26: Flags are stored and reused per target.
#[test]
fn per_target_flags_are_carried_per_target() {
    let root = fixture("flags");
    let flags = BTreeMap::from([("gb10/modelA/nvfp4".to_string(), vec!["-O3".to_string()])]);
    let a = attest(&root, "sm_121a", "nvcc 13.0.2", &flags);
    assert_eq!(a["gb10/modelA/nvfp4"].flags, vec!["-O3"]);
    assert!(a["gb10/modelB/nvfp4"].flags.is_empty());
    assert!(
        excuses(&root, &[SHARED.to_string()], &a),
        "recomputation must use each target's own flags"
    );
}
