// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the closure hash, on fixture trees under the temp dir.
//! The first two are cases a hash of the resolved file set would miss.
//!
//! Owner: metrale-closure.
//! Invariants: none beyond the types.

use super::*;

fn tmp() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "metrale-closure-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("common")).unwrap();
    std::fs::create_dir_all(dir.join("model/nvfp4")).unwrap();
    dir
}

fn write(p: &Path, s: &str) {
    std::fs::write(p, s).unwrap();
}

fn inputs(sources: Vec<PathBuf>) -> ClosureInputs {
    ClosureInputs {
        sources,
        configs: Vec::new(),
        flags: vec!["--fmad=false".into()],
        arch: "sm_121a".into(),
        compiler: "nvcc 13.0.2".into(),
    }
}

/// 2026-09-26: A leaf file that includes the common file it shadows changes
/// hash when the common file changes. The qwen3.6-27b nvfp4
/// `attn_prefill_paged_indirect.cu` has this shape.
#[test]
fn editing_a_common_file_a_shadow_includes_changes_the_hash() {
    let d = tmp();
    let common = d.join("common/prefill.cu");
    let shadow = d.join("model/nvfp4/prefill.cu");
    write(&common, "__global__ void k() { int tile = 64; }\n");
    write(
        &shadow,
        "#include \"../../common/prefill.cu\"\n// model tweak\n",
    );

    let before = hash(&d, &inputs(vec![shadow.clone()])).unwrap();
    write(&common, "__global__ void k() { int tile = 128; }\n");
    let after = hash(&d, &inputs(vec![shadow.clone()])).unwrap();

    assert_ne!(
        before, after,
        "a shadow that INCLUDES the common file must not look immune to \
         a change in it — this is the fail-open a file-set hash produces"
    );
}

/// 2026-09-26: An included header, which is in no resolved source list,
/// changes the hash when it changes.
#[test]
fn editing_an_included_header_changes_the_hash() {
    let d = tmp();
    let header = d.join("common/compute.cuh");
    let src = d.join("common/attn.cu");
    write(&header, "#define BR64 64\n");
    write(&src, "#include \"compute.cuh\"\n__global__ void a() {}\n");

    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();
    write(&header, "#define BR64 128\n");
    let after = hash(&d, &inputs(vec![src.clone()])).unwrap();

    assert_ne!(before, after, "a header edit must change the hash");
}

/// 2026-09-26: A leaf file that does not include the common file keeps its
/// hash when the common file changes.
#[test]
fn a_true_shadow_is_insulated_from_the_common_file() {
    let d = tmp();
    let common = d.join("common/gemm.cu");
    let shadow = d.join("model/nvfp4/gemm.cu");
    write(&common, "__global__ void g() { /* generic */ }\n");
    write(
        &shadow,
        "__global__ void g() { /* hand-tuned, standalone */ }\n",
    );

    let before = hash(&d, &inputs(vec![shadow.clone()])).unwrap();
    write(&common, "__global__ void g() { /* generic, edited */ }\n");
    let after = hash(&d, &inputs(vec![shadow.clone()])).unwrap();

    assert_eq!(
        before, after,
        "a shadow that does not include the common file is genuinely unaffected"
    );
}

#[test]
fn transitive_includes_are_followed() {
    let d = tmp();
    write(&d.join("common/deep.cuh"), "#define X 1\n");
    write(&d.join("common/mid.cuh"), "#include \"deep.cuh\"\n");
    let src = d.join("common/top.cu");
    write(&src, "#include \"mid.cuh\"\n");

    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();
    write(&d.join("common/deep.cuh"), "#define X 2\n");
    assert_ne!(
        before,
        hash(&d, &inputs(vec![src])).unwrap(),
        "two levels deep"
    );
}

/// 2026-09-26: An include naming no file is recorded, relative to the root,
/// and the hash still succeeds.
#[test]
fn an_unresolvable_include_is_recorded_rather_than_fatal() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(
        &src,
        "#include \"nowhere/absent.cuh\"\n__global__ void k() {}\n",
    );
    let closure = hash_with_report(&d, &inputs(vec![src])).expect("must not fail");
    assert_eq!(closure.unresolved.len(), 1, "{:?}", closure.unresolved);
    let entry = closure.unresolved.iter().next().unwrap();
    assert!(entry.contains("nowhere/absent.cuh"), "{entry}");
    assert!(
        entry.starts_with("common/x.cu"),
        "the report must be repo-relative, or two checkouts disagree: {entry}"
    );
}

/// 2026-09-26: A missing include that appears moves the digest with the
/// including file untouched, and its content is then hashed.
#[test]
fn resolving_a_previously_missing_include_changes_the_hash() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&src, "#include \"later.cuh\"\n__global__ void k() {}\n");
    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();

    write(&d.join("common/later.cuh"), "#define L 1\n");
    let after = hash(&d, &inputs(vec![src.clone()])).unwrap();
    assert_ne!(before, after, "the file appearing must move the hash");

    write(&d.join("common/later.cuh"), "#define L 2\n");
    assert_ne!(after, hash(&d, &inputs(vec![src])).unwrap());
}

/// 2026-09-26: Two files naming the same missing header give two entries.
#[test]
fn unresolved_entries_are_keyed_by_the_including_file() {
    let d = tmp();
    let a = d.join("common/a.cu");
    let b = d.join("common/b.cu");
    write(&a, "#include \"gone.cuh\"\n");
    write(&b, "#include \"gone.cuh\"\n");
    let closure = hash_with_report(&d, &inputs(vec![a, b])).unwrap();
    assert_eq!(
        closure.unresolved,
        BTreeSet::from([
            "common/a.cu -> gone.cuh".to_string(),
            "common/b.cu -> gone.cuh".to_string(),
        ])
    );
}

/// 2026-09-26: An include inside `#if` is walked whether or not the branch is
/// taken.
#[test]
fn includes_inside_untaken_conditionals_are_still_walked() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&d.join("common/dead.cuh"), "#define D 1\n");
    write(
        &src,
        "#if defined(NEVER)\n#include \"dead.cuh\"\n#endif\n__global__ void k() {}\n",
    );
    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();
    write(&d.join("common/dead.cuh"), "#define D 2\n");
    assert_ne!(
        before,
        hash(&d, &inputs(vec![src])).unwrap(),
        "an untaken branch is over-included, which is the safe direction"
    );
}

/// 2026-09-26: An angle-bracket include is neither followed nor reported,
/// even when a same-named file sits beside the source.
#[test]
fn angle_bracket_includes_are_not_followed() {
    let d = tmp();
    let src = d.join("common/x.cu");
    let toolchain_header = d.join("common/cuda_fp16.h");
    write(&src, "#include <cuda_fp16.h>\n__global__ void k() {}\n");
    write(&toolchain_header, "#define TOOLCHAIN_VALUE 1\n");

    let before = hash_with_report(&d, &inputs(vec![src.clone()])).unwrap();
    assert!(before.unresolved.is_empty(), "{:?}", before.unresolved);

    write(&toolchain_header, "#define TOOLCHAIN_VALUE 2\n");
    let after = hash_with_report(&d, &inputs(vec![src])).unwrap();
    assert!(after.unresolved.is_empty(), "{:?}", after.unresolved);
    assert_eq!(
        before.digest, after.digest,
        "angle-bracket headers are represented by compiler provenance, not local content"
    );
}

/// 2026-09-26: A `//`-commented include is neither followed nor reported.
#[test]
fn commented_out_includes_are_ignored() {
    let d = tmp();
    let src = d.join("common/x.cu");
    let commented_header = d.join("common/commented.cuh");
    write(
        &src,
        "// #include \"commented.cuh\"\n__global__ void k() {}\n",
    );
    write(&commented_header, "#define COMMENTED_VALUE 1\n");

    let before = hash_with_report(&d, &inputs(vec![src.clone()])).unwrap();
    assert!(before.unresolved.is_empty(), "{:?}", before.unresolved);

    write(&commented_header, "#define COMMENTED_VALUE 2\n");
    let after = hash_with_report(&d, &inputs(vec![src])).unwrap();
    assert!(after.unresolved.is_empty(), "{:?}", after.unresolved);
    assert_eq!(
        before.digest, after.digest,
        "a header named only by a commented include must stay outside the closure"
    );
}

/// 2026-09-26: Mutually including headers terminate.
#[test]
fn include_cycles_terminate() {
    let d = tmp();
    write(&d.join("common/a.cuh"), "#include \"b.cuh\"\n");
    write(&d.join("common/b.cuh"), "#include \"a.cuh\"\n");
    let src = d.join("common/x.cu");
    write(&src, "#include \"a.cuh\"\n");
    assert!(hash(&d, &inputs(vec![src])).is_ok());
}

/// 2026-09-26: Flags (and their order), arch, compiler and a config's presence
/// and content each move the hash on their own.
#[test]
fn non_source_inputs_each_move_the_hash() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&src, "__global__ void k() {}\n");
    let base = hash(&d, &inputs(vec![src.clone()])).unwrap();

    let mut flags = inputs(vec![src.clone()]);
    flags.flags.push("-O3".into());
    assert_ne!(base, hash(&d, &flags).unwrap(), "nvcc flags");

    let mut define_then_undefine = inputs(vec![src.clone()]);
    define_then_undefine.flags = vec!["-DSELECTED=1".into(), "-USELECTED".into()];
    let mut undefine_then_define = inputs(vec![src.clone()]);
    undefine_then_define.flags = vec!["-USELECTED".into(), "-DSELECTED=1".into()];
    assert_ne!(
        hash(&d, &define_then_undefine).unwrap(),
        hash(&d, &undefine_then_define).unwrap(),
        "nvcc flag order"
    );

    let mut arch = inputs(vec![src.clone()]);
    arch.arch = "sm_120a".into();
    assert_ne!(base, hash(&d, &arch).unwrap(), "arch");

    let mut cc = inputs(vec![src.clone()]);
    cc.compiler = "nvcc 12.9.0".into();
    assert_ne!(base, hash(&d, &cc).unwrap(), "compiler version");

    let mut cfg = inputs(vec![src.clone()]);
    let toml = d.join("model/MODEL.toml");
    write(&toml, "[behavior]\nthinking_default = true\n");
    cfg.configs.push(toml.clone());
    let with_cfg = hash(&d, &cfg).unwrap();
    assert_ne!(base, with_cfg, "config presence");
    write(&toml, "[behavior]\nthinking_default = false\n");
    assert_ne!(with_cfg, hash(&d, &cfg).unwrap(), "config CONTENT");
}

/// 2026-09-26: Identical bytes under a different stem, or under the same stem
/// in another directory, hash differently.
#[test]
fn identical_content_under_a_different_path_hashes_differently() {
    let d = tmp();
    let a = d.join("common/one.cu");
    let b = d.join("common/two.cu");
    write(&a, "__global__ void k() {}\n");
    write(&b, "__global__ void k() {}\n");
    assert_ne!(
        hash(&d, &inputs(vec![a])).unwrap(),
        hash(&d, &inputs(vec![b])).unwrap(),
        "different stems"
    );

    let common = d.join("common/same.cu");
    let model = d.join("model/nvfp4/same.cu");
    write(&common, "__global__ void same() {}\n");
    write(&model, "__global__ void same() {}\n");
    assert_ne!(
        hash(&d, &inputs(vec![common])).unwrap(),
        hash(&d, &inputs(vec![model])).unwrap(),
        "same stem in different source directories"
    );
}

/// 2026-09-26: The digest does not depend on the order of `sources` and is 64
/// hex characters.
#[test]
fn the_hash_is_order_independent_and_repeatable() {
    let d = tmp();
    let a = d.join("common/a.cu");
    let b = d.join("common/b.cu");
    write(&a, "__global__ void a() {}\n");
    write(&b, "__global__ void b() {}\n");

    let one = hash(&d, &inputs(vec![a.clone(), b.clone()])).unwrap();
    let two = hash(&d, &inputs(vec![b, a])).unwrap();
    assert_eq!(one, two, "source order must not matter");
    assert_eq!(one.len(), 64, "sha256 hex");
}

/// 2026-09-26: Two checkouts of the same files in different directories hash
/// the same, including when one root is given through an alias.
#[test]
fn the_hash_does_not_depend_on_the_checkout_location() {
    let d1 = tmp();
    let d2 = d1.parent().unwrap().join(format!(
        "{}-elsewhere",
        d1.file_name().unwrap().to_string_lossy()
    ));
    let _ = std::fs::remove_dir_all(&d2);
    std::fs::create_dir_all(d2.join("common")).unwrap();

    write(&d1.join("common/x.cu"), "__global__ void k() {}\n");
    write(&d2.join("common/x.cu"), "__global__ void k() {}\n");
    write(&d1.join("model/MODEL.toml"), "[model]\nname = \"same\"\n");
    std::fs::create_dir_all(d2.join("model")).unwrap();
    write(&d2.join("model/MODEL.toml"), "[model]\nname = \"same\"\n");

    // 2026-09-26: `model/..` stands in for a platform alias such as macOS
    // `/var` -> `/private/var`: the root is not in canonical form, the sources
    // are canonicalised.
    let aliased_d1 = d1.join("model/..");
    let mut first = inputs(vec![aliased_d1.join("common/x.cu")]);
    first.configs.push(aliased_d1.join("model/MODEL.toml"));
    let mut second = inputs(vec![d2.join("common/x.cu")]);
    second.configs.push(d2.join("model/MODEL.toml"));

    assert_eq!(
        hash(&aliased_d1, &first).unwrap(),
        hash(&d2, &second).unwrap(),
        "the same commit checked out twice must hash the same"
    );
}
