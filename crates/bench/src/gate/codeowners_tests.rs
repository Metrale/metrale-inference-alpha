// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the CODEOWNERS parser and matcher, including the committed file.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

/// 2026-09-26: Every pattern in the committed CODEOWNERS is supported; an unsupported one would
/// name nobody without any error.
#[test]
fn every_pattern_in_the_real_file_is_supported() {
    let rules = load(&repo_root());
    assert!(!rules.is_empty(), "CODEOWNERS did not parse");
    for rule in &rules {
        assert!(
            is_supported(&rule.pattern),
            "{:?} uses a construct this module does not implement — implement it \
             or its owners go unmentioned",
            rule.pattern
        );
    }
}

/// 2026-09-26: Real files of this repository resolve to the committed owner
/// set. Each path must exist, so a path left behind by a move fails here
/// instead of falling through to the `*` rule.
#[test]
fn real_paths_resolve_to_real_owners() {
    let rules = load(&repo_root());
    for path in [
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/model-layers/src/layers/ops/dispatch_helpers.rs",
        "crates/kernels/build.rs",
        ".github/workflows/ci.yml",
        "docs/adr/0012-closure-hash-cascade.md",
        "Cargo.toml",
    ] {
        assert!(
            repo_root().join(path).is_file(),
            "{path} is not a file of this repository"
        );
        assert_eq!(
            owners_of(&rules, path),
            ["@tbraun96", "@rsafier", "@SeedSource", "@TheTom"],
            "{path} must retain the committed owner set"
        );
    }
}

#[test]
fn a_directory_pattern_covers_its_whole_subtree() {
    let rules = parse("crates/model-layers/ @a\n");
    assert_eq!(owners_of(&rules, "crates/model-layers/src/lib.rs"), ["@a"]);
    assert_eq!(owners_of(&rules, "crates/model-layers/Cargo.toml"), ["@a"]);
    assert!(
        owners_of(&rules, "crates/server/src/lib.rs").is_empty(),
        "a sibling crate is not covered"
    );
}

#[test]
fn a_bare_filename_matches_at_any_depth() {
    let rules = parse("Cargo.toml @a\n");
    assert_eq!(owners_of(&rules, "Cargo.toml"), ["@a"]);
    assert_eq!(owners_of(&rules, "crates/server/Cargo.toml"), ["@a"]);
    assert!(owners_of(&rules, "crates/server/Cargo.lock").is_empty());
}

#[test]
fn a_pattern_with_a_slash_is_anchored_to_the_root() {
    let rules = parse("kernels/gb10/ @a\n");
    assert_eq!(owners_of(&rules, "kernels/gb10/common/x.cu"), ["@a"]);
    assert!(
        owners_of(&rules, "vendor/kernels/gb10/x.cu").is_empty(),
        "an anchored pattern must not match deeper"
    );
}

#[test]
fn a_star_matches_within_one_segment_only() {
    let rules = parse("/docs/*.md @a\n");
    assert_eq!(owners_of(&rules, "docs/readme.md"), ["@a"]);
    assert!(
        owners_of(&rules, "docs/adr/0001.md").is_empty(),
        "`*` must not cross a separator"
    );
}

#[test]
fn multiple_stars_within_one_segment_are_matched() {
    let rules = parse("/docs/a*b*.md @a\n");
    assert_eq!(owners_of(&rules, "docs/alpha-bench-check.md"), ["@a"]);
    assert!(owners_of(&rules, "docs/alpha-bench-check.txt").is_empty());
    assert!(owners_of(&rules, "docs/a/b/c.md").is_empty());
}

/// 2026-09-26: The last matching rule wins, so a specific rule after a catch-all overrides it.
#[test]
fn the_last_matching_rule_wins() {
    let rules = parse("* @everyone\ncrates/model-layers/ @model-owner\n");
    assert_eq!(owners_of(&rules, "README.md"), ["@everyone"]);
    assert_eq!(
        owners_of(&rules, "crates/model-layers/src/lib.rs"),
        ["@model-owner"]
    );
}

/// 2026-09-26: A pattern with no owners clears ownership rather than being ignored.
#[test]
fn a_pattern_with_no_owners_clears_ownership() {
    let rules = parse("* @everyone\nvendor/\n");
    assert!(
        owners_of(&rules, "vendor/thing.rs").is_empty(),
        "a bare pattern must clear, not fall through to the catch-all"
    );
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let rules = parse("# a comment\n\n  \n* @a # trailing comment\n");
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].pattern, "*");
    assert_eq!(rules[0].owners, ["@a"]);
}

#[test]
fn owners_across_paths_are_deduplicated_and_sorted() {
    let rules = parse("* @z\ncrates/ @a @z\n");
    let paths = vec![
        "crates/x/src/lib.rs".to_string(),
        "crates/y/src/lib.rs".to_string(),
        "README.md".to_string(),
    ];
    assert_eq!(owners_for_paths(&rules, &paths), ["@a", "@z"]);
}

#[test]
fn a_missing_codeowners_file_yields_no_rules_rather_than_failing() {
    let empty = std::env::temp_dir().join(format!("metrale-noowners-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    assert!(load(&empty).is_empty());
}

/// 2026-09-26: Each unsupported construct is reported by `is_supported` and matches nothing.
#[test]
fn unsupported_globs_are_reported_not_silently_wrong() {
    for pattern in [
        "docs/**/notes.md",
        "docs/note?.md",
        "docs/note[12].md",
        r"docs/note\*.md",
    ] {
        assert!(!is_supported(pattern), "{pattern}");
        let rules = parse(&format!("{pattern} @a\n"));
        assert!(owners_of(&rules, "docs/note1.md").is_empty(), "{pattern}");
    }
}
