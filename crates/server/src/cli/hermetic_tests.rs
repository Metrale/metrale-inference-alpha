// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `--hermetic`: the two resolvers, `expand`, and a source scan for reads of the raw fields.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::{CLOSED_KEYS, expand, mtp_gate_force, prefix_caching_enabled};

#[test]
fn hermetic_closes_the_prefix_cache_even_when_it_was_asked_for() {
    assert!(
        !prefix_caching_enabled(true, true),
        "--hermetic must close the prefix cache; it is the M2 channel"
    );
}

#[test]
fn without_hermetic_the_prefix_cache_is_exactly_what_was_asked_for() {
    assert!(prefix_caching_enabled(true, false));
    assert!(!prefix_caching_enabled(false, false));
}

#[test]
fn hermetic_forces_the_mtp_gate_rather_than_deferring_to_the_environment() {
    // 2026-09-26: `None` would leave the decision to METRALE_MTP_GATE_FORCE, so
    // under --hermetic the resolver must return `Some(true)`.
    assert_eq!(
        mtp_gate_force(None, true),
        Some(true),
        "--hermetic must PIN the gate, not leave it to the environment"
    );
}

#[test]
fn hermetic_wins_over_a_contradicting_gate_value() {
    // 2026-09-26: `validate_serve_args` refuses this pair; the resolver is checked
    // without relying on that.
    assert_eq!(mtp_gate_force(Some("auto"), true), Some(true));
}

#[test]
fn without_hermetic_the_gate_is_exactly_what_was_asked_for() {
    assert_eq!(
        mtp_gate_force(None, false),
        None,
        "absent means env decides"
    );
    assert_eq!(mtp_gate_force(Some("force"), false), Some(true));
    assert_eq!(mtp_gate_force(Some("auto"), false), Some(false));
}

/// 2026-09-26: No source file under `src/` other than the three in `ALLOWED` and the
/// `*_tests.rs` files may read `.enable_prefix_caching` or `.mtp_gate`. The resolved
/// prefix-cache value has readers that act on it (`serve_phases/build.rs`) and readers
/// that report it (`serve_phases/preflight.rs`, `tui/logo.rs`); all go through
/// `prefix_caching_enabled()`.
#[test]
fn no_production_code_reads_the_raw_fields_behind_hermetic() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    // 2026-09-26: The resolvers, the field declarations, and the validator, which
    // needs the raw request to detect a contradiction with `--hermetic`.
    const ALLOWED: &[&str] = &["cli/hermetic.rs", "cli/serve_args.rs", "cli/validate.rs"];
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for entry in walk(&src) {
        let rel = entry
            .strip_prefix(&src)
            .unwrap_or(&entry)
            .to_string_lossy()
            .replace('\\', "/");
        if rel.ends_with("_tests.rs") || ALLOWED.contains(&rel.as_str()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&entry) else {
            continue;
        };
        scanned += 1;
        for (n, line) in text.lines().enumerate() {
            for field in ["enable_prefix_caching", "mtp_gate"] {
                if reads_field(line, field) {
                    offenders.push(format!("{rel}:{} reads .{field}", n + 1));
                }
            }
        }
    }
    // 2026-09-26: The floor makes a walk that reaches no files fail rather than pass.
    assert!(
        scanned > 50,
        "the scan visited {scanned} files — it is not scanning the tree, so its \
         green means nothing"
    );
    assert!(
        offenders.is_empty(),
        "these read a raw --hermetic-controlled field instead of its resolver \
         (`args.prefix_caching_enabled()` / `args.mtp_gate_force()`), so --hermetic \
         would not reach them:\n  {}",
        offenders.join("\n  ")
    );
}

/// 2026-09-26: Whether `line` contains the field access `.<field>` as a whole token.
/// A substring match would flag `sched.levers.mtp_gate_force`, the resolved lever the
/// scheduler reads, because `.mtp_gate` is a prefix of it.
fn reads_field(line: &str, field: &str) -> bool {
    let needle = format!(".{field}");
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(i) = line[from..].find(&needle) {
        let start = from + i;
        let end = start + needle.len();
        let next_is_ident = bytes
            .get(end)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_');
        if !next_is_ident {
            return true;
        }
        from = end;
    }
    false
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out
}

/// 2026-09-26: `reads_field` flags `.mtp_gate` and `.enable_prefix_caching` but not
/// `.mtp_gate_force` or `.prefix_caching_enabled()`.
#[test]
fn the_scan_matches_whole_tokens_not_prefixes() {
    assert!(reads_field("if args.mtp_gate.is_some() {", "mtp_gate"));
    assert!(!reads_field("if sched.levers.mtp_gate_force {", "mtp_gate"));
    assert!(reads_field(
        "a.enable_prefix_caching,",
        "enable_prefix_caching"
    ));
    assert!(!reads_field(
        "args.prefix_caching_enabled()",
        "enable_prefix_caching"
    ));
}

fn map(pairs: &[(&str, &str)]) -> std::collections::BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// 2026-09-26: `hermetic=true` expands into every `CLOSED_KEYS` entry and keeps its own
/// key. The gate expands before the recipe renders (`bench_serve_plan.rs`), so a recipe
/// that turns the prefix cache on still renders a command line `validate_serve_args`
/// accepts; `recipe_tests.rs` checks that on the fixture recipes.
#[test]
fn hermetic_expands_into_the_keys_it_closes() {
    let out = expand(map(&[("hermetic", "true")]));
    assert_eq!(
        out.get("enable_prefix_caching").map(String::as_str),
        Some("false")
    );
    assert_eq!(out.get("mtp_gate").map(String::as_str), Some("force"));
    assert_eq!(
        out.get("hermetic").map(String::as_str),
        Some("true"),
        "and the regime keeps its name"
    );
}

/// 2026-09-26: Without `hermetic=true` the map comes back unchanged.
#[test]
fn nothing_expands_without_hermetic() {
    assert_eq!(expand(map(&[])), map(&[]));
    assert_eq!(
        expand(map(&[("ssm_cache_slots", "256")])),
        map(&[("ssm_cache_slots", "256")])
    );
    // 2026-09-26: `is_requested` accepts only the exact value `true`.
    assert_eq!(
        expand(map(&[("hermetic", "false")])),
        map(&[("hermetic", "false")])
    );
}

/// 2026-09-26: A key the caller already set keeps its value, so `validate_serve_args`
/// still sees the contradiction and refuses it. Recipe defaults are not in this map.
#[test]
fn expansion_never_overwrites_a_value_someone_named() {
    let out = expand(map(&[
        ("hermetic", "true"),
        ("enable_prefix_caching", "true"),
    ]));
    assert_eq!(
        out.get("enable_prefix_caching").map(String::as_str),
        Some("true"),
        "an explicit opposite must survive to be refused, not be quietly fixed"
    );
    let out = expand(map(&[("hermetic", "true"), ("mtp_gate", "auto")]));
    assert_eq!(out.get("mtp_gate").map(String::as_str), Some("auto"));
}

/// 2026-09-26: Every `CLOSED_KEYS` entry resolves, under `--hermetic`, to the value the
/// table gives, and a key with no resolver check here fails the test.
#[test]
fn hermetic_closures_match_the_resolvers() {
    for (key, value) in CLOSED_KEYS {
        match *key {
            "enable_prefix_caching" => {
                let want: bool = value.parse().expect("a bool");
                // 2026-09-26: The request is the opposite of the expected value,
                // so a resolver that ignored `hermetic` would fail.
                assert_eq!(
                    prefix_caching_enabled(!want, true),
                    want,
                    "CLOSED_KEYS says {key}={value}, the resolver disagrees"
                );
            }
            "mtp_gate" => {
                assert_eq!(
                    mtp_gate_force(Some("auto"), true),
                    Some(*value == "force"),
                    "CLOSED_KEYS says {key}={value}, the resolver disagrees"
                );
            }
            other => panic!(
                "CLOSED_KEYS gained `{other}` with no resolver check — add one here, or \
                 the disclosure and the enforcement can disagree about it"
            ),
        }
    }
}
