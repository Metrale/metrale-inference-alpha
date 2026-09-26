// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The lever table against the code that reads the levers, in both directions:
//! every `METRALE_*` name the code reads must be declared, and every declared name must
//! appear in the `reader` file the table gives.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use super::{Class, LeverSpec, PREFIX, all, check, lookup, undeclared};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn files(dir: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name()
                .is_some_and(|n| n == "target" || n == "node_modules")
            {
                continue;
            }
            files(&p, exts, out);
        } else if p
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| exts.contains(&x))
        {
            out.push(p);
        }
    }
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo())
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

fn is_test_file(rel: &str) -> bool {
    let base = rel.rsplit('/').next().unwrap_or(rel);
    base.ends_with("_tests.rs") || base == "tests.rs" || rel.contains("/tests/")
}

fn is_dev_file(rel: &str) -> bool {
    rel.contains("/examples/") || rel.contains("/benches/")
}

/// 2026-09-26: The line with any `//` comment that starts outside a string removed.
fn code_of(line: &str) -> String {
    let mut out = String::new();
    let mut in_str = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_str && c == '\\' {
            out.push(c);
            if let Some(n) = chars.next() {
                out.push(n);
            }
            continue;
        }
        if c == '"' {
            in_str = !in_str;
        }
        if !in_str && c == '/' && chars.peek() == Some(&'/') {
            break;
        }
        out.push(c);
    }
    out
}

/// 2026-09-26: Every whole `"METRALE_X"` string literal on a line of code.
fn literals(line: &str) -> Vec<String> {
    let code = code_of(line);
    let mut out = Vec::new();
    let mut rest = code.as_str();
    while let Some(i) = rest.find("\"METRALE_") {
        let tail = &rest[i + 1..];
        let end = tail
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(tail.len());
        if tail[end..].starts_with('"') && end > PREFIX.len() {
            out.push(tail[..end].to_string());
        }
        rest = &tail[end..];
    }
    out
}

/// 2026-09-26: `(name, file:line, production?)` for every literal outside a test file and
/// outside the lever tables; production means not under `examples/` or `benches/`.
fn rust_reads() -> Vec<(String, String, bool)> {
    let mut rs = Vec::new();
    files(&repo().join("crates"), &["rs"], &mut rs);
    assert!(
        rs.len() > 1000,
        "the scan found only {} Rust files",
        rs.len()
    );
    let mut out = Vec::new();
    for p in rs {
        let r = rel(&p);
        // 2026-09-26: The tables spell every name; they are what is being checked.
        if is_test_file(&r) || r.starts_with("crates/config/src/levers/") {
            continue;
        }
        let text = std::fs::read_to_string(&p).expect("readable source");
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with('*') {
                continue;
            }
            for name in literals(line) {
                out.push((name, format!("{r}:{}", i + 1), !is_dev_file(&r)));
            }
        }
    }
    out
}

/// 2026-09-26: Every `getenv("METRALE_X")` in the C/CUDA sources under `crates/` and `kernels/`.
fn c_reads() -> Vec<(String, String)> {
    let mut cs = Vec::new();
    for dir in ["crates", "kernels"] {
        files(
            &repo().join(dir),
            &["cu", "cuh", "c", "cc", "cpp", "h"],
            &mut cs,
        );
    }
    let mut out = Vec::new();
    for p in cs {
        let text = std::fs::read_to_string(&p).unwrap_or_default();
        for (i, line) in text.lines().enumerate() {
            if let Some(at) = line.find("getenv(\"METRALE_") {
                let name: String = line[at + 8..].chars().take_while(|c| *c != '"').collect();
                out.push((name, format!("{}:{}", rel(&p), i + 1)));
            }
        }
    }
    out
}

#[test]
fn names_are_well_formed_unique_and_in_order() {
    let names: Vec<&str> = all().map(|l| l.env).collect();
    assert!(names.len() > 500, "only {} levers declared", names.len());
    for w in names.windows(2) {
        assert!(
            w[0] < w[1],
            "{} and {} are out of order or repeated",
            w[0],
            w[1]
        );
    }
    for l in all() {
        let suffix = &l.env[PREFIX.len().min(l.env.len())..];
        assert!(
            l.env.starts_with(PREFIX)
                && !suffix.is_empty()
                && suffix
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'),
            "{} is not a METRALE_* name",
            l.env
        );
        assert!(
            l.path.ends_with(&suffix.to_ascii_lowercase()) && l.path.contains('.'),
            "{}: path {} does not name it",
            l.env,
            l.path
        );
        assert!(
            !l.doc.is_empty() && !l.default.is_empty(),
            "{}: empty field",
            l.env
        );
        if let Some(flag) = l.flag {
            assert!(flag.starts_with("--"), "{}: flag {flag}", l.env);
        }
    }
}

#[test]
fn every_lever_the_code_reads_is_declared() {
    let mut missing: Vec<String> = rust_reads()
        .into_iter()
        .filter(|(n, _, _)| lookup(n).is_none())
        .map(|(n, at, _)| format!("{n} ({at})"))
        .collect();
    missing.extend(
        c_reads()
            .into_iter()
            .filter(|(n, _)| lookup(n).is_none())
            .map(|(n, at)| format!("{n} ({at})")),
    );
    assert!(
        missing.is_empty(),
        "read but not declared in metrale_config::levers: {missing:?}"
    );
}

#[test]
fn every_declared_lever_is_read_where_the_table_says() {
    let prod: std::collections::BTreeSet<String> = rust_reads()
        .into_iter()
        .filter(|(_, _, prod)| *prod)
        .map(|(n, _, _)| n)
        .chain(c_reads().into_iter().map(|(n, _)| n))
        .collect();
    let mut bad = Vec::new();
    for l in all() {
        let text = std::fs::read_to_string(repo().join(l.reader)).unwrap_or_default();
        if !text.contains(l.env) {
            bad.push(format!("{}: {} does not read it", l.env, l.reader));
            continue;
        }
        let crates = l.reader.starts_with("crates/");
        let placed = match l.class {
            Class::Runtime => crates && !is_test_file(l.reader) && prod.contains(l.env),
            Class::Harness | Class::Build => crates && !is_test_file(l.reader),
            Class::Dev => is_test_file(l.reader) || is_dev_file(l.reader),
            Class::Tool => !crates,
        };
        if !placed {
            bad.push(format!(
                "{}: class {:?} but reader {}",
                l.env, l.class, l.reader
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn an_undeclared_name_is_refused_by_name() {
    let env = [
        "PATH",
        "METRALE_FOO",
        "METRALE_HOME",
        "METRALE_MTP_K_LADDER",
        "METRALE_FOO",
    ];
    assert_eq!(undeclared(env), ["METRALE_FOO"]);
    let err = check(env).unwrap_err();
    let shown = err.to_string();
    assert!(shown.contains("METRALE_FOO"), "{shown}");
    assert!(shown.contains("undeclared"), "{shown}");
    assert!(check(["PATH", "METRALE_HOME", "HOME"]).is_ok());
    assert_eq!(undeclared([PREFIX]), [PREFIX]);
}

#[test]
fn the_harness_set_is_exactly_the_nine_placement_variables() {
    // 2026-09-26: `serve_env::is_lever` treats these as the box's, never the recipe's, so
    // the set is pinned here, where it is declared.
    let harness: Vec<&str> = all()
        .filter(|l| l.class == Class::Harness)
        .map(|l: &LeverSpec| l.env)
        .collect();
    assert_eq!(
        harness,
        [
            "METRALE_HARNESS_PORT",
            "METRALE_HIPCC",
            "METRALE_HOME",
            "METRALE_SKIP_BUILD",
            "METRALE_TARGET_HW",
            "METRALE_TARGET_MODEL",
            "METRALE_TARGET_QUANT",
            "METRALE_TUI_LOG_FILE",
            "METRALE_WARM_TEMPLATE_DIR",
        ]
    );
}
