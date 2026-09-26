// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Checks that every kernel the workspace's Rust code looks up by
//! name, `module::func` with both written as literals, is declared by that
//! module in at least one target of the `kernels/` tree.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! A lookup names a module (a file stem, or its `[modules]` rename) and an
//! entry point. Nothing ties the two strings to the tree at compile time: a
//! renamed, removed or un-`use`d source is found only when a lookup fails at
//! boot on a GPU. This test resolves every target with
//! `metrale_closure::layout`, reads each module's entry points with the
//! scanner build.rs uses (`build_shadow.rs`), and fails on a lookup no target
//! can answer. It checks existence in some target, not in every target that
//! reaches the call: which targets reach a call is a runtime property.
//!
//! The lookups read are `.kernel(`, `try_kernel(`, `try_target_kernel(` and
//! `gated(`, whose last two arguments are the module and the entry point. A
//! module given as a `SCREAMING_CASE` constant is resolved through the
//! `const NAME: &str = "..."` items of the same crate. Test-only files and
//! inline `#[cfg(test)]` modules are skipped: their lookups go to a mock.

// 2026-09-26: Only `entry_points` is used here; the shadow comparison is
// `kernel_shadow_detector.rs`'s.
#[allow(dead_code)]
#[path = "../build_shadow.rs"]
mod build_shadow;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use metrale_closure::layout::{discover, walk};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

/// 2026-09-26: Every `(module, entry point)` some target declares.
fn declared_kernels(root: &Path) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    // 2026-09-26: Targets share most sources; scan each file once.
    let mut scanned: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for target in walk(root).expect("the tree resolves") {
        let layout = discover(root, &target).unwrap_or_else(|e| panic!("{target}: {e}"));
        let mut module_of: BTreeMap<String, String> = BTreeMap::new();
        for config in layout.configs() {
            let Ok(text) = std::fs::read_to_string(&config) else {
                continue;
            };
            let toml: toml::Value =
                toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", config.display()));
            for (stem, name) in toml
                .get("modules")
                .and_then(|m| m.as_table())
                .into_iter()
                .flatten()
            {
                if let Some(name) = name.as_str() {
                    module_of.insert(stem.clone(), name.to_string());
                }
            }
        }
        for (stem, entry) in layout.modules() {
            let module = module_of.get(&stem).cloned().unwrap_or(stem);
            let funcs = scanned
                .entry(entry.source.clone())
                .or_insert_with(|| build_shadow::entry_points(&entry.source));
            for func in funcs.iter() {
                out.insert((module.clone(), func.clone()));
            }
        }
    }
    out
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for e in read.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if name != "tests" && name != "target" {
                rust_files(&path, out);
            }
        } else if name.ends_with(".rs")
            && !name.ends_with("_tests.rs")
            && name != "tests.rs"
            && name != "mock.rs"
        {
            out.push(path);
        }
    }
}

/// 2026-09-26: The file's text up to its first inline `#[cfg(test)] mod x {`.
/// An out-of-line `#[cfg(test)] mod x;` is skipped: its file is a test file.
fn non_test_text(text: &str) -> &str {
    let mut at = 0;
    while let Some(i) = text[at..].find("#[cfg(test)]") {
        let after = text[at + i + "#[cfg(test)]".len()..].trim_start();
        if let Some(rest) = after.strip_prefix("mod ") {
            let rest = rest.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_');
            if rest.trim_start().starts_with('{') {
                return &text[..at + i];
            }
        }
        at += i + 1;
    }
    text
}

/// 2026-09-26: `NAME -> values` for every `const NAME: &str = "..."` in `text`.
fn str_consts(text: &str, out: &mut BTreeMap<String, BTreeSet<String>>) {
    for line in text.lines() {
        let Some(rest) = line.trim_start().split("const ").nth(1) else {
            continue;
        };
        let Some((name, rest)) = rest.split_once(':') else {
            continue;
        };
        let Some((ty, value)) = rest.split_once('=') else {
            continue;
        };
        if !matches!(ty.trim(), "&str" | "&'static str") {
            continue;
        }
        let value = value.trim().trim_end_matches(';');
        if let Some(v) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            out.entry(name.trim().to_string())
                .or_default()
                .insert(v.to_string());
        }
    }
}

/// 2026-09-26: The top-level comma-separated arguments of the call whose
/// opening parenthesis ends `text[..open]`, or `None` if unbalanced.
fn call_args(text: &str, open: usize) -> Option<Vec<&str>> {
    let bytes = text.as_bytes();
    let (mut depth, mut start, mut in_str) = (0usize, open + 1, false);
    let mut args = Vec::new();
    let mut i = open + 1;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            match c {
                b'\\' => i += 1,
                b'"' => in_str = false,
                _ => {}
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' if depth > 0 => depth -= 1,
                b')' => {
                    // 2026-09-26: A trailing comma leaves an empty last argument.
                    let last = text[start..i].trim();
                    if !last.is_empty() {
                        args.push(last);
                    }
                    return Some(args);
                }
                b',' if depth == 0 => {
                    args.push(text[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn literal(arg: &str) -> Option<&str> {
    let s = arg.strip_prefix('"')?.strip_suffix('"')?;
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')).then_some(s)
}

fn is_const_name(arg: &str) -> bool {
    arg.starts_with(|c: char| c.is_ascii_uppercase())
        && arg
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

/// 2026-09-26: One lookup: where it is, the candidate module names (one for a
/// literal, every value of the constant for a constant) and the entry point.
struct Lookup {
    site: String,
    modules: BTreeSet<String>,
    func: String,
}

const CALLS: &[&str] = &[".kernel(", "try_kernel(", "try_target_kernel(", "gated("];

fn lookups(root: &Path) -> Vec<Lookup> {
    let mut out = Vec::new();
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("crates/")
        .flatten()
        .map(|e| e.path())
        .collect();
    crates.sort();
    let sources: Vec<Vec<(PathBuf, String)>> = crates
        .iter()
        .map(|krate| {
            let mut files = Vec::new();
            rust_files(krate, &mut files);
            files.sort();
            files
                .into_iter()
                .map(|f| {
                    let t = std::fs::read_to_string(&f)
                        .unwrap_or_else(|e| panic!("{}: {e}", f.display()));
                    (f, t)
                })
                .collect()
        })
        .collect();
    // 2026-09-26: A constant resolves in its own crate first; one imported
    // from another crate (`KQUANT_MODULE` is model-layers', read by
    // model-arch) falls back to the workspace's.
    let mut workspace_consts = BTreeMap::new();
    for (_, t) in sources.iter().flatten() {
        str_consts(t, &mut workspace_consts);
    }
    for texts in &sources {
        let mut consts = BTreeMap::new();
        for (_, t) in texts {
            str_consts(t, &mut consts);
        }
        for (file, text) in texts {
            let text = non_test_text(text);
            for call in CALLS {
                let mut at = 0;
                while let Some(i) = text[at..].find(call) {
                    let pos = at + i;
                    at = pos + call.len();
                    // 2026-09-26: `try_kernel(` also matches inside `my_try_kernel(`.
                    let prev = text[..pos].chars().next_back();
                    if !call.starts_with('.')
                        && prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
                    {
                        continue;
                    }
                    let Some(args) = call_args(text, pos + call.len() - 1) else {
                        continue;
                    };
                    let [.., module, func] = args.as_slice() else {
                        continue;
                    };
                    let Some(func) = literal(func) else {
                        continue;
                    };
                    let modules: BTreeSet<String> = if let Some(m) = literal(module) {
                        [m.to_string()].into()
                    } else if is_const_name(module) {
                        match consts
                            .get(*module)
                            .or_else(|| workspace_consts.get(*module))
                        {
                            Some(values) => values.clone(),
                            None => continue,
                        }
                    } else {
                        continue;
                    };
                    let line = text[..pos].matches('\n').count() + 1;
                    out.push(Lookup {
                        site: format!("{}:{line}", file.strip_prefix(root).unwrap().display()),
                        modules,
                        func: func.to_string(),
                    });
                }
            }
        }
    }
    out
}

#[test]
fn every_literal_kernel_lookup_names_a_kernel_some_target_declares() {
    let root = workspace_root();
    let declared = declared_kernels(&root);
    let lookups = lookups(&root);
    assert!(
        lookups.len() > 500,
        "only {} lookups found — did the lookup API or the crate layout change?",
        lookups.len()
    );
    let mut missing: Vec<String> = lookups
        .iter()
        .filter(|l| {
            !l.modules
                .iter()
                .any(|m| declared.contains(&(m.clone(), l.func.clone())))
        })
        .map(|l| {
            let m: Vec<&str> = l.modules.iter().map(String::as_str).collect();
            format!("{}: {}::{}", l.site, m.join("|"), l.func)
        })
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "{} kernel lookup(s) name a module::func that no target under kernels/ declares \
         (renamed or removed source, a lost `[sources] use`, or a `[modules]` rename):\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
