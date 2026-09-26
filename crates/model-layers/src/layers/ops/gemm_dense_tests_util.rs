// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Text helpers for the dense GEMM source-contract tests:
//! delimiter matching and `//` comment stripping, with no parser dependency.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The extractors return `None` or an empty `Vec` when they find nothing.
//! `gemm_dense_tests.rs` asserts that it found what it looked for, directly or
//! through a minimum count, so a changed kernel or launcher fails the tests
//! instead of passing them vacuously.

use std::path::Path;

use crate::layers::ops::kernel_tree_tests_util::{KernelFile, compiled_cu_files};

/// 2026-09-25: Every `.cu` the build compiles, one entry per compiled location
/// (`compiled_cu_files`); panics if it finds 100 or fewer.
pub fn cu_files() -> Vec<KernelFile> {
    let files = compiled_cu_files();
    assert!(files.len() > 100, "kernel tree not found");
    files
}

/// 2026-09-25: The path from `kernels/` on, for assertion messages.
pub fn rel(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    match s.split_once("kernels/") {
        Some((_, tail)) => format!("kernels/{tail}"),
        None => s,
    }
}

/// 2026-09-25: Drop `//` line comments. Kernel signatures carry commas inside
/// comments (`// [K/2, N] transposed`), which a parameter count must not
/// see.
pub fn strip_line_comments(s: &str) -> String {
    s.lines()
        .map(|l| l.split_once("//").map_or(l, |(head, _)| head))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 2026-09-25: Index of the delimiter `c` that closes the `o` at `open`, or
/// `None` if it is never closed.
fn match_delim(s: &str, open: usize, o: char, c: char) -> Option<usize> {
    let mut depth = 0usize;
    for (i, ch) in s[open..].char_indices() {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return Some(open + i);
            }
        }
    }
    None
}

/// 2026-09-25: `(signature, body)` of the `__global__` kernel `name`, or `None`
/// if this file does not define it. `signature` spans `void NAME(` through its
/// `)`; `body` spans the matching braces.
pub fn kernel_sig_body(src: &str, name: &str) -> Option<(String, String)> {
    let pat = format!("void {name}(");
    let mut from = 0usize;
    while let Some(off) = src[from..].find(&pat) {
        let at = from + off;
        from = at + pat.len();
        // 2026-09-25: A declaration has `__global__` within the 160 bytes before
        // the name; a match without it is a call.
        let mut lo = at.saturating_sub(160);
        while lo > 0 && !src.is_char_boundary(lo) {
            lo -= 1;
        }
        if !src[lo..at].contains("__global__") {
            continue;
        }
        let open = at + pat.len() - 1;
        let close = match_delim(src, open, '(', ')')?;
        let sig = src[at..=close].to_string();
        let bopen = close + src[close..].find('{')?;
        let bclose = match_delim(src, bopen, '{', '}')?;
        return Some((sig, src[bopen..=bclose].to_string()));
    }
    None
}

/// 2026-09-25: Number of parameters in a signature from `kernel_sig_body`.
pub fn param_count(sig: &str) -> usize {
    let sig = strip_line_comments(sig);
    let open = sig.find('(').unwrap_or(0);
    let close = sig.rfind(')').unwrap_or(sig.len());
    split_top_level(&sig[open + 1..close]).len()
}

/// 2026-09-25: Split on commas at nesting depth zero, dropping blank entries
/// such as the one after a trailing comma.
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | '[' | '{' | '<' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' | '}' | '>' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.retain(|a| !a.trim().is_empty());
    out
}

/// 2026-09-25: Every index expression `arr[...]` in `body` outside `//`
/// comments, brackets stripped and whitespace collapsed.
pub fn indexed_exprs(body: &str, arr: &str) -> Vec<String> {
    let body = strip_line_comments(body);
    let pat = format!("{arr}[");
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(off) = body[from..].find(&pat) {
        let open = from + off + pat.len() - 1;
        let Some(close) = match_delim(&body, open, '[', ']') else {
            break;
        };
        out.push(
            body[open + 1..close]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        );
        from = close + 1;
    }
    out
}

/// 2026-09-25: Does `expr` multiply by the bare identifier `N`? `* N_TILE` and
/// `* NUM_x` do not count.
pub fn multiplies_by_bare_n(expr: &str) -> bool {
    let b = expr.as_bytes();
    for (i, &ch) in b.iter().enumerate() {
        if ch != b'*' {
            continue;
        }
        let mut j = i + 1;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if b.get(j) != Some(&b'N') {
            continue;
        }
        let after = b.get(j + 1).copied().unwrap_or(b' ');
        if !after.is_ascii_alphanumeric() && after != b'_' {
            return true;
        }
    }
    false
}

/// 2026-09-25: Body, braces included, of the first Rust function `name` in
/// `src`, or `None`.
pub fn fn_body(src: &str, name: &str) -> Option<String> {
    let pat = format!("fn {name}(");
    let at = src.find(&pat)?;
    let params_open = at + pat.len() - 1;
    let params_close = match_delim(src, params_open, '(', ')')?;
    let bopen = params_close + src[params_close..].find('{')?;
    let bclose = match_delim(src, bopen, '{', '}')?;
    Some(src[bopen..=bclose].to_string())
}

/// 2026-09-25: How many kernel arguments a launcher packs: the `.arg_`
/// occurrences in its body, comments included.
pub fn launcher_arg_count(src: &str, name: &str) -> Option<usize> {
    Some(fn_body(src, name)?.matches(".arg_").count())
}

/// 2026-09-25: Arguments of the first parenthesised group in `s`, outside `//`
/// comments.
pub fn call_args(s: &str) -> Vec<String> {
    let s = strip_line_comments(s);
    let Some(open) = s.find('(') else {
        return Vec::new();
    };
    let Some(close) = match_delim(&s, open, '(', ')') else {
        return Vec::new();
    };
    split_top_level(&s[open + 1..close])
}
