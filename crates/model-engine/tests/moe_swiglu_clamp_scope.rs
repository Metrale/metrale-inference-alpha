// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Keeps the SwiGLU clamp (`SWIGLU_LIMIT` in CUDA code) inside the
//! kernel directories of models whose config declares a limit.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! `swiglu_limit` is a per-checkpoint config value. The build merges a
//! hardware's `common/` kernels into every model target, so a clamp there
//! applies to models that declare no limit.

use std::path::{Path, PathBuf};

/// 2026-09-25: Model kernel directories allowed to define a SwiGLU clamp, because
/// their checkpoint's `config.json` declares `swiglu_limit` or `swiglu_limits`.
const DECLARES_A_SWIGLU_LIMIT: &[&str] = &["deepseek-v4-flash", "step3p7-flash"];

/// 2026-09-25: Kernels exempt from the rule. `moe_shared_expert_fused.cu`
/// clamps the routed experts in `common/`; the comment at its clamp explains why
/// it stays.
const KNOWN_INCONSISTENT: &[&str] = &["gb10/common/moe_shared_expert_fused.cu"];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model-engine is two levels below the workspace root")
        .join("kernels")
}

fn cuda_code_mentions_swiglu_limit(text: &str) -> bool {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment,
        Quoted(char),
    }

    let mut state = State::Code;
    let mut token = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match state {
            State::Code if ch == '/' && chars.peek() == Some(&'/') => {
                chars.next();
                token.clear();
                state = State::LineComment;
            }
            State::Code if ch == '/' && chars.peek() == Some(&'*') => {
                chars.next();
                token.clear();
                state = State::BlockComment;
            }
            State::Code if ch == '"' || ch == '\'' => {
                token.clear();
                state = State::Quoted(ch);
            }
            State::Code if ch.is_ascii_alphanumeric() || ch == '_' => token.push(ch),
            State::Code => {
                if token == "SWIGLU_LIMIT" {
                    return true;
                }
                token.clear();
            }
            State::LineComment if ch == '\n' => state = State::Code,
            State::BlockComment if ch == '*' && chars.peek() == Some(&'/') => {
                chars.next();
                state = State::Code;
            }
            State::Quoted(quote) if ch == '\\' => {
                chars.next();
            }
            State::Quoted(quote) if ch == quote => state = State::Code,
            _ => {}
        }
    }
    token == "SWIGLU_LIMIT"
}

fn is_known_inconsistent(path: &Path) -> bool {
    KNOWN_INCONSISTENT
        .iter()
        .any(|known| path == Path::new(known))
}

fn files_defining_a_clamp(root: &Path) -> Vec<PathBuf> {
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // 2026-09-25: Symlinked files and directories are skipped.
            if path.is_dir() && !path.is_symlink() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "cu" || e == "cuh")
                && !path.is_symlink()
                && std::fs::read_to_string(&path).is_ok_and(|t| cuda_code_mentions_swiglu_limit(&t))
            {
                hits.push(path);
            }
        }
    }
    hits.sort();
    hits
}

#[test]
fn clamp_scanner_distinguishes_code_from_comments_and_spacing() {
    assert!(cuda_code_mentions_swiglu_limit(
        "const float SWIGLU_LIMIT=10.0f;"
    ));
    assert!(cuda_code_mentions_swiglu_limit(
        "constexpr float\n SWIGLU_LIMIT\t = 7.0f;"
    ));
    assert!(!cuda_code_mentions_swiglu_limit(
        "// const float SWIGLU_LIMIT = 10.0f;\nfloat x = 1.0f;"
    ));
    assert!(!cuda_code_mentions_swiglu_limit(
        "/* SWIGLU_LIMIT = 10 */ const char* name = \"SWIGLU_LIMIT\";"
    ));
    assert!(is_known_inconsistent(Path::new(
        "gb10/common/moe_shared_expert_fused.cu"
    )));
    assert!(!is_known_inconsistent(Path::new(
        "gb10/another-model/moe_shared_expert_fused.cu"
    )));
}

/// 2026-09-25: A file whose code (not comments or strings) names `SWIGLU_LIMIT`
/// must sit under a directory in `DECLARES_A_SWIGLU_LIMIT` or be listed in
/// `KNOWN_INCONSISTENT`.
#[test]
fn the_swiglu_clamp_stays_in_models_that_declare_one() {
    let root = kernels_root();
    let mut stray = Vec::new();
    for path in files_defining_a_clamp(&root) {
        let rel = path.strip_prefix(&root).unwrap_or(path.as_path());
        let owned_by_a_declaring_model = rel
            .components()
            .any(|c| DECLARES_A_SWIGLU_LIMIT.contains(&c.as_os_str().to_string_lossy().as_ref()));
        let recorded = is_known_inconsistent(rel);
        if !owned_by_a_declaring_model && !recorded {
            stray.push(rel.display().to_string());
        }
    }
    assert!(
        stray.is_empty(),
        "SWIGLU_LIMIT is a per-checkpoint config value, and these files apply it \
         to every model that compiles them: {stray:?}. Put it in the shadow \
         directory of the model whose config.json declares it, or add that model \
         to DECLARES_A_SWIGLU_LIMIT once you have checked the config."
    );
}
