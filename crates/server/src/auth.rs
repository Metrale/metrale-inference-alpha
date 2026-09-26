// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Bearer-token validation for the HTTP API's `--require-auth` gate.
//!
//! Owner: server auth.
//! Invariants:
//! - An `AuthConfig` holds at least one non-empty token: both constructors
//!   return an error otherwise.
//! - `validate` compares the presented token with every loaded token, and
//!   compares equal-length tokens over every byte with no early exit.
//!
//! `build_auth_config` (`main_modules/serve.rs`) loads the tokens once at
//! startup, from `--auth-tokens-file` or `--auth-token`, and only when
//! `--require-auth` is set. For `--auth-token` it logs a warning that the
//! value is visible through `ps`/`/proc/<pid>/cmdline`. File permissions are
//! not checked.

use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// 2026-09-26: The loaded bearer tokens. The server shares one instance as
/// `Arc<AuthConfig>`.
#[derive(Debug)]
pub struct AuthConfig {
    /// 2026-09-26: The tokens as bytes; `from_file` sorts and de-duplicates them.
    tokens: Vec<Vec<u8>>,
}

impl AuthConfig {
    /// 2026-09-26: Load tokens from a file, one per line. Each line is trimmed
    /// at both ends; empty lines and lines starting with `#` are skipped.
    /// Errors when the file cannot be read or no token is left.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading auth tokens file {}", path.display()))?;
        let mut tokens: Vec<Vec<u8>> = raw
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !s.starts_with('#'))
            .map(|s| s.as_bytes().to_vec())
            .collect();
        tokens.sort();
        tokens.dedup();
        if tokens.is_empty() {
            return Err(anyhow!(
                "auth tokens file {} contains no usable tokens \
                 (lines must be non-empty and not start with `#`)",
                path.display()
            ));
        }
        Ok(Self { tokens })
    }

    /// 2026-09-26: One inline token, trimmed at both ends. An empty or
    /// all-whitespace token is an error.
    pub fn from_inline(token: &str) -> Result<Self> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("--auth-token must not be empty"));
        }
        Ok(Self {
            tokens: vec![trimmed.as_bytes().to_vec()],
        })
    }

    /// 2026-09-26: Number of distinct tokens loaded; the startup log in
    /// `build_auth_config` prints this count.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// 2026-09-26: `true` iff `presented` byte-equals one of the loaded tokens.
    /// Every loaded token is compared, even after a match, and each
    /// comparison is `ct_eq`, so the time does not depend on how many
    /// leading bytes match.
    pub fn validate(&self, presented: &[u8]) -> bool {
        let mut any_match = 0u8;
        for valid in &self.tokens {
            any_match |= ct_eq(presented, valid);
        }
        any_match == 1
    }
}

/// 2026-09-26: `1` if the slices are byte-equal, `0` otherwise. A length
/// mismatch returns `0` at once, so the length is observable through timing;
/// equal-length slices are compared over every byte with no early exit, so
/// the time does not show how many leading bytes match.
fn ct_eq(a: &[u8], b: &[u8]) -> u8 {
    if a.len() != b.len() {
        return 0;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    // 2026-09-26: 1 when `diff == 0`, else 0, without a branch: only
    // `0u32.wrapping_sub(1)` has bit 31 set.
    1u8 & ((diff as u32).wrapping_sub(1) >> 31) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_matches_only_when_equal() {
        assert_eq!(ct_eq(b"hello", b"hello"), 1);
        assert_eq!(ct_eq(b"hello", b"world"), 0);
        assert_eq!(ct_eq(b"hello", b"hellz"), 0);
        assert_eq!(ct_eq(b"hello", b"helloo"), 0);
        assert_eq!(ct_eq(b"", b""), 1);
        assert_eq!(ct_eq(b"a", b""), 0);
    }

    #[test]
    fn validates_single_inline_token() {
        let cfg = AuthConfig::from_inline("sk-test-token").unwrap();
        assert!(cfg.validate(b"sk-test-token"));
        assert!(!cfg.validate(b"sk-test-toke"));
        assert!(!cfg.validate(b"sk-test-tokenx"));
        assert!(!cfg.validate(b""));
        assert_eq!(cfg.token_count(), 1);
    }

    #[test]
    fn rejects_empty_inline() {
        assert!(AuthConfig::from_inline("").is_err());
        assert!(AuthConfig::from_inline("   ").is_err());
    }

    #[test]
    fn loads_file_with_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.txt");
        std::fs::write(
            &path,
            "# project A\n\
             alpha-token\n\
             \n\
             # project B\n\
             beta-token\n\
             alpha-token\n",
        )
        .unwrap();
        let cfg = AuthConfig::from_file(&path).unwrap();
        assert_eq!(cfg.token_count(), 2);
        assert!(cfg.validate(b"alpha-token"));
        assert!(cfg.validate(b"beta-token"));
        assert!(!cfg.validate(b"# project A"));
        assert!(!cfg.validate(b"gamma-token"));
    }

    #[test]
    fn empty_file_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "# only a comment\n\n   \n").unwrap();
        assert!(AuthConfig::from_file(&path).is_err());
    }
}
