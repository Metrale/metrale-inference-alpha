// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What makes one `met serve` "the same server" as another: the
//! bytes of the binary, the arguments it was started with, and its
//! `METRALE_*` serve levers.
//!
//! A benchmark that reuses a server somebody else started must know it is
//! measuring what it would have started itself, or its record names a config
//! it never ran. The server publishes the identity as digests
//! (`GET /serve-config`), never as the arguments themselves, because an argv
//! can carry `--auth-token`. Both sides compute the digests with the
//! functions here.
//!
//! The environment is part of the identity because levers are not recipe
//! keys: two servers with different levers can have byte-identical argv. The
//! digest is `serve_env::fingerprint` over `serve_env::process_levers()`,
//! which covers every `METRALE_*` variable except the harness class. The
//! harness computes the expected digest from the lever set the recipe
//! declares. The reuse check errs toward refusing: refusing a reusable server
//! costs one server start, while reusing a wrong one corrupts a record.
//!
//! Owner: bench (serve identity).
//! Invariants:
//! - Every digest this module computes is 64 lowercase hex characters;
//!   `this_process` reports `unavailable: …` as `binary_sha256` when the
//!   executable cannot be read.

use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// 2026-09-26: The SHA-256 of a server's arguments (everything after the
/// program name), each NUL-terminated so `["a b"]` and `["a", "b"]` differ.
#[must_use]
pub fn argv_fingerprint(args_after_program: &[String]) -> String {
    let mut h = Sha256::new();
    for a in args_after_program {
        h.update(a.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// 2026-09-26: The SHA-256 of a file's bytes; used as a binary's identity.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", h.finalize()))
}

/// 2026-09-26: The identity of this process, computed once: the digests of
/// its executable, its arguments after the program name and its levers, and
/// its pid.
pub fn this_process() -> &'static ServeIdentity {
    static ID: std::sync::OnceLock<ServeIdentity> = std::sync::OnceLock::new();
    ID.get_or_init(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        ServeIdentity {
            argv_sha256: argv_fingerprint(&args),
            binary_sha256: std::env::current_exe()
                .context("current_exe")
                .and_then(|p| file_sha256(&p))
                .unwrap_or_else(|e| format!("unavailable: {e:#}")),
            env_sha256: crate::serve_env::fingerprint(&crate::serve_env::process_levers()),
            pid: std::process::id(),
        }
    })
}

/// 2026-09-26: What `GET /serve-config` answers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServeIdentity {
    pub argv_sha256: String,
    pub binary_sha256: String,
    /// 2026-09-26: The digest of the `METRALE_*` serve levers in this
    /// server's environment (`serve_env::fingerprint` over
    /// `serve_env::process_levers()`). The argv says which recipe rendering
    /// the server runs; this says which levers it reads beside it.
    ///
    /// `serde(default)`, so a report without the field still parses, as the
    /// empty string, which [`env_is_unknown`] treats as unknown and the reuse
    /// check refuses: "cannot tell" must not read as "matches".
    #[serde(default)]
    pub env_sha256: String,
    pub pid: u32,
}

/// 2026-09-26: Whether a reported lever digest carries no information: the
/// empty string, which a report without [`ServeIdentity::env_sha256`] parses
/// to. A real digest is 64 hex characters even for an empty lever set.
#[must_use]
pub fn env_is_unknown(reported: &str) -> bool {
    reported.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprint_separates_arguments_and_orders_them() {
        let a = argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "1".into()]);
        let b = argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "2".into()]);
        let c = argv_fingerprint(&["serve m".into(), "--port".into(), "1".into()]);
        assert_ne!(a, b, "a different port is a different server");
        assert_ne!(a, c, "joined arguments are not the same argv");
        assert_eq!(
            a,
            argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "1".into()])
        );
        assert_eq!(a.len(), 64);
    }

    /// 2026-09-26: The identity carries the lever digest, and a report
    /// without the field still parses, with a digest that never equals a
    /// real one.
    #[test]
    fn the_identity_carries_the_lever_digest_and_an_older_server_reports_none() {
        let id = this_process();
        assert_eq!(
            id.env_sha256,
            crate::serve_env::fingerprint(&crate::serve_env::process_levers())
        );
        assert_eq!(id.env_sha256.len(), 64);
        let old: ServeIdentity =
            serde_json::from_str(r#"{"argv_sha256":"a","binary_sha256":"b","pid":1}"#).unwrap();
        assert_eq!(old.env_sha256, "");
        assert_ne!(
            old.env_sha256,
            crate::serve_env::fingerprint(&Default::default()),
            "even an empty lever set has a digest an old server cannot claim"
        );
    }

    #[test]
    fn a_file_digest_is_its_bytes() {
        let dir = std::env::temp_dir().join(format!("serve-identity-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bin");
        std::fs::write(&p, b"hello").unwrap();
        assert_eq!(
            file_sha256(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        std::fs::write(&p, b"hellO").unwrap();
        assert_ne!(
            file_sha256(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
