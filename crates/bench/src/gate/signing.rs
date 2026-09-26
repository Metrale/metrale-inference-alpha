// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Ed25519 signatures binding a gate record to the commit it
//! measured.
//!
//! The key is generated on the box on first use, so a signature does not
//! prove who ran a benchmark. It proves that the record file and the sha
//! passed to [`verify_record`] are the ones the key holder signed: editing
//! the file or re-pointing it at another commit breaks the signature. A new
//! signer's public key is a one-line `.pub` file under
//! `.github/record-signers/`, which a reviewer sees in the diff.
//!
//! Owner: bench gate (signing).
//! Invariants:
//! - Private key material is written only by `write_private`. On unix it
//!   creates a new file with mode `0600`; elsewhere it is a plain write.
//! - [`verify_record`] returns `Ok` only for a valid signature under a key
//!   file present in [`REGISTRY_DIR`], or for an unsigned record recorded
//!   before [`SIGNATURE_REQUIRED_AFTER`].

use anyhow::{Context, Result, bail};
use ring::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};
use std::path::{Path, PathBuf};

/// 2026-09-26: Where a signer's public key is published, relative to the repo
/// root.
pub const REGISTRY_DIR: &str = ".github/record-signers";

/// 2026-09-26: Sidecar format version. `sign_record` writes it and
/// `verify_record` refuses any other.
const SIG_VERSION: u32 = 1;

/// 2026-09-26: Unix seconds (2026-09-01T13:13:20Z). A record whose
/// `recorded_at` is below this and that has no `.sig` verifies as
/// [`Verified::Exempt`]; at or after it, a `.sig` is required.
/// `signing_tests::no_committed_record_escapes_the_cutover_unsigned` checks
/// every committed record against it.
pub const SIGNATURE_REQUIRED_AFTER: u64 = 1_788_268_400;

/// 2026-09-26: The signing identity for this machine.
pub struct Identity {
    keypair: Ed25519KeyPair,
    fingerprint: String,
}

impl Identity {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn public_key_bytes(&self) -> &[u8] {
        self.keypair.public_key().as_ref()
    }
}

/// 2026-09-26: Short name for a public key: the first 16 hex digits (8 bytes)
/// of its SHA-256. It names the key file and appears in messages; the full key
/// is what verifies.
pub fn fingerprint_of(public_key: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(public_key);
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// 2026-09-26: `<record>.json` → `<record>.json.sig`: appends `.sig` to the
/// whole file name.
pub fn sig_path(record: &Path) -> PathBuf {
    let mut s = record.to_path_buf().into_os_string();
    s.push(".sig");
    PathBuf::from(s)
}

/// 2026-09-26: The bytes a signature covers: the record file exactly as
/// written, then the commit sha.
fn message(record_bytes: &[u8], git_sha: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(record_bytes.len() + git_sha.len());
    msg.extend_from_slice(record_bytes);
    msg.extend_from_slice(git_sha.as_bytes());
    msg
}

/// 2026-09-26: Load this machine's identity from
/// `<metrale_home>/identity/ed25519.pk8`, generating and writing one on first
/// use. Prints nothing; an unreadable, unwritable or unusable key is an error.
pub fn load_or_create(metrale_home: &Path) -> Result<Identity> {
    let dir = metrale_home.join("identity");
    let key_path = dir.join("ed25519.pk8");

    let pkcs8 = if key_path.exists() {
        std::fs::read(&key_path).with_context(|| format!("reading {}", key_path.display()))?
    } else {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let rng = ring::rand::SystemRandom::new();
        let doc = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|_| anyhow::anyhow!("generating an Ed25519 key"))?;
        write_private(&key_path, doc.as_ref())?;
        doc.as_ref().to_vec()
    };

    let keypair = Ed25519KeyPair::from_pkcs8(&pkcs8)
        .map_err(|e| anyhow::anyhow!("{} is not a usable Ed25519 key: {e}", key_path.display()))?;
    let fingerprint = fingerprint_of(keypair.public_key().as_ref());
    Ok(Identity {
        keypair,
        fingerprint,
    })
}

/// 2026-09-26: Write key material to a new file created with mode `0600`, so
/// the key is never readable by others between creation and a later chmod.
#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    f.write_all(bytes)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

/// 2026-09-26: Write this identity's public key to
/// `<root>/.github/record-signers/<fingerprint>.pub` unless that file exists.
/// Returns whether it wrote, so the caller asks for a commit only once.
pub fn register(root: &Path, identity: &Identity) -> Result<bool> {
    let dir = root.join(REGISTRY_DIR);
    let path = dir.join(format!("{}.pub", identity.fingerprint()));
    if path.exists() {
        return Ok(false);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let armored = format!(
        "# Metrale Engine record signer {}\n# Ed25519 public key, base64. Added automatically on first use.\n{}\n",
        identity.fingerprint(),
        b64(identity.public_key_bytes())
    );
    std::fs::write(&path, armored).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

/// 2026-09-26: Fingerprints of the `.pub` files git tracks under
/// `.github/record-signers/`. Asks `git ls-files`, so a key `register` wrote
/// but nobody committed is absent. Errors when git cannot answer.
pub fn committed_signers(root: &Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--", REGISTRY_DIR])
        .output()
        .with_context(|| format!("listing tracked files under {REGISTRY_DIR}"))?;
    if !out.status.success() {
        bail!(
            "git ls-files {REGISTRY_DIR} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.rsplit('/').next())
        .filter_map(|f| f.strip_suffix(".pub"))
        .map(str::to_owned)
        .collect())
}

/// 2026-09-26: The operator notice when `fingerprint` is not among the
/// `committed` signers, or `None` when it is. Pure.
pub fn signer_notice(committed: &[String], fingerprint: &str) -> Option<String> {
    if committed.iter().any(|c| c == fingerprint) {
        return None;
    }
    Some(format!(
        "gate: NOTE — this box signs as {fingerprint}, which is not committed in \
         {REGISTRY_DIR} ({} signer(s) are).\n\
         gate: a record signed by an uncommitted key needs its one-line .pub \
         committed alongside, AND every record one PR adds must carry the SAME \
         fingerprint — CI rejects a set spanning signers, so a campaign split \
         across boxes cannot be certified.",
        committed.len()
    ))
}

/// 2026-09-26: Sign a record file already on disk with its sha; writes and
/// returns the `.sig` sidecar path.
pub fn sign_record(identity: &Identity, record: &Path, git_sha: &str) -> Result<PathBuf> {
    let bytes = std::fs::read(record).with_context(|| format!("reading {}", record.display()))?;
    let sig = identity.keypair.sign(&message(&bytes, git_sha));
    let out = sig_path(record);
    let body = format!(
        "{{\"v\":{SIG_VERSION},\"key\":\"{}\",\"sig\":\"{}\"}}\n",
        identity.fingerprint(),
        b64(sig.as_ref())
    );
    std::fs::write(&out, body).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

/// 2026-09-26: The outcome of a successful [`verify_record`].
#[derive(Debug, PartialEq, Eq)]
pub enum Verified {
    /// 2026-09-26: The signature verifies under a key in [`REGISTRY_DIR`].
    Signed { fingerprint: String },
    /// 2026-09-26: No `.sig`, and recorded before
    /// [`SIGNATURE_REQUIRED_AFTER`].
    Exempt,
}

/// 2026-09-26: Verify a record's `.sig` sidecar against `git_sha`.
///
/// `recorded_at` is used only for the [`SIGNATURE_REQUIRED_AFTER`] exemption.
/// Errors name the cause: no sidecar after the cutover, an unknown format
/// version, a key file missing from [`REGISTRY_DIR`], or a signature that
/// does not match. `check_one` adds any error to the record's failures.
pub fn verify_record(
    root: &Path,
    record: &Path,
    git_sha: &str,
    recorded_at: u64,
) -> Result<Verified> {
    let sidecar = sig_path(record);
    if !sidecar.exists() {
        if recorded_at < SIGNATURE_REQUIRED_AFTER {
            return Ok(Verified::Exempt);
        }
        bail!(
            "{} has no signature ({}). Re-run the benchmark, or commit the .sig \
             the CLI wrote beside the record.",
            record.file_name().unwrap_or_default().to_string_lossy(),
            sidecar.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    let raw = std::fs::read_to_string(&sidecar)
        .with_context(|| format!("reading {}", sidecar.display()))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", sidecar.display()))?;
    let version = parsed.get("v").and_then(serde_json::Value::as_u64);
    if version != Some(u64::from(SIG_VERSION)) {
        bail!(
            "{} is signature format v{:?}; this build understands v{SIG_VERSION}",
            sidecar.display(),
            version
        );
    }
    let fingerprint = parsed
        .get("key")
        .and_then(serde_json::Value::as_str)
        .context("signature names no key")?;
    let sig = unb64(
        parsed
            .get("sig")
            .and_then(serde_json::Value::as_str)
            .context("signature has no sig field")?,
    )?;

    let key_path = root.join(REGISTRY_DIR).join(format!("{fingerprint}.pub"));
    if !key_path.exists() {
        bail!(
            "{} is signed by {fingerprint}, which is not in {REGISTRY_DIR}. A new \
             signer must be committed there — it is one line, and reviewing it is \
             the point.",
            record.file_name().unwrap_or_default().to_string_lossy()
        );
    }
    let public = read_registered_key(&key_path)?;

    let bytes = std::fs::read(record).with_context(|| format!("reading {}", record.display()))?;
    UnparsedPublicKey::new(&ED25519, &public)
        .verify(&message(&bytes, git_sha), &sig)
        .map_err(|_| {
            anyhow::anyhow!(
                "{} does not match its signature. The record, or the commit it \
                 names, was changed after it was measured.",
                record.file_name().unwrap_or_default().to_string_lossy()
            )
        })?;

    Ok(Verified::Signed {
        fingerprint: fingerprint.to_string(),
    })
}

/// 2026-09-26: The base64 key on the first line of a registered key file
/// that is neither blank nor a `#` comment.
fn read_registered_key(path: &Path) -> Result<Vec<u8>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .with_context(|| format!("{} holds no key line", path.display()))?;
    unb64(line)
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn unb64(s: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .context("decoding base64")
}
