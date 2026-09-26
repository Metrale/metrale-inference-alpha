// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Download tests against the real Hub; each is `#[ignore = "network"]` and runs only with `--ignored`.
//!
//! Owner: server (model download).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: A temp cache root, removed on drop.
struct Cache(PathBuf);

impl Cache {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("metrale-dlnet-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("temp cache");
        Self(p)
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 2026-09-26: Download a small public model and resolve it from the cache.
#[test]
#[ignore = "network"]
fn a_real_download_produces_a_model_the_resolver_accepts() {
    let c = Cache::new("real");
    let repo = "hf-internal-testing/tiny-random-gpt2";
    let h = start(repo, c.0.clone());

    let mut planned = None;
    let mut done = None;
    for msg in h.rx.iter() {
        match msg {
            DownloadMsg::Planned {
                files, total_bytes, ..
            } => {
                eprintln!("planned {files} files, {total_bytes} bytes");
                planned = Some(files);
            }
            DownloadMsg::Done { snapshot, revision } => {
                eprintln!("done {revision} -> {}", snapshot.display());
                done = Some(snapshot);
            }
            DownloadMsg::Failed(e) => panic!("download failed: {}", e.hint()),
            _ => {}
        }
    }
    assert!(planned.unwrap_or(0) > 0, "something was planned");
    let snap = done.expect("the download completed");
    assert!(snap.join("config.json").exists());

    let resolved = crate::model_resolver::resolve_model_dir(repo, Some(&c.0))
        .expect("a freshly downloaded model must resolve");
    assert_eq!(resolved, snap);
}

/// 2026-09-26: A download cancelled at once reports `Cancelled` and writes no
/// `refs/main`.
#[test]
#[ignore = "network"]
fn a_cancelled_download_leaves_no_refs_main() {
    let c = Cache::new("cancel");
    let repo = "hf-internal-testing/tiny-random-gpt2";
    let h = start(repo, c.0.clone());
    h.cancel();
    let mut cancelled = false;
    for msg in h.rx.iter() {
        match msg {
            DownloadMsg::Cancelled { .. } => cancelled = true,
            DownloadMsg::Done { .. } => panic!("a cancelled download must not publish"),
            _ => {}
        }
    }
    assert!(cancelled, "the worker reported the cancellation");
    assert!(
        hf::local_revision(&c.0, repo).is_none(),
        "a cancelled download must not be loadable"
    );
}

/// 2026-09-26: For a gated repo without a token, this expects `repo_info` to
/// succeed and `fetch_file` to fail as `Gated { had_token: false }`, before
/// anything is written.
#[test]
#[ignore = "network"]
fn a_gated_repo_is_reported_as_gated_and_not_as_a_network_fault() {
    const GATED: &str = "meta-llama/Llama-3.2-1B";

    let (revision, files) =
        hf::repo_info(GATED, None).expect("a gated repo still describes itself");
    assert!(!revision.is_empty());
    assert!(!files.is_empty());

    let c = Cache::new("gated");
    let dest = c.0.join("config.json");
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let err = hf::fetch_file(
        GATED,
        &revision,
        "config.json",
        &dest,
        None,
        &cancel,
        &mut |_| {},
    )
    .expect_err("without a token the weights must be refused");

    match &err {
        DownloadError::Gated { repo, had_token } => {
            assert_eq!(repo, GATED);
            assert!(!had_token, "no token was supplied");
        }
        other => panic!("a gated repo must not read as {other:?}"),
    }
    let hint = err.hint();
    assert!(
        hint.contains("HF_TOKEN") || hint.contains("login"),
        "gated hint must name how to authenticate: {hint}"
    );
    assert!(!dest.exists(), "nothing should have been written");
}

/// 2026-09-26: With an invalid token, a public repo still lists and downloads
/// (the anonymous retry in `hf::get_or_anon`).
#[test]
#[ignore = "network"]
fn a_stale_token_still_resolves_and_fetches_a_public_repo() {
    let repo = "hf-internal-testing/tiny-random-gpt2";
    let bad = Some("hf_thistokenisdefinitelynotvalid");

    let (revision, files) =
        hf::repo_info(repo, bad).expect("public metadata must survive a bad token");
    assert!(!revision.is_empty());
    assert!(!files.is_empty());

    let c = Cache::new("staletoken");
    let dest = c.0.join("config.json");
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let done = hf::fetch_file(
        repo,
        &revision,
        "config.json",
        &dest,
        bad,
        &cancel,
        &mut |_| {},
    )
    .expect("public files must survive a bad token");
    assert!(done);
    assert!(dest.exists());
}
