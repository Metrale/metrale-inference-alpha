// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Offline tests of the download module: publish ordering, resolver agreement, error hints, the space check, `.part` naming and token precedence.
//!
//! Owner: server (model download).
//! Invariants: none beyond the types.

use super::*;
use std::path::Path;

/// 2026-09-26: A temp cache root, removed on drop.
struct Cache(PathBuf);

impl Cache {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("metrale-dl-{name}"));
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

/// 2026-09-26: Write a snapshot's files by hand, without `refs/main`.
fn place(cache: &Path, repo: &str, rev: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let snap = hf::repo_dir(cache, repo).join("snapshots").join(rev);
    std::fs::create_dir_all(&snap).unwrap();
    for (name, body) in files {
        std::fs::write(snap.join(name), body).unwrap();
    }
    snap
}

#[test]
fn refs_main_is_not_written_for_a_partial_download() {
    let c = Cache::new("partial");
    place(&c.0, "org/m", "rev1", &[("config.json", b"{}")]);
    assert!(
        hf::publish(&c.0, "org/m", "rev1").is_err(),
        "a snapshot without weights must not be published"
    );
    assert!(
        hf::local_revision(&c.0, "org/m").is_none(),
        "refs/main must not exist after a refused publish"
    );
}

#[test]
fn refs_main_is_not_written_without_a_config() {
    let c = Cache::new("noconfig");
    place(&c.0, "org/m", "rev1", &[("model.safetensors", b"w")]);
    assert!(hf::publish(&c.0, "org/m", "rev1").is_err());
    assert!(hf::local_revision(&c.0, "org/m").is_none());
}

#[test]
fn a_complete_snapshot_publishes_and_reads_back() {
    let c = Cache::new("complete");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[("config.json", b"{}"), ("model.safetensors", b"weights")],
    );
    hf::publish(&c.0, "org/m", "rev1").expect("a complete snapshot publishes");
    assert_eq!(hf::local_revision(&c.0, "org/m").as_deref(), Some("rev1"));
}

#[test]
fn a_published_download_is_one_the_resolver_accepts() {
    // 2026-09-26: The download's cache layout and `model_resolver` must agree.
    let c = Cache::new("resolves");
    let snap = place(
        &c.0,
        "org/m",
        "rev1",
        &[("config.json", b"{}"), ("model.safetensors", b"weights")],
    );
    hf::publish(&c.0, "org/m", "rev1").unwrap();

    let resolved = crate::model_resolver::resolve_model_dir("org/m", Some(&c.0))
        .expect("the resolver must accept what we just wrote");
    assert_eq!(resolved, snap);
}

#[test]
fn the_resolver_refuses_a_model_whose_publish_never_happened() {
    let c = Cache::new("unpublished");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[("config.json", b"{}"), ("model.safetensors", b"w")],
    );
    // 2026-09-26: Every file is present, but `refs/main` was never written.
    assert!(
        crate::model_resolver::resolve_model_dir("org/m", Some(&c.0)).is_err(),
        "without refs/main the model is not finished and must not load"
    );
}

#[test]
fn repo_dir_matches_the_hub_naming_the_resolver_expects() {
    let c = Cache::new("naming");
    assert_eq!(
        hf::repo_dir(&c.0, "nvidia/Qwen3.6-27B-NVFP4")
            .file_name()
            .unwrap(),
        "models--nvidia--Qwen3.6-27B-NVFP4"
    );
}

#[test]
fn every_error_names_an_action() {
    let errs = [
        DownloadError::Offline("no route".into()),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: false,
        },
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: true,
        },
        DownloadError::NotFound {
            repo: "org/m".into(),
        },
        DownloadError::RateLimited,
        DownloadError::DiskFull,
        DownloadError::NotEnoughSpace {
            need: 28_100_000_000,
            free: 6_400_000_000,
        },
        DownloadError::NoSafetensors {
            repo: "org/m".into(),
        },
        DownloadError::Http {
            repo: "org/m".into(),
            status: 500,
        },
        DownloadError::Io("disk on fire".into()),
    ];
    for e in errs {
        let h = e.hint();
        assert!(!h.is_empty(), "{e:?} has no hint");
        assert!(h.len() > 15, "{e:?} hint is too terse to act on: {h}");
    }
}

#[test]
fn a_gated_repo_reads_differently_with_and_without_a_token() {
    // 2026-09-26: Without a token the hint says to log in; with one it says to
    // accept the licence.
    let without = DownloadError::Gated {
        repo: "org/m".into(),
        had_token: false,
    }
    .hint();
    let with = DownloadError::Gated {
        repo: "org/m".into(),
        had_token: true,
    }
    .hint();
    assert_ne!(without, with);
    assert!(
        without.contains("HF_TOKEN") || without.contains("login"),
        "{without}"
    );
    assert!(
        with.contains("licence") || with.contains("accepted"),
        "{with}"
    );
}

#[test]
fn not_enough_space_states_both_numbers() {
    let h = DownloadError::NotEnoughSpace {
        need: 28_100_000_000,
        free: 6_400_000_000,
    }
    .hint();
    assert!(h.contains("28.1"), "{h}");
    assert!(h.contains("6.4"), "{h}");
}

#[test]
fn free_bytes_reports_something_for_a_real_directory() {
    let c = Cache::new("statvfs");
    let free = hf::free_bytes(&c.0).expect("temp dir is on a real filesystem");
    assert!(free > 0, "a writable filesystem has some space");
}

#[test]
fn free_bytes_measures_the_cache_root_even_before_it_exists() {
    // 2026-09-26: Before a first download the cache path does not exist yet;
    // the measurement walks up to an existing ancestor.
    let c = Cache::new("statvfs-missing");
    let not_yet = c.0.join("models--org--m/snapshots/rev1");
    assert!(!not_yet.exists());
    let free = hf::free_bytes(&not_yet).expect("walks up to a real ancestor");
    assert!(free > 0);
    let root_free = hf::free_bytes(&c.0).expect("root exists");
    assert!(
        free.abs_diff(root_free) < root_free / 100,
        "the answer must describe the filesystem the files will land on"
    );
}

#[test]
fn free_bytes_is_none_only_when_nothing_up_the_tree_can_be_measured() {
    // 2026-09-26: "/" always exists, so any absolute path is measurable.
    assert!(hf::free_bytes(Path::new("/definitely/not/here/at/all")).is_some());
}

#[test]
fn the_space_check_refuses_only_when_it_genuinely_will_not_fit() {
    use super::fits;
    assert!(fits(100, 0, 100).is_ok());
    match fits(101, 0, 100) {
        Err(DownloadError::NotEnoughSpace { need, free }) => {
            assert_eq!((need, free), (101, 100));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    // 2026-09-26: Only the remainder of a resumed download must fit.
    assert!(fits(1_000, 950, 100).is_ok());
    // 2026-09-26: More on disk than the total must not underflow `need`.
    assert!(fits(100, 250, 0).is_ok());
}

#[test]
fn part_files_cannot_collide_between_siblings() {
    // 2026-09-26: `with_extension("part")` would map both names onto one
    // `.part`; `part_path` appends instead.
    let dir = Path::new("/tmp/x");
    let a = super::hf::part_path(&dir.join("model-00001-of-00002.safetensors"));
    let b = super::hf::part_path(&dir.join("model-00001-of-00002.json"));
    assert_ne!(a, b, "sibling files must not share a .part");
    assert!(a.to_string_lossy().ends_with(".safetensors.part"), "{a:?}");

    let c = super::hf::part_path(&dir.join("model.safetensors-00001-of-00001.safetensors"));
    let d = super::hf::part_path(&dir.join("model.safetensors.index.json"));
    assert_ne!(c, d);
}

#[test]
fn an_index_without_its_shards_is_not_publishable() {
    // 2026-09-26: `model_resolver::snapshot_has_weights` accepts an index
    // alone; `publish` needs a real shard.
    let c = Cache::new("indexonly");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[
            ("config.json", b"{}"),
            ("model.safetensors.index.json", b"{}"),
        ],
    );
    assert!(crate::model_resolver::snapshot_has_weights(
        &hf::repo_dir(&c.0, "org/m").join("snapshots").join("rev1")
    ));
    assert!(
        hf::publish(&c.0, "org/m", "rev1").is_err(),
        "an index naming absent shards is not a downloaded model"
    );
    assert!(hf::local_revision(&c.0, "org/m").is_none());
}

#[test]
fn a_part_file_does_not_count_as_a_shard() {
    let c = Cache::new("partonly");
    place(
        &c.0,
        "org/m",
        "rev1",
        &[
            ("config.json", b"{}"),
            ("model-00001-of-00001.safetensors.part", b"half"),
        ],
    );
    assert!(hf::publish(&c.0, "org/m", "rev1").is_err());
}

#[test]
fn the_token_precedence_rule_prefers_the_environment_then_the_login_file() {
    use super::hf::pick_token;
    let some = |s: &str| Some(s.to_string());

    assert_eq!(pick_token(&[None, None], None), None);

    assert_eq!(
        pick_token(&[None, None], Some("from-file")).as_deref(),
        Some("from-file")
    );

    // 2026-09-26: Order: the first variable, the second, then the file.
    assert_eq!(
        pick_token(&[some("env-a"), None], Some("from-file")).as_deref(),
        Some("env-a")
    );
    assert_eq!(
        pick_token(&[some("env-a"), some("env-b")], None).as_deref(),
        Some("env-a")
    );
    assert_eq!(
        pick_token(&[None, some("env-b")], None).as_deref(),
        Some("env-b")
    );
}

#[test]
fn a_blank_token_reads_as_absent_rather_than_as_a_credential() {
    use super::hf::pick_token;
    assert_eq!(pick_token(&[Some(String::new()), None], None), None);
    assert_eq!(pick_token(&[Some("   ".into()), None], None), None);
    assert_eq!(pick_token(&[None, None], Some("\n")), None);
    // 2026-09-26: Surrounding whitespace is trimmed from a real token.
    assert_eq!(
        pick_token(&[None, None], Some("hf_realtoken\n")).as_deref(),
        Some("hf_realtoken")
    );
    // 2026-09-26: A blank variable does not shadow the file.
    assert_eq!(
        pick_token(&[Some("  ".into()), None], Some("hf_realtoken")).as_deref(),
        Some("hf_realtoken")
    );
}

#[test]
fn hub_statuses_map_to_causes_a_reader_can_act_on() {
    use super::hf::classify;
    // 2026-09-26: 401 and 403 both map to `Gated`; `had_token` is carried
    // through, and it alone picks the hint.
    assert_eq!(
        classify("org/m", 401, false),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: false
        }
    );
    assert_eq!(
        classify("org/m", 403, true),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: true
        }
    );
    assert_eq!(
        classify("org/m", 404, false),
        DownloadError::NotFound {
            repo: "org/m".into()
        }
    );
    assert_eq!(classify("org/m", 429, false), DownloadError::RateLimited);
    // 2026-09-26: Any other status is kept as a number.
    assert_eq!(
        classify("org/m", 502, false),
        DownloadError::Http {
            repo: "org/m".into(),
            status: 502
        }
    );
}

#[test]
fn a_full_disk_is_reported_as_a_full_disk() {
    use super::hf::write_error;
    // 2026-09-26: Raw Linux errno values (ENOSPC 28, EACCES 13, EFBIG 27), so
    // the test also covers std's errno-to-`ErrorKind` mapping that
    // `write_error` relies on.
    assert_eq!(
        write_error(std::io::Error::from_raw_os_error(28)),
        DownloadError::DiskFull
    );
    match write_error(std::io::Error::from_raw_os_error(13)) {
        DownloadError::Io(m) => assert!(!m.is_empty(), "permission denied keeps its text"),
        other => panic!("EACCES is not a full disk: {other:?}"),
    }
    match write_error(std::io::Error::from_raw_os_error(27)) {
        DownloadError::Io(_) => {}
        other => panic!("EFBIG is not a full disk: {other:?}"),
    }
}
