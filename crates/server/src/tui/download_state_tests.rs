// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `DownloadState`: refusal, cancel, the fraction, and
//! the freshness answers. Every accepted `start` spawns the real download
//! worker; only the two `#[ignore = "network"]` tests wait for its answer.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

fn root() -> std::path::PathBuf {
    let p = std::env::temp_dir().join("metrale-dlstate");
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn a_second_download_is_refused_by_name_not_queued() {
    let mut s = DownloadState::default();
    let (_, err) = s.start("org/first", root());
    assert!(!err, "the first one starts");

    let (text, err) = s.start("org/second", root());
    assert!(err, "the second is refused");
    assert!(text.contains("org/first"), "names the running one: {text}");
    assert!(text.contains('x'), "and how to stop it: {text}");
    assert!(s.is_downloading("org/first"));
    assert!(!s.is_downloading("org/second"));
}

#[test]
fn cancelling_twice_abandons_the_job() {
    // 2026-09-26: The first call sets the worker's cancel flag and marks the
    // job; the second stops tracking it.
    let mut s = DownloadState::default();
    s.start("org/m", root());

    let (text, _) = s.cancel().expect("something is running");
    assert!(text.contains("stopping"), "{text}");
    assert!(
        s.job.as_ref().is_some_and(|j| j.cancelling),
        "still tracked, now marked"
    );

    let (text, _) = s.cancel().expect("still running");
    assert!(text.contains("abandoned"), "{text}");
    assert!(s.job.is_none(), "no longer tracked");
    assert!(s.cancel().is_none(), "nothing left to cancel");
}

#[test]
fn cancelling_nothing_is_not_an_error() {
    let mut s = DownloadState::default();
    assert!(s.cancel().is_none());
}

#[test]
fn a_job_without_sizes_reports_no_fraction_rather_than_zero() {
    let mut s = DownloadState::default();
    s.start("org/m", root());
    let job = s.job.as_mut().unwrap();
    job.total = 0;
    job.done = 5_000;
    assert_eq!(job.fraction(), None);

    job.total = 10_000;
    job.done = 2_500;
    assert_eq!(job.fraction(), Some(0.25));
}

#[test]
fn the_fraction_never_exceeds_one() {
    let mut s = DownloadState::default();
    s.start("org/m", root());
    let job = s.job.as_mut().unwrap();
    job.total = 100;
    job.done = 250;
    assert_eq!(job.fraction(), Some(1.0));
}

#[test]
fn freshness_defaults_to_nothing_rather_than_current() {
    let s = DownloadState::default();
    assert!(!s.freshness.contains_key("org/m"));
}

#[test]
fn pump_with_no_job_is_quiet() {
    let mut s = DownloadState::default();
    assert!(s.pump().is_none());
}

#[test]
#[ignore = "network"]
fn a_failed_download_settles_and_explains_itself() {
    let mut s = DownloadState::default();
    s.start("definitely-not/a-real-model-xyz", root());

    let mut settled = None;
    for _ in 0..600 {
        if let Some(x) = s.pump() {
            settled = Some(x);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let settled = settled.expect("the job must reach a terminal state");
    match settled {
        Settled::Stopped(r) => assert_eq!(r, "definitely-not/a-real-model-xyz"),
        Settled::Finished(_) => panic!("a nonexistent repo must not report success"),
    }
    assert!(s.job.is_none(), "the job is cleared so the bar stops");
    let (text, error) = s.last_message.take().expect("it says what went wrong");
    assert!(error, "and marks it as an error: {text}");
    assert!(
        text.contains("definitely-not/a-real-model-xyz"),
        "naming the model: {text}"
    );
}

#[test]
#[ignore = "network"]
fn a_real_download_settles_as_finished() {
    let cache = std::env::temp_dir().join("metrale-dlstate-real");
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();

    let mut s = DownloadState::default();
    s.start("hf-internal-testing/tiny-random-gpt2", cache.clone());

    let mut settled = None;
    let mut saw_progress = false;
    for _ in 0..1200 {
        if let Some(job) = s.job.as_ref()
            && job.done > 0
        {
            saw_progress = true;
        }
        if let Some(x) = s.pump() {
            settled = Some(x);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    match settled.expect("the download must finish") {
        Settled::Finished(r) => assert_eq!(r, "hf-internal-testing/tiny-random-gpt2"),
        Settled::Stopped(_) => panic!("a real model should download"),
    }
    assert!(saw_progress, "progress must be observable while it runs");
    assert!(s.job.is_none());
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn a_download_for_a_model_not_being_downloaded_reads_as_not_downloading() {
    let mut s = DownloadState::default();
    s.start("org/m", root());
    assert!(!s.is_downloading("org/other"));
    assert!(!DownloadState::default().is_downloading("org/m"));
}

#[test]
fn a_freshness_answer_reports_every_outcome_not_only_stale() {
    use crate::model_download::stale::Freshness;
    let cases = [
        (Freshness::Current, "up to date", false),
        (
            Freshness::Stale {
                local: "aaaaaaaaaaaaaaaaaaaa".into(),
                remote: "bbbbbbbbbbbbbbbbbbbb".into(),
            },
            "has an update",
            false,
        ),
        (Freshness::Missing, "nothing on disk", false),
        (Freshness::Unknown, "could not reach the Hub", true),
    ];
    for (f, needle, wants_error_tone) in cases {
        let mut s = DownloadState::default();
        let (tx, rx) = std::sync::mpsc::channel();
        s.pending_check = Some(rx);
        s.checking = Some("org/m".into());
        tx.send(("org/m".to_string(), f.clone())).unwrap();
        s.pump();
        let (text, error) = s.last_message.take().expect("the check must say something");
        assert!(text.contains("org/m"), "names the model: {text}");
        assert!(text.contains(needle), "{f:?}: {text}");
        assert_eq!(error, wants_error_tone, "{f:?}: {text}");
        assert!(s.checking.is_none(), "the skeleton is cleared");
        assert!(s.pending_check.is_none());
        assert_eq!(
            s.freshness.get("org/m"),
            Some(&f),
            "the badge map still fills"
        );
    }
}

/// 2026-09-26: `Missing` names the `d` key; `Current` and `Unknown` do not.
#[test]
fn only_the_actionable_outcomes_name_the_d_key() {
    use crate::model_download::stale::Freshness;
    for (f, should_teach_d) in [
        (Freshness::Current, false),
        (Freshness::Missing, true),
        (Freshness::Unknown, false),
    ] {
        let mut s = DownloadState::default();
        let (tx, rx) = std::sync::mpsc::channel();
        s.pending_check = Some(rx);
        tx.send(("org/m".to_string(), f)).unwrap();
        s.pump();
        let (text, _) = s.last_message.take().unwrap();
        assert_eq!(text.contains("d downloads"), should_teach_d, "{text}");
    }
}

#[test]
fn a_freshness_worker_that_died_settles_out_loud() {
    use crate::model_download::stale::Freshness;
    let mut s = DownloadState::default();
    let (tx, rx) = std::sync::mpsc::channel::<(String, Freshness)>();
    s.pending_check = Some(rx);
    s.checking = Some("org/m".into());
    drop(tx);
    s.pump();
    assert!(s.pending_check.is_none(), "settled, not stuck checking");
    assert!(s.checking.is_none());
    let (text, error) = s.last_message.take().expect("says so");
    assert!(error, "{text}");
    assert!(text.contains("u retries"), "and names the way back: {text}");
}
