// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the log writer and tee accessors in `init.rs`, without installing a subscriber.
//!
//! `install_tty_subscriber` sets the global dispatcher and fills a
//! `OnceLock`, so no test calls it.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Puts the process-global [`TUI_ACTIVE`] back to its saved value on drop.
struct RestoreActive(bool);

impl Drop for RestoreActive {
    fn drop(&mut self) {
        TUI_ACTIVE.store(self.0, Ordering::Relaxed);
    }
}

/// 2026-09-26: The only test that toggles the process-global [`TUI_ACTIVE`], so parallel tests cannot race on it.
///
/// Non-empty lines are written only while the flag is set, so nothing reaches
/// the harness's stdout.
#[test]
fn the_writer_reports_every_byte_as_written_and_never_fails() {
    // 2026-09-26: `SwitchableIo::write` reports the whole buffer written, whatever the tee or stdout did.
    let restore = RestoreActive(TUI_ACTIVE.load(Ordering::Relaxed));
    TUI_ACTIVE.store(true, Ordering::Relaxed);

    let line = b"INFO metrale: a log line\n";
    assert_eq!(
        SwitchableIo.write(line).expect("the writer cannot fail"),
        line.len()
    );
    SwitchableIo.flush().expect("flush cannot fail");

    use tracing_subscriber::fmt::MakeWriter as _;
    let mut a = SwitchableWriter.make_writer();
    let mut b = SwitchableWriter.make_writer();
    assert_eq!(a.write(line).expect("write"), line.len());
    assert_eq!(b.write(line).expect("write"), line.len());

    TUI_ACTIVE.store(false, Ordering::Relaxed);
    assert_eq!(SwitchableIo.write(b"").expect("empty is fine"), 0);
    SwitchableIo.flush().expect("flush cannot fail");

    // 2026-09-26: Dropping the claim clears the flag without an explicit call.
    {
        let _claim = ActiveClaim::claim();
        assert!(
            TUI_ACTIVE.load(Ordering::SeqCst),
            "a claim means the TUI owns the screen and logs stay off stdout"
        );
    }
    assert!(
        !TUI_ACTIVE.load(Ordering::SeqCst),
        "dropping the claim hands stdout back — the two bail-out paths in the \
         event loop depend on this happening without being asked"
    );
    drop(restore);
}

#[test]
fn with_no_tee_installed_there_is_nothing_to_name_and_no_fd_to_redirect() {
    if TEE.get().is_none() {
        assert!(tee_file_path().is_none(), "nothing to name");
        assert!(tee_raw_fd().is_none(), "and no fd to redirect stderr onto");
    }
    flush_tee();
}

#[test]
fn the_tee_path_follows_its_environment_override_when_one_is_set() {
    match std::env::var("METRALE_TUI_LOG_FILE") {
        Ok(explicit) => assert_eq!(tee_path(), std::path::PathBuf::from(explicit)),
        Err(_) => {
            let p = tee_path();
            let name = p.file_name().expect("a file name").to_string_lossy();
            assert!(name.starts_with("met-serve-"), "{name}");
            assert!(
                name.contains(&std::process::id().to_string()),
                "named by pid so concurrent runs do not collide: {name}"
            );
            assert!(name.ends_with(".log"), "{name}");
            assert!(
                p.parent()
                    .expect("a parent")
                    .ends_with(".cache/metrale/logs"),
                "{}",
                p.display()
            );
        }
    }
}

#[test]
fn two_tee_paths_taken_in_the_same_second_are_the_same_file() {
    if std::env::var("METRALE_TUI_LOG_FILE").is_err() {
        let a = tee_path();
        let b = tee_path();
        assert_eq!(
            a.parent(),
            b.parent(),
            "the directory is fixed even if the second ticks over"
        );
    }
}

#[test]
fn the_default_filter_is_info_when_the_environment_says_nothing() {
    if std::env::var("RUST_LOG").is_err() {
        let spec = env_filter().to_string();
        assert!(spec.contains("info"), "{spec}");
    }
}
