// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `bench_lease`: the reuse decision, the lease file,
//! orphan release and the boot-failure message.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use super::*;

fn lease() -> Lease {
    Lease {
        pid: 4242,
        port: 40001,
        model: "Qwen/Qwen3.8-27B".into(),
        recipe_id: "qwen/qwen3.8-27b".into(),
        argv_sha256: "a".repeat(64),
        binary_sha256: "b".repeat(64),
        env_sha256: "e".repeat(64),
        owner_pid: 1,
        started_at: 0,
    }
}

fn reported() -> ServeIdentity {
    ServeIdentity {
        argv_sha256: "a".repeat(64),
        binary_sha256: "b".repeat(64),
        env_sha256: "e".repeat(64),
        pid: 4242,
    }
}

fn want() -> Expected {
    Expected {
        argv_sha256: "a".repeat(64),
        binary_sha256: "b".repeat(64),
        env_sha256: "e".repeat(64),
    }
}

/// 2026-09-26: Reused when pid, binary, rendering and model all match; each
/// negative control changes one of them and is refused with its own message.
#[test]
fn a_server_is_reused_only_when_every_digest_matches() {
    let want = want();
    assert_eq!(
        mismatch(&lease(), &reported(), &want, "Qwen/Qwen3.8-27B"),
        None
    );

    let mut r = reported();
    r.pid = 4243;
    assert!(
        mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B")
            .unwrap()
            .contains("pid 4243")
    );

    let mut r = reported();
    r.binary_sha256 = "c".repeat(64);
    assert!(
        mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B")
            .unwrap()
            .contains("another binary")
    );

    let hermetic = Expected {
        argv_sha256: "d".repeat(64),
        ..want.clone()
    };
    assert!(
        mismatch(&lease(), &reported(), &hermetic, "Qwen/Qwen3.8-27B")
            .unwrap()
            .contains("another rendering")
    );

    assert!(
        mismatch(&lease(), &reported(), &want, "Qwen/Qwen3.6-35B")
            .unwrap()
            .contains("this run needs")
    );
}

/// 2026-09-26: Levers never reach argv, so argv, binary and model can all
/// match while the server runs under other `METRALE_*` levers. The server's
/// own lever digest is compared, never the lease's copy.
#[test]
fn a_server_under_another_serve_environment_is_refused() {
    let want = want();

    let mut r = reported();
    r.env_sha256 = "f".repeat(64);
    let why = mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B").expect("refused");
    assert!(why.contains("another METRALE_* serve environment"), "{why}");
    assert!(why.contains("qwen/qwen3.8-27b"), "names the recipe: {why}");
    assert!(
        why.contains("env-only"),
        "the message must say argv and binary matched, so a reader is not sent \
         looking for a recipe difference that does not exist: {why}"
    );

    // 2026-09-26: A server that reports "" is refused as unknown, with a message
    // distinct from the mismatch one.
    let mut r = reported();
    r.env_sha256 = String::new();
    let why = mismatch(&lease(), &r, &want, "Qwen/Qwen3.8-27B").expect("refused");
    assert!(
        why.contains("does not report its METRALE_* serve environment"),
        "{why}"
    );
    assert!(!why.contains("env-only"), "{why}");

    // 2026-09-26: A different lease digest beside a matching server report is
    // still reused: the lease's copy does not decide.
    let mut stale = lease();
    stale.env_sha256 = "0".repeat(64);
    assert_eq!(
        mismatch(&stale, &reported(), &want, "Qwen/Qwen3.8-27B"),
        None
    );
}

/// 2026-09-26: The lease round-trips through its file, and a file that is
/// not a lease is an error rather than "no lease".
#[test]
fn the_lease_file_round_trips_and_a_bad_one_is_refused() {
    let dir = std::env::temp_dir().join(format!("serve-lease-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = ArtifactStore::with_root(dir.clone());
    assert_eq!(read(&store).unwrap(), None);
    write(&store, &lease()).unwrap();
    assert_eq!(read(&store).unwrap(), Some(lease()));
    std::fs::write(lease_path(&store), "not json").unwrap();
    assert!(read(&store).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2026-09-26: A lease whose owner is dead is released; one whose owner lives
/// is kept. Pid 1 is always alive; the large pids (near 4,000,000,000) are far
/// above `/proc/sys/kernel/pid_max`, which read 4,194,304 on 2026-09-26.
#[test]
#[cfg(target_os = "linux")]
fn an_orphaned_lease_is_released_and_a_live_one_kept() {
    let dir = std::env::temp_dir().join(format!("serve-lease-orphan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = ArtifactStore::with_root(dir.clone());
    // 2026-09-26: The server pid is dead too, so release signals nothing real.
    let dead_server = Lease {
        pid: 4_000_000_000 - 7,
        owner_pid: 1,
        ..lease()
    };
    write(&store, &dead_server).unwrap();
    assert_eq!(release_if_orphaned(&store).unwrap(), None);
    assert!(lease_path(&store).exists());
    let orphan = Lease {
        owner_pid: 4_000_000_000 - 9,
        ..dead_server
    };
    write(&store, &orphan).unwrap();
    assert_eq!(release_if_orphaned(&store).unwrap(), Some(orphan));
    assert!(!lease_path(&store).exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2026-09-26: A lease file without `env_sha256` reads with an empty digest,
/// and a full lease round-trips it.
#[test]
fn an_older_lease_file_reads_with_an_empty_env_digest() {
    let old: Lease = serde_json::from_str(
        r#"{"pid":1,"port":2,"model":"m","recipe_id":"r","argv_sha256":"a","binary_sha256":"b","owner_pid":3,"started_at":4}"#,
    )
    .unwrap();
    assert_eq!(old.env_sha256, "");
    let back: Lease = serde_json::from_str(&serde_json::to_string(&lease()).unwrap()).unwrap();
    assert_eq!(back.env_sha256, "e".repeat(64));
}

/// 2026-09-26: A leased serve that dies at boot quotes its own final `Error:`
/// block, indented, and says so when the log tail has none.
#[test]
fn a_serve_that_dies_at_boot_reports_its_own_cause() {
    let tail = "native FP8 dense residency: weights 23.42 GB ...\n\
Error: Failed to build model\n\n\
Caused by:\n    No memory left for KV cache: total GPU = 121.7 GB, \
--gpu-memory-utilization 70% → budget 85.2 GB, but 69.9 GB already consumed + \
23.7 GB inference reserve = 93.7 GB committed.\n";
    let msg = exited_before_serving("exit status: 1", "unsloth/Qwen3.8-27B-NVFP4", tail);
    assert!(
        msg.starts_with("the leased server exited (exit status: 1) before it began serving"),
        "{msg}"
    );
    assert!(msg.contains("No memory left for KV cache"), "{msg}");
    assert!(
        msg.contains("    Error: Failed to build model"),
        "the block is indented: {msg}"
    );
    assert!(
        !msg.contains("native FP8 dense residency"),
        "only the error block: {msg}"
    );
    // 2026-09-26: Negative control: a tail with no error block is not quoted as one.
    let msg = exited_before_serving("signal: 9", "m", "loading shard 3/17\n");
    assert!(msg.contains("carries no `Error:` block"), "{msg}");
    assert!(!msg.contains("loading shard"), "{msg}");
}
