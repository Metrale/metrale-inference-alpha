// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Running one unit and classifying the record it produced.
//!
//! Owner: server CLI (`met benchmark certify`).
//! A local unit runs as a child `met benchmark run … --pull-request-gate`,
//! because a process self-starts at most one server
//! (`bench_selfstart::claim_start_slot`).
//! Invariants: `classify` reads the verdict only from the record; the exit code
//! only words the reason and decides whether a missing record is retryable.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::plan::Unit;

/// 2026-09-26: What the campaign hands the runner for every unit.
pub struct RunCtx<'a> {
    pub root: &'a Path,
    pub anchor: &'a str,
    pub hardware: &'a str,
    pub yes: bool,
    pub deadline: Duration,
    pub log_dir: &'a Path,
}

/// 2026-09-26: The facts about a record that decide a unit's outcome. Read by a
/// [`Records`] so the classification is testable without a real record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordFacts {
    pub path: PathBuf,
    pub git_sha: String,
    pub verdict_passes: bool,
    pub frame_completed: bool,
    /// 2026-09-26: The record names a shard and carries per-subset tallies.
    pub is_shard_with_tallies: bool,
}

/// 2026-09-26: Where a unit's newest record is read from.
pub trait Records: Send {
    /// 2026-09-26: The newest record for `id` and this `shard` recorded at or after
    /// `since` (unix secs). The shard identity is matched, never assumed:
    /// two shards of one group at one commit land in one directory.
    fn newest_since(
        &self,
        root: &Path,
        id: &str,
        shard: Option<(usize, usize)>,
        since: u64,
    ) -> Option<RecordFacts>;
}

/// 2026-09-26: The real `.benchmarks/<id>/` reader.
pub struct RepoRecords;

impl Records for RepoRecords {
    fn newest_since(
        &self,
        root: &Path,
        id: &str,
        shard: Option<(usize, usize)>,
        since: u64,
    ) -> Option<RecordFacts> {
        use metrale_bench::gate;
        gate::records_newest_first(root, id)
            .into_iter()
            .filter_map(|path| gate::read_record(&path).ok().map(|r| (path, r)))
            .find(|(_, r)| r.benchmark_id == id && r.shard() == shard && r.recorded_at >= since)
            .map(|(path, r)| facts_of(path, &r))
    }
}

/// 2026-09-26: The facts the classifier reads, from a parsed record.
pub fn facts_of(path: PathBuf, r: &metrale_bench::gate::GateRecord) -> RecordFacts {
    let tallies =
        metrale_bench::benchmarks::bfcl::aggregate::tallies_from_metrics(&r.metrics).is_some();
    RecordFacts {
        is_shard_with_tallies: r.shard().is_some() && tallies,
        verdict_passes: r.verdict_passes(),
        frame_completed: !r.frame_status_failed(),
        git_sha: r.git_sha.clone(),
        path,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// 2026-09-26: A passing record at the anchor.
    Passed {
        record: PathBuf,
    },
    /// 2026-09-26: A shard's record, with tallies, at the anchor; the group's verdict
    /// comes later.
    MemberDone {
        record: PathBuf,
    },
    /// 2026-09-26: A record at the anchor whose run failed or whose verdict does not pass.
    VerdictFail {
        record: Option<PathBuf>,
        reason: String,
    },
    /// 2026-09-26: No usable record: none was written, it names another commit, or the
    /// runner itself failed. `retryable` says whether a second attempt may help.
    Harness {
        reason: String,
        retryable: bool,
    },
    TimedOut,
    Cancelled,
}

/// 2026-09-26: Classify a finished unit. `killed_for` wins when set; `exit` is `None`
/// when the child ended by a signal.
pub fn classify(
    unit: &Unit,
    anchor: &str,
    exit: Option<i32>,
    record: Option<RecordFacts>,
    killed_for: Option<RunOutcome>,
) -> RunOutcome {
    if let Some(k) = killed_for {
        return k;
    }
    let Some(r) = record else {
        return RunOutcome::Harness {
            reason: format!(
                "the child exited {} and wrote no record for {} at this commit",
                exit.map_or("by signal".to_string(), |c| c.to_string()),
                unit.label()
            ),
            retryable: exit != Some(0),
        };
    };
    if !(r.git_sha.starts_with(anchor) || anchor.starts_with(&r.git_sha)) {
        return RunOutcome::Harness {
            reason: format!(
                "{} names commit {}, not the anchor {anchor}: the tree moved under the run",
                r.path.display(),
                r.git_sha
            ),
            retryable: false,
        };
    }
    if !r.frame_completed {
        return RunOutcome::VerdictFail {
            record: Some(r.path),
            reason: "the run itself failed (frame status Failed)".into(),
        };
    }
    if r.verdict_passes {
        return RunOutcome::Passed { record: r.path };
    }
    if unit.shard.is_some() && r.is_shard_with_tallies {
        return RunOutcome::MemberDone { record: r.path };
    }
    RunOutcome::VerdictFail {
        record: Some(r.path),
        reason: match exit {
            Some(2) => "the gate said no (exit 2)".into(),
            Some(0) => "the run recorded an Info verdict, which is not a pass".into(),
            other => format!("verdict is not PASS (exit {other:?})"),
        },
    }
}

pub trait GateRunner {
    /// 2026-09-26: Run one unit; `on_line` receives its output lines as they arrive.
    fn run(&mut self, unit: &Unit, ctx: &RunCtx, on_line: &mut dyn FnMut(&str)) -> RunOutcome;
}

/// 2026-09-26: Spawns `<exe> benchmark run <id> --pull-request-gate …` and reads the
/// record it leaves behind.
pub struct LocalChild {
    pub exe: PathBuf,
    pub records: Box<dyn Records>,
    /// 2026-09-26: The campaign's cancel flag; `supervise` checks it every loop.
    pub cancel: Arc<AtomicBool>,
    /// 2026-09-26: Extra arguments after the standard ones.
    pub extra_args: Vec<String>,
}

impl LocalChild {
    /// 2026-09-26: `--serve-reuse --serve-lease-owner <this pid>`, so consecutive units
    /// may share a leased server (`bench_lease`) and a lease whose driver is gone
    /// is released. Empty under `--no-serve-reuse`.
    pub fn reuse_args(no_serve_reuse: bool) -> Vec<String> {
        if no_serve_reuse {
            vec![]
        } else {
            vec![
                "--serve-reuse".to_string(),
                "--serve-lease-owner".to_string(),
                std::process::id().to_string(),
            ]
        }
    }

    pub fn argv(&self, unit: &Unit, ctx: &RunCtx) -> Vec<String> {
        let mut v = vec![
            "benchmark".to_string(),
            "run".to_string(),
            unit.id.to_string(),
            "--pull-request-gate".to_string(),
            "--hardware".to_string(),
            ctx.hardware.to_string(),
        ];
        if ctx.yes {
            v.push("--yes".into());
        }
        if let Some(p) = unit.shard_param() {
            v.push("--param".into());
            v.push(p);
        }
        v.extend(self.extra_args.iter().cloned());
        v
    }
}

/// 2026-09-26: How long to wait, after the child exits, for its output readers to reach
/// EOF. Bounded because a process the child leaves running can hold the pipes
/// open.
const READER_DRAIN: Duration = Duration::from_secs(5);

/// 2026-09-26: Wait for the child while streaming its stderr and stdout (prefixed
/// `stdout: `) to the log and `on_line`, enforcing the deadline and the cancel
/// flag: SIGTERM first, SIGKILL 30 s later. Returns the exit code and the
/// reason it was killed, if it was.
fn supervise(
    mut child: std::process::Child,
    deadline: Duration,
    cancel: &AtomicBool,
    log: &mut std::fs::File,
    on_line: &mut dyn FnMut(&str),
) -> (Option<i32>, Option<RunOutcome>) {
    use std::io::{BufRead, BufReader, Write};
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let stderr = child.stderr.take().expect("stderr is piped");
    let stdout = child.stdout.take().expect("stdout is piped");
    let tx2 = tx.clone();
    let readers = [
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        }),
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx2.send(format!("stdout: {line}")).is_err() {
                    break;
                }
            }
        }),
    ];
    let started = Instant::now();
    let mut killed: Option<RunOutcome> = None;
    let mut kill_at: Option<Instant> = None;
    loop {
        while let Ok(line) = rx.try_recv() {
            let _ = writeln!(log, "{line}");
            on_line(&line);
        }
        if let Ok(Some(status)) = child.try_wait() {
            // 2026-09-26: Wait for the readers to reach EOF, bounded by `READER_DRAIN`,
            // then drain what they sent after the exit.
            let drain_until = Instant::now() + READER_DRAIN;
            while readers.iter().any(|r| !r.is_finished()) && Instant::now() < drain_until {
                std::thread::sleep(Duration::from_millis(20));
            }
            while let Ok(line) = rx.try_recv() {
                let _ = writeln!(log, "{line}");
                on_line(&line);
            }
            return (status.code(), killed);
        }
        if killed.is_none() {
            if cancel.load(Ordering::SeqCst) {
                killed = Some(RunOutcome::Cancelled);
            } else if started.elapsed() > deadline {
                killed = Some(RunOutcome::TimedOut);
            }
            if killed.is_some() {
                terminate(&mut child);
                kill_at = Some(Instant::now() + Duration::from_secs(30));
            }
        } else if kill_at.is_some_and(|t| Instant::now() > t) {
            let _ = child.kill();
            kill_at = None;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// 2026-09-26: SIGTERM the child's whole process group (it may have started a server),
/// then the child alone.
#[cfg(unix)]
fn terminate(child: &mut std::process::Child) {
    let pid = child.id();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .status();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
}

/// 2026-09-26: Without process groups: kill the child alone, at once.
#[cfg(not(unix))]
fn terminate(child: &mut std::process::Child) {
    let _ = child.kill();
}

/// 2026-09-26: Put the child in its own process group so `terminate` can reach the
/// server it starts. A no-op where process groups do not exist.
fn in_own_group(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0)
    }
    #[cfg(not(unix))]
    {
        cmd
    }
}

/// 2026-09-26: `Command::spawn`, retried while exec fails with `ETXTBSY` (at most
/// `SPAWN_BUSY_RETRIES` times, `SPAWN_BUSY_WAIT` apart). An executable written moments ago
/// can still be open for writing in a child another thread forked before the writer closed
/// it; that child drops the descriptor when it execs, so the error clears on its own.
pub(super) fn spawn_retrying_busy(
    cmd: &mut std::process::Command,
) -> std::io::Result<std::process::Child> {
    let mut attempts = 0;
    loop {
        match cmd.spawn() {
            Err(e) if is_text_file_busy(&e) && attempts < SPAWN_BUSY_RETRIES => {
                attempts += 1;
                std::thread::sleep(SPAWN_BUSY_WAIT);
            }
            other => return other,
        }
    }
}

const SPAWN_BUSY_RETRIES: u32 = 50;
const SPAWN_BUSY_WAIT: std::time::Duration = std::time::Duration::from_millis(10);

#[cfg(unix)]
fn is_text_file_busy(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::ETXTBSY)
}

#[cfg(not(unix))]
fn is_text_file_busy(_: &std::io::Error) -> bool {
    false
}

impl GateRunner for LocalChild {
    fn run(&mut self, unit: &Unit, ctx: &RunCtx, on_line: &mut dyn FnMut(&str)) -> RunOutcome {
        let since = super::lockfile::now_unix();
        let _ = std::fs::create_dir_all(ctx.log_dir);
        let log_path = ctx.log_dir.join(format!("{}.log", unit.file_stem()));
        let mut log = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                return RunOutcome::Harness {
                    reason: format!("cannot open {}: {e}", log_path.display()),
                    retryable: false,
                };
            }
        };
        let mut cmd = std::process::Command::new(&self.exe);
        cmd.args(self.argv(unit, ctx))
            .current_dir(ctx.root)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let child = spawn_retrying_busy(in_own_group(&mut cmd));
        let child = match child {
            Ok(c) => c,
            Err(e) => {
                return RunOutcome::Harness {
                    reason: format!("cannot spawn {}: {e}", self.exe.display()),
                    retryable: false,
                };
            }
        };
        let (exit, killed) = supervise(child, ctx.deadline, &self.cancel, &mut log, on_line);
        let record = self
            .records
            .newest_since(ctx.root, unit.id, unit.shard, since);
        classify(unit, ctx.anchor, exit, record, killed)
    }
}

#[cfg(all(test, unix))]
#[path = "runner_tests.rs"]
mod runner_tests;
