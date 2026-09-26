// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The leased server: one `met serve` that outlives a
//! `met benchmark run --pull-request-gate --serve-reuse` run, so the next such
//! run on this box measures against it instead of loading the checkpoint again.
//!
//! The lease is `serve-lease.json` under the metrale home, and the child's
//! output appends to `serve-lease.log` beside it. `met benchmark serve-release`
//! stops the server; `met benchmark certify` releases an orphaned lease before
//! its units and the lease after them (`bench_certify/mod.rs`).
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - Only the server the lease file names is ever reused, never another
//!   `met serve` found on the box.
//! - It is reused only when its `/serve-config` reports the lease's pid, this
//!   binary's digest, the plan's argv digest and the plan's `METRALE_*` lever
//!   digest, the lease names the plan's model, and `/v1/models` lists that
//!   model (`mismatch`, `probe`). Any other answer, or a probe error, stops it
//!   and starts a fresh one.
//! - A harness whose `METRALE_*` levers the plan does not declare, or declares
//!   at another value, gets no server (`ServePlan::reconcile_env` runs first).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use metrale_bench::serve_identity::{ServeIdentity, argv_fingerprint, env_is_unknown, file_sha256};
use metrale_bench::{ArtifactStore, TargetEndpoint, serve_env};

use super::bench_cause;
use super::bench_selfstart::SelfServed;
use super::bench_serve_plan::ServePlan;

const POLL: Duration = Duration::from_millis(500);
/// 2026-09-26: `stop` sends SIGTERM, waits up to this long, then sends SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(60);

/// 2026-09-26: The leased server, as recorded in `serve-lease.json`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Lease {
    pub pid: u32,
    pub port: u16,
    pub model: String,
    pub recipe_id: String,
    pub argv_sha256: String,
    pub binary_sha256: String,
    /// 2026-09-26: `serve_env::fingerprint` of the lever set the server was
    /// started under. Recorded only: `mismatch` compares the server's own
    /// report, not this copy. Empty when the file lacks the field.
    #[serde(default)]
    pub env_sha256: String,
    /// 2026-09-26: The `--serve-lease-owner` pid (a campaign driver), or the
    /// run's own pid when none was given.
    pub owner_pid: u32,
    pub started_at: u64,
}

/// 2026-09-26: What a reused server must report: this binary, the plan's
/// rendering on the leased port, and the plan's lever set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    pub argv_sha256: String,
    pub binary_sha256: String,
    pub env_sha256: String,
}

pub fn lease_path(store: &ArtifactStore) -> PathBuf {
    store.root().join("serve-lease.json")
}

pub fn log_path(store: &ArtifactStore) -> PathBuf {
    store.root().join("serve-lease.log")
}

/// 2026-09-26: The lease on file, if any. A malformed file is an error, not
/// "no lease": a server it named may be running.
pub fn read(store: &ArtifactStore) -> Result<Option<Lease>> {
    let path = lease_path(store);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn write(store: &ArtifactStore, lease: &Lease) -> Result<()> {
    let path = lease_path(store);
    std::fs::write(&path, serde_json::to_string_pretty(lease)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// 2026-09-26: Whether `/proc/<pid>` exists. Off Linux every pid reads as
/// gone, so a lease is never reused there.
fn pid_alive(pid: u32) -> bool {
    cfg!(target_os = "linux") && Path::new(&format!("/proc/{pid}")).exists()
}

/// 2026-09-26: [`Expected`] for `plan` on `port`, from this binary.
///
/// The lever digest is over the plan's declaration, not this process's
/// environment: `METRALE_*` levers are not in the argv, so only this digest
/// tells two servers with the same rendering apart. A child that `start`
/// spawns after `ServePlan::reconcile_env` runs under exactly the declared set.
fn expected(plan: &ServePlan, port: u16) -> Result<Expected> {
    let argv = plan.argv(port)?;
    let mine = std::env::current_exe().context("current_exe")?;
    Ok(Expected {
        argv_sha256: argv_fingerprint(&argv[1..]),
        binary_sha256: file_sha256(&mine)?,
        env_sha256: serve_env::fingerprint(&plan.serve_env),
    })
}

/// 2026-09-26: Why a leased server is not the one this run needs, or `None`
/// when it is. Pure.
pub fn mismatch(
    lease: &Lease,
    reported: &ServeIdentity,
    expected: &Expected,
    model: &str,
) -> Option<String> {
    if reported.pid != lease.pid {
        return Some(format!(
            "pid {} answered on port {}, the lease names pid {}",
            reported.pid, lease.port, lease.pid
        ));
    }
    if reported.binary_sha256 != expected.binary_sha256 {
        return Some("it was built from another binary".into());
    }
    if reported.argv_sha256 != expected.argv_sha256 {
        return Some(format!(
            "it serves {} under another rendering (recipe {}, overrides or hermetic set differ)",
            lease.model, lease.recipe_id
        ));
    }
    // 2026-09-26: The recipe renders to flags, checked above; the `METRALE_*`
    // levers never reach argv, so they are checked here, separately, and the
    // message says which half differs. The server's own report is compared,
    // not the lease's copy, and a server that reports none is refused.
    if env_is_unknown(&reported.env_sha256) {
        return Some(
            "it does not report its METRALE_* serve environment (a server older than the \
             env digest), so its levers cannot be verified"
                .into(),
        );
    }
    if reported.env_sha256 != expected.env_sha256 {
        return Some(format!(
            "it was started under another METRALE_* serve environment than recipe {} is \
             measured under (a server carrying levers this run did not declare, or lacking \
             ones it did; argv and binary match, so this is env-only)",
            lease.recipe_id
        ));
    }
    if lease.model != model {
        return Some(format!("it serves {}, this run needs {model}", lease.model));
    }
    None
}

/// 2026-09-26: Take the leased server if it is the one `plan` would start,
/// else stop it and start one. Either way the returned server is left
/// running when dropped.
pub async fn acquire(plan: ServePlan, owner_pid: Option<u32>) -> Result<SelfServed> {
    // 2026-09-26: Before the probe: a harness whose levers the plan does not
    // declare, or contradict it, gets no server, reused or fresh.
    let reconciled = plan.reconcile_env()?;
    let store = ArtifactStore::discover()?;
    if let Some(lease) = read(&store)? {
        if pid_alive(lease.pid) {
            let target = TargetEndpoint::local(lease.port, &plan.model);
            let verdict = match probe(&target, &lease, &plan).await {
                Ok(None) => None,
                Ok(Some(why)) => Some(why),
                Err(e) => Some(format!("{e:#}")),
            };
            match verdict {
                None => {
                    eprintln!(
                        "gate: reusing the leased server (pid {}, port {}, recipe {}) — same binary, \
                         same rendering",
                        lease.pid, lease.port, lease.recipe_id
                    );
                    let resolved = plan.disclosed(lease.port)?;
                    return Ok(SelfServed::external(
                        target,
                        plan.recipe_id,
                        plan.requested,
                        resolved,
                        reconciled.env,
                        plan.entry,
                    ));
                }
                Some(why) => {
                    eprintln!(
                        "gate: the leased server (pid {}, port {}) is not this run's: {why}; replacing it",
                        lease.pid, lease.port
                    );
                    stop(&lease);
                }
            }
        } else {
            eprintln!(
                "gate: the lease names pid {}, which is gone; starting afresh",
                lease.pid
            );
        }
        let _ = std::fs::remove_file(lease_path(&store));
    }
    start(
        &store,
        plan,
        reconciled,
        owner_pid.unwrap_or_else(std::process::id),
    )
    .await
}

async fn probe(target: &TargetEndpoint, lease: &Lease, plan: &ServePlan) -> Result<Option<String>> {
    let doc =
        metrale_bench::http::get_json(target, "/serve-config", Duration::from_secs(10)).await?;
    let reported: ServeIdentity = serde_json::from_value(doc).context("parsing /serve-config")?;
    let want = expected(plan, lease.port)?;
    if let Some(why) = mismatch(lease, &reported, &want, &plan.model) {
        return Ok(Some(why));
    }
    let models = metrale_bench::http::list_models(target, Duration::from_secs(10)).await?;
    if !models.contains(&plan.model) {
        return Ok(Some(format!(
            "it is serving {models:?}, not {}",
            plan.model
        )));
    }
    Ok(None)
}

/// 2026-09-26: Start `met serve` as a child (in its own process group on
/// Unix), record the lease, and wait for the model. If it does not come up,
/// the child is stopped and the lease removed.
///
/// The child inherits this process's environment plus `reconciled.missing`,
/// the declared levers this process does not carry. After `reconcile` that
/// makes the child's lever set exactly the declaration, which is what the
/// lease and the server's own `env_sha256` both fingerprint.
async fn start(
    store: &ArtifactStore,
    plan: ServePlan,
    reconciled: serve_env::Reconciled,
    owner_pid: u32,
) -> Result<SelfServed> {
    let port = metrale_bench::benchmarks::agentic::score::free_port()?;
    let serve_args = plan.serve_args(port)?;
    super::bench_selfstart::check_box_is_free_enough(
        serve_args.gpu_memory_utilization,
        &plan.recipe_id,
        plan.limits.memory.min_free_fraction,
    )?;
    let argv = plan.argv(port)?;
    let exe = std::env::current_exe().context("current_exe")?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path(store))
        .with_context(|| format!("opening {}", log_path(store).display()))?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(&argv[1..])
        .envs(&reconciled.missing)
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {} serve", exe.display()))?;
    let lease = Lease {
        pid: child.id(),
        port,
        model: plan.model.clone(),
        recipe_id: plan.recipe_id.clone(),
        argv_sha256: argv_fingerprint(&argv[1..]),
        binary_sha256: file_sha256(&exe)?,
        env_sha256: serve_env::fingerprint(&reconciled.env),
        owner_pid,
        started_at: super::bench_certify::lockfile::now_unix(),
    };
    write(store, &lease)?;
    eprintln!(
        "gate: serving {} from recipe {} on port {port} as a LEASED server (pid {}); it stays up \
         after this run — `met benchmark serve-release` stops it",
        plan.model, plan.recipe_id, lease.pid
    );
    eprintln!(
        "gate: the leased server's METRALE_* serve env is {} ({} handed to the child)",
        if reconciled.env.is_empty() {
            "empty".to_string()
        } else {
            reconciled
                .env
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        },
        reconciled.missing.len()
    );
    let target = TargetEndpoint::local(port, &plan.model);
    let boot_timeout = Duration::from_secs(plan.limits.timing.boot_timeout_s);
    if let Err(e) = await_serving(
        &target,
        &plan.model,
        &mut child,
        boot_timeout,
        &log_path(store),
    )
    .await
    {
        stop(&lease);
        let _ = std::fs::remove_file(lease_path(store));
        return Err(e);
    }
    eprintln!("gate: endpoint is serving {}", plan.model);
    let resolved = plan.disclosed(port)?;
    Ok(SelfServed::external(
        target,
        plan.recipe_id,
        plan.requested,
        resolved,
        reconciled.env,
        plan.entry,
    ))
}

/// 2026-09-26: The refusal for a leased serve that died during startup,
/// quoting the final `Error:` block of its log tail. The block is indented so
/// this message's own `Error:` line stays the outermost one
/// `bench_cause::final_error_block` finds. Pure over the log tail.
pub(super) fn exited_before_serving(status: &str, model: &str, log_tail: &str) -> String {
    match bench_cause::final_error_block(log_tail) {
        Some(block) => format!(
            "the leased server exited ({status}) before it began serving {model:?} — \
             serve-lease.log ends with:\n{}",
            bench_cause::indented(&block)
        ),
        None => format!(
            "the leased server exited ({status}) before it began serving {model:?} — see \
             serve-lease.log (its tail carries no `Error:` block)"
        ),
    }
}

/// 2026-09-26: Block until `/v1/models` names `model`, or fail at
/// `boot_timeout`. A child that exits first fails at once, quoting the end
/// of `log`.
async fn await_serving(
    target: &TargetEndpoint,
    model: &str,
    child: &mut std::process::Child,
    boot_timeout: Duration,
    log: &Path,
) -> Result<()> {
    let deadline = Instant::now() + boot_timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            let tail = bench_cause::tail_of_file(log, bench_cause::TAIL_BYTES)
                .unwrap_or_else(|e| format!("(serve-lease.log could not be read: {e})"));
            bail!(
                "{}",
                exited_before_serving(&status.to_string(), model, &tail)
            );
        }
        let last = match metrale_bench::http::list_models(target, Duration::from_secs(5)).await {
            Ok(models) if models.iter().any(|m| m == model) => return Ok(()),
            Ok(models) => format!("the endpoint is serving {models:?}"),
            Err(e) => format!("{e:#}"),
        };
        if Instant::now() >= deadline {
            bail!(
                "{model:?} did not come up within {}s — {last}",
                boot_timeout.as_secs()
            );
        }
        tokio::time::sleep(POLL).await;
    }
}

/// 2026-09-26: SIGTERM the leased server's process group and pid, wait up to
/// `STOP_GRACE`, then SIGKILL the group if the pid is still alive.
pub fn stop(lease: &Lease) {
    let pid = lease.pid;
    let _ = std::process::Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .status();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    let until = Instant::now() + STOP_GRACE;
    while pid_alive(pid) && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(250));
    }
    if pid_alive(pid) {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
    }
}

/// 2026-09-26: Stop the leased server, if any, and remove the lease. Returns
/// what was released.
pub fn release(store: &ArtifactStore) -> Result<Option<Lease>> {
    let Some(lease) = read(store)? else {
        return Ok(None);
    };
    if pid_alive(lease.pid) {
        stop(&lease);
    }
    std::fs::remove_file(lease_path(store))
        .with_context(|| format!("removing {}", lease_path(store).display()))?;
    Ok(Some(lease))
}

/// 2026-09-26: [`release`], but only when the lease's `owner_pid` is gone.
pub fn release_if_orphaned(store: &ArtifactStore) -> Result<Option<Lease>> {
    match read(store)? {
        Some(l) if !pid_alive(l.owner_pid) => release(store),
        _ => Ok(None),
    }
}

/// 2026-09-26: `met benchmark serve-release`. Exits 0 with or without a lease.
pub fn release_cmd() -> Result<i32> {
    let store = ArtifactStore::discover()?;
    match release(&store)? {
        Some(l) => {
            eprintln!(
                "released the leased server (pid {}, port {}, {})",
                l.pid, l.port, l.model
            );
            Ok(0)
        }
        None => {
            eprintln!("no leased server ({})", lease_path(&store).display());
            Ok(0)
        }
    }
}

#[cfg(test)]
#[path = "bench_lease_tests.rs"]
mod tests;
