// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: One shutdown path for every trigger. [`request`] is called by
//! the `SIGINT`/`SIGTERM` task from [`install_signal_listeners`], by the TUI
//! on Ctrl+C (raw mode delivers it as a key event), by `/quit`, by the TUI's
//! own exit, by the scheduler when the fault latch reports a fault, and by the
//! bench self-start teardown.
//!
//! During startup a request is sent on `main`'s one-shot escape. Once the
//! router's accept loop has called [`disarm_startup_escape`], a request wakes
//! [`wait`] in that loop, which stops accepting, drains in-flight requests for
//! a bounded grace and returns, so `serve()` unwinds normally.
//!
//! Owner: server tui.
//! Invariants: the shutdown latch is one-way: once [`requested`] is true it
//! stays true for the life of the process.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Notify, oneshot};

/// 2026-09-26: Where the process is in its shutdown sequence.
struct Phase {
    /// 2026-09-26: Set once a shutdown has been requested, from any trigger.
    requested: AtomicBool,
    /// 2026-09-26: Whether a request still takes the startup escape.
    /// Cleared by [`disarm_startup_escape`].
    in_startup: AtomicBool,
}

// 2026-09-26: Static because a shutdown request is process-wide, and the
// callers of `request` (the signal task, the TUI thread, the scheduler's fault
// paths, the bench self-start) share no object it could be carried on.
static PHASE: Phase = Phase {
    requested: AtomicBool::new(false),
    in_startup: AtomicBool::new(true),
};

/// 2026-09-26: The wakeup for [`wait`] and the startup escape. `request`
/// always wakes the waiters, and while `in_startup` is set it also sends on
/// the escape.
#[derive(Default)]
struct Channels {
    notify: Notify,
    escape: std::sync::Mutex<Option<oneshot::Sender<&'static str>>>,
}

static CHANNELS: OnceLock<Channels> = OnceLock::new();

fn channels() -> &'static Channels {
    CHANNELS.get_or_init(Channels::default)
}

fn notify() -> &'static Notify {
    &channels().notify
}

/// 2026-09-26: The startup escape: the sending half of `main`'s one-shot.
/// The first request during startup takes it; later ones find the slot empty.
///
/// `main` races `serve()` against the receiving half, so a shutdown during
/// model load does not wait for the accept loop. Once
/// [`disarm_startup_escape`] has run, a request goes through the draining
/// path instead, so a live server does not exit with requests in flight.
fn escape() -> &'static std::sync::Mutex<Option<oneshot::Sender<&'static str>>> {
    &channels().escape
}

/// 2026-09-26: Arm the startup escape with `main`'s sender. `main` calls it
/// once, before `serve()`.
pub fn arm_startup_escape(tx: oneshot::Sender<&'static str>) {
    *escape().lock().expect("shutdown escape poisoned") = Some(tx);
}

/// 2026-09-26: Close the startup escape once the server is accepting: from
/// here on, a shutdown means "stop accepting and drain", not "return from
/// `main`".
///
/// It only clears `in_startup` and leaves the sender parked, so the channel
/// never closes and `main`'s receiver never resolves with `Err`.
pub fn disarm_startup_escape() {
    PHASE.in_startup.store(false, Ordering::SeqCst);
}

/// 2026-09-26: Request a clean shutdown. Idempotent; callable from any
/// thread.
pub fn request(reason: &'static str) {
    if !PHASE.requested.swap(true, Ordering::SeqCst) {
        tracing::info!("Shutdown requested ({reason}) — draining in-flight requests");
    }
    // 2026-09-26: In startup, hand the reason to `main`'s `select!`. Only the
    // first request takes the sender. After disarming, the sender stays
    // parked and the notification below does the work.
    if PHASE.in_startup.load(Ordering::SeqCst)
        && let Ok(mut slot) = escape().lock()
        && let Some(tx) = slot.take()
    {
        let _ = tx.send(reason);
    }
    notify().notify_waiters();
}

/// 2026-09-26: Whether a shutdown has been requested.
pub fn requested() -> bool {
    PHASE.requested.load(Ordering::SeqCst)
}

/// 2026-09-26: Resolves when a shutdown is requested, at once if one
/// already was.
pub async fn wait() {
    if requested() {
        return;
    }
    // 2026-09-26: Create the `Notified` future before re-checking, so a
    // `request` between the two checks still wakes it.
    let notified = notify().notified();
    if requested() {
        return;
    }
    notified.await;
}

/// 2026-09-26: Install `SIGINT` and `SIGTERM` listeners on the tokio
/// runtime (`SIGINT` only off Unix). Called from `serve()` and from the bench
/// runner. `SIGTERM` still arrives while the TUI owns the keyboard.
pub fn install_signal_listeners() {
    tokio::spawn(async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            let mut term =
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("SIGTERM listener unavailable: {e}");
                        if ctrl_c.await.is_ok() {
                            request("SIGINT");
                        }
                        return;
                    }
                };
            tokio::select! {
                r = ctrl_c => { if r.is_ok() { request("SIGINT"); } }
                _ = term.recv() => { request("SIGTERM"); }
            }
        }
        #[cfg(not(unix))]
        {
            if ctrl_c.await.is_ok() {
                request("SIGINT");
            }
        }
    });
}

/// 2026-09-26: After the accept loop stops, wait until the
/// `REQUESTS_ACTIVE` gauge reads zero or the grace window expires, polling
/// every 200 ms.
pub async fn drain_in_flight(grace: Duration) {
    let start = std::time::Instant::now();
    loop {
        let active = crate::metrics::REQUESTS_ACTIVE.get();
        if active <= 0 {
            tracing::info!("Drain complete — no active requests");
            return;
        }
        if start.elapsed() >= grace {
            tracing::warn!(
                "Drain grace ({}s) expired with {active} request(s) still active — exiting",
                grace.as_secs()
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[cfg(test)]
#[path = "shutdown_tests.rs"]
mod tests;
