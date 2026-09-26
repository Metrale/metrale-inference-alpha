// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Run one-shot work off the render thread and answer on a
//! channel, which the render loop drains with `try_recv`
//! (`.github/workflows/tui-threading.yml` rejects `block_on` under `tui/`).
//! Callers: the recipe index fetch and date lookup (`recipe/fetch.rs`), the
//! library scan (`data/library.rs`) and the freshness check
//! (`download_state.rs`).
//!
//! Owner: server tui.
//! Invariants: the returned receiver always resolves: with the work's
//! result, with `on_spawn_failure`'s value when the thread cannot be spawned,
//! or as disconnected if `work` panics.

use std::sync::mpsc::{Receiver, channel};

/// 2026-09-26: Run `work` on a thread named `name`; the result arrives on the
/// returned channel. `on_spawn_failure` supplies the answer when the thread
/// cannot be created, so the caller needs no second code path for it.
pub fn spawn<T, W, F>(name: &str, work: W, on_spawn_failure: F) -> Receiver<T>
where
    T: Send + 'static,
    W: FnOnce() -> T + Send + 'static,
    F: FnOnce(std::io::Error) -> T,
{
    let (tx, rx) = channel();
    let spawned = std::thread::Builder::new().name(name.to_string()).spawn({
        let tx = tx.clone();
        move || {
            // 2026-09-26: A dropped receiver means the UI moved on; the send
            // error is ignored.
            let _ = tx.send(work());
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("could not spawn {name}: {e}");
        let _ = tx.send(on_spawn_failure(e));
    }
    rx
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
