// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Demand promotion of LoRA adapters, HTTP side. A request naming
//! a stageable adapter (registered, not resident) has it loaded into a cache
//! pool slot, from the `METRALE_LORA_PEER` weight peer or from local disk, and
//! is routed to it. The load and the victim choice run on the scheduler thread
//! (`LoraCommand::Promote` / `PromoteDisk`); this module holds the peer
//! registry entry type, the rejection type, and the single-flight that turns
//! concurrent misses for one adapter into one promote.
//!
//! Owner: server LoRA API.
//! Invariants:
//! - At most one `leader` future runs per adapter name at a time, and the
//!   name's entry is removed when it finishes, is cancelled or panics.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::oneshot;

/// 2026-09-26: A peer-stageable adapter: its id on the weight peer, plus the
/// peft config the peer does not send, parsed from the adapter's local
/// `adapter_config.json` at startup.
#[derive(Clone, Debug)]
pub struct StageableAdapter {
    pub peer_stage_id: String,
    pub peft: metrale_config::PeftAdapterConfig,
}

/// 2026-09-26: Why a demand promotion did not yield a slot. `Clone` because
/// every coalesced waiter gets a copy. `api/lora_control.rs` maps `PoolFull`
/// to 503 (retryable) and `Peer` to 502.
#[derive(Clone, Debug)]
pub enum PromoteReject {
    /// 2026-09-26: Every cache slot is busy, or the promote timed out waiting
    /// for the scheduler; retry once in-flight work drains.
    PoolFull(String),
    /// 2026-09-26: The peer or disk load, or the scheduler control channel,
    /// failed.
    Peer(String),
}

impl std::fmt::Display for PromoteReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromoteReject::PoolFull(m) | PromoteReject::Peer(m) => f.write_str(m),
        }
    }
}

/// 2026-09-26: Followers parked on one in-flight promotion, each awaiting the
/// leader's result over its own oneshot.
type PromoteWaiters = Vec<oneshot::Sender<Result<i32, PromoteReject>>>;

/// 2026-09-26: Single-flight coordinator for demand promotions, one map entry
/// per adapter name being promoted. The first caller for a name is the leader
/// and runs the promote; later callers are followers that await its result.
///
/// The inner `Mutex` is held only for map inserts and removals, never across
/// the leader's `.await`, and only this type touches it, so the scheduler
/// thread cannot deadlock against it.
#[derive(Default)]
pub struct PromotionManager {
    inflight: Mutex<HashMap<String, PromoteWaiters>>,
}

enum Role {
    Leader,
    Follower(oneshot::Receiver<Result<i32, PromoteReject>>),
}

/// 2026-09-26: Leadership guard. The leader claims the map entry and runs its
/// promote across an `.await`. If that future is cancelled (axum drops the
/// handler task when the client disconnects) or panics, `Drop` removes the
/// entry, so the name is never wedged, and dropping the waiter `Sender`s wakes
/// the followers with a `RecvError`. On success the leader removes the entry,
/// broadcasts, and calls `disarm()`, so `Drop` does nothing.
struct LeaderGuard<'a> {
    mgr: &'a PromotionManager,
    name: String,
    armed: bool,
}

impl LeaderGuard<'_> {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LeaderGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // 2026-09-26: Cancelled or panicked before broadcasting.
            let _ = self.mgr.inflight.lock().unwrap().remove(&self.name);
        }
    }
}

impl PromotionManager {
    /// 2026-09-26: Coalesced promote: if a promote for `name` is in flight,
    /// await its result; otherwise become the leader, run `leader` once,
    /// broadcast a clone of the result to every waiter that joined meanwhile,
    /// and remove the entry on success and on failure.
    ///
    /// Distinct names run independent leaders.
    pub async fn coalesce<F, Fut>(&self, name: &str, leader: F) -> Result<i32, PromoteReject>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<i32, PromoteReject>>,
    {
        let role = {
            let mut map = self.inflight.lock().unwrap();
            match map.get_mut(name) {
                Some(waiters) => {
                    let (tx, rx) = oneshot::channel();
                    waiters.push(tx);
                    Role::Follower(rx)
                }
                None => {
                    // 2026-09-26: Inserting the empty waiter list claims
                    // leadership; the lock is released before the promote runs.
                    map.insert(name.to_string(), Vec::new());
                    Role::Leader
                }
            }
        };

        match role {
            Role::Follower(rx) => rx.await.unwrap_or_else(|_| {
                Err(PromoteReject::Peer(
                    "promotion leader dropped without a result".to_string(),
                ))
            }),
            Role::Leader => {
                // 2026-09-26: Armed before the await, so a dropped handler task
                // cannot leave the entry behind.
                let mut guard = LeaderGuard {
                    mgr: self,
                    name: name.to_string(),
                    armed: true,
                };
                let result = leader().await;
                // 2026-09-26: Remove the entry and notify everyone who joined
                // meanwhile, then disarm.
                let waiters = self
                    .inflight
                    .lock()
                    .unwrap()
                    .remove(name)
                    .unwrap_or_default();
                guard.disarm();
                for w in waiters {
                    let _ = w.send(result.clone());
                }
                result
            }
        }
    }

    /// 2026-09-26: Number of in-flight entries; 0 at rest.
    #[cfg(test)]
    fn inflight_len(&self) -> usize {
        self.inflight.lock().unwrap().len()
    }
}

/// 2026-09-26: Whether a miss should attempt a promote: promotion is enabled
/// and the name is registered stageable. Only this module's tests call it.
pub fn should_attempt_promote(promotion_enabled: bool, is_stageable: bool) -> bool {
    promotion_enabled && is_stageable
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn attempt_gate() {
        assert!(should_attempt_promote(true, true));
        assert!(!should_attempt_promote(false, true));
        assert!(!should_attempt_promote(true, false));
        assert!(!should_attempt_promote(false, false));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_flight_same_name_one_promote() {
        let mgr = Arc::new(PromotionManager::default());
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..100 {
            let mgr = Arc::clone(&mgr);
            let calls = Arc::clone(&calls);
            handles.push(tokio::spawn(async move {
                mgr.coalesce("lyra", || async {
                    // 2026-09-26: Held open long enough for every follower to
                    // register.
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Ok(5)
                })
                .await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1, "coalesced to one promote");
        for r in results {
            assert_eq!(r.unwrap().unwrap(), 5);
        }
        assert_eq!(mgr.inflight_len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_flight_failure_broadcasts_and_clears() {
        let mgr = Arc::new(PromotionManager::default());
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..50 {
            let mgr = Arc::clone(&mgr);
            let calls = Arc::clone(&calls);
            handles.push(tokio::spawn(async move {
                mgr.coalesce("cold", || async {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Err::<i32, _>(PromoteReject::PoolFull("all busy".to_string()))
                })
                .await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1, "one promote attempt");
        for r in results {
            match r.unwrap() {
                Err(PromoteReject::PoolFull(m)) => assert_eq!(m, "all busy"),
                other => panic!("expected same PoolFull, got {other:?}"),
            }
        }
        assert_eq!(mgr.inflight_len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_names_promote_independently() {
        let mgr = Arc::new(PromotionManager::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let names = ["a", "b", "c"];
        let mut handles = Vec::new();
        for n in names {
            for _ in 0..10 {
                let mgr = Arc::clone(&mgr);
                let calls = Arc::clone(&calls);
                handles.push(tokio::spawn(async move {
                    mgr.coalesce(n, || async {
                        calls.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                        Ok(7)
                    })
                    .await
                }));
            }
        }
        let _ = futures::future::join_all(handles).await;
        assert_eq!(calls.load(Ordering::SeqCst), names.len());
        assert_eq!(mgr.inflight_len(), 0);
    }
}
