// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Ownership table of SSM snapshot pool slots per session, with LRU and TTL eviction.
//!
//! Owner: server scheduler.
//! Invariants: a session never lists the same slot twice.
//!
//! Sessions are keyed by `compute_session_hash` of the prompt. Outside this
//! module only `evict_expired` (when the scheduler run finishes),
//! `session_count` and `total_slots` are called; nothing calls `save_snapshot`,
//! so the table stays empty and gates no snapshot.

use std::collections::HashMap;
use std::time::Instant;

/// 2026-09-26: Tracks SSM snapshot ownership per session, with LRU and TTL eviction.
pub struct SessionSsmManager {
    sessions: HashMap<u64, SessionState>,
    /// 2026-09-26: TTL in seconds: `evict_expired` drops sessions idle longer than this.
    ttl_secs: u64,
}

/// 2026-09-26: Per-session state.
struct SessionState {
    /// 2026-09-26: SSM snapshot pool slot IDs owned by this session.
    snapshot_slots: Vec<usize>,
    /// 2026-09-26: Last `save_snapshot`, or `owns_snapshot` that returned true.
    last_access: Instant,
}

impl SessionSsmManager {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            sessions: HashMap::new(),
            ttl_secs,
        }
    }

    /// 2026-09-26: Register a snapshot slot as belonging to a session.
    pub fn save_snapshot(&mut self, session_hash: u64, slot_id: usize) {
        let entry = self
            .sessions
            .entry(session_hash)
            .or_insert_with(|| SessionState {
                snapshot_slots: Vec::new(),
                last_access: Instant::now(),
            });
        entry.last_access = Instant::now();
        if !entry.snapshot_slots.contains(&slot_id) {
            entry.snapshot_slots.push(slot_id);
        }
    }

    /// 2026-09-26: Check if a snapshot slot belongs to the given session.
    /// Returns true and updates `last_access` if owned.
    pub fn owns_snapshot(&mut self, session_hash: u64, slot_id: usize) -> bool {
        if let Some(state) = self.sessions.get_mut(&session_hash)
            && state.snapshot_slots.contains(&slot_id)
        {
            state.last_access = Instant::now();
            return true;
        }
        false
    }

    /// 2026-09-26: Evict the least-recently-used session and return its freed snapshot
    /// slot IDs, or `None` if there are no sessions.
    pub fn evict_lru(&mut self) -> Option<Vec<usize>> {
        if self.sessions.is_empty() {
            return None;
        }
        let lru_hash = *self
            .sessions
            .iter()
            .min_by_key(|(_, state)| state.last_access)?
            .0;
        self.evict_session(lru_hash)
    }

    /// 2026-09-26: Evict a specific session and return its freed snapshot slot IDs.
    pub fn evict_session(&mut self, session_hash: u64) -> Option<Vec<usize>> {
        let state = self.sessions.remove(&session_hash)?;
        if !state.snapshot_slots.is_empty() {
            tracing::info!(
                "Session {session_hash:#x} evicted: freed {} SSM snapshot slot(s)",
                state.snapshot_slots.len(),
            );
        }
        Some(state.snapshot_slots)
    }

    /// 2026-09-26: Evict every session idle longer than the TTL and return all freed
    /// snapshot slot IDs.
    pub fn evict_expired(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let ttl = std::time::Duration::from_secs(self.ttl_secs);
        let mut freed = Vec::new();
        self.sessions.retain(|hash, state| {
            if now.duration_since(state.last_access) > ttl {
                tracing::info!(
                    "Session {hash:#x} expired ({} slots freed)",
                    state.snapshot_slots.len(),
                );
                freed.extend_from_slice(&state.snapshot_slots);
                false
            } else {
                true
            }
        });
        freed
    }

    /// 2026-09-26: Number of sessions in the table.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// 2026-09-26: Total snapshot slots across all sessions.
    pub fn total_slots(&self) -> usize {
        self.sessions.values().map(|s| s.snapshot_slots.len()).sum()
    }
}

/// 2026-09-26: Hash the first 1024 prompt tokens (fewer if the prompt is shorter)
/// with `DefaultHasher`.
pub fn compute_session_hash(prompt_tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let n = prompt_tokens.len().min(1024);
    for &tok in &prompt_tokens[..n] {
        tok.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_owns() {
        let mut mgr = SessionSsmManager::new(600);
        mgr.save_snapshot(0xAABB, 5);
        mgr.save_snapshot(0xAABB, 7);
        assert!(mgr.owns_snapshot(0xAABB, 5));
        assert!(mgr.owns_snapshot(0xAABB, 7));
        assert!(!mgr.owns_snapshot(0xAABB, 9));
        assert!(!mgr.owns_snapshot(0xCCDD, 5));
    }

    #[test]
    fn test_evict_lru() {
        let mut mgr = SessionSsmManager::new(600);
        mgr.save_snapshot(0x1111, 1);
        // 2026-09-26: Age session 0x1111 by 100 s.
        mgr.sessions.get_mut(&0x1111).unwrap().last_access =
            Instant::now() - std::time::Duration::from_secs(100);
        mgr.save_snapshot(0x2222, 2);
        // 2026-09-26: 0x1111 is older, so it is evicted.
        let freed = mgr.evict_lru().unwrap();
        assert_eq!(freed, vec![1]);
        assert_eq!(mgr.session_count(), 1);
        assert!(mgr.owns_snapshot(0x2222, 2));
    }

    #[test]
    fn test_evict_expired() {
        let mut mgr = SessionSsmManager::new(5);
        mgr.save_snapshot(0xAAAA, 10);
        mgr.sessions.get_mut(&0xAAAA).unwrap().last_access =
            Instant::now() - std::time::Duration::from_secs(10);
        mgr.save_snapshot(0xBBBB, 20);
        let freed = mgr.evict_expired();
        assert_eq!(freed, vec![10]);
        assert_eq!(mgr.session_count(), 1);
    }

    #[test]
    fn test_duplicate_slot_save() {
        let mut mgr = SessionSsmManager::new(600);
        mgr.save_snapshot(0x1234, 3);
        mgr.save_snapshot(0x1234, 3);
        mgr.save_snapshot(0x1234, 4);
        let freed = mgr.evict_session(0x1234).unwrap();
        assert_eq!(freed, vec![3, 4]);
    }

    #[test]
    fn test_session_hash_stability() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let h1 = compute_session_hash(&tokens);
        let h2 = compute_session_hash(&tokens);
        assert_eq!(h1, h2);

        let different = vec![1u32, 2, 3, 4, 6];
        let h3 = compute_session_hash(&different);
        assert_ne!(h1, h3);
    }
}
