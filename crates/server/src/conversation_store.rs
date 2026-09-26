// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: In-memory conversation store behind `/v1/conversations` and the
//! Responses API's `conversation` field, which prepends the stored items to a
//! request and appends the turn's items after it (`api/responses.rs`,
//! `api/responses_translate.rs`).
//!
//! Owner: server API.
//! Invariants:
//! - Ids are `conv_<uuid>`. After `create` returns, the store holds at most
//!   `max_entries` conversations; the one evicted is the least recently
//!   created or read by `get`.
//! - An entry idle for longer than `ttl` is never returned by `get`,
//!   `update_metadata` or `add_items`; they remove it on sight.
//!
//! Items are kept as raw `serde_json::Value`, so any item type round-trips
//! unchanged apart from the `id` that `stamp_id` adds.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// 2026-09-26: One conversation: metadata and its items in insertion order.
pub struct Conversation {
    pub id: String,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
    /// 2026-09-26: Items as wire JSON. Every object item has a top-level `id`
    /// (`stamp_id`).
    pub items: Vec<serde_json::Value>,
    /// 2026-09-26: The TTL clock, reset by `get`, `update_metadata`,
    /// `add_items` and a `remove_item` that removed something. Eviction order
    /// is `Inner::order`, not this.
    last_access: Instant,
}

pub struct ConversationStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    max_entries: usize,
}

struct Inner {
    map: HashMap<String, Conversation>,
    order: VecDeque<String>,
}

/// 2026-09-26: Most items one `add_items` call accepts (`TooMany` above it);
/// the `/v1/conversations` create handler applies the same cap to initial
/// items.
pub const MAX_ITEMS_PER_INSERT: usize = 20;

impl ConversationStore {
    /// 2026-09-26: Build the store from `METRALE_CONVERSATION_MAX_ENTRIES`
    /// (default 10 000, minimum 1) and `METRALE_CONVERSATION_TTL_SECONDS`
    /// (default 86 400, minimum 1).
    ///
    /// # Errors
    /// When either variable is set to something that is not a whole number
    /// at or above its minimum (`env_config::parse_min`).
    pub fn from_env() -> Result<Arc<Self>, String> {
        let max_entries = crate::env_config::parse_min(
            "METRALE_CONVERSATION_MAX_ENTRIES",
            std::env::var("METRALE_CONVERSATION_MAX_ENTRIES")
                .ok()
                .as_deref(),
            1_usize,
            "how many conversations to keep before evicting the coldest",
        )?
        .unwrap_or(10_000);
        let ttl_secs = crate::env_config::parse_min(
            "METRALE_CONVERSATION_TTL_SECONDS",
            std::env::var("METRALE_CONVERSATION_TTL_SECONDS")
                .ok()
                .as_deref(),
            1_u64,
            "how long a conversation stays retrievable, in seconds",
        )?
        .unwrap_or(86_400_u64);
        Ok(Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl: Duration::from_secs(ttl_secs),
            max_entries,
        }))
    }

    #[cfg(test)]
    pub fn with_config(max_entries: usize, ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl,
            max_entries,
        })
    }

    /// 2026-09-26: Create a conversation with the given items and metadata,
    /// evicting the oldest entries of `order` above `max_entries`. Returns the
    /// new id.
    pub fn create(
        &self,
        initial_items: Vec<serde_json::Value>,
        metadata: HashMap<String, String>,
    ) -> String {
        let id = format!("conv_{}", crate::ids::uuid_v4());
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let items = initial_items
            .into_iter()
            .enumerate()
            .map(|(i, v)| stamp_id(v, &id, i))
            .collect();
        let entry = Conversation {
            id: id.clone(),
            created_at: now_unix,
            metadata,
            items,
            last_access: Instant::now(),
        };
        let mut inner = self.inner.lock();
        inner.map.insert(id.clone(), entry);
        inner.order.push_back(id.clone());
        while inner.map.len() > self.max_entries {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            } else {
                break;
            }
        }
        id
    }

    /// 2026-09-26: A snapshot of the conversation, which moves to the back of
    /// the eviction order. An entry past its TTL is removed and gives `None`.
    pub fn get(&self, id: &str) -> Option<ConversationSnapshot> {
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return None;
        }
        inner.order.retain(|k| k != id);
        inner.order.push_back(id.to_string());
        let c = inner.map.get_mut(id).expect("entry present");
        c.last_access = Instant::now();
        Some(ConversationSnapshot {
            id: c.id.clone(),
            created_at: c.created_at,
            metadata: c.metadata.clone(),
            items: c.items.clone(),
        })
    }

    /// 2026-09-26: Merge `patch` into the conversation's metadata. Returns the
    /// updated snapshot; `None` when the id is unknown or past its TTL.
    pub fn update_metadata(
        &self,
        id: &str,
        patch: HashMap<String, String>,
    ) -> Option<ConversationSnapshot> {
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return None;
        }
        let c = inner.map.get_mut(id).expect("entry present");
        c.last_access = Instant::now();
        for (k, v) in patch {
            c.metadata.insert(k, v);
        }
        Some(ConversationSnapshot {
            id: c.id.clone(),
            created_at: c.created_at,
            metadata: c.metadata.clone(),
            items: c.items.clone(),
        })
    }

    /// 2026-09-26: Append items and return them with their ids. More than
    /// `MAX_ITEMS_PER_INSERT` is `TooMany`; an unknown or expired id is
    /// `NotFound`.
    pub fn add_items(
        &self,
        id: &str,
        new_items: Vec<serde_json::Value>,
    ) -> Result<Vec<serde_json::Value>, AddItemsError> {
        if new_items.len() > MAX_ITEMS_PER_INSERT {
            return Err(AddItemsError::TooMany(new_items.len()));
        }
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return Err(AddItemsError::NotFound),
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return Err(AddItemsError::NotFound);
        }
        let c = inner.map.get_mut(id).expect("entry present");
        let start = c.items.len();
        let stamped: Vec<serde_json::Value> = new_items
            .into_iter()
            .enumerate()
            .map(|(i, v)| stamp_id(v, id, start + i))
            .collect();
        c.items.extend(stamped.iter().cloned());
        c.last_access = Instant::now();
        Ok(stamped)
    }

    /// 2026-09-26: Remove every item whose `id` is `item_id`; `true` when at
    /// least one was removed. The TTL is not checked.
    pub fn remove_item(&self, conv_id: &str, item_id: &str) -> bool {
        let mut inner = self.inner.lock();
        let Some(c) = inner.map.get_mut(conv_id) else {
            return false;
        };
        let before = c.items.len();
        c.items
            .retain(|v| v.get("id").and_then(|v| v.as_str()).unwrap_or("") != item_id);
        let removed = c.items.len() < before;
        if removed {
            c.last_access = Instant::now();
        }
        removed
    }

    pub fn delete(&self, id: &str) -> bool {
        let mut inner = self.inner.lock();
        if inner.map.remove(id).is_some() {
            inner.order.retain(|k| k != id);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}

pub struct ConversationSnapshot {
    pub id: String,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
    pub items: Vec<serde_json::Value>,
}

#[derive(Debug)]
pub enum AddItemsError {
    NotFound,
    TooMany(usize),
}

/// 2026-09-26: Give an object item without an `id` the id
/// `item_<conv id without "conv_">_<idx>`, where `idx` is its position in
/// `items` at insert. A client-supplied `id` and non-object items are left
/// as they are.
fn stamp_id(mut v: serde_json::Value, conv_id: &str, idx: usize) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut()
        && !obj.contains_key("id")
    {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(format!(
                "item_{}_{}",
                conv_id.trim_start_matches("conv_"),
                idx
            )),
        );
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn create_and_retrieve() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(
            vec![json!({"type": "message", "role": "user", "content": "hi"})],
            HashMap::new(),
        );
        let snap = store.get(&id).expect("hit");
        assert_eq!(snap.items.len(), 1);
        assert_eq!(snap.items[0]["role"], "user");
        assert!(snap.items[0]["id"].as_str().unwrap().starts_with("item_"));
    }

    #[test]
    fn add_items_respects_cap() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(Vec::new(), HashMap::new());
        let twenty_one: Vec<_> = (0..21)
            .map(|i| json!({"type": "message", "role": "user", "content": format!("m{i}")}))
            .collect();
        let err = store.add_items(&id, twenty_one).unwrap_err();
        assert!(matches!(err, AddItemsError::TooMany(21)));
    }

    #[test]
    fn delete_item_by_id() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(
            vec![
                json!({"type": "message", "role": "user", "content": "a"}),
                json!({"type": "message", "role": "user", "content": "b"}),
            ],
            HashMap::new(),
        );
        let snap = store.get(&id).unwrap();
        let item_id = snap.items[0]["id"].as_str().unwrap().to_string();
        assert!(store.remove_item(&id, &item_id));
        let snap2 = store.get(&id).unwrap();
        assert_eq!(snap2.items.len(), 1);
        assert_eq!(snap2.items[0]["content"], "b");
    }

    #[test]
    fn update_metadata_merges() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let mut md = HashMap::new();
        md.insert("k1".to_string(), "v1".to_string());
        let id = store.create(Vec::new(), md);
        let mut patch = HashMap::new();
        patch.insert("k2".to_string(), "v2".to_string());
        let snap = store.update_metadata(&id, patch).unwrap();
        assert_eq!(snap.metadata["k1"], "v1");
        assert_eq!(snap.metadata["k2"], "v2");
    }

    #[test]
    fn ttl_evicts() {
        let store = ConversationStore::with_config(16, Duration::from_millis(10));
        let id = store.create(Vec::new(), HashMap::new());
        std::thread::sleep(Duration::from_millis(30));
        assert!(store.get(&id).is_none());
    }

    #[test]
    fn delete_removes_entry() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(Vec::new(), HashMap::new());
        assert!(store.delete(&id));
        assert!(store.get(&id).is_none());
        assert!(!store.delete(&id));
    }
}
