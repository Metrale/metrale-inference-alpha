// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: LRU + TTL store for Responses API resume (`previous_response_id`) and
//! Chat Completions `store: true`.
//!
//! Owner: server API.
//! Invariants:
//! - Every entry is `Response` or `ChatCompletion`, and `get`/`delete` answer
//!   only for the kind asked, so a chat id passed as `previous_response_id`
//!   finds nothing.
//! - A disk write or delete failure is logged, never returned to a request.
//!
//! Two evictions: an entry idle longer than the TTL is dropped when `get`
//! finds it (`insert` does not check the TTL), and past the capacity `insert`
//! drops the least recently used entry. With `METRALE_STORE_DIR` set, each
//! `insert` writes a JSON file and each removal deletes it; at startup the
//! directory is replayed, deleting files older than the TTL. The runtime
//! values come from `ResponseStore::from_env` (`store_impl.rs`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::openai::IncomingMessage;

/// 2026-09-26: What kind of object is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoredKind {
    Response,
    ChatCompletion,
}

impl StoredKind {
    pub fn id_prefix(self) -> &'static str {
        match self {
            StoredKind::Response => "resp_",
            StoredKind::ChatCompletion => "chatcmpl-",
        }
    }
}

/// 2026-09-26: One stored entry. The in-memory copy carries `last_access`, an
/// `Instant`; the on-disk copy carries `persisted_at_unix` instead, so replay
/// can apply the TTL across a restart.
pub struct StoredEntry {
    pub id: String,
    pub kind: StoredKind,
    pub model: String,
    pub created_at: u64,
    pub messages: Vec<IncomingMessage>,
    pub body: serde_json::Value,
    pub last_access: Instant,
}

/// 2026-09-26: Disk layout: `StoredEntry` with a wall-clock `persisted_at_unix` in
/// place of the `Instant`. `messages` is the JSON that `replay` deserializes
/// back into `IncomingMessage`.
#[derive(Serialize, Deserialize)]
struct DiskEntry {
    id: String,
    kind: StoredKind,
    model: String,
    created_at: u64,
    messages: serde_json::Value,
    body: serde_json::Value,
    persisted_at_unix: u64,
}

/// 2026-09-26: Persistence backend. `forget` can be called with the store's lock
/// held (eviction, expiry, delete), so implementations must not block on I/O
/// there; `FilesystemBackend` queues its disk work to a thread.
pub trait StoreBackend: Send + Sync {
    fn persist(&self, entry: &StoredEntry);
    fn forget(&self, id: &str);
    /// 2026-09-26: Called once when the store is built; returns the entries on disk
    /// whose TTL has not elapsed.
    fn replay(&self, ttl: Duration) -> Vec<StoredEntry>;
}

/// 2026-09-26: No-op backend used when persistence is disabled.
struct NoopBackend;
impl StoreBackend for NoopBackend {
    fn persist(&self, _entry: &StoredEntry) {}
    fn forget(&self, _id: &str) {}
    fn replay(&self, _ttl: Duration) -> Vec<StoredEntry> {
        Vec::new()
    }
}

/// 2026-09-26: Queue depth for pending disk operations, bounded so a stalled disk
/// cannot grow it without limit. A full queue does not drop an op: `submit` runs
/// it inline.
const DISK_QUEUE_DEPTH: usize = 1024;

enum DiskOp {
    Persist(Box<DiskEntry>),
    Forget(String),
}

/// 2026-09-26: Persists entries from a dedicated thread, one JSON file per entry at
/// `{dir}/{sanitize_id(id)}.json`.
///
/// `persist` and `forget` are reached from async request handlers
/// (`finalize_responses_stream`, `translate_chat_response_to_responses`) through
/// the sync `ResponseStore::insert`, so the disk work leaves the request thread.
/// Writes and deletes share one FIFO queue, so while it has room they run in
/// submission order; the inline fallback on a full queue can run an op ahead of
/// queued ones.
pub struct FilesystemBackend {
    dir: PathBuf,
    /// 2026-09-26: `None` only once `Drop` has closed the queue.
    tx: Mutex<Option<std::sync::mpsc::SyncSender<DiskOp>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// 2026-09-26: Latches the "disk queue full" warning, so it is logged once per
    /// backend.
    queue_full_warned: std::sync::atomic::AtomicBool,
}

fn write_to_disk(dir: &std::path::Path, disk: &DiskEntry) {
    let path = dir.join(format!("{}.json", sanitize_id(&disk.id)));
    let tmp = path.with_extension("json.tmp");
    let bytes = match serde_json::to_vec(disk) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("response_store: serialize failed for {}: {e}", disk.id);
            return;
        }
    };
    // 2026-09-26: Write-then-rename for crash-atomicity.
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        tracing::warn!("response_store: write {}: {e}", tmp.display());
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        tracing::warn!("response_store: rename {}: {e}", path.display());
    }
}

fn remove_from_disk(dir: &std::path::Path, id: &str) {
    let path = dir.join(format!("{}.json", sanitize_id(id)));
    if let Err(e) = std::fs::remove_file(&path)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!("response_store: remove {}: {e}", path.display());
    }
}

impl FilesystemBackend {
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<DiskOp>(DISK_QUEUE_DEPTH);
        let worker_dir = dir.clone();
        let worker = std::thread::Builder::new()
            .name("metrale-respstore".into())
            .spawn(move || {
                while let Ok(op) = rx.recv() {
                    match op {
                        DiskOp::Persist(d) => write_to_disk(&worker_dir, &d),
                        DiskOp::Forget(id) => remove_from_disk(&worker_dir, &id),
                    }
                }
            })?;
        Ok(Self {
            dir,
            tx: Mutex::new(Some(tx)),
            worker: Mutex::new(Some(worker)),
            queue_full_warned: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// 2026-09-26: Enqueue a disk op, or run it inline if the queue is full.
    ///
    /// Unlike the request dumper, this never drops an op: a lost write means a
    /// resumable response disappears after a restart. Under overload the caller
    /// takes the blocking write instead, and the first time is logged.
    fn submit(&self, op: DiskOp) {
        let guard = self.tx.lock();
        let Some(tx) = guard.as_ref() else {
            return;
        };
        if let Err(std::sync::mpsc::TrySendError::Full(op)) = tx.try_send(op) {
            if !self
                .queue_full_warned
                .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                tracing::warn!(
                    "response_store: disk queue full — falling back to inline writes                      (persistence is keeping up poorly; requests will see the write latency)"
                );
            }
            match op {
                DiskOp::Persist(d) => write_to_disk(&self.dir, &d),
                DiskOp::Forget(id) => remove_from_disk(&self.dir, &id),
            }
        }
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", sanitize_id(id)))
    }
}

/// 2026-09-26: Maximum stem length, in characters, whatever a client sends. The
/// stem is ASCII, so a file name is at most 96 + `.json.tmp` = 105 bytes.
const MAX_STEM: usize = 96;

/// 2026-09-26: Map a client-supplied response id to a safe filename stem.
///
/// Response ids reach this from request bodies, so the result must not be able
/// to leave the store directory (CWE-22). The allowlist is `[A-Za-z0-9_-]`;
/// every other char, including `.`, `/`, `\\` and NUL, becomes `_`. With no
/// dots, `..` cannot be expressed, so traversal is impossible by construction.
/// The mapping is lossy, which is safe for reading: `replay` takes the id from
/// the JSON body, not the file name.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .take(MAX_STEM)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 2026-09-26: Convert `Vec<IncomingMessage>` to a JSON array that replay reads back
/// through the same `IncomingMessage` deserializer that serves inbound requests
/// (`IncomingMessage` derives no `Serialize`). Written: role, text, image URIs,
/// `tool_calls`, `tool_call_id` and `name`; videos and `reasoning_content` are
/// not written.
fn messages_to_disk_json(msgs: &[IncomingMessage]) -> serde_json::Value {
    serde_json::Value::Array(
        msgs.iter()
            .map(|m| {
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), serde_json::Value::String(m.role.clone()));
                if !m.content.has_images() {
                    obj.insert(
                        "content".into(),
                        serde_json::Value::String(m.content.text.clone()),
                    );
                } else {
                    // 2026-09-26: Multi-part content: a text part, then each image URI
                    // as an `image_url` part, the chat content array shape the
                    // deserializer reads.
                    let mut parts: Vec<serde_json::Value> = Vec::new();
                    if !m.content.text.is_empty() {
                        parts.push(serde_json::json!({
                            "type": "text",
                            "text": m.content.text,
                        }));
                    }
                    for img in m.content.images() {
                        parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": img },
                        }));
                    }
                    obj.insert("content".into(), serde_json::Value::Array(parts));
                }
                if let Some(tc) = &m.tool_calls
                    && let Ok(v) = serde_json::to_value(tc)
                {
                    obj.insert("tool_calls".into(), v);
                }
                if let Some(id) = &m.tool_call_id {
                    obj.insert("tool_call_id".into(), serde_json::Value::String(id.clone()));
                }
                if let Some(n) = &m.name {
                    obj.insert("name".into(), serde_json::Value::String(n.clone()));
                }
                serde_json::Value::Object(obj)
            })
            .collect(),
    )
}

impl Drop for FilesystemBackend {
    /// 2026-09-26: Close the queue and join the writer, so a dropped store has
    /// finished its disk work before a fresh store replays the same directory.
    fn drop(&mut self) {
        self.tx.lock().take();
        if let Some(h) = self.worker.lock().take() {
            let _ = h.join();
        }
    }
}

impl StoreBackend for FilesystemBackend {
    fn persist(&self, entry: &StoredEntry) {
        let disk = DiskEntry {
            id: entry.id.clone(),
            kind: entry.kind,
            model: entry.model.clone(),
            created_at: entry.created_at,
            messages: messages_to_disk_json(&entry.messages),
            body: entry.body.clone(),
            persisted_at_unix: now_unix(),
        };
        self.submit(DiskOp::Persist(Box::new(disk)));
    }

    fn forget(&self, id: &str) {
        self.submit(DiskOp::Forget(id.to_string()));
    }

    fn replay(&self, ttl: Duration) -> Vec<StoredEntry> {
        let mut out = Vec::new();
        let rd = match std::fs::read_dir(&self.dir) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("response_store: read_dir {}: {e}", self.dir.display());
                return out;
            }
        };
        let now = now_unix();
        let ttl_s = ttl.as_secs();
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let bytes = match std::fs::read(&p) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("response_store: read {}: {e}", p.display());
                    continue;
                }
            };
            let disk: DiskEntry = match serde_json::from_slice(&bytes) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("response_store: parse {}: {e}", p.display());
                    // 2026-09-26: Leave the file for an operator to inspect.
                    continue;
                }
            };
            if now.saturating_sub(disk.persisted_at_unix) > ttl_s {
                // 2026-09-26: Expired on disk: remove and skip.
                let _ = std::fs::remove_file(&p);
                continue;
            }
            let messages: Vec<IncomingMessage> = match serde_json::from_value(disk.messages) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(
                        "response_store: messages shape drifted for {}: {e}",
                        disk.id
                    );
                    continue;
                }
            };
            out.push(StoredEntry {
                id: disk.id,
                kind: disk.kind,
                model: disk.model,
                created_at: disk.created_at,
                messages,
                body: disk.body,
                last_access: Instant::now(),
            });
        }
        out
    }
}

pub struct ResponseStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    max_entries: usize,
    backend: Box<dyn StoreBackend>,
    /// 2026-09-26: True when the filesystem backend is attached; read through
    /// `is_persistent` for startup logging.
    persistent: bool,
    persist_dir: Option<PathBuf>,
}

struct Inner {
    map: HashMap<String, StoredEntry>,
    order: std::collections::VecDeque<String>,
}

pub struct GetResult {
    pub model: String,
    pub created_at: u64,
    pub messages: Vec<IncomingMessage>,
    pub body: serde_json::Value,
}

mod store_impl;

#[cfg(test)]
mod tests;
