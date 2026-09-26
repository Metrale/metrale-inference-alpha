// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Request / response dumper for the `--dump` CLI flag.
//!
//! Owner: server API.
//! Invariants:
//! - A request and its response entry carry the same `seq`, reserved once by
//!   `next_seq`.
//! - Emitting never waits on the disk and never fails a request: a full queue
//!   drops the entry, and a write error is logged once.
//!
//! For `/v1/chat/completions` and `/v1/messages`, the server appends one JSONL
//! entry per request whose body parses as JSON, and one per response. The file
//! I/O runs on a dedicated thread fed by a bounded queue, because an emit point
//! can be a sync fn inside a streaming future (`handle_done`), which cannot
//! await a `spawn_blocking`.

use parking_lot::Mutex;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::time::{SystemTime, UNIX_EPOCH};

/// 2026-09-26: Queue depth for pending dump lines; bounded so a stalled disk cannot
/// grow it without limit. Overflow drops entries and logs a warning on the
/// first drop.
const DUMP_QUEUE_DEPTH: usize = 4096;

/// 2026-09-26: Shared handle installed in `AppState::dump_writer` when `--dump` is
/// set and the file opened. Clones share one `Arc`, so one queue and one writer
/// thread.
#[derive(Clone)]
pub struct DumpHandle {
    inner: Arc<DumpInner>,
}

struct DumpInner {
    /// 2026-09-26: `None` only once `Drop` has closed the queue.
    tx: Mutex<Option<SyncSender<String>>>,
    writer: Mutex<Option<std::thread::JoinHandle<()>>>,
    path: std::path::PathBuf,
    seq: AtomicU64,
    /// 2026-09-26: Entries lost to a full queue. The first drop logs that the file is
    /// incomplete.
    dropped: AtomicU64,
    drop_logged: std::sync::atomic::AtomicBool,
}

/// 2026-09-26: Owns the file. Runs until every `DumpHandle` is gone, then flushes and exits.
fn writer_loop(
    rx: std::sync::mpsc::Receiver<String>,
    file: std::fs::File,
    path: std::path::PathBuf,
) {
    let mut w = std::io::BufWriter::new(file);
    let mut io_error_logged = false;
    while let Ok(line) = rx.recv() {
        // 2026-09-26: Flush per entry: a dump is read while the server is still
        // running, and a crash must not swallow the entry that explains it.
        if let Err(e) = w.write_all(line.as_bytes()).and_then(|_| w.flush())
            && !io_error_logged
        {
            io_error_logged = true;
            tracing::warn!(path = %path.display(), "dump write failed: {e}");
        }
    }
    let _ = w.flush();
}

impl DumpHandle {
    /// 2026-09-26: Open `path` in append mode, creating it if missing, and start the
    /// writer thread. An error returns to the caller; at startup it is logged and
    /// the server runs without dumping.
    pub fn open(path: std::path::PathBuf) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let (tx, rx) = sync_channel::<String>(DUMP_QUEUE_DEPTH);
        let thread_path = path.clone();
        let handle = std::thread::Builder::new()
            .name("metrale-dump".into())
            .spawn(move || writer_loop(rx, file, thread_path))?;
        Ok(Self {
            inner: Arc::new(DumpInner {
                tx: Mutex::new(Some(tx)),
                writer: Mutex::new(Some(handle)),
                path,
                seq: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
                drop_logged: std::sync::atomic::AtomicBool::new(false),
            }),
        })
    }

    /// 2026-09-26: Path of the dump file as given or resolved (for startup logging).
    pub fn path(&self) -> &std::path::Path {
        &self.inner.path
    }

    /// 2026-09-26: Reserve a sequence number. The caller passes the same `seq` when
    /// emitting the matching response entry.
    pub fn next_seq(&self) -> u64 {
        self.inner.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// 2026-09-26: Queue a `{"kind":"request",...}` entry. The handlers pass the raw
    /// request body parsed as a `serde_json::Value`.
    pub fn dump_request<T: serde::Serialize>(&self, endpoint: &str, seq: u64, body: &T) {
        self.write_entry("request", endpoint, seq, body, None);
    }

    /// 2026-09-26: Queue a `{"kind":"response",...}` entry. `is_stream` is true for a
    /// streamed response, whose body is the captured event list (`/v1/messages`)
    /// or a summary synthesized at the end of the stream (`/v1/chat/completions`).
    pub fn dump_response<T: serde::Serialize>(
        &self,
        endpoint: &str,
        seq: u64,
        body: &T,
        is_stream: bool,
    ) {
        self.write_entry("response", endpoint, seq, body, Some(is_stream));
    }

    fn write_entry<T: serde::Serialize>(
        &self,
        kind: &str,
        endpoint: &str,
        seq: u64,
        body: &T,
        is_stream: Option<bool>,
    ) {
        // 2026-09-26: Build the line before taking the lock, to keep the critical
        // section short.
        let mut obj = serde_json::Map::with_capacity(6);
        obj.insert("ts".into(), serde_json::Value::String(iso8601_now()));
        obj.insert("kind".into(), serde_json::Value::String(kind.into()));
        obj.insert(
            "endpoint".into(),
            serde_json::Value::String(endpoint.into()),
        );
        obj.insert("seq".into(), serde_json::Value::Number(seq.into()));
        if let Some(s) = is_stream {
            obj.insert("stream".into(), serde_json::Value::Bool(s));
        }
        match serde_json::to_value(body) {
            Ok(v) => {
                obj.insert("body".into(), v);
            }
            Err(e) => {
                // 2026-09-26: A body that fails to serialise is recorded as the error
                // string, so the entry still lands in the file.
                obj.insert(
                    "body".into(),
                    serde_json::Value::String(format!("<serialization error: {e}>")),
                );
            }
        }
        let mut line = match serde_json::to_string(&serde_json::Value::Object(obj)) {
            Ok(s) => s,
            Err(_) => return, // 2026-09-26: A `Value::Object` always serialises.
        };
        line.push('\n');

        // 2026-09-26: Hand off to the writer thread with `try_send`: a full queue
        // drops the entry rather than stalling a request on disk.
        let guard = self.inner.tx.lock();
        let Some(tx) = guard.as_ref() else { return };
        if let Err(TrySendError::Full(_)) = tx.try_send(line) {
            let n = self.inner.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            if !self.inner.drop_logged.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    path = %self.inner.path.display(),
                    "dump queue full — dropping entries ({n} so far); the dump file is INCOMPLETE"
                );
            }
        }
    }
}

impl Drop for DumpInner {
    /// 2026-09-26: Close the queue and join the writer, so every queued entry is
    /// flushed once the last handle is gone.
    fn drop(&mut self) {
        self.tx.lock().take();
        if let Some(h) = self.writer.lock().take() {
            let _ = h.join();
        }
    }
}

/// 2026-09-26: ISO-8601 UTC timestamp with milliseconds.
fn iso8601_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let days = secs.div_euclid(86400);
    let time_secs = secs.rem_euclid(86400) as u32;

    // 2026-09-26: Civil-from-days algorithm (Howard Hinnant). Unix epoch is day 0
    // = 1970-01-01.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };

    let hh = time_secs / 3600;
    let mm = (time_secs % 3600) / 60;
    let ss = time_secs % 60;

    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{millis:03}Z")
}

/// 2026-09-26: The `--dump` value that asks for a fresh timestamped file in the temp dir.
pub const DUMP_AUTO: &str = "auto";

/// 2026-09-26: Resolve the `--dump` argument to a file path. [`DUMP_AUTO`] maps to
/// `metrale-dump-<unix secs>.jsonl` in `std::env::temp_dir()`; anything else is
/// an explicit path.
pub fn resolve_path(arg: &str) -> std::path::PathBuf {
    if arg == DUMP_AUTO {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        std::env::temp_dir().join(format!("metrale-dump-{ts}.jsonl"))
    } else {
        std::path::PathBuf::from(arg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_auto_goes_to_tmp() {
        let p = resolve_path(DUMP_AUTO);
        assert!(p.starts_with(std::env::temp_dir()));
        assert!(
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("metrale-dump-")
        );
    }

    #[test]
    fn resolve_explicit_is_verbatim() {
        let p = resolve_path("/tmp/my-dump.jsonl");
        assert_eq!(p, std::path::PathBuf::from("/tmp/my-dump.jsonl"));
    }

    #[test]
    fn dump_writes_pair_with_shared_seq() {
        let tmp =
            std::env::temp_dir().join(format!("metrale-dump-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let h = DumpHandle::open(tmp.clone()).expect("open");

        let seq = h.next_seq();
        #[derive(serde::Serialize)]
        struct Req {
            model: &'static str,
        }
        #[derive(serde::Serialize)]
        struct Resp {
            ok: bool,
        }
        h.dump_request("/v1/chat/completions", seq, &Req { model: "test" });
        h.dump_response("/v1/chat/completions", seq, &Resp { ok: true }, false);

        drop(h); // 2026-09-26: Joins the writer thread, which flushes.
        let contents = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "two JSONL lines expected");
        let a: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let b: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(a["kind"], "request");
        assert_eq!(b["kind"], "response");
        assert_eq!(a["seq"], b["seq"], "request and response share seq");
        assert_eq!(a["body"]["model"], "test");
        assert_eq!(b["body"]["ok"], true);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn iso8601_has_expected_shape() {
        let s = iso8601_now();
        // 2026-09-26: YYYY-MM-DDTHH:MM:SS.sssZ is 24 chars.
        assert_eq!(s.len(), 24, "{s}");
        assert!(s.ends_with('Z'));
        assert_eq!(s.as_bytes()[10], b'T');
    }
}
