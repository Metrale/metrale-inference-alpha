// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Launch trace: record the kernel launches, async memsets and
//! device-to-device copies a region enqueues, so two executions of it can be
//! diffed op by op ([`end_and_diff`]). The verify paths trace their captured
//! region this way to find a host value that differs between two steps. Off
//! until [`begin`] is called.
//!
//! Owner: telemetry.
//! Invariants: [`record`] stores an op only between [`begin`] and the next
//! [`end_and_diff`].

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// 2026-09-26: One enqueued op. `kind` separates kernels from memsets and
/// copies, so an op that appears or vanishes shows as an op mismatch.
#[derive(Clone, PartialEq, Eq)]
pub struct Entry {
    pub kind: &'static str,
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    /// 2026-09-26: Args as u64 words: a buffer is its raw address, a byte
    /// argument its first 8 bytes, little-endian.
    pub args: Vec<u64>,
}

static ON: AtomicBool = AtomicBool::new(false);
static TRACE: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
static PREV: Mutex<Option<Vec<Entry>>> = Mutex::new(None);
static NAMES: Mutex<Option<HashMap<u64, String>>> = Mutex::new(None);

#[inline(always)]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// 2026-09-26: Remember a kernel handle's name. The CUDA backend's
/// `GpuBackend::kernel` calls it on every successful lookup.
pub fn name_kernel(handle: u64, module: &str, func: &str) {
    let mut g = NAMES.lock().unwrap();
    g.get_or_insert_with(HashMap::new)
        .insert(handle, format!("{module}::{func}"));
}

/// 2026-09-26: The `module::func` a kernel handle was looked up as, if it was named.
pub fn kernel_name(handle: u64) -> Option<String> {
    NAMES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&handle).cloned())
}

fn name_of(handle: u64) -> String {
    kernel_name(handle).unwrap_or_else(|| format!("fn@{handle:#x}"))
}

pub fn begin() {
    TRACE.lock().unwrap().clear();
    ON.store(true, Ordering::Relaxed);
}

#[inline(always)]
pub fn record(e: Entry) {
    if on() {
        TRACE.lock().unwrap().push(e);
    }
}

/// 2026-09-26: Stop recording and diff this trace against the previous one.
/// Returns `None` on the first call, else a report of how many ops differ,
/// detailing at most `max_report` of them.
pub fn end_and_diff(max_report: usize) -> Option<String> {
    ON.store(false, Ordering::Relaxed);
    let cur = std::mem::take(&mut *TRACE.lock().unwrap());
    let prev = PREV.lock().unwrap().replace(cur.clone())?;

    let mut out = String::new();
    let mut n = 0usize;
    if prev.len() != cur.len() {
        out.push_str(&format!(
            "OP COUNT differs: prev {} vs cur {}\n",
            prev.len(),
            cur.len()
        ));
    }
    for (i, (p, c)) in prev.iter().zip(cur.iter()).enumerate() {
        if p == c {
            continue;
        }
        n += 1;
        if n > max_report {
            continue;
        }
        let mut what = Vec::new();
        if p.kind != c.kind || p.func != c.func {
            what.push(format!("op {} -> {}", name_of(p.func), name_of(c.func)));
        }
        if p.grid != c.grid {
            what.push(format!("grid {:?} -> {:?}", p.grid, c.grid));
        }
        if p.block != c.block {
            what.push(format!("block {:?} -> {:?}", p.block, c.block));
        }
        if p.smem != c.smem {
            what.push(format!("smem {} -> {}", p.smem, c.smem));
        }
        for (a, (pv, cv)) in p.args.iter().zip(c.args.iter()).enumerate() {
            if pv != cv {
                what.push(format!(
                    "arg{a} {pv:#x} -> {cv:#x} (Δ {})",
                    *cv as i64 - *pv as i64
                ));
            }
        }
        if p.args.len() != c.args.len() {
            what.push(format!("argc {} -> {}", p.args.len(), c.args.len()));
        }
        out.push_str(&format!(
            "#{i} {} [{}] {}\n",
            name_of(c.func),
            c.kind,
            what.join("; ")
        ));
    }
    Some(format!(
        "{n} differing op(s) of {}\n{out}",
        cur.len().min(prev.len())
    ))
}
