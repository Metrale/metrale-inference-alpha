// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`RunDumps`]: the diagnostic file sinks one scheduler run writes to.
//!
//! Serving builds it with [`RunDumps::from_env`] (through
//! `SchedIo::serving`); `RunDumps::default()` opens nothing.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::sync::Mutex;

/// 2026-09-25: Diagnostic file sinks for one scheduler run.
#[derive(Debug, Default)]
pub struct RunDumps {
    /// 2026-09-25: `METRALE_LOGIT_DUMP=path`: the JSONL file
    /// `logit_dump::record` appends to.
    pub logits: Option<Mutex<BufWriter<File>>>,
    /// 2026-09-25: `METRALE_ADADEC_DIAGNOSTIC=dir`: `dir/adadec_entropy.jsonl`,
    /// the adaptive-decode entropy trace.
    pub adadec: Option<Mutex<File>>,
    /// 2026-09-25: `METRALE_DUMP_LOGITS_PATH=dir`: raw FP32 logits rows are
    /// appended to `logits_seq.bin` and `logits_stok.bin` under it. `None` =
    /// no raw dump.
    pub raw_logits_dir: Option<std::path::PathBuf>,
}

impl RunDumps {
    /// 2026-09-25: Open whichever sinks this run's environment asks for.
    pub fn from_env() -> Self {
        Self {
            raw_logits_dir: std::env::var("METRALE_DUMP_LOGITS_PATH")
                .ok()
                .map(std::path::PathBuf::from),
            logits: Self::open_append("METRALE_LOGIT_DUMP", |p| p.to_path_buf())
                .map(|f| Mutex::new(BufWriter::new(f))),
            adadec: Self::open_append("METRALE_ADADEC_DIAGNOSTIC", |p| {
                p.join("adadec_entropy.jsonl")
            })
            .map(Mutex::new),
        }
    }

    /// 2026-09-25: Resolve `var` to a path, run `to_file` over it, and open
    /// for append. `None` when the variable is unset or empty; an open
    /// failure is logged and treated as unset, so a bad diagnostic path
    /// never fails a serve.
    fn open_append(
        var: &str,
        to_file: impl FnOnce(&std::path::Path) -> std::path::PathBuf,
    ) -> Option<File> {
        let raw = std::env::var(var).ok()?;
        if raw.is_empty() {
            return None;
        }
        let base = std::path::Path::new(&raw);
        let path = to_file(base);
        if let Some(dir) = path.parent()
            && dir != std::path::Path::new("")
            && raw.as_str() == base.to_string_lossy()
            && path != base
        {
            let _ = std::fs::create_dir_all(dir);
        }
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => Some(f),
            Err(e) => {
                tracing::error!("{var}: cannot open {}: {e}", path.display());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_environment_opens_nothing() {
        let d = RunDumps::default();
        assert!(d.logits.is_none() && d.adadec.is_none());
    }

    #[test]
    fn two_runs_hold_independent_sinks() {
        let a = RunDumps::default();
        let b = RunDumps::default();
        assert!(a.logits.is_none());
        assert!(b.logits.is_none());
    }
}
