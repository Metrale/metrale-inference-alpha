// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SpillIo`] over files: the swap files a sequence is spilled
//! to. The device router's `SaveSequenceState` effect streams the state into
//! the writer this router hands out; the router owns the files, their ids
//! and the usage ledger.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::io::{Read, Write};

use anyhow::Result;
use metrale_cache::kv_spill::KvSpillManager;
use metrale_scheduler::SpillIo;

/// 2026-09-25: The on-disk spill store, one directory per process.
pub struct FileSpill(parking_lot::Mutex<KvSpillManager>);

impl FileSpill {
    pub fn new(manager: KvSpillManager) -> Self {
        Self(parking_lot::Mutex::new(manager))
    }
}

impl SpillIo for FileSpill {
    fn create(&self) -> Result<(u64, Box<dyn Write + Send>)> {
        let (id, writer) = self.0.lock().create_file()?;
        Ok((id, Box::new(writer)))
    }
    fn open(&self, id: u64) -> Result<Box<dyn Read + Send>> {
        Ok(Box::new(self.0.lock().open_file(id)?))
    }
    fn remove(&self, id: u64) -> Result<()> {
        self.0.lock().remove_file(id)
    }
    fn record_usage(&self, id: u64) {
        self.0.lock().record_usage(id);
    }
}
