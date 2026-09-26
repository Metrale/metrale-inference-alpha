// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SpillIo`], the swap files a sequence is spilled to when the KV pool
//! runs dry. The router owns the files, their ids and the usage ledger.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::io::{Read, Write};

use anyhow::Result;

pub trait SpillIo: Send + Sync {
    /// 2026-09-25: A fresh swap file: its id and a writer the device state streams into.
    fn create(&self) -> Result<(u64, Box<dyn Write + Send>)>;
    /// 2026-09-25: The reader for swap file `id`.
    fn open(&self, id: u64) -> Result<Box<dyn Read + Send>>;
    fn remove(&self, id: u64) -> Result<()>;
    /// 2026-09-25: Account the finished write of `id` against the swap budget.
    fn record_usage(&self, id: u64);
}
