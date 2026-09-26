// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The rail-count check `RailSet::complete` runs before it pairs
//! server params with rails.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

/// 2026-09-26: Refuse a server param count that differs from the rail count.
/// `complete` pairs them with `zip`, which would stop at the shorter list and
/// leave rails unconnected.
pub(crate) fn check_rail_count(server_len: usize, rails_len: usize, peer: &str) -> Result<()> {
    if server_len != rails_len {
        bail!(
            "{peer}: server returned {server_len} rail params for {rails_len} client rails — \
             refusing to leave rails unconnected (a zip would silently truncate)"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "railset_tests.rs"]
mod tests;
