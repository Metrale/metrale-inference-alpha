// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Environment lookups over a key list the caller passes: the RDMA
//! clients read their device names, GID indices and depths through these. No
//! key or fallback chain is fixed here.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

/// 2026-09-26: The value of the first key that is set, even to the empty
/// string; `default` when none is.
pub fn first_set(keys: &[&str], default: &str) -> String {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            return v;
        }
    }
    default.to_string()
}

/// 2026-09-26: The value of the first key that is set and non-empty;
/// `default` when none is.
pub fn first_nonempty(keys: &[&str], default: &str) -> String {
    for k in keys {
        if let Ok(v) = std::env::var(k)
            && !v.is_empty()
        {
            return v;
        }
    }
    default.to_string()
}

/// 2026-09-26: The first value that parses as `u32`; a set but unparseable key
/// is skipped. `default` when none parses.
pub fn first_set_u32(keys: &[&str], default: u32) -> u32 {
    for k in keys {
        if let Some(v) = std::env::var(k).ok().and_then(|s| s.parse().ok()) {
            return v;
        }
    }
    default
}
