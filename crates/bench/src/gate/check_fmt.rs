// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Message rendering for the gate check: a list of paths as one readable line.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types. It decides no verdict, so it is not a boundary file.

/// 2026-09-26: Paths for a one-line message: up to three names, then "and N more".
pub(super) fn summarize_paths(paths: &[String]) -> String {
    const SHOWN: usize = 3;
    if paths.len() <= SHOWN {
        return paths.join(", ");
    }
    format!(
        "{} and {} more",
        paths[..SHOWN].join(", "),
        paths.len() - SHOWN
    )
}
