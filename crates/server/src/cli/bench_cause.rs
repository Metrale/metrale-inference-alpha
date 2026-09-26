// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The final `Error:` block of a child's log, quoted into the error that reports it.
//!
//! Owner: server CLI (`met benchmark`).
//! Callers: the serve lease (`bench_lease::exited_before_serving`) quotes
//! `serve-lease.log`, and certify placement (`bench_certify/remote/place.rs`)
//! quotes the gate child's log.
//! Invariants: none beyond the types.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// 2026-09-26: How many bytes of a log's tail the callers read to find its final block.
pub(crate) const TAIL_BYTES: u64 = 64 * 1024;

/// 2026-09-26: The longest block, in bytes, that [`final_error_block`] returns uncut.
pub(crate) const CAUSE_CAP: usize = 4096;

/// 2026-09-26: The last `Error:` block in `text`: from the final line that starts with
/// `Error:` to the end, trimmed. A block longer than [`CAUSE_CAP`] bytes is cut
/// on a char boundary and ends with `… (+N bytes in the log)`. `None` when no
/// line starts with `Error:`.
pub(crate) fn final_error_block(text: &str) -> Option<String> {
    let start = text
        .rmatch_indices("Error:")
        .map(|(i, _)| i)
        .find(|&i| i == 0 || text.as_bytes()[i - 1] == b'\n')?;
    let block = text[start..].trim();
    if block.len() <= CAUSE_CAP {
        return Some(block.to_string());
    }
    let mut cut = CAUSE_CAP;
    while !block.is_char_boundary(cut) {
        cut -= 1;
    }
    Some(format!(
        "{}… (+{} bytes in the log)",
        &block[..cut],
        block.len() - cut
    ))
}

/// 2026-09-26: The last `bytes` of a file, lossily decoded. A file no longer than
/// `bytes` is read whole. Otherwise the window drops everything up to and
/// including its first newline, unless that newline is the window's last byte,
/// in which case the window is returned as read.
pub(crate) fn tail_of_file(path: &Path, bytes: u64) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut text = String::new();
    if len > bytes {
        f.seek(SeekFrom::Start(len - bytes))?;
        let mut raw = Vec::new();
        f.read_to_end(&mut raw)?;
        let cut = match raw.iter().position(|b| *b == b'\n') {
            Some(i) if i + 1 < raw.len() => i + 1,
            _ => 0,
        };
        text = String::from_utf8_lossy(&raw[cut..]).into_owned();
    } else {
        f.read_to_string(&mut text)?;
    }
    Ok(text)
}

/// 2026-09-26: `block` with every line indented by four spaces, so no line of it starts
/// with `Error:` and [`final_error_block`] on the embedding error finds that
/// error's own `Error:` line.
pub(crate) fn indented(block: &str) -> String {
    block
        .lines()
        .map(|l| format!("    {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[path = "bench_cause_tests.rs"]
mod tests;
