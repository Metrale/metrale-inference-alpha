// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Copy text to the terminal's clipboard by writing an OSC 52
//! escape sequence to stdout. Nothing is read back, so a successful copy means
//! the sequence was written, not that the terminal applied it.
//!
//! Owner: server tui.
//! Invariants:
//! - `osc52` returns `None` for empty text and for text whose base64 is longer
//!   than `MAX_BYTES`; nothing longer is written.

use base64::Engine as _;

/// 2026-09-26: The longest base64 payload `osc52` will encode; longer text is
/// refused rather than sent.
const MAX_BYTES: usize = 74_994;

/// 2026-09-26: The bytes to write to the terminal to set its clipboard to
/// `text`: `ESC ] 52 ; c ; <standard base64> BEL`. `None` for empty text or a
/// payload over `MAX_BYTES`.
pub fn osc52(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() {
        return None;
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    if b64.len() > MAX_BYTES {
        return None;
    }
    Some(format!("\x1b]52;c;{b64}\x07").into_bytes())
}

/// 2026-09-26: Would this non-empty text be refused for being too large?
pub fn too_large(text: &str) -> bool {
    !text.is_empty() && osc52(text).is_none()
}

/// 2026-09-26: Write `text` to the terminal's clipboard. `Ok` carries the
/// character count written; `Err` carries the message to show (empty text, too
/// large, or a failed write).
pub fn copy(text: &str) -> Result<usize, String> {
    if text.is_empty() {
        return Err("nothing selected".into());
    }
    let Some(seq) = osc52(text) else {
        return Err(format!(
            "selection is too large to copy ({} chars; the limit is about {} )",
            text.chars().count(),
            MAX_BYTES / 4 * 3
        ));
    };
    write_raw(&seq).map_err(|e| format!("could not write to the terminal: {e}"))?;
    Ok(text.chars().count())
}

/// 2026-09-26: Write to stdout directly, not through the ratatui frame buffer,
/// and flush.
fn write_raw(bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    out.write_all(bytes)?;
    out.flush()
}

#[cfg(test)]
#[path = "clipboard_tests.rs"]
mod tests;
