// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: How the dashboard prints byte counts, rates, percentages, the MTP gate state and wrapped text.
//!
//! Owner: server tui.
//! Invariants:
//! - [`bytes`] and [`rate`] scale by powers of 1024, so their `GB` is 1024³ bytes.

use metrale_speculative::snapshot::MtpModeSnap;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: u64 = 1024 * 1024;
const KIB: f64 = 1024.0;

/// 2026-09-26: A byte count for a human: `18.6 GB` at or above 1024³ bytes, whole `MB` below.
///
/// Below a gibibyte the value is truncated, not rounded, so one byte short of
/// a gibibyte reads `1023 MB`. `GB` here means 1024³, the same divisor as the
/// memory figures in `data/metrics_poll.rs`.
pub fn bytes(n: u64) -> String {
    let g = n as f64 / GIB;
    if g >= 1.0 {
        format!("{g:.1} GB")
    } else {
        format!("{} MB", n / MIB)
    }
}

/// 2026-09-26: A completion fraction as a percentage: one decimal below 10%, none from 10% up.
///
/// The decimal lets a download below 1% show progress (`0.4%`) instead of
/// `0%`. The input is clamped to 0..=1, and the result is unpadded.
pub fn percent(frac: f64) -> String {
    let pct = frac.clamp(0.0, 1.0) * 100.0;
    if pct < 10.0 {
        format!("{pct:.1}%")
    } else {
        format!("{pct:.0}%")
    }
}

/// 2026-09-26: A byte-per-second rate for a human: `92 MB/s`, `3.0 MB/s`, `2.0 KB/s`, `0 B/s`.
///
/// 1024-based like [`bytes`], so a rate divides into the size printed beside
/// it. One decimal below ten (never for `B/s`), none above; negative or NaN
/// input prints `0 B/s`.
pub fn rate(bytes_per_sec: f64) -> String {
    let (n, unit) = match bytes_per_sec {
        b if b >= GIB => (b / GIB, "GB"),
        b if b >= MIB as f64 => (b / MIB as f64, "MB"),
        b if b >= KIB => (b / KIB, "KB"),
        b => (b.max(0.0), "B"),
    };
    if n < 10.0 && unit != "B" {
        format!("{n:.1} {unit}/s")
    } else {
        format!("{n:.0} {unit}/s")
    }
}

/// 2026-09-26: What the scheduler's MTP gate is doing, in words.
///
/// Not `{:?}`: a variant rename would silently change what the dashboard says.
pub fn mtp_mode_label(mode: MtpModeSnap) -> &'static str {
    match mode {
        MtpModeSnap::Mtp => "speculative",
        MtpModeSnap::Serial => "serial",
        MtpModeSnap::Probing => "probing",
        MtpModeSnap::Off => "off",
    }
}

/// 2026-09-26: Word-wrap one paragraph to `width` columns, as owned lines; empty for width 0.
///
/// `render::wrap` and [`wrap_help`] both wrap through this loop. Width is
/// measured in bytes, so non-ASCII text wraps early rather than late. A word
/// wider than `width` is split at character boundaries.
pub fn wrap_words(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        if word.len() > width {
            for ch in word.chars() {
                if !current.is_empty() && current.len() + ch.len_utf8() > width {
                    lines.push(std::mem::take(&mut current));
                }
                current.push(ch);
            }
        } else {
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// 2026-09-26: Wrap multi-paragraph help text to `width`, keeping one blank line between paragraphs.
///
/// [`wrap_words`] collapses all whitespace, which would merge the paragraphs
/// of a multi-paragraph clap doc such as `--kv-cache-dtype`'s. Leading and
/// trailing blank lines are dropped.
pub fn wrap_help(text: &str, width: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut paragraph = String::new();
    let flush = |paragraph: &mut String, out: &mut Vec<String>| {
        if !paragraph.is_empty() {
            out.extend(wrap_words(paragraph, width));
            paragraph.clear();
        }
    };
    for line in text.lines() {
        if line.trim().is_empty() {
            flush(&mut paragraph, &mut out);
            if out.last().is_some_and(|l| !l.is_empty()) {
                out.push(String::new());
            }
        } else {
            if !paragraph.is_empty() {
                paragraph.push(' ');
            }
            paragraph.push_str(line.trim());
        }
    }
    flush(&mut paragraph, &mut out);
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    out
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
