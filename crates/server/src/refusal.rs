// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Heuristic refusal detector: a prefix matcher over the assistant's opening text.
//!
//! Owner: server API.
//! Invariants:
//! - `detect` returns `None` whenever `METRALE_DISABLE_REFUSAL_DETECTION=1`.
//! - A returned sentence is a leading slice of the input, trimmed, in the
//!   model's original case.
//!
//! It fills `message.refusal` on the blocking chat path and the refusal delta
//! on the streaming path, both only when no tool call was found. It is not a
//! safety classifier: it misses refusals that open differently and fires on
//! content that opens by quoting one. `/v1/moderations` answers 501.

/// 2026-09-26: Prefixes matched against the lowercased first 48 characters of the
/// message after leading whitespace. Any match counts, so order does not matter.
const REFUSAL_PREFIXES: &[&str] = &[
    "i cannot ",
    "i can't help with ",
    "i can't assist with ",
    "i'm not able to ",
    "i am not able to ",
    "i'm unable to ",
    "i am unable to ",
    "i must decline",
    "i won't assist",
    "i will not assist",
    "i won't help",
    "i will not help",
    "sorry, i cannot",
    "sorry, but i can't",
    "sorry, but i cannot",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i apologize, but i can't",
    "i apologize, but i cannot",
    "as an ai, i cannot",
    "as an ai, i can't",
    "as an ai language model, i cannot",
    "as an ai language model, i can't",
];

/// 2026-09-26: Returns the refusal sentence when `content` opens with one of the
/// known patterns, else `None`. The sentence ends at the first `.`, `?` or `!`
/// within 512 characters, else at the first newline, and is trimmed. When
/// `METRALE_DISABLE_REFUSAL_DETECTION=1`, always returns `None`.
pub fn detect(content: &str) -> Option<String> {
    if std::env::var("METRALE_DISABLE_REFUSAL_DETECTION").as_deref() == Ok("1") {
        return None;
    }
    let trimmed = content.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    // 2026-09-26: Compare against a lowercase view but return the original-cased
    // sentence, so the client sees the model's exact words.
    let head: String = trimmed
        .chars()
        .take(48)
        .flat_map(|c| c.to_lowercase())
        .collect();
    let matched = REFUSAL_PREFIXES.iter().any(|p| head.starts_with(p));
    if !matched {
        return None;
    }
    // 2026-09-26: The first sentence ends at the first terminal punctuation within
    // 512 characters; with none, fall back to the first line.
    let end_idx = trimmed
        .char_indices()
        .take(512)
        .find(|(_, c)| matches!(c, '.' | '?' | '!'))
        .map(|(i, c)| i + c.len_utf8());
    let sentence = match end_idx {
        Some(i) => &trimmed[..i],
        None => trimmed
            .split_once('\n')
            .map(|(line, _)| line)
            .unwrap_or(trimmed),
    };
    Some(sentence.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // 2026-09-26: Unit tests run in parallel threads of one process, and env vars
    // are process-wide. `kill_switch_returns_none` sets
    // METRALE_DISABLE_REFUSAL_DETECTION, so every test here that calls `detect()`
    // holds this lock.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn matches_canonical_refusal() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let r = detect("I cannot help with that request. Here is why…").unwrap();
        assert_eq!(r, "I cannot help with that request.");
    }

    #[test]
    fn matches_with_leading_whitespace() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let r = detect("   I'm sorry, but I can't assist with weapons design.").unwrap();
        assert_eq!(r, "I'm sorry, but I can't assist with weapons design.");
    }

    #[test]
    fn mixed_case_matches() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(detect("As AN ai, I cannot provide that.").is_some());
    }

    #[test]
    fn non_refusal_returns_none() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(detect("Sure, here's how to do that.").is_none());
        assert!(detect("").is_none());
        assert!(detect("I can do that for you.").is_none());
    }

    #[test]
    fn kill_switch_returns_none() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 2026-09-26: SAFETY: serialized by ENV_LOCK against this module's tests only.
        unsafe {
            std::env::set_var("METRALE_DISABLE_REFUSAL_DETECTION", "1");
        }
        let got = detect("I cannot help with that.");
        // 2026-09-26: SAFETY: serialized by ENV_LOCK against this module's tests only.
        unsafe {
            std::env::remove_var("METRALE_DISABLE_REFUSAL_DETECTION");
        }
        assert!(got.is_none());
    }

    #[test]
    fn no_terminator_falls_back_to_line() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let r = detect("I cannot answer that\nnext paragraph").unwrap();
        assert_eq!(r, "I cannot answer that");
    }
}
