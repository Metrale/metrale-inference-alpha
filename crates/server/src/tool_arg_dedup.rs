// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-call dedup: trips when the same call (name plus arguments) repeats.
//!
//! Owner: server API.
//! Invariants: arguments that parse as JSON are hashed in canonical form (sorted
//! keys, no whitespace), so formatting does not separate equal calls; anything
//! else is hashed as the raw string.
//!
//! This catches a repeated call even when the narration around it varies. The
//! ring holds the last `cap` hashes. A call trips when, counting itself, it
//! ends a run of `threshold_consec` identical hashes, or when it and its copies
//! in the ring number `threshold_window`. `new()` uses cap 8, consec 5 and
//! window 6; the streaming chat path also builds one with 4, 2 and 3.

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

const DEFAULT_CAP: usize = 8;
const DEFAULT_CONSEC: u32 = 5;
const DEFAULT_WINDOW: u32 = 6;

/// 2026-09-26: Tool-call argument hash dedup guard.
#[derive(Debug)]
pub struct ToolArgDedup {
    recent: VecDeque<u64>,
    cap: usize,
    threshold_consec: u32,
    threshold_window: u32,
}

impl Default for ToolArgDedup {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolArgDedup {
    /// 2026-09-26: Construct with cap 8, consec 5, window 6.
    pub fn new() -> Self {
        Self::with_params(DEFAULT_CAP, DEFAULT_CONSEC, DEFAULT_WINDOW)
    }

    /// 2026-09-26: Construct with explicit ring capacity (at least 1) and trip thresholds.
    pub fn with_params(cap: usize, threshold_consec: u32, threshold_window: u32) -> Self {
        Self {
            recent: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
            threshold_consec,
            threshold_window,
        }
    }

    /// 2026-09-26: Check whether `(name, args_json)` trips a threshold, then push its
    /// hash onto the ring either way.
    pub fn check(&mut self, name: &str, args_json: &str) -> bool {
        let h = hash_call(name, args_json);
        let trip = self.would_trip(h);
        if self.recent.len() >= self.cap {
            self.recent.pop_front();
        }
        self.recent.push_back(h);
        trip
    }

    /// 2026-09-26: Whether pushing `h` would trip a threshold, counting `h` itself.
    fn would_trip(&self, h: u64) -> bool {
        // 2026-09-26: Consecutive: the run of `h` at the ring's tail, plus the new push.
        let mut consec = 1u32;
        for &p in self.recent.iter().rev() {
            if p == h {
                consec += 1;
            } else {
                break;
            }
        }
        if consec >= self.threshold_consec {
            return true;
        }
        // 2026-09-26: Window: occurrences of `h` in the ring before the push, plus the
        // new one.
        let occ = self.recent.iter().filter(|&&p| p == h).count() as u32 + 1;
        occ >= self.threshold_window
    }

    /// 2026-09-26: Drop all stored hashes.
    pub fn reset(&mut self) {
        self.recent.clear();
    }

    /// 2026-09-26: Number of hashes currently stored.
    pub fn len(&self) -> usize {
        self.recent.len()
    }
}

/// 2026-09-26: Hash `(name, canonical_args)` to a u64 with `DefaultHasher`. The args
/// are re-serialised in canonical (sorted-key, no-whitespace) form, or hashed as
/// the raw string if they do not parse as JSON.
fn hash_call(name: &str, args_json: &str) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(name.as_bytes());
    h.write_u8(0); // 2026-09-26: Separator, so "ab" + "c" != "a" + "bc".
    let canonical = canonicalize_json(args_json).unwrap_or_else(|| args_json.to_string());
    h.write(canonical.as_bytes());
    h.finish()
}

/// 2026-09-26: Re-serialise a JSON value with sorted object keys and no whitespace.
/// Returns `None` if `s` is not valid JSON.
fn canonicalize_json(s: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(canonical_value_to_string(&v))
}

fn canonical_value_to_string(v: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(&mut out, v);
    out
}

fn write_canonical(out: &mut String, v: &serde_json::Value) {
    use std::fmt::Write;
    match v {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            let _ = write!(out, "{}", n);
        }
        serde_json::Value::String(s) => {
            let _ = write!(out, "{}", serde_json::Value::String(s.clone()));
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, x) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, x);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{}", serde_json::Value::String(k.clone()));
                out.push(':');
                write_canonical(out, &map[k]);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_consecutive_identical_calls_trip() {
        let mut g = ToolArgDedup::new();
        for i in 0..4 {
            assert!(
                !g.check("Bash", r#"{"command":"mkdir -p /tmp/foo"}"#),
                "iteration {i} must not trip"
            );
        }
        assert!(
            g.check("Bash", r#"{"command":"mkdir -p /tmp/foo"}"#),
            "5th identical call must trip"
        );
    }

    #[test]
    fn whitespace_and_key_order_normalised() {
        let mut g = ToolArgDedup::new();
        // 2026-09-26: The first three differ only in whitespace and key order.
        assert!(!g.check("Write", r#"{"path":"/tmp/x","content":"hi"}"#));
        assert!(!g.check("Write", r#"{ "path" : "/tmp/x" , "content" : "hi" }"#));
        assert!(!g.check("Write", r#"{"content":"hi","path":"/tmp/x"}"#));
        assert!(!g.check("Write", r#"{"path":"/tmp/x","content":"hi"}"#));
        assert!(
            g.check("Write", r#"{"path":"/tmp/x","content":"hi"}"#),
            "5th identical (modulo whitespace+key-order) call must trip"
        );
    }

    #[test]
    fn distinct_calls_do_not_trip() {
        let mut g = ToolArgDedup::new();
        for i in 0..8 {
            let args = format!(r#"{{"command":"mkdir -p /tmp/test-{i}"}}"#);
            assert!(
                !g.check("Bash", &args),
                "8 distinct calls must not trip (i={i})"
            );
        }
    }

    #[test]
    fn six_of_eight_window_trips() {
        let mut g = ToolArgDedup::new();
        // 2026-09-26: 6 dupes interleaved with 2 distinct calls within the 8-slot ring.
        let dup = r#"{"command":"ls"}"#;
        let other_a = r#"{"command":"pwd"}"#;
        let other_b = r#"{"command":"whoami"}"#;
        // 2026-09-26: The run of dups never exceeds 2, so the 6th dup (index 7) trips
        // on the window.
        let mut tripped_at = None;
        for (i, args) in [dup, dup, other_a, dup, dup, other_b, dup, dup]
            .iter()
            .enumerate()
        {
            if g.check("Bash", args) {
                tripped_at = Some(i);
                break;
            }
        }
        assert!(
            tripped_at.is_some(),
            "must trip on 6/8 window pattern; ring={:?}",
            g.recent
        );
    }

    #[test]
    fn name_difference_breaks_dedup() {
        let mut g = ToolArgDedup::new();
        let args = r#"{"x":1}"#;
        // 2026-09-26: Same args under five different names: the hashes differ, so
        // nothing trips.
        for name in ["A", "B", "C", "D", "E"] {
            assert!(!g.check(name, args), "name {name} must not trip");
        }
    }

    #[test]
    fn invalid_json_falls_back_to_raw_string() {
        let mut g = ToolArgDedup::new();
        // 2026-09-26: Malformed JSON is hashed as the raw string, and identical raw
        // strings still trip the consecutive threshold.
        for _ in 0..4 {
            assert!(!g.check("Bash", "not valid {{ json"));
        }
        assert!(g.check("Bash", "not valid {{ json"));
    }

    #[test]
    fn reset_clears_state() {
        let mut g = ToolArgDedup::new();
        for _ in 0..4 {
            g.check("Bash", r#"{"x":1}"#);
        }
        assert_eq!(g.len(), 4);
        g.reset();
        assert_eq!(g.len(), 0);
        // 2026-09-26: After reset, four more identical calls stay below the
        // consecutive threshold.
        for _ in 0..4 {
            assert!(!g.check("Bash", r#"{"x":1}"#));
        }
    }
}
