// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Streaming-detector tests: `METRALE_BUFFER_TOOL_ARGS` turns off
//! live argument fragments, and a bare `<function=…>` call streams live
//! without also being emitted whole. Fixtures come from `streaming_frag`.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;
use super::streaming_frag::{args_from_outputs, collect_fragments, write_and_bash_tools};

// 2026-09-26: Ignored by default: it sets the process-wide
// `METRALE_BUFFER_TOOL_ARGS`, which `StreamingToolDetector::new_with_tools`
// reads, and parallel tests that build detectors expect it unset. Run it
// alone:
//   cargo test -p metrale-server --bin met -- --ignored --test-threads=1 \
//       tool_parser::tests::streaming_frag_env::kill_switch
#[test]
#[ignore = "mutates process-global METRALE_BUFFER_TOOL_ARGS; run serially with --ignored --test-threads=1"]
fn kill_switch_buffers_full_args_no_fragments() {
    // 2026-09-26: With the variable at `1`, the arguments come as one
    // `ToolCallDelta` at the close and no `ToolCallArgsFragment` is emitted.
    let _guard = env_guard::set("METRALE_BUFFER_TOOL_ARGS", "1");
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let chunks = [
        "<tool_call>",
        "<function=Write>",
        "<parameter=file_path>",
        "/tmp/x.rs",
        "</parameter>",
        "<parameter=content>",
        "hello",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let frag_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .count();
    let delta_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallDelta { .. }))
        .count();
    assert_eq!(frag_count, 0, "kill-switch must emit NO live fragments");
    assert_eq!(
        delta_count, 1,
        "kill-switch must emit exactly one buffered ToolCallDelta"
    );
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/tmp/x.rs");
    assert_eq!(args["content"], "hello");
}

/// 2026-09-26: Sets an env var for the guard's lifetime and restores the
/// previous value on drop. A process-wide mutex serialises the tests that use
/// this guard; tests that read the var without it are kept apart only by
/// `#[ignore]` and a serial run.
mod env_guard {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // 2026-09-26: A static, because what it guards, the process environment,
    // is shared by every test thread of the binary.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    pub struct Guard {
        key: &'static str,
        prev: Option<String>,
        _lock: MutexGuard<'static, ()>,
    }

    pub fn set(key: &'static str, val: &str) -> Guard {
        let lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var(key).ok();
        // 2026-09-26: SAFETY: sound only in the serial run named above:
        // `ENV_LOCK` serialises the guard's users, but other tests read this
        // var through `StreamingToolDetector::new_with_tools` without it.
        unsafe {
            std::env::set_var(key, val);
        }
        Guard {
            key,
            prev,
            _lock: lock,
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            // 2026-09-26: SAFETY: as in `set`; `_lock` still holds `ENV_LOCK`.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }
}

#[test]
fn bare_function_streams_incrementally_and_emits_call_once() {
    // 2026-09-26: A bare `<function=…>` with no envelope streams its header
    // and each completed parameter (`process_bare_function`), and a call
    // streamed that way is closed with `ToolCallEnd`, not emitted again whole.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let full = "<function=Write>\n\
                <parameter=file_path>\n/tmp/x.rs\n</parameter>\n\
                <parameter=content>\nhello\n</parameter>\n\
                </function>";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    let start_pos = outputs
        .iter()
        .position(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .expect("bare <function=> must emit ToolCallStart before the block closes");
    let frag_positions: Vec<usize> = outputs
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(
        frag_positions.len() >= 2,
        "expected MULTIPLE incremental fragments for a bare function, got {}",
        frag_positions.len()
    );
    assert!(
        frag_positions.iter().all(|&p| p > start_pos),
        "fragments must follow ToolCallStart"
    );

    let whole_calls = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(..)))
        .count();
    assert_eq!(
        whole_calls, 0,
        "a call streamed incrementally must not ALSO be emitted whole (duplicate delivery)"
    );

    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    assert_eq!(
        args,
        serde_json::json!({"file_path": "/tmp/x.rs", "content": "hello"})
    );
}
