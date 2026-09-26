// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Process-wide Prometheus metrics of the server, and the guard for the active-request gauge.
//!
//! Owner: server.
//! Invariants: none beyond the types.

use lazy_static::lazy_static;
use prometheus::{
    HistogramVec, IntCounter, IntCounterVec, IntGauge, register_histogram_vec,
    register_int_counter, register_int_counter_vec, register_int_gauge,
};

lazy_static! {
    pub static ref REQUESTS_TOTAL: IntCounter =
        register_int_counter!("metrale_requests_total", "Total requests processed").unwrap();
    pub static ref REQUESTS_ACTIVE: IntGauge =
        register_int_gauge!("metrale_requests_active", "Currently active requests").unwrap();
    /// 2026-09-26: Time to first token, labelled by model. The server can swap
    /// models (`main_modules/auto_swap.rs`), and one histogram pooling two
    /// models would give quantiles of neither. The counters in this file are
    /// process totals and are never reset.
    pub static ref TTFT_SECONDS: HistogramVec = register_histogram_vec!(
        "metrale_time_to_first_token_seconds",
        "Time to first token",
        &["model"],
        vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    .unwrap();
    pub static ref GENERATION_TOKENS_TOTAL: IntCounter =
        register_int_counter!("metrale_generation_tokens_total", "Total tokens generated").unwrap();
    /// 2026-09-26: Incremented once per token by the streaming chat handler
    /// (`api/chat_stream/handle_token.rs`), so its rate is live.
    /// `GENERATION_TOKENS_TOTAL` is added once per request at completion, and
    /// blocking requests add only to that one, so the two are not
    /// interchangeable.
    pub static ref DECODED_TOKENS_TOTAL: IntCounter =
        register_int_counter!(
            "metrale_decoded_tokens_total",
            "Tokens decoded, counted as they are produced (rate-friendly)"
        ).unwrap();
    // 2026-09-26: Updated by `main_modules/byte_count.rs`: bytes in from the
    // request's `Content-Length` header, bytes out as each response body frame
    // is polled, so streamed responses are counted too.
    pub static ref HTTP_BYTES_IN: IntCounter =
        register_int_counter!("metrale_http_bytes_in_total", "Total HTTP request body bytes")
            .unwrap();
    pub static ref HTTP_BYTES_OUT: IntCounter =
        register_int_counter!("metrale_http_bytes_out_total", "Total HTTP response body bytes")
            .unwrap();
    pub static ref PROMPT_TOKENS_TOTAL: IntCounter =
        register_int_counter!("metrale_prompt_tokens_total", "Total prompt tokens processed")
            .unwrap();

    // 2026-09-26: One count per loop-detector verdict (`api/chat/loop_detect.rs`).
    // `verdict` is none, hint, suppress or failing_repeat; `channel` is text,
    // tools, combined, or n/a for none; `spinning` is 0 or 1.
    pub static ref LOOP_DETECTOR_VERDICTS: IntCounterVec =
        register_int_counter_vec!(
            "metrale_loop_detector_verdicts_total",
            "Loop detector verdicts emitted, by verdict + channel + spinning flag",
            &["verdict", "channel", "spinning"]
        ).unwrap();

    // 2026-09-26: Speculative-decode verify outcomes. `k` is the draft path
    // ("2" or "dflash") and `outcome` is accept/reject or
    // accept_all/accept_partial (`scheduler/verify_k2_step.rs`,
    // `verify_dflash_step.rs`).
    pub static ref SPEC_DECODE_VERIFY: IntCounterVec =
        register_int_counter_vec!(
            "metrale_spec_decode_verify_total",
            "MTP draft verify outcomes by K and result",
            &["k", "outcome"]
        ).unwrap();

    // 2026-09-26: Tool calls emitted by the streaming and blocking chat paths.
    // No tool-name label, to keep the series count bounded.
    pub static ref TOOL_CALLS_TOTAL: IntCounter =
        register_int_counter!(
            "metrale_tool_calls_total",
            "Total successful tool calls emitted by the server"
        ).unwrap();
}

/// 2026-09-26: Holds one count of `metrale_requests_active`: increments on
/// construction and decrements once on drop, so a handler future dropped on
/// client disconnect still releases it. A streaming request moves the guard
/// into `StreamCtx`, which the stream's `flat_map` closure owns, so it drops
/// with the stream.
pub struct ActiveRequestGuard(());

impl ActiveRequestGuard {
    pub fn new() -> Self {
        REQUESTS_ACTIVE.inc();
        Self(())
    }
}

impl Default for ActiveRequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        REQUESTS_ACTIVE.dec();
    }
}
