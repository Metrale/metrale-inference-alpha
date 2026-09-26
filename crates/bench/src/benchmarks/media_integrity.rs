// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Cross-request and cross-path integrity checks, used by the
//! vision and video benchmarks. They target machinery a single well-formed
//! request cannot reach, whose failure is a fluent answer to the wrong input
//! rather than an error:
//!
//! * [`heterogeneous_concurrency`]: several different requests in flight at
//!   once, each required to get its own answer.
//! * [`cache_leak`]: the same prompt text with a different image, back to
//!   back.
//! * [`long_prompt_path`]: the same image and question with filler prepended.
//! * [`media_in_history`]: media in an earlier turn or a tool result.
//! * [`stream_parity`]: the same request streamed and blocking.
//! * [`multi_choice`]: `n > 1` with an image.
//! * `responses_parity` and `thinking_parity` (the `responses` submodule).
//!
//! Owner: bench, media integrity.
//! Invariants: a check whose request failed never returns `Cell::Pass`.

use std::time::Duration;

use serde_json::{Value, json};

use crate::http;
use crate::plugin::PluginHandle;

/// 2026-09-26: One concurrent subject: the request, a predicate scoring its
/// own reply, and a label for the failure message.
pub type Subject = (Value, Box<dyn Fn(&str) -> bool + Send + Sync>, String);

/// 2026-09-26: Outcome of one integrity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cell {
    Pass { id: &'static str, detail: String },
    Fail { id: &'static str, detail: String },
    Skipped { id: &'static str, why: String },
    Error { id: &'static str, msg: String },
}

impl Cell {
    pub fn id(&self) -> &'static str {
        match self {
            Cell::Pass { id, .. }
            | Cell::Fail { id, .. }
            | Cell::Skipped { id, .. }
            | Cell::Error { id, .. } => id,
        }
    }
    pub fn passed(&self) -> bool {
        matches!(self, Cell::Pass { .. })
    }
    pub fn measured(&self) -> bool {
        matches!(self, Cell::Pass { .. } | Cell::Fail { .. })
    }
    pub fn line(&self) -> String {
        match self {
            Cell::Pass { id, detail } => format!("{id}: {detail}"),
            Cell::Fail { id, detail } => format!("{id}: FAILED — {detail}"),
            Cell::Skipped { id, why } => format!("{id}: skipped — {why}"),
            Cell::Error { id, msg } => format!("{id}: {msg}"),
        }
    }
}

/// 2026-09-26: One image part plus a prompt, as a streaming chat body with
/// thinking off.
pub fn image_request(
    model: &str,
    mime: &str,
    bytes: &[u8],
    prompt: &str,
    max_tokens: usize,
) -> Value {
    use base64::Engine;
    let mut uri = format!("data:{mime};base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut uri);
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// 2026-09-26: Several different requests at once, each scored against its own
/// expected answer.
///
/// With `METRALE_VISION_CODISPATCH` on, image requests admitted in one tick
/// share one encoder output and each reads its slice through per-request base
/// offsets (`ModelVision::set_vision_slice_base`). When every request is
/// identical, every offset is interchangeable and an off-by-one is invisible;
/// with different content, a mis-sliced offset hands request A the answer to
/// request B. Fewer than two subjects is a skip.
pub async fn heterogeneous_concurrency(
    handle: &PluginHandle,
    subjects: Vec<Subject>,
    timeout: Duration,
) -> Cell {
    const ID: &str = "heterogeneous-concurrency";
    if subjects.len() < 2 {
        return Cell::Skipped {
            id: ID,
            why: "fewer than two subjects".to_string(),
        };
    }
    let n = subjects.len();
    let futures: Vec<_> = subjects
        .iter()
        .map(|(body, _, _)| http::chat_stream(handle.target(), body, timeout))
        .collect();
    let outs = futures::future::join_all(futures).await;

    let mut wrong = Vec::new();
    let mut errors = Vec::new();
    for (out, (_, want, label)) in outs.into_iter().zip(subjects.iter()) {
        match out {
            Ok(o) => {
                if !want(o.text.trim()) {
                    wrong.push(format!(
                        "{label} got \"{}\"",
                        crate::benchmarks::one_line(o.text.chars().take(60).collect::<String>())
                    ));
                }
            }
            Err(e) => errors.push(format!(
                "{label}: {}",
                crate::benchmarks::one_line(format!("{e:#}"))
            )),
        }
    }
    if !errors.is_empty() {
        return Cell::Error {
            id: ID,
            msg: errors.join("; "),
        };
    }
    if wrong.is_empty() {
        Cell::Pass {
            id: ID,
            detail: format!("{n} different requests in flight, each got its own answer"),
        }
    } else {
        Cell::Fail {
            id: ID,
            detail: format!(
                "{}/{n} replies did not match their own input — {}",
                wrong.len(),
                wrong.join("; ")
            ),
        }
    }
}

/// 2026-09-26: The same prompt text with a different image, back to back.
///
/// Vision prompts must not be served from the prefix cache. The model-engine
/// guard is `tokens_have_vision_pad`, which must recognise both the image and
/// the video pad; if it missed one, the second request would match the
/// first's cached prefix and answer about the previous image. `first_marker`
/// tells that case apart from a merely wrong reply.
pub async fn cache_leak(
    handle: &PluginHandle,
    first: Value,
    second: Value,
    second_want: &(dyn Fn(&str) -> bool + Sync),
    first_marker: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "prefix-cache-isolation";
    let a = match http::chat_stream(handle.target(), &first, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let b = match http::chat_stream(handle.target(), &second, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let b_text = b.text.trim();
    if second_want(b_text) {
        return Cell::Pass {
            id: ID,
            detail: format!(
                "identical prompt, different media: second reply is its own ({} then {} prompt \
                 tokens)",
                a.prompt_tokens, b.prompt_tokens
            ),
        };
    }
    // 2026-09-26: Distinguish "answered the first image", a cache hit, from
    // merely wrong.
    let detail = if first_marker(b_text) {
        format!(
            "the second request was answered with the FIRST image's content — the prefix cache \
             served a vision prompt. Reply: \"{}\"",
            crate::benchmarks::one_line(b_text.chars().take(80).collect::<String>())
        )
    } else {
        format!(
            "the second reply matched neither expectation: \"{}\"",
            crate::benchmarks::one_line(b_text.chars().take(80).collect::<String>())
        )
    };
    Cell::Fail { id: ID, detail }
}

/// 2026-09-26: The same image and question, short and with a large block of
/// filler text prepended; both replies must satisfy `want`.
///
/// With `METRALE_VISION_CODISPATCH` on, `phase_start_prefills` admits an image
/// request to the batched encode only when its prompt fits
/// `max_prefill_tokens`; a longer prompt encodes its own images
/// (`prepare_vision_embed`), so the two prompts take different paths.
pub async fn long_prompt_path(
    handle: &PluginHandle,
    short: Value,
    long: Value,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "long-prompt-splice";
    let s = match http::chat_stream(handle.target(), &short, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let l = match http::chat_stream(handle.target(), &long, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let short_ok = want(s.text.trim());
    let long_ok = want(l.text.trim());
    if short_ok && long_ok {
        Cell::Pass {
            id: ID,
            detail: format!(
                "same answer on both paths ({} vs {} prompt tokens)",
                s.prompt_tokens, l.prompt_tokens
            ),
        }
    } else {
        Cell::Fail {
            id: ID,
            detail: format!(
                "short path {}, long path {} ({} vs {} prompt tokens) — the two splices disagree",
                if short_ok { "ok" } else { "WRONG" },
                if long_ok { "ok" } else { "WRONG" },
                s.prompt_tokens,
                l.prompt_tokens
            ),
        }
    }
}

/// 2026-09-26: Media that is not in the message asking the question: in an
/// earlier turn or in a tool result. The server collects media from every
/// message, tool results included (`api/chat/msg_entry.rs`,
/// `collect_message_media`); a regression that scanned only the final message
/// would pass the legs that put the image and the question in one turn.
pub async fn media_in_history(
    handle: &PluginHandle,
    body: Value,
    want: &(dyn Fn(&str) -> bool + Sync),
    id: &'static str,
    timeout: Duration,
) -> Cell {
    match http::chat_stream(handle.target(), &body, timeout).await {
        Ok(o) => {
            let text = o.text.trim();
            if want(text) {
                Cell::Pass {
                    id,
                    detail: format!("answered from history ({} prompt tokens)", o.prompt_tokens),
                }
            } else {
                Cell::Fail {
                    id,
                    detail: format!(
                        "did not answer about the earlier media: \"{}\"",
                        crate::benchmarks::one_line(text.chars().take(80).collect::<String>())
                    ),
                }
            }
        }
        Err(e) => Cell::Error {
            id,
            msg: crate::benchmarks::one_line(format!("{e:#}")),
        },
    }
}

/// 2026-09-26: The same request streamed and not streamed.
///
/// The blocking path builds its response separately (`api/chat_blocking.rs`),
/// so a server can stream correctly and assemble a blocking reply wrongly.
/// Both must answer correctly and agree on `prompt_tokens`: the request is
/// the same apart from `stream`, so a difference there means the two paths
/// built different prompts from it.
pub async fn stream_parity(
    handle: &PluginHandle,
    mut body: Value,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "stream-blocking-parity";
    body["stream"] = Value::Bool(true);
    let streamed = match http::chat_stream(handle.target(), &body, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("streaming: {e:#}")),
            };
        }
    };
    body["stream"] = Value::Bool(false);
    let blocking = match http::chat_blocking(handle.target(), &body, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("blocking: {e:#}")),
            };
        }
    };
    let b_text = blocking.choices.first().cloned().unwrap_or_default();
    let s_ok = want(streamed.text.trim());
    let b_ok = want(b_text.trim());
    if !s_ok || !b_ok {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "streaming {}, blocking {} — the two response paths disagree about the same \
                 image",
                if s_ok { "ok" } else { "WRONG" },
                if b_ok { "ok" } else { "WRONG" }
            ),
        };
    }
    if streamed.prompt_tokens != blocking.prompt_tokens {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "both answered correctly but built DIFFERENT prompts: {} tokens streaming vs {} \
                 blocking, from a byte-identical request",
                streamed.prompt_tokens, blocking.prompt_tokens
            ),
        };
    }
    Cell::Pass {
        id: ID,
        detail: format!(
            "both paths correct and agree on {} prompt tokens",
            streamed.prompt_tokens
        ),
    }
}

/// 2026-09-26: `n > 1` with an image, on the blocking path.
///
/// `api/chat_blocking.rs` gives the image pixels to choice 0 and an empty
/// vector to every later choice. If a later choice loses the image rather than
/// sharing choice 0's encode, it answers about nothing. Every choice must be
/// present and answer about the picture.
pub async fn multi_choice(
    handle: &PluginHandle,
    mut body: Value,
    n: usize,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "multi-choice-image";
    body["stream"] = Value::Bool(false);
    body["n"] = Value::from(n);
    match http::chat_blocking(handle.target(), &body, timeout).await {
        Ok(o) => {
            if o.choices.len() != n {
                return Cell::Fail {
                    id: ID,
                    detail: format!("asked for n={n}, got {} choices", o.choices.len()),
                };
            }
            let bad: Vec<usize> = o
                .choices
                .iter()
                .enumerate()
                .filter(|(_, c)| !want(c.trim()))
                .map(|(i, _)| i)
                .collect();
            if bad.is_empty() {
                Cell::Pass {
                    id: ID,
                    detail: format!("all {n} choices answered about the image"),
                }
            } else {
                Cell::Fail {
                    id: ID,
                    detail: format!(
                        "choice(s) {bad:?} did not answer about the image — the later choices \
                         are not seeing it"
                    ),
                }
            }
        }
        // 2026-09-26: `n > 1` may not be supported; an error containing "400"
        // or "not supported" is a skip rather than a failure.
        Err(e) => {
            let msg = crate::benchmarks::one_line(format!("{e:#}"));
            let unsupported = msg.contains("400") || msg.to_lowercase().contains("not supported");
            if unsupported {
                Cell::Skipped { id: ID, why: msg }
            } else {
                Cell::Error { id: ID, msg }
            }
        }
    }
}

#[path = "media_integrity_responses.rs"]
mod responses;
pub use responses::{
    responses_image_request, responses_parity, thinking_image_request, thinking_parity,
};

#[cfg(test)]
#[path = "media_integrity_tests.rs"]
mod media_integrity_tests;
