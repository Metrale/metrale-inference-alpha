// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What a concurrency cell sends: the prompt fixture, the prompt
//! identities of its warm-up and measured rounds, and the request body.
//!
//! Owner: bench (concurrency).
//! Invariants: every warm-up round sends exactly the measured round's tags.

use super::{CODE_TASK, ConcurrencySweep, ESSAY_TASK, PromptMode, essay_nonce_tag, json, stats};

/// 2026-09-26: Which prompt a cell sends. `Natural` and `Count` are the shared
/// `stats::PromptMode` fixtures; `Essay` is this driver's own. The essay
/// fixture also changes the request's sampling pins, so it is chosen only by
/// name through the `prompt_mode` parameter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Fixture {
    /// 2026-09-26: `Default` only so `ConcurrencySweep::default()` exists
    /// before `configure`, which always sets it from `prompt_mode`.
    #[default]
    Natural,
    Count,
    /// 2026-09-26: The essay ask, a fixed-width nonce, and `presence_penalty` /
    /// `frequency_penalty` pinned to 0.0. Qwen3.8-27B's `[sampling.non_thinking]`
    /// preset in MODEL.toml sets `presence_penalty = 1.5`, and the published
    /// legs sent both at 0.0 (`harness_shas` in `bench/ladder38/published.json`).
    Essay,
}

impl Fixture {
    pub(super) fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "essay" => Some(Fixture::Essay),
            other => PromptMode::parse(other).map(|mode| match mode {
                PromptMode::Natural => Fixture::Natural,
                PromptMode::Count => Fixture::Count,
            }),
        }
    }

    /// 2026-09-26: The prompt identity for request `i` of a cell: `c{i}` for
    /// the natural and count fixtures, the harness's nonce for the essay one.
    pub(super) fn prefix_tag(self, i: usize) -> String {
        match self {
            Fixture::Natural | Fixture::Count => format!("c{i}"),
            Fixture::Essay => essay_nonce_tag(i),
        }
    }
}

/// 2026-09-26: The prompt identities one cell sends before and during
/// measurement, as a value the tests can inspect.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PromptPlan {
    pub(super) warmup_rounds: Vec<Vec<String>>,
    pub(super) measured: Vec<String>,
}

pub(super) fn prompt_plan(conc: usize, warmup: usize, fixture: Fixture) -> PromptPlan {
    let measured: Vec<String> = (0..conc).map(|i| fixture.prefix_tag(i)).collect();
    PromptPlan {
        warmup_rounds: vec![measured.clone(); warmup],
        measured,
    }
}

impl ConcurrencySweep {
    /// 2026-09-26: The ISL filler from [`stats::make_prompt`] first, the
    /// fixture's task last.
    pub(super) fn cell_prompt(&self, isl: usize, prefix_tag: &str) -> String {
        match self.fixture {
            Fixture::Count => stats::make_prompt(isl, PromptMode::Count, prefix_tag),
            Fixture::Natural => {
                let mut p = stats::make_prompt(isl, PromptMode::Natural, prefix_tag);
                p.push(' ');
                p.push_str(CODE_TASK);
                p
            }
            // 2026-09-26: Byte for byte the harness's `make_prompt` output for
            // the same nonce (pinned in `concurrency_essay_tests.rs`).
            Fixture::Essay => {
                let mut p = stats::make_prompt(isl, PromptMode::Natural, prefix_tag);
                p.push_str(ESSAY_TASK);
                p
            }
        }
    }

    /// 2026-09-26: Pure, so the tests can check each fixture's sampling pins
    /// without an endpoint.
    pub(super) fn request_body(
        &self,
        model: &str,
        isl: usize,
        prefix_tag: &str,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": model,
            "stream": true,
            "max_tokens": self.osl,
            // 2026-09-26: Pinned sampling, so the draw adds no run-to-run noise.
            "temperature": 0.0,
            "seed": 0,
            // 2026-09-26: On a thinking-on serve the output budget would
            // otherwise go to reasoning.
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": self.cell_prompt(isl, prefix_tag)}],
        });
        if self.fixture == Fixture::Essay {
            // 2026-09-26: Only for the essay fixture (see `Fixture::Essay`); the
            // natural and count requests leave both to the server's preset.
            body["presence_penalty"] = json!(0.0);
            body["frequency_penalty"] = json!(0.0);
        }
        body
    }
}
