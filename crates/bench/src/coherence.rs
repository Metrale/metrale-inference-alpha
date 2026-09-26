// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A known-answer probe of the endpoint, run by the executor before
//! a benchmark loads.
//!
//! [`crate::http::probe`] only checks that `/v1/models` answers with a 2xx
//! status line; it never parses the body. This probe also reads the model list
//! (a wrong model name), checks the served model against the benchmark's
//! [`crate::benchmark::ModelExpectation`] (a wrong family), and asks the two
//! [`CHECKS`] questions (an endpoint that does not generate sense). It is not a
//! quality measurement.
//!
//! Owner: bench.
//! Invariants:
//! - [`probe`] and [`probe_for`] never return an error: a transport failure is
//!   recorded in [`Report::transport_error`].

use anyhow::Result;
use serde_json::json;
use std::time::Duration;

use crate::http;
use crate::plugin::TargetEndpoint;

/// 2026-09-26: Whether a run probes the endpoint before it starts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoherencePolicy {
    /// 2026-09-26: Probe and report. A failed probe is a warning in the run
    /// log, never a refusal (`executor::RunTask::probe_coherence`): a base
    /// checkpoint, or a model that phrases answers unusually, is a legitimate
    /// thing to benchmark on purpose.
    #[default]
    Probe,
    /// 2026-09-26: Do not probe at all.
    Skip,
}

/// 2026-09-26: A question whose answer is not a matter of opinion.
#[derive(Clone, Copy, Debug)]
pub struct Check {
    pub label: &'static str,
    pub prompt: &'static str,
    /// 2026-09-26: Lower-cased terms; the answer passes when it contains one
    /// of them as a whole word (see `judge`). Several entries are spellings of
    /// the same fact.
    pub accept: &'static [&'static str],
}

/// 2026-09-26: Two facts: one arithmetic, one recall.
pub const CHECKS: &[Check] = &[
    Check {
        label: "arithmetic",
        prompt: "What is 2+2? Reply with only the number.",
        accept: &["4", "four"],
    },
    Check {
        label: "recall",
        prompt: "What is the capital of France? Reply with only the city name.",
        accept: &["paris"],
    },
];

/// 2026-09-26: What one check produced, kept so a failure can quote it back.
#[derive(Clone, Debug)]
pub struct Answer {
    pub label: &'static str,
    pub answer: String,
    pub passed: bool,
}

/// 2026-09-26: The outcome of a probe: what was asked, what came back, and
/// whether it fit.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub answers: Vec<Answer>,
    /// 2026-09-26: Set when a question's request failed (`http::chat_stream`
    /// returned an error). Reported with its own wording, not as a wrong answer.
    pub transport_error: Option<String>,
    /// 2026-09-26: Set when the served model is not one the benchmark is
    /// defined on. Distinct from `served_instead`: the name may be exactly
    /// what was asked for and still be outside the benchmark's families.
    pub wrong_family: Option<String>,
    /// 2026-09-26: What `/v1/models` lists, when the requested model is not
    /// among them. Only this catches a wrong model name: the server answers a
    /// chat completion for a name it does not serve with its live model
    /// (`api::chat`; with `--auto-swap` a catalogue name loads first), so the
    /// questions cannot see the mistake.
    pub served_instead: Option<Vec<String>>,
}

impl Report {
    /// 2026-09-26: True when every check passed and no other field is set.
    pub fn is_clean(&self) -> bool {
        self.transport_error.is_none()
            && self.wrong_family.is_none()
            && self.served_instead.is_none()
            && self.answers.iter().all(|a| a.passed)
    }

    /// 2026-09-26: One message naming the most important thing wrong, or
    /// `None` when nothing is. Priority: transport error, wrong family, wrong
    /// or missing model, then failed answers.
    pub fn concern(&self, target: &TargetEndpoint) -> Option<String> {
        if let Some(e) = &self.transport_error {
            return Some(format!(
                "{} did not answer a test request: {e}",
                target.base_url
            ));
        }
        if let Some(note) = &self.wrong_family {
            return Some(note.clone());
        }
        // 2026-09-26: Before the answers: a wrong model name explains any odd
        // answer, so it is the cause to report.
        if let Some(served) = &self.served_instead {
            // 2026-09-26: The server lists nothing only when no model is
            // loaded (`api::models`), and its chat handler then answers 503
            // (`api::chat`), so the run will produce no numbers; the
            // wrong-model wording would be false.
            if served.is_empty() {
                return Some(format!(
                    "{} has no model loaded, so this run will produce no numbers — every \
                     request will be refused. Load a model first: in the dashboard open the \
                     Library (press 4), choose a model and a recipe, and start it.",
                    target.base_url
                ));
            }
            return Some(format!(
                "{} is serving {} — not {:?}, which this benchmark is set to request. \
                 Metrale Engine answers whatever model name it is sent, so the run WILL produce \
                 numbers; they will just be for a different model than the one named.",
                target.base_url,
                served.join(", "),
                target.model
            ));
        }
        let failed: Vec<&Answer> = self.answers.iter().filter(|a| !a.passed).collect();
        if failed.is_empty() {
            return None;
        }
        let detail = failed
            .iter()
            .map(|a| match a.answer.trim() {
                "" => format!("{} answered nothing", a.label),
                text => format!("{} answered {:?}", a.label, truncate(text, 60)),
            })
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "{} is serving {:?}, which did not answer as expected ({detail}). \
             This benchmark may be aimed at a different model, or the checkpoint \
             may be a base (non-instruct) one — the run is still valid, but read \
             the numbers with that in mind.",
            target.base_url, target.model
        ))
    }
}

/// 2026-09-26: [`probe_for`] without a model expectation.
pub async fn probe(target: &TargetEndpoint, timeout: Duration) -> Report {
    probe_for(target, None, timeout).await
}

/// 2026-09-26: Read the model list, check the served model against
/// `expectation` when one is given, then ask every [`CHECKS`] question.
///
/// The first failed question request stops the questions and is stored as
/// [`Report::transport_error`]; nothing here returns an error.
pub async fn probe_for(
    target: &TargetEndpoint,
    expectation: Option<crate::benchmark::ModelExpectation>,
    timeout: Duration,
) -> Report {
    let mut report = Report::default();

    match http::list_models(target, timeout).await {
        Ok(served) if !served.iter().any(|m| m == &target.model) => {
            report.served_instead = Some(served);
        }
        Ok(_) => {}
        // 2026-09-26: An unreadable list is only logged at debug level; the
        // questions below still run.
        Err(e) => tracing::debug!("could not read the model list: {e:#}"),
    }

    // 2026-09-26: The family is checked against the first listed model when
    // the requested one is not listed, and against the requested name
    // otherwise, including when the list could not be read.
    if let Some(expect) = expectation {
        let actual = report
            .served_instead
            .as_ref()
            .and_then(|s| s.first().cloned())
            .unwrap_or_else(|| target.model.clone());
        if !expect.accepts(&actual) {
            report.wrong_family = Some(format!(
                "{} is serving {actual}, which this benchmark is not defined on. {}",
                target.base_url, expect.note
            ));
        }
    }

    for check in CHECKS {
        match ask(target, check, timeout).await {
            Ok(answer) => report.answers.push(answer),
            Err(e) => {
                report.transport_error = Some(one_line(&format!("{e:#}")));
                break;
            }
        }
    }
    report
}

/// 2026-09-26: Collapse an error chain to one line of at most 280 characters,
/// a bound against an unbounded chain rather than a line width.
fn one_line(s: &str) -> String {
    let flat = s.lines().map(str::trim).collect::<Vec<_>>().join(" ");
    truncate(&flat, 280)
}

/// 2026-09-26: One question. A transport or HTTP error is returned as `Err`
/// rather than counted as a failed answer, so the two get different messages.
async fn ask(target: &TargetEndpoint, check: &Check, timeout: Duration) -> Result<Answer> {
    let body = json!({
        "model": target.model,
        "stream": true,
        // 2026-09-26: Thinking off, and 96 tokens so that a server which
        // ignores `enable_thinking` still has room to reach the answer.
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 96,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": check.prompt}],
    });
    let outcome = http::chat_stream(target, &body, timeout).await?;
    let (passed, answer) = judge(&outcome.text, &outcome.reasoning, check.accept);
    Ok(Answer {
        label: check.label,
        passed,
        answer,
    })
}

/// 2026-09-26: Did the reply contain the expected fact, and what should be
/// quoted back?
///
/// Passes when either the text or the reasoning contains an accepted term as a
/// whole word (no letter, digit or `_` on either side), so a server that
/// ignores `enable_thinking` does not fail a healthy model. The quoted answer
/// is the text, or the reasoning when the text is blank.
fn judge(text: &str, reasoning: &str, accept: &[&str]) -> (bool, String) {
    let matched = |s: &str| {
        let lowered = s.to_lowercase();
        accept.iter().any(|term| {
            lowered.match_indices(term).any(|(at, _)| {
                let before = lowered[..at].chars().next_back();
                let after = lowered[at + term.len()..].chars().next();
                let continues_word = |ch: char| ch.is_alphanumeric() || ch == '_';
                !before.is_some_and(continues_word) && !after.is_some_and(continues_word)
            })
        })
    };
    let passed = matched(text) || matched(reasoning);
    let answer = if text.trim().is_empty() {
        reasoning.to_string()
    } else {
        text.to_string()
    };
    (passed, answer)
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
#[path = "coherence_tests.rs"]
mod tests;
