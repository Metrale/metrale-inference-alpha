// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the tool-call path comparison and the probe's request.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants: none beyond the types.

use super::*;
use crate::benchmarks::transcript::Transcript;

fn transcript(calls: &[(&str, &str)], text: &str) -> Transcript {
    Transcript {
        reasoning: String::new(),
        text: text.to_string(),
        tool_calls: calls
            .iter()
            .map(|(n, a)| (n.to_string(), a.to_string()))
            .collect(),
        finish_reason: Some("stop".to_string()),
        completion_tokens: 32,
        cached_prompt_tokens: 0,
    }
}

fn result(path: Path, calls: &[(&str, &str)], text: &str) -> PathResult {
    PathResult {
        path,
        target: transcript(calls, text),
    }
}

/// 2026-09-26: Every path answered the target with no call.
#[test]
fn no_call_on_every_path_is_no_divergence() {
    let r = vec![
        result(Path::Direct, &[], "Closed membership means ..."),
        result(Path::AfterWeather, &[], "Closed membership means ..."),
        result(Path::AfterSearch, &[], "Closed membership means ..."),
    ];
    assert!(divergences(&r).is_empty());
    assert!(!reference_called(&r));
}

/// 2026-09-26: The direct path makes no call, and the path after the weather
/// turn calls `get_current_weather` on the target.
#[test]
fn a_call_inherited_from_a_predecessor_is_a_divergence() {
    let r = vec![
        result(Path::Direct, &[], "Closed membership means ..."),
        result(
            Path::AfterWeather,
            &[("get_current_weather", r#"{"location":"San Francisco, CA"}"#)],
            "",
        ),
        result(Path::AfterSearch, &[], "Closed membership means ..."),
    ];
    let d = divergences(&r);
    assert_eq!(d.len(), 1, "only the perturbed path should diverge");
    assert_eq!(d[0].path, "after-weather");
    assert!(d[0].reference_calls.is_empty());
    assert_eq!(d[0].path_calls[0].0, "get_current_weather");
    assert!(
        d[0].describe().contains("reference made no call"),
        "the reading must say what the reference did: {}",
        d[0].describe()
    );
}

/// 2026-09-26: The reverse: the reference calls and a perturbed path does not.
#[test]
fn a_call_that_disappears_is_also_a_divergence() {
    let r = vec![
        result(Path::Direct, &[("web_search", r#"{"query":"x"}"#)], ""),
        result(Path::AfterWeather, &[], "prose instead"),
    ];
    let d = divergences(&r);
    assert_eq!(d.len(), 1);
    assert!(d[0].path_calls.is_empty());
    assert!(d[0].describe().contains("this path made no call"));
}

/// 2026-09-26: Same function, different arguments.
#[test]
fn the_same_tool_with_different_arguments_is_a_divergence() {
    let r = vec![
        result(
            Path::Direct,
            &[(
                "web_search",
                r#"{"query":"IP address to company data API"}"#,
            )],
            "",
        ),
        result(
            Path::AfterSearch,
            &[("web_search", r#"{"query":"IP address to company data"}"#)],
            "",
        ),
    ];
    let d = divergences(&r);
    assert_eq!(
        d.len(),
        1,
        "arguments are part of the answer, not decoration"
    );
    assert_eq!(d[0].path_calls[0].0, "web_search");
}

/// 2026-09-26: Reworded prose with identical calls is not a divergence: only the
/// calls are compared.
#[test]
fn reworded_prose_with_identical_calls_is_not_a_divergence() {
    let r = vec![
        result(
            Path::Direct,
            &[],
            "Closed membership means a node joins once.",
        ),
        result(
            Path::AfterWeather,
            &[],
            "It means that a node joins exactly one time.",
        ),
    ];
    assert!(
        divergences(&r).is_empty(),
        "different prose with the same (absent) calls must pass"
    );
}

/// 2026-09-26: Call order is part of the comparison: the same two calls
/// transposed diverge.
#[test]
fn transposed_calls_are_a_divergence() {
    let a = ("order_status_check", r#"{"order_id":"282828"}"#);
    let b = ("get_product_details", r#"{"product_id":"282828"}"#);
    let r = vec![
        result(Path::Direct, &[a, b], ""),
        result(Path::AfterWeather, &[b, a], ""),
    ];
    assert_eq!(
        divergences(&r).len(),
        1,
        "the reference emitted these calls in one order; the other path did not"
    );
}

/// 2026-09-26: A reference that itself calls on the target is reported by
/// `reference_called`.
#[test]
fn a_reference_that_calls_is_reported_separately() {
    let r = vec![
        result(
            Path::Direct,
            &[("get_current_weather", r#"{"location":"SF"}"#)],
            "",
        ),
        result(
            Path::AfterWeather,
            &[("get_current_weather", r#"{"location":"SF"}"#)],
            "",
        ),
    ];
    assert!(
        divergences(&r).is_empty(),
        "the paths agree, so path-independence holds"
    );
    assert!(
        reference_called(&r),
        "but the reference hallucinated a call, and that must not be silent"
    );
}

/// 2026-09-26: With no reference path, `divergences` returns nothing and
/// `reference_called` is false.
#[test]
fn a_missing_reference_yields_no_verdict_rather_than_a_pass() {
    let r = vec![result(Path::AfterWeather, &[("x", "{}")], "")];
    assert!(
        divergences(&r).is_empty(),
        "no reference means no comparison"
    );
    assert!(
        !reference_called(&r),
        "and no reference means no reference-called finding either"
    );
}

/// 2026-09-26: `Path::ALL` holds three distinct paths, two of them interposing a
/// turn.
#[test]
fn all_three_paths_are_distinct_and_two_interpose_a_call() {
    assert_eq!(Path::ALL.len(), 3);
    assert!(Path::Direct.interposed().is_none());
    assert_eq!(Path::AfterWeather.interposed(), Some(CALLS_WEATHER));
    assert_eq!(Path::AfterSearch.interposed(), Some(CALLS_SEARCH));
    let labels: Vec<_> = Path::ALL.iter().map(|p| p.label()).collect();
    let mut uniq = labels.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        uniq.len(),
        labels.len(),
        "path labels must be distinguishable"
    );
}

/// 2026-09-26: The offered tools include `get_current_weather` and `web_search`,
/// the two the interposed turns ask for.
#[test]
fn the_offered_tools_are_the_ones_the_contamination_used() {
    let names: Vec<String> = tools()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"get_current_weather".to_string()));
    assert!(names.contains(&"web_search".to_string()));
}

/// 2026-09-26: The request offers both tools, leaves the choice to the model,
/// and is greedy with seed 0.
#[test]
fn the_request_offers_tools_and_leaves_the_choice_to_the_model() {
    let body = request_body("m", &[json!({"role": "user", "content": "hi"})], 256);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["seed"], 0);
}
