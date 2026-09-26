// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the driver's configuration and of its functions that
//! need no server: `request_body`, `to_outcome`, `rung_slots` and `scored`.
//!
//! Owner: bench (contamination).
//! Invariants: none beyond the types.

use super::super::prompts;
use super::super::transcript::{RequestOutcome, Transcript};
use super::{CrossContamination, Phase, request_body, rung_slots, to_outcome};
use crate::benchmark::Benchmark;
use crate::params::{ParamValue, ParamValues};
use crate::result::VerdictKind;

fn configured() -> CrossContamination {
    let mut b = CrossContamination::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b
}

fn reply(text: &str) -> RequestOutcome {
    RequestOutcome::Ok(Box::new(Transcript {
        text: text.into(),
        finish_reason: Some("stop".into()),
        completion_tokens: 40,
        ..Default::default()
    }))
}

fn honest(i: usize) -> RequestOutcome {
    let c = prompts::PROBES[i].canary;
    reply(&format!(
        "{c}\nthree steady sentences about topic {i}.\n{c}"
    ))
}

#[test]
fn defaults_validate_and_are_the_documented_ladder() {
    let b = configured();
    assert_eq!(b.concurrencies, vec![2, 4, 8]);
    assert_eq!(b.min_completion_tokens, 16);
    assert_eq!(b.max_tokens, 256);
    assert_eq!(b.timeout, std::time::Duration::from_secs(300));
}

#[test]
fn a_floor_at_or_above_the_budget_is_rejected() {
    for floor in [256, 257] {
        let mut b = CrossContamination::default();
        let mut v = ParamValues::defaults(&b.parameters());
        v.set("min_completion_tokens", ParamValue::Int(floor));
        v.set("max_tokens", ParamValue::Int(256));
        let err = b.configure(&v).unwrap_err().to_string();
        assert!(err.contains("floor"), "{err}");
        assert!(err.contains("Unmeasured by construction"), "{err}");
    }
}

#[test]
fn reconfiguring_clears_prior_run_state() {
    let mut b = configured();
    b.ref_a.push(honest(0));
    b.ref_b.push(honest(0));
    b.rungs.push(("c2".into(), vec![honest(0)]));
    b.post.push(honest(0));
    b.phase = Phase::Done;
    b.probed = true;
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    assert_eq!(b.phase, Phase::Prime);
    assert!(!b.probed);
    assert_eq!(b.rung_cursor, 0);
    assert!(b.ref_a.is_empty());
    assert!(b.ref_b.is_empty());
    assert!(b.rungs.is_empty());
    assert!(b.post.is_empty());
}

#[test]
fn rung_slots_lead_with_every_probe_then_cycle() {
    let n = prompts::PROBES.len();
    let slots = rung_slots(n, 8);
    assert_eq!(slots.len(), 8);
    assert_eq!(
        &slots[..n],
        &(0..n).collect::<Vec<_>>()[..],
        "measured slots are the identity"
    );
    for p in 0..n {
        assert!(
            slots.iter().filter(|s| **s == p).count() >= 2,
            "probe {p} must also appear as ballast at conc 8"
        );
    }
    assert_eq!(rung_slots(n, 1).len(), n);
}

#[test]
fn to_outcome_keeps_errors_as_errors() {
    let ok = to_outcome(Ok(crate::http::ChatOutcome {
        text: "body".into(),
        reasoning: "thought".into(),
        tool_calls: vec![crate::http::ToolCall {
            id: "call-1".into(),
            name: "lookup".into(),
            arguments: "{\"x\":1}".into(),
        }],
        finish_reason: Some("tool_calls".into()),
        completion_tokens: 21,
        cached_prompt_tokens: 13,
        ..Default::default()
    }));
    match &ok {
        RequestOutcome::Ok(t) => {
            assert_eq!(t.text, "body");
            assert_eq!(t.reasoning, "thought");
            assert_eq!(
                t.tool_calls,
                vec![("lookup".to_string(), "{\"x\":1}".to_string())]
            );
            assert_eq!(t.finish_reason.as_deref(), Some("tool_calls"));
            assert_eq!(t.completion_tokens, 21);
            assert_eq!(t.cached_prompt_tokens, 13);
        }
        RequestOutcome::Error(e) => panic!("a completed request read as an error: {e}"),
    }
    let err = to_outcome(Err(anyhow::anyhow!("connection reset")));
    match &err {
        RequestOutcome::Error(e) => assert!(e.contains("connection reset")),
        RequestOutcome::Ok(_) => panic!("a transport failure read as a transcript"),
    }
}

#[test]
fn request_body_is_greedy_streamed_and_carries_the_prompt() {
    let prompt = prompts::PROBES[0].prompt();
    let b = request_body("m-x", &prompt, 256);
    assert_eq!(
        b["temperature"], 0.0,
        "temp 0 or transcripts cannot be compared"
    );
    assert_eq!(b["stream"], true);
    assert_eq!(b["max_tokens"], 256);
    assert_eq!(b["model"], "m-x");
    assert_eq!(
        b["messages"],
        serde_json::json!([{"role": "user", "content": prompt}])
    );
}

/// 2026-09-26: Through the driver's own `Legs` assembly: the leaked-canary
/// case fails as contaminated only if the driver passes the canaries in.
#[test]
fn scored_passes_clean_legs_and_fails_a_leaked_canary() {
    let mut b = configured();
    b.ref_a = vec![honest(0), honest(1)];
    b.ref_b = vec![honest(0), honest(1)];
    b.rungs = vec![
        ("c2".into(), vec![honest(0), honest(1)]),
        ("c4".into(), vec![honest(0), honest(1)]),
    ];
    b.post = vec![honest(0), honest(1)];
    let (s, v) = b.scored();
    assert_eq!(s.identical, s.compared);
    assert!(matches!(v.kind, VerdictKind::Pass), "{}", v.reason);
    assert!(
        v.reason.contains("tokens compared"),
        "a pass must state its evidence volume: {}",
        v.reason
    );

    let foreign = prompts::PROBES[1].canary;
    b.rungs[1].1[0] = reply(&format!(
        "{}\nthree steady sentences about topic 0.\n{foreign}",
        prompts::PROBES[0].canary
    ));
    let (s, v) = b.scored();
    assert_eq!(s.contaminated, 1, "leakage, not a generic diff");
    assert!(matches!(v.kind, VerdictKind::Fail));
    assert!(v.reason.contains("CONTAMINATED"), "{}", v.reason);
}

#[test]
fn scored_classifies_a_post_leg_divergence_as_persistent() {
    let mut b = configured();
    b.ref_a = vec![honest(0), honest(1)];
    b.ref_b = vec![honest(0), honest(1)];
    b.rungs = vec![("c2".into(), vec![honest(0), honest(1)])];
    b.post = vec![
        reply("corrupted state, thirty tokens of it either way"),
        honest(1),
    ];
    let (s, v) = b.scored();
    assert_eq!(s.persistent, 1);
    assert_eq!(s.diverged, 0, "persistent is its own, worse class");
    assert!(v.reason.contains("PERSISTENT"), "{}", v.reason);
}
