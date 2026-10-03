// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Prompt-lookup decoding through the real scheduler loop, against
//! the recording model: the client bytes never depend on the flag, the copies
//! really are verified where the history repeats, and nothing changes where
//! it does not.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Mutex;

use metrale_speculative::prompt_lookup::PromptLookupConfig;

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};

static SERIAL: Mutex<()> = Mutex::new(());

const PL: PromptLookupConfig = PromptLookupConfig {
    ngram: 2,
    max_drafts: 3,
    max_seqs: 8,
    min_match: 2,
    miss_backoff: 0,
};

fn traced(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

/// 2026-10-02: The harness prompt after its first token is the period
/// `3 4 5 6 7 8 2` (`runner::build_request`); `n` tokens continuing it from
/// prompt index `from`.
fn period_from(from: usize, n: usize) -> Vec<u32> {
    (from..from + n).map(|i| 2 + (i as u32 % 7)).collect()
}

fn scenario(reqs: Vec<ReqSpec>, num_drafts: usize, pl: Option<PromptLookupConfig>) -> Scenario {
    Scenario {
        name: "prompt_lookup",
        cfg: ModelCfg {
            has_proposer: true,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            use_speculative: true,
            num_drafts,
            prompt_lookup: pl,
            ..RunOptions::default()
        },
        reqs,
    }
}

/// 2026-10-02: What each client received, with the response's accepted-draft
/// count (`acc=`, usage metadata that counts speculation, not output) cut out.
fn outputs(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.starts_with("out s"))
        .map(|l| {
            l.split(", ")
                .filter(|f| !f.starts_with("acc="))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

fn verifies(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|l| l.starts_with("decode_verify"))
        .count()
}

/// 2026-10-02: A request whose output continues the prompt's period for 16
/// tokens, then leaves it, then returns to it; the drafter is wrong every
/// `wrong_every`-th token, so copies beat it on the repeating stretches.
fn repeating_with(id: u64, wrong_every: usize) -> ReqSpec {
    let prompt_len = 15;
    let mut toks = period_from(prompt_len, 16);
    toks.extend([40, 41, 42, 43, 44]);
    toks.extend(period_from(prompt_len + 2, 9));
    toks.push(EOS);
    let mut r = ReqSpec::new(id, prompt_len, toks);
    r.draft_wrong_every = Some(wrong_every);
    r
}

fn repeating(id: u64) -> ReqSpec {
    repeating_with(id, 4)
}

#[test]
fn copies_change_the_verify_count_not_the_output() {
    // 2026-10-02: A 3-draft chain that misses every 2nd token: a 3-token copy
    // takes the round only where it matches, and is right on the stretches.
    let off = traced(&scenario(vec![repeating_with(1, 2)], 3, None));
    let on = traced(&scenario(vec![repeating_with(1, 2)], 3, Some(PL)));
    assert_eq!(outputs(&on), outputs(&off));
    // 2026-10-02: The lever moved: fewer verifies deliver the same tokens.
    assert!(
        verifies(&on) < verifies(&off),
        "prompt lookup verified {} times, MTP alone {}",
        verifies(&on),
        verifies(&off)
    );
}

#[test]
fn batched_copies_change_the_verify_count_not_the_output() {
    let reqs = || (1..=3).map(|i| repeating_with(i, 2)).collect::<Vec<_>>();
    let off = traced(&scenario(reqs(), 3, None));
    let on = traced(&scenario(reqs(), 3, Some(PL)));
    assert_eq!(outputs(&on), outputs(&off));
    assert!(
        on.iter().any(|l| l.starts_with("decode_verify_batched")),
        "the copies must be verified inside the batch"
    );
    assert!(verifies(&on) < verifies(&off));
}

#[test]
fn copies_that_are_always_wrong_never_change_the_output() {
    // 2026-10-02: Every time the history ends on a pair of the prompt's
    // period the output leaves it, so each copy fails at its first token.
    let prompt_len = 15;
    let mut toks = Vec::new();
    for k in 0..6u32 {
        toks.extend(period_from(prompt_len, 2));
        toks.extend([50 + k, 57 + k]);
    }
    toks.push(EOS);
    let mut r = ReqSpec::new(1, prompt_len, toks);
    r.draft_wrong_every = Some(5);
    let mk = |pl| scenario(vec![r.clone()], 3, pl);
    let off = traced(&mk(None));
    let on = traced(&mk(Some(PL)));
    assert_eq!(outputs(&on), outputs(&off));
    assert!(
        on.iter().any(|l| l.contains("na=0")),
        "a refused copy must trim the drafter as a full reject"
    );
}

#[test]
fn with_nothing_to_match_the_trace_is_unchanged() {
    // 2026-10-02: Generated tokens never seen before and an n-gram longer than
    // any run the prompt shares with them: no lookup can hit, so the model
    // sees exactly the calls it sees with the flag off. The only extra calls
    // are the lookup's slot-capacity reads.
    let toks: Vec<u32> = (40..60).chain([EOS]).collect();
    let mut r = ReqSpec::new(1, 9, toks);
    r.draft_wrong_every = Some(3);
    let pl = PromptLookupConfig { ngram: 3, ..PL };
    let strip = |v: Vec<String>| -> Vec<String> {
        v.into_iter()
            .filter(|l| !l.starts_with("mtp_slot_draft_capacity("))
            .collect()
    };
    let off = strip(traced(&scenario(vec![r.clone()], 3, None)));
    let on = strip(traced(&scenario(vec![r], 3, Some(pl))));
    assert_eq!(on, off);
}

#[test]
fn a_one_draft_drafter_gets_three_token_copies() {
    // 2026-10-02: `--num-drafts 1` (the MoE recipe) verifies 2 rows per MTP
    // round; the slots here hold 8 drafts, so a copy may verify 4.
    let off = traced(&scenario(vec![repeating(1)], 1, None));
    let on = traced(&scenario(vec![repeating(1)], 1, Some(PL)));
    assert_eq!(outputs(&on), outputs(&off));
    assert!(
        on.iter()
            .any(|l| l.starts_with("decode_verify_graphed_k4(")),
        "a 3-token copy takes the K4 verify"
    );
    assert!(
        !off.iter()
            .any(|l| l.starts_with("decode_verify_graphed_k4("))
    );
    assert!(verifies(&on) < verifies(&off));
}

#[test]
fn a_one_draft_slot_caps_the_copy() {
    // 2026-10-02: A verify slot that holds one draft bounds a copy to one.
    let mut sc = scenario(vec![repeating(1)], 1, Some(PL));
    sc.cfg.slot_draft_capacity = 1;
    let on = traced(&sc);
    assert!(
        !on.iter().any(|l| l.starts_with("decode_verify_graphed_k4(")
            || l.starts_with("decode_verify_graphed_k3(")),
        "no verify wider than the slot"
    );
}

/// 2026-10-02: `(drafts, picks)` of a serial K4 verify line
/// `decode_verify_graphed_k4(tokens=[l, d0, d1, d2], ..) -> [v0, v1, v2, v3]`.
fn k4_line(l: &str) -> Option<(Vec<u32>, Vec<u32>)> {
    let nums = |s: &str| -> Vec<u32> { s.split(", ").filter_map(|t| t.parse().ok()).collect() };
    let rest = l.strip_prefix("decode_verify_graphed_k4(tokens=[")?;
    let (toks, rest) = rest.split_once(']')?;
    let picks = rest.split_once("-> [")?.1.trim_end_matches(']');
    Some((nums(toks)[1..].to_vec(), nums(picks)))
}

#[test]
fn a_fully_accepted_copy_still_trims_the_drafter_as_a_full_reject() {
    let on = traced(&scenario(vec![repeating_with(1, 2)], 3, Some(PL)));
    // 2026-10-02: The control run needs full accepts of the drafter's own
    // chain, so its drafter misses only every 4th token.
    let off = traced(&scenario(vec![repeating(1)], 3, None));
    // 2026-10-02: For every fully accepted K4 verify, the drafter trim that
    // follows it: with the flag on some are copies (na=0 although all three
    // drafts matched); with it off every one is the drafter's own (na=3).
    let trims_after_full_accepts = |lines: &[String]| -> Vec<String> {
        lines
            .iter()
            .enumerate()
            .filter(|(_, l)| k4_line(l).is_some_and(|(d, v)| d[..] == v[..3]))
            .filter_map(|(i, _)| {
                lines[i..]
                    .iter()
                    .find(|l| l.starts_with("trim_proposer_state("))
                    .map(|l| {
                        l.rsplit_once("na=")
                            .map_or(String::new(), |t| t.1.to_string())
                    })
            })
            .collect()
    };
    let on_trims = trims_after_full_accepts(&on);
    let off_trims = trims_after_full_accepts(&off);
    assert!(!off_trims.is_empty() && off_trims.iter().all(|t| t.starts_with("3,")));
    assert!(
        on_trims.iter().any(|t| t.starts_with("0,")),
        "a fully accepted copy must trim the drafter by na=0: {on_trims:?}"
    );
}

#[test]
fn a_lone_sequence_verifies_a_long_copy_in_one_pass() {
    // 2026-10-02: An 8-draft copy ceiling on a slot that holds 8: a lone
    // sequence takes the K-row verify, with the same client bytes.
    let pl = PromptLookupConfig {
        max_drafts: 8,
        ..PL
    };
    let off = traced(&scenario(vec![repeating_with(1, 2)], 1, None));
    let on = traced(&scenario(vec![repeating_with(1, 2)], 1, Some(pl)));
    assert_eq!(outputs(&on), outputs(&off));
    assert!(
        on.iter()
            .any(|l| l.starts_with("decode_verify_graphed_kgamma(")),
        "a copy of 4+ drafts takes the K-row MTP verify"
    );
    assert!(
        !on.iter().any(|l| l.starts_with("decode_verify_dflash(")),
        "never the DFlash verify on an MTP serve"
    );
    assert!(verifies(&on) < verifies(&off));
}

#[test]
fn inside_a_batch_a_long_copy_is_cut_to_the_four_row_verify() {
    let pl = PromptLookupConfig {
        max_drafts: 8,
        ..PL
    };
    let reqs = || (1..=3).map(|i| repeating_with(i, 2)).collect::<Vec<_>>();
    let off = traced(&scenario(reqs(), 3, None));
    let on = traced(&scenario(reqs(), 3, Some(pl)));
    assert_eq!(outputs(&on), outputs(&off));
    // 2026-10-02: Once the batch drains to one sequence it may verify a long
    // copy; while three are active every verify stays within 4 rows.
    let first_kgamma = on
        .iter()
        .position(|l| l.starts_with("decode_verify_graphed_kgamma("));
    let last_batched = on
        .iter()
        .rposition(|l| l.starts_with("decode_verify_batched("));
    if let (Some(k), Some(b)) = (first_kgamma, last_batched) {
        assert!(k > b, "a long copy only after the batch has drained");
    }
    for l in on
        .iter()
        .filter(|l| l.starts_with("decode_verify_batched("))
    {
        let ks = l
            .split("ks=[")
            .nth(1)
            .and_then(|r| r.split(']').next())
            .unwrap_or("");
        assert!(
            ks.split(", ")
                .all(|k| k.parse::<usize>().is_ok_and(|k| (2..=4).contains(&k))),
            "batched rows out of 2..=4: {l}"
        );
    }
}
