// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The probe: a fixed conversation script that every round replays
//! from scratch against the same server, and the anchors the reference round
//! must meet.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants:
//! - The script is constant text, with no run id or `unique_prefix_tag`, so
//!   every round's turn 1 sends the same prompt prefix (`probe_tests` pins
//!   its SHA-256).

use serde_json::{Value, json};

use crate::benchmarks::transcript::Transcript;

/// 2026-09-26: The document that opens turn 1. Every round sends it unchanged,
/// so a replay's turn-1 prefill can be restored from the prefix cache.
pub const LONG_PREFIX: &str = "\
SYSTEM DOCUMENT 7741-C, revision 9, frozen text — quote from it, never extend it.

Section 1, Invariants. The ledger defines three invariants that every batch
must preserve end to end. First, monotonic sequence: each record carries a
sequence number exactly one greater than its predecessor, and gaps are
treated as corruption, never as reordering. Second, bounded drift: the
clock offset between any two nodes in the batch must stay under forty
milliseconds for the duration of the batch, and an offset beyond that bound
invalidates every record written after the breach. Third, closed membership:
a node joins the batch through a single signed admission record and leaves
through a single signed departure record, and no record may reference a node
outside its admission and departure interval.

Section 2, Batch lifecycle. A batch opens when the coordinator writes the
genesis record, assigning the batch id and the initial sequence window.
Nodes acknowledge the genesis record, after which the coordinator admits
writes. The batch runs until either the coordinator writes the seal record
or the drift invariant is breached. When a batch seals, every node flushes
its local ledger segment to the archive tier and reports the segment digest.
The digest is the hash of the concatenated record envelopes in sequence
order, excluding payloads. Two nodes that sealed the same batch must report
the same digest; a mismatch is escalated as a split-brain event and the
batch is quarantined.

Section 3, Recovery. Recovery replays the archived segments in sequence
order against an empty ledger until the digest matches the sealed value.
Replay is deterministic by construction: the envelope fixes the record
bytes, and the sequence order fixes their application order. Recovery never
consults a live node; a replay that cannot reproduce the sealed digest
marks the segment as untrusted and routes it to manual review. The recovery
window is bounded at ninety minutes per segment, and a segment that exceeds
the window is split at the nearest checkpoint boundary and retried in
halves.

Section 4, Checksum rule. The envelope checksum is computed over the header
fields in their serialized order: batch id, sequence number, node id,
timestamp, then payload length. The checksum excludes the payload itself,
which is carried separately and verified against the declared length only.
When a record is archived, the archive tier recomputes the checksum from the
stored header and refuses any record whose recomputed value differs from
the recorded one; such a record is quarantined with its recomputed value
attached. The checksum is sixteen bytes and is never truncated in transit.

Section 5, Escalation ladder. A drift breach escalates to the coordinator
within one heartbeat interval. A digest mismatch escalates immediately and
pauses admission. A split-brain event escalates to the on-call engineer and
freezes the archive tier until the quarantined batch is resolved. Escalation
records carry the breached invariant by name, the observed values, and the
bounds from this document, in that order.

Section 6, Admission and departure. Admission is a two-phase exchange. The
candidate node first presents a signed intent record naming the batch and its
own node id; the coordinator validates the signature against the roster key
and, if the batch is open, returns an admission record carrying the assigned
sequence window. The candidate becomes a member only after it acknowledges
the admission record. Departure is symmetric: a member submits a departure
intent, the coordinator drains the member's outstanding writes, then issues
the departure record. A node that leaves before its writes are drained is
marked delinquent, and its unacknowledged records are re-assigned to the
coordinator for replay. Membership changes are appended to the batch journal
in order and are themselves covered by the envelope checksum, so a roster
tamper is detected at the next archive recomputation.

Section 7, Archival layout. The archive tier stores segments in fixed-size
blocks aligned on checksum boundaries; a segment smaller than one block is
padded with a terminal envelope carrying a zero payload length. Block
headers repeat the batch id and the first and last sequence numbers of the
contained records, which lets a recovery scan locate a segment by sequence
number alone without reading payloads. The tier keeps two copies of every
block on distinct media and compares them on read; a copy disagreement
triggers a re-fetch from the member that sealed the batch, and if that
member is gone the block is marked degraded and excluded from digest
recomputation until restored. Degraded blocks are reported in the daily
integrity summary with their batch id, block offset, and the media that
disagreed.";

/// 2026-09-26: The user turns of the script, in order. Turn 1 is prefixed with
/// [`LONG_PREFIX`]; each later turn is sent after the earlier turns and the
/// model's replies to them.
pub const TURNS: [&str; 4] = [
    // 2026-09-26: Turn 1: acknowledge the document and count its sections.
    "Reply with exactly one line: ACK 7741-C and the number of sections \
     listed in the document.",
    // 2026-09-26: Turn 2: Section 1's invariants as three numbered lines.
    "List the three invariants from Section 1 of the document, numbered 1 \
     to 3, one per line, each in at most ten words.",
    // 2026-09-26: Turn 3: rewrite Section 4 as two sentences.
    "Rewrite Section 4 of the document as exactly two sentences, preserving \
     every rule it states.",
    // 2026-09-26: Turn 4: three questions answered from Section 4.
    "Answer from the document: (a) what does the envelope checksum cover, in \
     field order; (b) what is excluded from it; (c) what happens when the \
     archive tier recomputes a mismatching checksum. One short paragraph.",
];

/// 2026-09-26: The request every turn sends: greedy, seed 0, fixed `max_tokens`.
/// `stream_options.include_usage` asks for usage on the stream; without it
/// `ChatOutcome` counts deltas for `completion_tokens` and reads 0 cached
/// tokens. There is no `chat_template_kwargs`: the gate's BENCH.toml entry
/// turns thinking off with the `disable_thinking` serve override.
pub(super) fn request_body(model: &str, messages: &[Value], max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "messages": messages,
    })
}

/// 2026-09-26: The full user message of turn 1.
pub(super) fn first_turn() -> String {
    format!("{LONG_PREFIX}\n\n{}", TURNS[0])
}

/// 2026-09-26: The number of sections in [`LONG_PREFIX`], counted from its
/// `Section <n>,` headings; turn 1's anchor checks the reply states it.
fn section_count() -> usize {
    (1..)
        .take_while(|n| LONG_PREFIX.contains(&format!("Section {n},")))
        .count()
}

fn contains_token(text: &str, expected: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case(expected))
}

fn has_number_prefix(line: &str, number: usize) -> bool {
    line.strip_prefix(&number.to_string())
        .and_then(|rest| rest.chars().next())
        .is_some_and(|c| c.is_whitespace() || matches!(c, '.' | ')' | ':'))
}

fn contains_in_order(text: &str, phrases: &[&str]) -> bool {
    let text = text.to_lowercase();
    let mut cursor = 0;
    phrases.iter().all(|phrase| {
        let Some(found) = text[cursor..].find(phrase) else {
            return false;
        };
        cursor += found + phrase.len();
        true
    })
}

/// 2026-09-26: Anchors on the reference round; every violation found, or empty.
/// `compare.rs` holds replays to round 0 only, so these hold round 0 to the
/// script: every turn ends with `stop`, turn 1 has the ACK and the section
/// count, turn 2 is three numbered lines, and turns 3 and 4 name the checksum
/// fields in order, the payload exclusion, recompute and quarantine (turn 3
/// also in exactly two sentences). A turn that hit the token budget is a
/// violation because its length cannot anchor the collapse ratios.
pub(super) fn validate_reference(reference: &[Transcript]) -> Vec<String> {
    let mut violations = Vec::new();
    if reference.len() != TURNS.len() {
        violations.push(format!(
            "reference has {} turn(s), script has {}",
            reference.len(),
            TURNS.len()
        ));
        return violations;
    }
    for (i, t) in reference.iter().enumerate() {
        match t.finish_reason.as_deref() {
            Some("stop") => {}
            Some("length") => violations.push(format!(
                "turn {}: reference hit the token budget (finish_reason=length) — a \
                 truncated reference cannot anchor the collapse ratios",
                i + 1
            )),
            other => violations.push(format!(
                "turn {}: reference did not finish normally (finish_reason={other:?})",
                i + 1
            )),
        }
    }
    let t1 = &reference[0].text;
    if !t1.contains("ACK 7741-C") {
        violations.push("turn 1: missing the demanded 'ACK 7741-C' acknowledgement".into());
    }
    // 2026-09-26: The count is searched outside the document id, because
    // "7741-C" contains a 7 of its own. A digit or the spelled-out word counts.
    let sections = section_count();
    let words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let stripped = t1.replace("7741-C", "");
    let has_digit = contains_token(&stripped, &sections.to_string());
    let has_word = words
        .get(sections)
        .is_some_and(|word| contains_token(&stripped, word));
    if !has_digit && !has_word {
        violations.push(format!(
            "turn 1: does not state the document's section count ({sections})"
        ));
    }
    let lines: Vec<&str> = reference[1]
        .text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() != 3 {
        violations.push(format!(
            "turn 2: expected exactly 3 numbered lines, got {}",
            lines.len()
        ));
    } else {
        for (i, line) in lines.iter().enumerate() {
            if !has_number_prefix(line, i + 1) {
                violations.push(format!(
                    "turn 2: line {} does not start with its number: {:?}",
                    i + 1,
                    line
                ));
            }
        }
    }
    let t3 = reference[2].text.to_lowercase();
    let t3_sentences = t3
        .split_terminator(['.', '!', '?'])
        .filter(|sentence| !sentence.trim().is_empty())
        .count();
    // 2026-09-26: Each anchor is named once and used both to decide and to
    // report, so the message says which one failed.
    let t3_fields = contains_in_order(
        &t3,
        &[
            "batch id",
            "sequence number",
            "node id",
            "timestamp",
            "payload length",
        ],
    );
    // 2026-09-26: Stem and noun are checked apart: Section 4 says "The checksum
    // excludes the payload itself", with "the" between the two words.
    let t3_exclusion = t3.contains("exclud") && t3.contains("payload");
    let t3_recompute = t3.contains("recompute");
    let t3_quarantine = t3.contains("quarantin");
    if t3_sentences != 2 || !t3_fields || !t3_exclusion || !t3_recompute || !t3_quarantine {
        violations.push(format!(
            "turn 3: Section 4 rewrite failed an anchor (sentences={t3_sentences}, \
             fields_in_order={t3_fields}, exclusion={t3_exclusion}, \
             recompute={t3_recompute}, quarantine={t3_quarantine})"
        ));
    }
    let t4 = reference[3].text.to_lowercase();
    if !contains_in_order(
        &t4,
        &[
            "batch id",
            "sequence number",
            "node id",
            "timestamp",
            "payload length",
        ],
        // 2026-09-26: Stemmed as in turn 3; `payload` is required on its own.
    ) || !t4.contains("exclud")
        || !t4.contains("payload")
        || !t4.contains("recompute")
        || !t4.contains("quarantin")
    {
        violations.push("turn 4: does not answer every checksum question".into());
    }
    violations
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod probe_tests;
