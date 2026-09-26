// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Card rendering tests, aimed at cards that come out wrong without an error: a
//! metric key no record carries, a slot with no data, a model id with an `&` in it.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.
use super::card::{Fmt, parse_args, render, spec_for};
use super::record::GateRecord;
use super::tests::{SHA, hw, run_record};
use crate::result::Verdict;

/// 2026-09-26: A record for `id` carrying `metrics`, built with `GateRecord::from_run`.
fn rec(id: &str, metrics: &[(&str, f64)]) -> GateRecord {
    let mut r = run_record(
        metrics
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect(),
        Verdict::pass("ok"),
    );
    r.benchmark_id = id.to_string();
    GateRecord::from_run(&r, hw(), SHA.to_string(), Vec::new(), None).expect("record")
}

fn template() -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    std::fs::read_to_string(root.join("assets/cards/result-card.svg"))
        .expect("the card template is committed")
}

/// 2026-09-26: Every metric key a card spec names must exist on the last record file (by name)
/// of each `.benchmarks/<gate>` directory; a missing key would hide its box.
#[test]
fn every_card_key_exists_on_a_real_record() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let benchmarks = root.join(".benchmarks");
    if !benchmarks.exists() {
        return;
    }
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(&benchmarks).unwrap().flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let mut files: Vec<_> = std::fs::read_dir(entry.path())
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "json")
                    && p.file_name().is_some_and(|n| n != "BASELINE.json")
            })
            .collect();
        files.sort();
        let Some(newest) = files.last() else { continue };
        let Ok(rec) = super::record::read_record(newest) else {
            continue;
        };
        let spec = spec_for(&id);
        if !spec.hero_key.is_empty() && !rec.metrics.contains_key(spec.hero_key) {
            missing.push(format!("{id}: hero key `{}` absent", spec.hero_key));
        }
        for s in spec.slots.iter().flatten() {
            if !rec.metrics.contains_key(s.key) {
                missing.push(format!("{id}: slot `{}` key `{}` absent", s.label, s.key));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "card specs name metrics no record carries — these render as blank boxes:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn a_missing_hero_metric_renders_an_em_dash_not_a_zero() {
    // 2026-09-26: The template's hero placeholder is `000.0`; a record with no metrics must
    // not leave it in place.
    let r = rec("decode-floor", &[]);
    let svg = render(&template(), &r, &Default::default());
    assert!(svg.contains(">—<"), "expected an em dash for the hero");
    assert!(!svg.contains(">000.0<"), "the placeholder survived");
}

/// 2026-09-26: The group of an unused slot gets `display="none"`, which hides the box, its
/// accent bar and its label together.
#[test]
fn an_unused_slot_hides_its_whole_box() {
    let r = rec(
        "decode-floor",
        &[
            ("server_decode_tok_s", 22.7),
            ("accept_len_mean", 2.68),
            ("output_tokens", 795.0),
            ("runs", 3.0),
        ],
    );
    let svg = render(&template(), &r, &Default::default());
    assert!(
        hidden(&svg, "field-m4"),
        "slot 4 has no data, box still visible"
    );
    assert!(
        !hidden(&svg, "field-m1"),
        "slot 1 HAS data, must not be hidden"
    );
}

/// 2026-09-26: Whether `display="none"` is set on the element carrying this id.
fn hidden(svg: &str, id: &str) -> bool {
    let Some(at) = svg.find(&format!("id=\"{id}\"")) else {
        return false;
    };
    let open = svg[..at].rfind('<').unwrap_or(0);
    let close = svg[at..].find('>').map(|i| at + i).unwrap_or(svg.len());
    svg[open..close].contains("display=\"none\"")
}

/// 2026-09-26: The value of `attr` on the element carrying `id`. Reads that element only: the
/// same green also fills other template elements, so a substring check would pass without
/// a chip.
fn attr_of(svg: &str, id: &str, attr: &str) -> String {
    let Some(at) = svg.find(&format!("id=\"{id}\"")) else {
        return String::new();
    };
    let open = svg[..at].rfind('<').unwrap_or(0);
    let close = svg[at..].find('>').map(|i| at + i).unwrap_or(svg.len());
    let tag = &svg[open..close];
    let key = format!("{attr}=\"");
    match tag.find(&key) {
        Some(k) => {
            let v = &tag[k + key.len()..];
            v[..v.find('"').unwrap_or(0)].to_string()
        }
        None => String::new(),
    }
}

#[test]
fn a_passing_run_gets_a_green_chip_and_a_failing_one_gets_gold() {
    // 2026-09-26: `verdict_passes()` matches "PASS" exactly, so the fixture uses that spelling.
    let mut pass = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    pass.verdict = Some("PASS".to_string());
    let svg = render(&template(), &pass, &Default::default());
    assert!(svg.contains(">PASS<"), "no PASS text on a passing record");
    assert_eq!(attr_of(&svg, "verdict-chip", "fill"), "#12B981");

    let mut fail = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    fail.verdict = Some("FAIL".to_string());
    let svg = render(&template(), &fail, &Default::default());
    assert!(svg.contains(">FAIL<"), "no FAIL text on a failing record");
    assert_eq!(attr_of(&svg, "verdict-chip", "fill"), "#EFB338");
    assert_eq!(attr_of(&svg, "value-toks", "fill"), "#82868F");
}

/// 2026-09-26: A record with no verdict hides the verdict chip.
#[test]
fn a_record_with_no_verdict_hides_the_chip_entirely() {
    let mut r = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    r.verdict = None;
    let svg = render(&template(), &r, &Default::default());
    assert!(hidden(&svg, "field-verdict"), "the chip is still showing");
}

#[test]
fn absent_attribution_hides_its_box() {
    let r = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    let svg = render(&template(), &r, &parse_args("author=Ada").unwrap());
    assert!(!hidden(&svg, "field-author"), "author was given");
    assert!(hidden(&svg, "field-handle"), "handle was not given");
    assert!(hidden(&svg, "field-site"), "website was not given");
}

#[test]
fn an_ampersand_in_a_model_id_does_not_break_the_svg() {
    let mut r = rec("decode-floor", &[("server_decode_tok_s", 1.0)]);
    r.target_model = "vendor/model&<thing>".to_string();
    let svg = render(&template(), &r, &Default::default());
    assert!(
        svg.contains("vendor/model&amp;&lt;thing&gt;"),
        "not escaped"
    );
}

#[test]
fn args_parse_and_a_missing_equals_is_refused() {
    let ok = parse_args("author=Ada Lovelace, handle=@ada ,website=ada.dev,").unwrap();
    assert_eq!(ok.get("author").map(String::as_str), Some("Ada Lovelace"));
    assert_eq!(ok.get("handle").map(String::as_str), Some("@ada"));
    assert_eq!(ok.get("website").map(String::as_str), Some("ada.dev"));
    assert!(parse_args("authorAda").is_err());
    assert!(parse_args("=nokey").is_err());
}

#[test]
fn formats_round_the_way_a_reader_expects() {
    assert_eq!(Fmt::Int.apply(114.6), "115");
    assert_eq!(Fmt::One.apply(22.68), "22.7");
    assert_eq!(Fmt::Two.apply(84.219), "84.22");
    assert_eq!(Fmt::Ms.apply(8288.4), "8288 ms");
}
