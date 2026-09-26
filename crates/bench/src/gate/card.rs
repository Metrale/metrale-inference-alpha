// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Shareable result cards: a benchmark record rendered onto the SVG card template
//! (`assets/cards/result-card.svg`).
//!
//! Owner: bench gate.
//! Invariants:
//! - One template for every benchmark: [`spec_for`] maps a benchmark's metrics onto one hero
//!   number and four detail slots.
//! - Every card prints the model, quantization, recipe, benchmark id and hardware beside the
//!   number; an unknown quantization or recipe renders as `—`.
use crate::gate::record::GateRecord;
use std::collections::BTreeMap;

/// 2026-09-26: How a raw `f64` becomes card text.
#[derive(Clone, Copy, Debug)]
pub enum Fmt {
    /// 2026-09-26: Rounded to an integer: `115`.
    Int,
    /// 2026-09-26: One decimal: `22.7`.
    One,
    /// 2026-09-26: Two decimals: `84.22`.
    Two,
    /// 2026-09-26: Rounded milliseconds: `8288 ms`.
    Ms,
}

impl Fmt {
    pub fn apply(self, v: f64) -> String {
        match self {
            Fmt::Int => format!("{}", v.round() as i64),
            Fmt::One => format!("{v:.1}"),
            Fmt::Two => format!("{v:.2}"),
            Fmt::Ms => format!("{} ms", v.round() as i64),
        }
    }
}

/// 2026-09-26: One of the four detail boxes: its label, the metric key and the format.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub label: &'static str,
    pub key: &'static str,
    pub fmt: Fmt,
}

const fn slot(label: &'static str, key: &'static str, fmt: Fmt) -> Option<Slot> {
    Some(Slot { label, key, fmt })
}

/// 2026-09-26: What a benchmark puts on a card.
#[derive(Clone, Debug)]
pub struct CardSpec {
    pub hero_label: &'static str,
    pub hero_key: &'static str,
    pub hero_note: &'static str,
    pub hero_fmt: Fmt,
    pub slots: [Option<Slot>; 4],
}

/// 2026-09-26: The per-benchmark mapping. Its keys are metric names that committed records
/// carry: `every_card_key_exists_on_a_real_record` fails if the last record file (by name)
/// of a gate lacks one, since a missing key hides its box.
pub fn spec_for(benchmark_id: &str) -> CardSpec {
    match benchmark_id {
        "decode-floor" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "server_decode_tok_s",
            hero_note: "decode, steady state",
            hero_fmt: Fmt::One,
            slots: [
                slot("Accept length", "accept_len_mean", Fmt::Two),
                slot("Output tokens", "output_tokens", Fmt::Int),
                slot("Runs", "runs", Fmt::Int),
                None,
            ],
        },
        "concurrency-sweep" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "peak_aggregate_tok_s",
            hero_note: "aggregate, best rung of C=1..128",
            hero_fmt: Fmt::One,
            slots: [
                slot("C=1", "c1_aggregate_tok_s", Fmt::One),
                slot("C=16", "c16_aggregate_tok_s", Fmt::One),
                slot("C=128", "c128_aggregate_tok_s", Fmt::One),
                slot("TTFT p50, C=1", "c1_ttft_p50_ms", Fmt::Ms),
            ],
        },
        // 2026-09-26: This gate's BENCH.toml entry runs C=1..16 (batch cap 16), so there is
        // no C=128 slot to show.
        "concurrency-sweep-dflash2" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "peak_aggregate_tok_s",
            hero_note: "aggregate, best rung of C=1..16, DFlash2 armed",
            hero_fmt: Fmt::One,
            slots: [
                slot("C=1", "c1_aggregate_tok_s", Fmt::One),
                slot("C=4", "c4_aggregate_tok_s", Fmt::One),
                slot("C=16", "c16_aggregate_tok_s", Fmt::One),
                slot("TTFT p50, C=1", "c1_ttft_p50_ms", Fmt::Ms),
            ],
        },
        // 2026-09-26: This gate's BENCH.toml entry pins C=1..16, ISL 128, OSL 1024, so there is
        // no C=128 slot to show.
        "concurrency-sweep-moe" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "peak_aggregate_tok_s",
            hero_note: "aggregate, best rung of C=1..16, ISL 128 / OSL 1024",
            hero_fmt: Fmt::One,
            slots: [
                slot("C=1", "c1_aggregate_tok_s", Fmt::One),
                slot("C=4", "c4_aggregate_tok_s", Fmt::One),
                slot("C=16", "c16_aggregate_tok_s", Fmt::One),
                slot("TTFT p50, C=1", "c1_ttft_p50_ms", Fmt::Ms),
            ],
        },
        "bfcl-subset" | "bfcl-subset-echolp" => CardSpec {
            hero_label: "Overall accuracy",
            hero_key: "overall_accuracy",
            hero_note: "BFCL single-turn — see samples for the draw",
            hero_fmt: Fmt::Two,
            slots: [
                slot("Normalized ST", "normalized_single_turn_score", Fmt::Two),
                slot("Samples", "samples", Fmt::Int),
                None,
                None,
            ],
        },
        "agentic-webserver" => CardSpec {
            hero_label: "Webserver OK",
            hero_key: "webserver_ok",
            hero_note: "runs that built, served and tore down cleanly",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Followed directions", "followed_directions", Fmt::Int),
                slot("Iterations", "iterations", Fmt::Int),
                slot("Seconds / turn", "s_per_turn", Fmt::Two),
                slot("Decode tok/s", "decode_tps", Fmt::One),
            ],
        },
        "ttft-cold-gate" => CardSpec {
            hero_label: "TTFT median",
            hero_key: "median_ms",
            hero_note: "cold, first token after load",
            hero_fmt: Fmt::Ms,
            slots: [
                slot("p90", "p90_ms", Fmt::Ms),
                slot("Samples", "samples", Fmt::Int),
                None,
                None,
            ],
        },
        "ttft-warm-gate" => CardSpec {
            hero_label: "TTFT median",
            hero_key: "median_ms",
            hero_note: "warm, prefix cache primed",
            hero_fmt: Fmt::Ms,
            slots: [
                slot("p90", "p90_ms", Fmt::Ms),
                slot("Samples", "samples", Fmt::Int),
                None,
                None,
            ],
        },
        "vision-fidelity" => CardSpec {
            hero_label: "Geometry cells matched",
            hero_key: "geometry_matched",
            hero_note: "vision fidelity, control held",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Cells asserted", "geometry_asserted", Fmt::Int),
                slot("Probes passed", "probes_passed", Fmt::Int),
                slot("Probes total", "probes_total", Fmt::Int),
                None,
            ],
        },
        "video-fidelity" => CardSpec {
            hero_label: "Legs passed",
            hero_key: "legs_passed",
            hero_note: "video fidelity, control held",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Legs asserted", "legs_asserted", Fmt::Int),
                slot("Skipped", "legs_skipped", Fmt::Int),
                None,
                None,
            ],
        },
        "ssm-state-poisoning-gate" => CardSpec {
            hero_label: "Replays byte-identical",
            hero_key: "invariant",
            hero_note: "SSM state isolation under interleaving",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Rounds", "rounds", Fmt::Int),
                slot("Collapsed", "collapsed", Fmt::Int),
                slot("Jittered", "jittered", Fmt::Int),
                None,
            ],
        },
        // 2026-09-26: Unknown benchmark: the hero is the alphabetically first metric (the
        // metrics are a `BTreeMap`), so the same record always renders the same card.
        _ => CardSpec {
            hero_label: "Result",
            hero_key: "",
            hero_note: "first metric, alphabetically — no card mapping for this benchmark yet",
            hero_fmt: Fmt::Two,
            slots: [None, None, None, None],
        },
    }
}

/// 2026-09-26: `author=Ada,website=ada.dev` -> map, with keys lowercased. Later keys win; spaces
/// and empty pairs are ignored. A part without `=`, or with an empty key, is an error rather
/// than dropped, so a typo cannot silently leave a field off the card.
pub fn parse_args(raw: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for part in raw.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let (k, v) = p
            .split_once('=')
            .ok_or_else(|| format!("`{p}` is not key=value"))?;
        let k = k.trim();
        if k.is_empty() {
            return Err(format!("`{p}` has an empty key"));
        }
        out.insert(k.to_lowercase(), v.trim().to_string());
    }
    Ok(out)
}

/// 2026-09-26: Escapes `&`, `<` and `>` for XML text; an unescaped `&` in a model id makes the
/// SVG invalid.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// 2026-09-26: Set an attribute on the element carrying `id`, replacing it if present; no-op
/// when the id is absent. Used to hide a whole box with `display="none"` on its group.
fn set_attr(svg: &mut String, id: &str, attr: &str, value: &str) {
    let needle = format!("id=\"{id}\"");
    let Some(at) = svg.find(&needle) else { return };
    let Some(open) = svg[..at].rfind('<') else {
        return;
    };
    let Some(close) = svg[at..].find('>').map(|i| at + i) else {
        return;
    };
    let existing = format!(" {attr}=\"");
    if let Some(k) = svg[open..close].find(&existing) {
        let vstart = open + k + existing.len();
        let Some(vend) = svg[vstart..close].find('"').map(|i| vstart + i) else {
            return;
        };
        svg.replace_range(vstart..vend, value);
        return;
    }
    svg.insert_str(close, &format!(" {attr}=\"{value}\""));
}

/// 2026-09-26: Replace the text body of `<... id="ID" ...>body</...>` with `value`, XML-escaped;
/// no-op when the id is absent.
fn set(svg: &mut String, id: &str, value: &str) {
    let needle = format!("id=\"{id}\"");
    let Some(at) = svg.find(&needle) else { return };
    let Some(gt) = svg[at..].find('>').map(|i| at + i + 1) else {
        return;
    };
    let Some(lt) = svg[gt..].find('<').map(|i| gt + i) else {
        return;
    };
    svg.replace_range(gt..lt, &esc(value));
}

/// 2026-09-26: Render a record onto the card template.
pub fn render(template: &str, record: &GateRecord, args: &BTreeMap<String, String>) -> String {
    let spec = spec_for(&record.benchmark_id);
    let mut svg = template.to_string();
    let m = &record.metrics;

    let hero = if spec.hero_key.is_empty() {
        m.iter().next().map(|(k, v)| (k.clone(), *v))
    } else {
        m.get(spec.hero_key)
            .map(|v| (spec.hero_key.to_string(), *v))
    };
    match hero {
        Some((key, v)) => {
            set(&mut svg, "value-toks", &spec.hero_fmt.apply(v));
            let label = if spec.hero_key.is_empty() {
                key
            } else {
                spec.hero_label.to_string()
            };
            set(&mut svg, "label-toks", &label);
        }
        // 2026-09-26: No hero metric (a failed run): print `—`, not the template's `000.0`.
        None => {
            set(&mut svg, "value-toks", "—");
            set(&mut svg, "label-toks", spec.hero_label);
        }
    }
    set(&mut svg, "note-toks", spec.hero_note);

    match record.verdict.as_deref() {
        Some(_) if record.verdict_passes() => {
            set(&mut svg, "value-verdict", "PASS");
            set_attr(&mut svg, "verdict-chip", "fill", "#12B981");
        }
        Some(_) => {
            set(&mut svg, "value-verdict", "FAIL");
            set_attr(&mut svg, "verdict-chip", "fill", "#EFB338");
            // 2026-09-26: Grey the hero so a failed run's number or dash does not read as a result.
            set_attr(&mut svg, "value-toks", "fill", "#82868F");
        }
        // 2026-09-26: An ungated run has no verdict; hide the chip.
        None => set_attr(&mut svg, "field-verdict", "display", "none"),
    }

    set(&mut svg, "value-model", &record.target_model);
    set(
        &mut svg,
        "value-quant",
        record
            .params
            .get("quant")
            .or_else(|| record.params.get("quantization"))
            .map_or_else(|| quant_from_model(&record.target_model), |q| q.clone())
            .as_str(),
    );
    set(
        &mut svg,
        "value-recipe",
        record.served_by.as_deref().unwrap_or("—"),
    );
    set(&mut svg, "value-test", &record.benchmark_id);
    set(&mut svg, "value-hardware", &hardware_line(record));

    for (i, s) in spec.slots.iter().enumerate() {
        let (lid, vid) = (format!("label-m{}", i + 1), format!("value-m{}", i + 1));
        match s.and_then(|s| m.get(s.key).map(|v| (s, *v))) {
            Some((s, v)) => {
                set(&mut svg, &lid, s.label);
                set(&mut svg, &vid, &s.fmt.apply(v));
            }
            // 2026-09-26: Hide the whole box of an unused slot, rather than leave the template's
            // placeholder text or an empty bordered box.
            None => set_attr(&mut svg, &format!("field-m{}", i + 1), "display", "none"),
        }
    }

    for (field, value_id, value) in [
        ("field-author", "value-author", args.get("author")),
        ("field-handle", "value-handle", args.get("handle")),
        (
            "field-site",
            "value-site",
            args.get("website").or_else(|| args.get("site")),
        ),
    ] {
        match value {
            Some(v) if !v.is_empty() => set(&mut svg, value_id, v),
            _ => set_attr(&mut svg, field, "display", "none"),
        }
    }
    svg
}

/// 2026-09-26: The quantization read off the checkpoint id (`unsloth/Qwen3.8-27B-NVFP4`) when
/// the record has no `quant`/`quantization` param; `—` when no known tag appears.
fn quant_from_model(model: &str) -> String {
    let up = model.to_uppercase();
    for tag in ["NVFP4", "FP8", "W4A4", "W4A16", "BF16", "INT8", "FP16"] {
        if up.contains(tag) {
            return tag.to_string();
        }
    }
    "—".to_string()
}

fn hardware_line(record: &GateRecord) -> String {
    let hw = &record.hardware;
    let gpu = hw.gate_key();
    match record
        .hardware_state
        .as_ref()
        .and_then(|s| s.before.sm_clock_mhz)
    {
        Some(mhz) => format!("{gpu}, SM {mhz:.0} MHz"),
        None => gpu,
    }
}
