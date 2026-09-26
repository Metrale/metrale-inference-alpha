// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `apply_param_overrides`: the baseline's
//! `[benchmarks.param_overrides]` pins against the real `concurrency-sweep`
//! schema.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use metrale_bench::gate;

/// 2026-09-26: A baseline entry with no metrics and only the given param pins.
fn entry_with_pins(pins: &[(&str, &str)]) -> gate::ModelBaseline {
    gate::ModelBaseline {
        recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".to_string()),
        label: String::new(),
        note: String::new(),
        metrics: BTreeMap::new(),
        serve_overrides: BTreeMap::new(),
        param_overrides: pins
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        serve_env: BTreeMap::new(),
    }
}

/// 2026-09-26: All three precedence arms: a pin replaces the schema default,
/// an explicit `--param` outranks a pin, and an unpinned key keeps its
/// schema default.
#[test]
fn param_overrides_pin_the_instrument_and_yield_to_an_explicit_param() {
    let descriptor = metrale_bench::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[
        ("concurrencies", "1,4,8,16"),
        ("isls", "512"),
        ("osl", "320"),
    ]);

    let mut values = metrale_bench::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let applied =
        apply_param_overrides(descriptor, &specs, &mut values, &entry, &[]).expect("applies");
    assert_eq!(applied.len(), 3, "{applied:?}");
    assert_eq!(values.int_list("concurrencies").unwrap(), &[1, 4, 8, 16]);
    assert_eq!(values.int_list("isls").unwrap(), &[512]);
    assert_eq!(values.usize("osl").unwrap(), 320);
    assert_eq!(values.usize("warmup").unwrap(), 1);

    let explicit = vec![("osl".to_string(), "512".to_string())];
    let mut values =
        metrale_bench::ParamValues::from_overrides(&specs, vec![("osl", "512")]).unwrap();
    let applied =
        apply_param_overrides(descriptor, &specs, &mut values, &entry, &explicit).expect("applies");
    assert!(
        applied.iter().all(|(k, _)| k != "osl"),
        "stated intent is never overridden: {applied:?}"
    );
    assert_eq!(values.usize("osl").unwrap(), 512);
    assert_eq!(
        values.int_list("concurrencies").unwrap(),
        &[1, 4, 8, 16],
        "the other pins still apply"
    );
}

/// 2026-09-26: A pin naming no schema parameter is an error naming the key.
#[test]
fn a_param_override_for_an_unknown_key_is_a_loud_error() {
    let descriptor = metrale_bench::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[("no_such_knob", "7")]);
    let mut values = metrale_bench::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_param_overrides(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("must refuse");
    let msg = format!("{err:#}");
    assert!(msg.contains("no_such_knob"), "{msg}");
    assert!(msg.contains("drifted"), "{msg}");
}

/// 2026-09-26: A pin naming a threshold-coupled parameter is refused.
#[test]
fn a_param_override_cannot_name_a_threshold_coupled_param() {
    let descriptor = metrale_bench::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[("min_c16", "94.0")]);
    let mut values = metrale_bench::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_param_overrides(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("must refuse");
    let msg = format!("{err:#}");
    assert!(msg.contains("min_c16"), "{msg}");
    assert!(msg.contains("threshold"), "{msg}");
}

/// 2026-09-26: An out-of-domain pin fails in the spec's own parser, as a
/// typed --param does.
#[test]
fn a_param_override_goes_through_the_kinds_own_parser() {
    let descriptor = metrale_bench::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[("osl", "0")]);
    let mut values = metrale_bench::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_param_overrides(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("must refuse");
    assert!(format!("{err:#}").contains("osl=0"), "{err:#}");
}
