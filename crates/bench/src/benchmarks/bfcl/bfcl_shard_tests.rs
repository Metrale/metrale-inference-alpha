// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the BFCL `shard` parameter: parsing, refusals, and how
//! `configure` treats a constructor-selected slice.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: The `--param shard=i/n` surface, including every way it is
/// refused; each message says what was wrong with the value.
#[test]
fn the_shard_parameter_parses_and_refuses_precisely() {
    use super::dataset::Shard;
    assert_eq!(Shard::parse("2/7"), Ok(Shard { index: 2, count: 7 }));
    assert_eq!(Shard::parse(" 0 / 4 "), Ok(Shard { index: 0, count: 4 }));
    // 2026-09-26: The whole draw as a value is `0/1`; `1/1` is index 1 of one
    // shard and is refused.
    assert_eq!(Shard::parse("0/1"), Ok(Shard { index: 0, count: 1 }));
    assert!(
        Shard::parse("1/1").is_err(),
        "1/1 is index 1 of 1, not the whole draw"
    );

    // 2026-09-26: Indices are 0-based, and the message says so.
    let e = Shard::parse("4/4").expect_err("index 4 of 4 is out of range");
    assert!(e.contains("0-based"), "{e}");
    assert!(e.contains("the last is 3"), "{e}");

    let e = Shard::parse("0/0").expect_err("zero shards");
    assert!(e.contains("at least 1"), "{e}");

    let e = Shard::parse("2-7").expect_err("wrong separator");
    assert!(e.contains("index/count"), "{e}");

    let e = Shard::parse("a/4").expect_err("non-numeric index");
    assert!(e.contains("not a number"), "{e}");
}

/// 2026-09-26: `shard` defaults to `inherit` so that `configure` with default
/// parameters keeps a slice set by `Bfcl::sharded`; a default meaning "the
/// whole draw" would turn every shard into a copy of the whole draw.
///
/// A mutation control must change only the ParamSpec's default (for example to
/// `ParamValue::Text("0/1")`) and leave `INHERIT_SHARD` alone: the default is
/// built from that constant and `configure` compares against it, so changing
/// the constant moves both sides and the test stays green.
#[test]
fn a_shard_member_keeps_its_slice_under_default_parameters() {
    use super::dataset::Shard;
    let mut b = Bfcl::sharded(Variant::Subset, 2, 4);
    let values = ParamValues::defaults(&b.parameters());
    b.configure(&values).expect("defaults must configure");
    assert_eq!(
        b.shard,
        Some(Shard { index: 2, count: 4 }),
        "an empty `shard` param must leave the constructor's slice alone"
    );
}

/// 2026-09-26: The whole-draw benchmark stays whole under defaults, and an
/// explicit value overrides both it and a shard's constructor slice.
#[test]
fn an_explicit_shard_value_overrides_and_an_empty_one_does_not() {
    use super::dataset::Shard;
    let mut whole = Bfcl::new(Variant::Subset);
    let defaults = ParamValues::defaults(&whole.parameters());
    whole.configure(&defaults).expect("defaults");
    assert_eq!(whole.shard, None, "the whole draw stays whole");

    let mut values = defaults.clone();
    values.set("shard", ParamValue::Text("3/9".to_string()));
    whole.configure(&values).expect("explicit shard");
    assert_eq!(whole.shard, Some(Shard { index: 3, count: 9 }));

    let mut member = Bfcl::sharded(Variant::Subset, 0, 4);
    member
        .configure(&values)
        .expect("explicit shard on a member");
    assert_eq!(member.shard, Some(Shard { index: 3, count: 9 }));
}

/// 2026-09-26: `configure` refuses a bad value rather than carrying it into
/// the run.
#[test]
fn configure_refuses_a_bad_shard_rather_than_ignoring_it() {
    let mut b = Bfcl::new(Variant::Subset);
    let mut values = ParamValues::defaults(&b.parameters());
    values.set("shard", ParamValue::Text("9/4".to_string()));
    let e = b.configure(&values).expect_err("9/4 is out of range");
    assert!(e.to_string().contains("out of range"), "{e}");
}
