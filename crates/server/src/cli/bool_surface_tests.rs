// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Every boolean on the `met` command line is a presence flag,
//! and no flag may be given without its value.
//!
//! A value-taking boolean makes "given false" and "absent" two configurations
//! that read alike in a recipe and render differently. The first two tests
//! walk clap's description of every subcommand; the third parses every
//! `serve` boolean.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::Cli;
use clap::builder::ValueParser;
use clap::{CommandFactory, Parser};

/// 2026-09-26: Every argument of `cmd` and its subcommands, with the path
/// that reaches it.
fn walk(cmd: &clap::Command, path: &str, out: &mut Vec<(String, clap::Arg)>) {
    for arg in cmd.get_arguments() {
        out.push((path.to_string(), arg.clone()));
    }
    for sub in cmd.get_subcommands() {
        walk(sub, &format!("{path} {}", sub.get_name()), out);
    }
}

fn all_args() -> Vec<(String, clap::Arg)> {
    let mut out = Vec::new();
    // 2026-09-26: Built, so every argument reports its resolved arity.
    let mut cli = Cli::command();
    cli.build();
    walk(&cli, "met", &mut out);
    assert!(
        out.len() > 150,
        "the walk found only {} arguments",
        out.len()
    );
    out
}

fn is_bool(arg: &clap::Arg) -> bool {
    arg.get_value_parser().type_id() == ValueParser::bool().type_id()
}

#[test]
fn every_bool_flag_is_presence_only() {
    let offenders: Vec<String> = all_args()
        .iter()
        .filter(|(_, a)| is_bool(a) && !matches!(a.get_action(), clap::ArgAction::SetTrue))
        .map(|(p, a)| format!("{p} --{}", a.get_long().unwrap_or("?")))
        .collect();
    assert!(
        offenders.is_empty(),
        "boolean flags that take a value (make them ArgAction::SetTrue, named for the \
         non-default state, or an auto/on/off enum): {offenders:?}"
    );
}

/// 2026-09-26: A long flag that takes a value must be given one: no minimum
/// arity of 0, which would let a bare `--flag` stand for a hidden value.
#[test]
fn no_value_flag_can_be_given_without_its_value() {
    let offenders: Vec<String> = all_args()
        .iter()
        .filter(|(_, a)| a.get_action().takes_values())
        .filter(|(_, a)| a.get_long().is_some())
        .filter(|(_, a)| a.get_num_args().is_some_and(|r| r.min_values() == 0))
        .map(|(p, a)| format!("{p} --{}", a.get_long().unwrap_or("?")))
        .collect();
    assert!(
        offenders.is_empty(),
        "flags that accept being given without a value: {offenders:?}"
    );
}

/// 2026-09-26: Every serve boolean parses bare and refuses `--flag true`,
/// `--flag false`, `--flag=true` and `--flag=false`. The MODEL positional is
/// filled, so a stray `true` cannot be taken as the model.
#[test]
fn a_serve_boolean_refuses_a_value() {
    let serve = Cli::command();
    let serve = serve
        .find_subcommand("serve")
        .expect("serve subcommand")
        .clone();
    let bools: Vec<String> = serve
        .get_arguments()
        .filter(|a| is_bool(a))
        .filter_map(|a| a.get_long().map(str::to_string))
        .collect();
    assert!(
        bools.len() > 25,
        "only {} serve booleans found",
        bools.len()
    );
    for flag in &bools {
        let bare = ["met", "serve", "org/model", &format!("--{flag}")].map(String::from);
        assert!(
            Cli::try_parse_from(&bare).is_ok(),
            "--{flag} bare must parse"
        );
        for value in ["true", "false"] {
            let spaced = ["met", "serve", "org/model", &format!("--{flag}"), value];
            assert!(
                Cli::try_parse_from(spaced).is_err(),
                "--{flag} {value} was accepted"
            );
            let joined = ["met", "serve", "org/model", &format!("--{flag}={value}")];
            assert!(
                Cli::try_parse_from(joined).is_err(),
                "--{flag}={value} was accepted"
            );
        }
    }
}
