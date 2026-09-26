// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `KernelFlagPlan::from_args` on parsed, validated command lines.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::{GdnPlan, KernelFlagPlan};
use crate::cli::{Cli, Command, validate_serve_args};
use clap::Parser;

fn plan(flags: &[&str]) -> KernelFlagPlan {
    let mut argv = vec!["met", "serve", "org/model"];
    argv.extend_from_slice(flags);
    let Command::Serve(args) = Cli::try_parse_from(argv).expect("parses").command else {
        unreachable!("a serve command")
    };
    validate_serve_args(&args).expect("valid");
    KernelFlagPlan::from_args(&args)
}

#[test]
fn an_empty_command_line_publishes_nothing_the_environment_owns() {
    // 2026-09-26: A `Some` here would seal that cell's `METRALE_*` fallback on
    // every boot.
    assert_eq!(
        plan(&[]),
        KernelFlagPlan {
            gdn: None,
            w4a4_downcast: false,
            w4a4_downcast_wide: false,
            prefill_codispatch: None,
            prefill_varlen: None,
            ssm_tail_midchunk: None,
            hermetic: false,
        }
    );
    // 2026-09-26: `auto` is the absence of a pin, not a GDN flag.
    assert_eq!(plan(&["--ssm-batched-recurrent", "auto"]).gdn, None);
}

#[test]
fn any_gdn_flag_hands_the_whole_cell_to_the_command_line() {
    let fused = GdnPlan {
        h_f16: false,
        h_f16_pool: false,
        fused_norm: true,
        batched_recurrent: None,
        exact_verify: false,
    };
    assert_eq!(plan(&["--gdn-fused-norm"]).gdn, Some(fused));
    // 2026-09-26: A GDN flag that does not name batched recurrence leaves it
    // `None` (the target default); an explicit pin is carried either way.
    assert_eq!(
        plan(&["--ssm-batched-recurrent", "off"]).gdn,
        Some(GdnPlan {
            fused_norm: false,
            batched_recurrent: Some(false),
            ..fused
        })
    );
    assert_eq!(
        plan(&["--gdn-fused-norm", "--ssm-batched-recurrent", "on"]).gdn,
        Some(GdnPlan {
            batched_recurrent: Some(true),
            ..fused
        })
    );
    assert_eq!(
        plan(&["--exact-verify"]).gdn,
        Some(GdnPlan {
            fused_norm: false,
            exact_verify: true,
            ..fused
        })
    );
    assert_eq!(
        plan(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"]).gdn,
        Some(GdnPlan {
            h_f16: true,
            ..fused
        })
    );
}

#[test]
fn each_presence_flag_publishes_its_non_default_state_only() {
    let p = plan(&[
        "--prefill-codispatch",
        "--prefill-varlen-batch",
        "--no-ssm-tail-midchunk",
        "--hermetic",
    ]);
    assert_eq!(p.prefill_codispatch, Some(true));
    assert_eq!(p.prefill_varlen, Some(true));
    assert_eq!(p.ssm_tail_midchunk, Some(false));
    assert!(p.hermetic);
}

#[test]
fn the_wide_downcast_needs_the_downcast() {
    let alone = plan(&["--w4a4-downcast-wide"]);
    assert!(!alone.w4a4_downcast && !alone.w4a4_downcast_wide);
    let both = plan(&["--w4a4-downcast", "--w4a4-downcast-wide"]);
    assert!(both.w4a4_downcast && both.w4a4_downcast_wide);
}
