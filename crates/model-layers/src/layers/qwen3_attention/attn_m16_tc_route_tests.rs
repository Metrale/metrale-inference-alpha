// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU tests for the `METRALE_ATTN_M16_TC` route lines: each latch fires once per
//! model, on its own key only, and each message names the lever, the kernel, the row band and the
//! off switch.
//!
//! The `tc` arm selection itself is inline in `qkv_fp8_batch.rs` and `attn/o_proj.rs` and is not
//! exercised here.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use super::{
    O_PROJ_M16_TC_ROUTE_KEY, O_PROJ_M16_TC_ROUTE_MSG, QKV_M16_TC_ROUTE_KEY, QKV_M16_TC_ROUTE_MSG,
    log_o_proj_m16_tc_route, log_qkv_m16_tc_route,
};
use crate::layers::ops::ModelStats;

#[test]
fn qkv_route_fires_exactly_once_per_model() {
    let stats = ModelStats::new();
    log_qkv_m16_tc_route(&stats);
    log_qkv_m16_tc_route(&stats);
    log_qkv_m16_tc_route(&stats);
    // 2026-09-25: The calls above consumed the latch, so a direct probe of the same key returns
    // false.
    assert!(
        !stats.once(QKV_M16_TC_ROUTE_KEY),
        "three calls must consume the latch exactly once, not three times \
         or zero times"
    );
}

#[test]
fn o_proj_route_fires_exactly_once_per_model() {
    let stats = ModelStats::new();
    log_o_proj_m16_tc_route(&stats);
    log_o_proj_m16_tc_route(&stats);
    assert!(!stats.once(O_PROJ_M16_TC_ROUTE_KEY));
}

#[test]
fn each_tier_fires_its_own_arm_only() {
    // 2026-09-25: Firing the Q/K/V route must not consume the o_proj key, and the reverse, so an
    // operator can tell the two tiers apart.
    let stats = ModelStats::new();
    log_qkv_m16_tc_route(&stats);
    assert!(
        stats.once(O_PROJ_M16_TC_ROUTE_KEY),
        "the o_proj key must still be unconsumed after only the q/k/v route fired"
    );

    let stats = ModelStats::new();
    log_o_proj_m16_tc_route(&stats);
    assert!(
        stats.once(QKV_M16_TC_ROUTE_KEY),
        "the q/k/v key must still be unconsumed after only the o_proj route fired"
    );
}

#[test]
fn qkv_message_names_the_lever_kernel_band_and_off_switch() {
    let msg = QKV_M16_TC_ROUTE_MSG;
    assert!(msg.contains("METRALE_ATTN_M16_TC"), "must name the lever");
    assert!(
        msg.contains("w8a16_gemm_m16_strided"),
        "must name the kernel this tier actually launches"
    );
    assert!(msg.contains("5..=16"), "must name the row band");
    assert!(
        msg.contains("w8a16_gemv_batch16_strided"),
        "must name the bit-exact kernel it displaces"
    );
    assert!(
        msg.contains("REASSOCIATED"),
        "must state the numeric consequence"
    );
    assert!(
        msg.contains("Unset it to restore the bit-exact tier"),
        "must name the off switch"
    );
    assert!(
        msg.contains("#927") && msg.contains("cell W"),
        "must cite the receipt"
    );
}

#[test]
fn o_proj_message_names_the_lever_kernel_band_and_off_switch() {
    let msg = O_PROJ_M16_TC_ROUTE_MSG;
    assert!(msg.contains("METRALE_ATTN_M16_TC"), "must name the lever");
    assert!(
        msg.contains("w8a16_gemm_m16") && !msg.contains("w8a16_gemm_m16_strided"),
        "must name the CONTIGUOUS kernel, not the strided one the q/k/v tier uses"
    );
    assert!(msg.contains("5..=16"), "must name the row band");
    assert!(
        msg.contains("w8a16_gemv_batch16") && !msg.contains("w8a16_gemv_batch16_strided"),
        "must name the CONTIGUOUS bit-exact kernel it displaces"
    );
    assert!(
        msg.contains("REASSOCIATED"),
        "must state the numeric consequence"
    );
    assert!(
        msg.contains("Unset it to restore the bit-exact tier"),
        "must name the off switch"
    );
}
