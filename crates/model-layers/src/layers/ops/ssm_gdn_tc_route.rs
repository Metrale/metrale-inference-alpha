// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GDN prefill state spines' entry names, and the route lines
//! built from them.
//!
//! # Why the name is a constant
//!
//! The tensor-core module holds two entries, `…_tcfuse` (one bf16 limb of
//! `S_c`) and `…_tcfuse_x2` (two limbs), and the lever ships `_x2`. A route
//! line that names only the family, such as
//!
//! ```text
//! GDN state spine: gated_delta_rule_chunk_delta_h_tcfuse (METRALE_GDN_PREFILL_TC; …)
//! ```
//!
//! reads as the 1-limb entry. So the name exists once, here:
//! `qwen3_ssm::init_kernels` binds the handle with it and
//! [`gdn_tc_spine_route_line`] prints it.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

/// 2026-09-25: The spine entry the `gdn_prefill_tc` lever launches: two bf16
/// limbs of `S_c`. Both entries take the same arguments, grid, block and
/// shared memory, so nothing downstream depends on which one is bound.
pub const GDN_TC_SPINE_ENTRY: &str = "gated_delta_rule_chunk_delta_h_tcfuse_x2";

/// 2026-09-25: The module the entry lives in:
/// `kernels/hopper/common/gated_delta_rule_chunk_tc.cu`.
pub const GDN_TC_SPINE_MODULE: &str = "gated_delta_rule_chunk_tc";

/// 2026-09-25: The scalar spine entries `init_kernels::fused_spine_kernel`
/// binds: this one under `METRALE_GDN_PIPE=1`, the next under
/// `METRALE_GDN_VTILE=1`, the last by default. The init route line and the
/// handle are built from the same strings.
pub const GDN_SCALAR_SPINE_PIPE: &str = "gated_delta_rule_chunk_delta_h_pipe";
/// 2026-09-25: SPLIT=4, 512 threads. Not the default; `fused_spine_kernel`'s
/// doc says why.
pub const GDN_SCALAR_SPINE_VTILE: &str = "gated_delta_rule_chunk_delta_h_vtile";
/// 2026-09-25: SPLIT=2, 256 threads: the default scalar spine.
pub const GDN_SCALAR_SPINE_VFUSED: &str = "gated_delta_rule_chunk_delta_h_vfused";

/// 2026-09-25: `GDN state spine: …`, the line `qwen3_ssm::init` prints for
/// each layer as it binds the handles, before any prefill has run.
///
/// # Why it is not simply the scalar entry's name
///
/// With the tensor-core handle bound, the prefill launches
/// [`GDN_TC_SPINE_ENTRY`], and the scalar entry stays bound only as the
/// fallback the prefill's shape guards drop to. An init line that named the
/// scalar entry would contradict the prefill's line:
///
/// ```text
///    48  qwen3_ssm::init: GDN state spine: gated_delta_rule_chunk_delta_h_vfused
/// 14400  GDN state spine: gated_delta_rule_chunk_delta_h_tcfuse_x2 (…)
/// ```
///
/// So the line follows the handle the probe resolved: bound, it names
/// [`GDN_TC_SPINE_ENTRY`]; unbound, `scalar_entry`.
pub fn gdn_init_spine_line(tc_spine_bound: bool, scalar_entry: &str) -> String {
    if tc_spine_bound {
        format!(
            "GDN state spine: {GDN_TC_SPINE_ENTRY} ([defaults] gdn_prefill_tc; the \
             scalar spine stays bound as the fallback the prefill's shape guards \
             drop to, and the prefill logs the entry it launches)"
        )
    } else {
        format!("GDN state spine: {scalar_entry}")
    }
}

/// 2026-09-25: `GDN state spine: …`, the line the prefill logs when the
/// tensor-core spine runs, built from [`GDN_TC_SPINE_ENTRY`]. Pure and
/// returning a `String`, so the tests below can check it.
pub fn gdn_tc_spine_route_line(num_v_heads: u32, batch_size: u32, smem_bytes: u32) -> String {
    format!(
        "GDN state spine: {GDN_TC_SPINE_ENTRY} (METRALE_GDN_PREFILL_TC; bf16 mma.sync \
         operands, f32 accumulator = the recurrent state, h stays f32) \
         grid=[{num_v_heads},{batch_size}] block=256 smem={smem_bytes}B"
    )
}

#[cfg(test)]
mod tests {
    use super::{
        GDN_SCALAR_SPINE_PIPE, GDN_SCALAR_SPINE_VFUSED, GDN_SCALAR_SPINE_VTILE, GDN_TC_SPINE_ENTRY,
        gdn_init_spine_line, gdn_tc_spine_route_line,
    };

    /// 2026-09-25: The line names the `_x2` entry, never the bare family name
    /// (`…_tcfuse` followed by a space).
    #[test]
    fn the_route_line_names_the_entry_that_is_launched() {
        let line = gdn_tc_spine_route_line(48, 1, 88_324);
        assert!(line.contains(GDN_TC_SPINE_ENTRY), "{line}");
        assert!(
            !line.contains("gated_delta_rule_chunk_delta_h_tcfuse "),
            "the line must not name the FAMILY where the binary launches the \
             `_x2` member — round 12 stage 4b:\n{line}"
        );
    }

    /// 2026-09-25: The line carries the launch geometry.
    #[test]
    fn the_route_line_carries_the_geometry() {
        let line = gdn_tc_spine_route_line(48, 2, 88_324);
        for field in [
            "grid=[48,2]",
            "block=256",
            "smem=88324B",
            "METRALE_GDN_PREFILL_TC",
        ] {
            assert!(line.contains(field), "missing `{field}` in:\n{line}");
        }
    }

    /// 2026-09-25: With the tensor-core handle bound, the init line names that
    /// entry, not the scalar spine bound behind it.
    #[test]
    fn the_init_line_names_the_tc_entry_when_its_handle_is_bound() {
        let line = gdn_init_spine_line(true, GDN_SCALAR_SPINE_VFUSED);
        assert!(line.contains(GDN_TC_SPINE_ENTRY), "{line}");
        assert!(
            !line.contains(GDN_SCALAR_SPINE_VFUSED),
            "the init line must not NAME the scalar spine where the probe bound \
             the tensor-core one:\n{line}"
        );
    }

    /// 2026-09-25: The init line and the prefill's line name the same entry.
    #[test]
    fn the_two_route_lines_agree_on_the_entry() {
        let init = gdn_init_spine_line(true, GDN_SCALAR_SPINE_VFUSED);
        let dispatch = gdn_tc_spine_route_line(48, 1, 88_324);
        for line in [&init, &dispatch] {
            assert!(line.contains(GDN_TC_SPINE_ENTRY), "{line}");
        }
    }

    /// 2026-09-25: With the probe off (by default, every target but
    /// `kernels/hopper`), the line names the scalar entry, for each of the
    /// three.
    #[test]
    fn the_init_line_names_the_scalar_entry_when_the_probe_is_off() {
        for entry in [
            GDN_SCALAR_SPINE_PIPE,
            GDN_SCALAR_SPINE_VTILE,
            GDN_SCALAR_SPINE_VFUSED,
        ] {
            assert_eq!(
                gdn_init_spine_line(false, entry),
                format!("GDN state spine: {entry}"),
            );
        }
    }
}
