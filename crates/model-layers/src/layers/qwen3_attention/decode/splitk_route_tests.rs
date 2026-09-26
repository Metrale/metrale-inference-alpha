// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the paged-decode dispatch: the route line as text, the GQA-packed route's
//! conditions, and the packed launchers' argument counts against the kernel sources.
//!
//! The FP8 route line is pinned whole: a format checked piece by piece can be reordered without a
//! test noticing.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use super::splitk_dispatch::{
    ROUTE_NONSPLIT_BF16, ROUTE_NONSPLIT_FP8, ROUTE_SPLITK_BF16, ROUTE_SPLITK_FP8, route_line,
};
use metrale_kernels::attn_splitk::SplitkPolicy;

/// 2026-09-25: The whole line for the FP8 Hopper split-K kernel under `auto`. `sm_count` is
/// expected as `metrale_kernels::TARGET_SM_COUNT`, the compiled target's, so the test holds on
/// every target.
#[test]
fn the_fp8_route_line_names_the_kernel_the_split_count_and_the_policy() {
    let sm = metrale_kernels::TARGET_SM_COUNT;
    assert_eq!(
        route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto),
        format!(
            "paged decode attention: paged_decode_attn_splitk_fp8_hopper num_splits=11 \
             sm_count={sm} policy=auto (METRALE_ATTN_DECODE_SPLITK)"
        ),
    );
}

/// 2026-09-25: The BF16 split-K kernel gets the same line, naming its own kernel.
#[test]
fn the_bf16_twin_reports_its_own_arm() {
    let line = route_line(ROUTE_SPLITK_BF16, 11, SplitkPolicy::Auto);
    assert!(
        line.contains("paged_decode_attn_splitk_bf16_hopper"),
        "{line}"
    );
    assert!(line.contains("num_splits=11"), "{line}");
    assert_ne!(line, route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto));
}

/// 2026-09-25: `METRALE_ATTN_DECODE_SPLITK=0` resolves to `Pinned(1)`, so the line names the
/// non-split kernel at one split under a policy that renders as `1`.
#[test]
fn the_zero_control_reports_the_non_split_kernel_at_one_split() {
    assert_eq!(
        route_line(ROUTE_NONSPLIT_FP8, 1, SplitkPolicy::Pinned(1)),
        format!(
            "paged decode attention: paged_decode_attn_fp8 num_splits=1 sm_count={} \
             policy=1 (METRALE_ATTN_DECODE_SPLITK)",
            metrale_kernels::TARGET_SM_COUNT,
        ),
    );
    // 2026-09-25: The policy field uses `SplitkPolicy::label`, the boot line's spelling.
    for (policy, label) in [
        (SplitkPolicy::Legacy, "legacy"),
        (SplitkPolicy::Auto, "auto"),
        (SplitkPolicy::Pinned(6), "6"),
    ] {
        assert!(
            route_line(ROUTE_NONSPLIT_BF16, 1, policy).contains(&format!("policy={label} ")),
            "policy {policy:?} must render as {label}",
        );
    }
}

/// 2026-09-25: The line leads with the kernel and ends by naming the environment variable that
/// moves it.
#[test]
fn the_line_names_its_lever_and_leads_with_the_kernel() {
    let line = route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto);
    assert!(line.ends_with("(METRALE_ATTN_DECODE_SPLITK)"), "{line}");
    assert!(
        line.starts_with("paged decode attention: paged_decode_attn_splitk_fp8_hopper "),
        "{line}"
    );
}

use super::splitk_dispatch::{ROUTE_GQA_BF16, ROUTE_GQA_FP8, gqa_pack_kernel, gqa_pack_route};
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-09-25: The packed arms name their own kernel in the route line, at `num_splits=1`. The
/// packed and unpacked kernels are meant to produce identical output, so the log line is how a run
/// shows which one launched.
#[test]
fn the_packed_arms_report_their_own_kernel_at_one_split() {
    for kernel in [ROUTE_GQA_FP8, ROUTE_GQA_BF16] {
        let line = route_line(kernel, 1, SplitkPolicy::Legacy);
        assert!(line.contains(kernel), "{line}");
        assert!(line.contains("num_splits=1"), "{line}");
    }
    assert_ne!(ROUTE_GQA_FP8, ROUTE_NONSPLIT_FP8);
    assert_ne!(ROUTE_GQA_BF16, ROUTE_NONSPLIT_BF16);
}

/// 2026-09-25: The packed route needs the lever, the shape and the handle.
///
/// Tested through `gqa_pack_route` with `armed` passed in: `gqa_pack_kernel` reads the lever
/// resolved once per process, which is false unless `METRALE_ATTN_DECODE_GQA_PACK` arms it, and
/// would then return `None` before reaching the shape check.
#[test]
fn the_packed_route_needs_the_lever_the_shape_and_the_handle() {
    let handle = Some(KernelHandle(7));

    // 2026-09-25: The lever. Declared off, so a default build takes the unpacked kernel.
    const { assert!(!metrale_kernels::attn_splitk::DECODE_GQA_PACK_DECLARED) };
    assert!(gqa_pack_route(false, handle, 24, 4, 256).is_none());
    assert_eq!(
        gqa_pack_kernel(handle, 24, 4, 256).is_some(),
        metrale_kernels::attn_splitk::gqa_pack_enabled(),
        "the public entry must be the route function at the resolved lever"
    );

    // 2026-09-25: The shape, with the lever armed. `(25, 4)` catches a truncating check:
    // 25 / 4 == 6 in integer division while 25 != 4 * 6.
    for (nq, nkv, hd) in [
        (32u32, 4u32, 256u32),
        (16, 4, 256),
        (24, 24, 256),
        (24, 1, 256),
        (25, 4, 256),
        (24, 4, 128),
        (24, 4, 512),
        (0, 0, 256),
    ] {
        assert!(
            gqa_pack_route(true, handle, nq, nkv, hd).is_none(),
            "nq={nq} nkv={nkv} hd={hd} must not reach the packed kernel"
        );
    }
    // 2026-09-25: The one shape it serves, so the loop above does not pass by refusing everything.
    assert_eq!(
        gqa_pack_route(true, handle, 24, 4, 256).map(|h| h.0),
        handle.map(|h| h.0)
    );

    assert!(gqa_pack_route(true, None, 24, 4, 256).is_none());
}

/// 2026-09-25: Reads `kernels/gb10/common/<name>`, relative to this crate's manifest dir.
fn kernel_src(name: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// 2026-09-25: The parameter count of an `extern "C" __global__` entry, by name.
///
/// Line comments are stripped first: the signatures carry `// [num_seqs, num_q_heads, head_dim]`
/// shape notes whose commas would otherwise count as separators.
fn cuda_param_count(src: &str, kernel: &str) -> usize {
    let at = src
        .find(&format!("{kernel}(\n"))
        .unwrap_or_else(|| panic!("no entry `{kernel}(` in source"));
    let mut lines = src[at..].lines();
    lines.next();
    let mut params = String::new();
    let mut closed = false;
    for line in lines {
        let code = line.split("//").next().unwrap_or("");
        // 2026-09-25: The closing `) {` is at column 0 in the unpacked kernels and indented in the
        // packed ones. Match either.
        if code.trim_start().starts_with(')') {
            closed = true;
            break;
        }
        params.push_str(code);
        params.push('\n');
    }
    assert!(closed, "`{kernel}`: parameter list has no closing paren");
    params.matches(',').count() + 1
}

/// 2026-09-25: The number of `.arg_*` calls in a launcher function in `ops/prefill_attn_a.rs`.
fn launcher_arg_count(name: &str) -> usize {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/layers/ops/prefill_attn_a.rs"),
    )
    .expect("read prefill_attn_a.rs");
    let at = src
        .find(&format!("pub fn {name}(\n"))
        .unwrap_or_else(|| panic!("no launcher `{name}`"));
    let body = &src[at..];
    let end = body[1..].find("\npub fn ").map_or(body.len(), |e| e + 1);
    body[..end].matches(".arg_").count()
}

/// 2026-09-25: Launcher argument count equals the compiled kernel's parameter count.
///
/// `cuLaunchKernel`'s `void**` form reads one host pointer per compiled parameter, so a launcher
/// that passes fewer makes the driver read past the end of the argument array. The compiled-binary
/// check in `crates/kernels/tests/kernel_arity.rs` is `#[ignore]`d (it needs nvcc), so these entry
/// points are checked here against the sources.
///
/// Each packed kernel takes the same argument list as the unpacked one it replaces, so each pair's
/// counts are asserted equal too.
#[test]
fn every_packed_launcher_passes_exactly_the_kernel_parameter_count() {
    let cases = [
        (
            "paged_decode_attn_fp8_gqa.cu",
            "paged_decode_attn_fp8_gqa",
            "paged_decode_attn_fp8.cu",
            "paged_decode_attn_fp8",
        ),
        (
            "paged_decode_attn_bf16_gqa.cu",
            "paged_decode_attn_bf16_gqa",
            "paged_decode_attn.cu",
            "paged_decode_attn",
        ),
    ];
    for (packed_file, packed_entry, base_file, base_entry) in cases {
        let params = cuda_param_count(&kernel_src(packed_file), packed_entry);
        let args = launcher_arg_count(packed_entry);
        assert_eq!(
            args, params,
            "{packed_entry}: launcher passes {args} args, kernel declares {params} params"
        );
        assert_eq!(
            params,
            cuda_param_count(&kernel_src(base_file), base_entry),
            "{packed_entry} must take the same argument list as {base_entry} — \
             it is a drop-in for it, differing only in grid"
        );
    }
}
