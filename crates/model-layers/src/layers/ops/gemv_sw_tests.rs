// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the W4A16 decode GEMV launchers and source-contract tests on the kernel files they launch.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;
use crate::layers::ops::kernel_tree_tests_util::{KernelFile, compiled_cu_files};
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use std::fs;
use std::path::Path;

#[test]
fn gemv_sw_ships_on_and_only_the_one_value_kills() {
    assert!(gemv_sw_from(None), "unset → ON");
    assert!(gemv_sw_from(Some("0")), "`=0` is NOT off");
    assert!(gemv_sw_from(Some("")), "empty is NOT off");
    assert!(!gemv_sw_from(Some("1")), "`=1` is the kill");
}

#[test]
fn sw_requires_both_the_lever_and_a_live_handle() {
    assert!(use_gemv_sw(true, KernelHandle(1)));
    assert!(
        !use_gemv_sw(true, KernelHandle(0)),
        "missing kernel falls back"
    );
    assert!(!use_gemv_sw(false, KernelHandle(1)), "kill switch wins");
    assert!(!use_gemv_sw(false, KernelHandle(0)));
}

#[test]
fn sw_grid_covers_every_output_and_is_half_base_when_n_divisible_by_8() {
    for n in 1..=64 {
        assert!(w4a16_gemv_sw_grid_x(n) * W4A16_GEMV_SW_OUTS_PER_BLOCK >= n);
        assert!(w4a16_gemv_grid_x(n) * W4A16_GEMV_OUTS_PER_BLOCK >= n);
    }
    for n in [8u32, 16, 256, 5120, 14336] {
        assert_eq!(
            w4a16_gemv_sw_grid_x(n) * 2,
            w4a16_gemv_grid_x(n),
            "N={n}: SW is 8 outs/block, base is 4 — grid_x must be half"
        );
    }
}

#[test]
fn decode_dispatch_uses_the_selected_handle_and_matching_grid() {
    for (lever, sw_handle, expected_handle, expected_grid_x) in [
        (true, KernelHandle(22), 22, 2),
        (false, KernelHandle(22), 11, 3),
        (true, KernelHandle(0), 11, 3),
    ] {
        let gpu = MockGpuBackend::new();
        w4a16_decode_gemv(
            &gpu,
            KernelHandle(11),
            sw_handle,
            lever,
            DevicePtr::NULL,
            &QuantizedWeight::null(),
            DevicePtr::NULL,
            9,
            128,
            0,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].func, expected_handle);
        assert_eq!(launches[0].grid, [expected_grid_x, 1, 1]);
        assert_eq!(launches[0].block, [256, 1, 1]);
    }
}

/// 2026-09-25: Every place the build compiles `file_name`.
fn named_cu(file_name: &str) -> Vec<KernelFile> {
    compiled_cu_files()
        .into_iter()
        .filter(|f| f.path.file_name().is_some_and(|n| n == file_name))
        .collect()
}

/// 2026-09-25: Every compiled copy of w4a16_gemv.cu defines `N_PER_BLOCK` and `N_PER_BLOCK_SW`
/// with the launcher's values, and every copy of w4a16_gemv_fused.cu defines
/// `N_PER_BLOCK_SW`. A mismatch would not raise a CUDA error; the grid would cover the wrong
/// outputs. Changing either `#define` in one copy fails this test.
#[test]
fn cuda_n_per_block_matches_rust_ssot() {
    let gemv = named_cu("w4a16_gemv.cu");
    assert!(
        gemv.len() >= 3,
        "expected gb10 + strix + strix-hip copies, got {gemv:?}"
    );
    let want_base = format!("#define N_PER_BLOCK {W4A16_GEMV_OUTS_PER_BLOCK}");
    let want_sw = format!("#define N_PER_BLOCK_SW {W4A16_GEMV_SW_OUTS_PER_BLOCK}");
    for p in &gemv {
        let src = fs::read_to_string(&p.source).unwrap();
        assert!(
            src.contains(&want_base),
            "{} missing {want_base}",
            p.path.display()
        );
        assert!(
            src.contains(&want_sw),
            "{} missing {want_sw}",
            p.path.display()
        );
    }
    let fused = named_cu("w4a16_gemv_fused.cu");
    assert!(
        !fused.is_empty(),
        "dual_sw / silu_input_sw live in w4a16_gemv_fused.cu"
    );
    for p in &fused {
        let src = fs::read_to_string(&p.source).unwrap();
        assert!(
            src.contains(&want_sw),
            "{} missing {want_sw}",
            p.path.display()
        );
    }
}

/// 2026-09-25: The single-warp and base kernels share one partial-sum loop:
/// `w4a16_gemv_partial` starts `k16` at `orig_lane * 2u` with the `K16 + 1u` bound, and the
/// dual kernels share `w4a16_dual_partial`. A `k16 += 64u` loop in w4a16_gemv.cu, or losing
/// `orig_lane * 2u`, fails this test.
#[test]
fn sw_partial_shares_pipelined_k16_loop() {
    for p in named_cu("w4a16_gemv.cu") {
        let src = fs::read_to_string(&p.source).unwrap();
        assert!(
            src.contains("orig_lane * 2u"),
            "{}: w4a16_gemv_partial must start k16 at orig_lane*2",
            p.path.display()
        );
        assert!(
            src.contains("k16 < K16 + 1u"),
            "{}: pipelined K16+1 bound missing",
            p.path.display()
        );
        assert!(
            !src.contains("k16 += 64u"),
            "{}: stride-64 sequential loop drifted back in",
            p.path.display()
        );
    }
    for p in named_cu("w4a16_gemv_fused.cu") {
        let src = fs::read_to_string(&p.source).unwrap();
        assert!(
            src.contains("w4a16_dual_partial"),
            "{}: dual and dual_sw must share w4a16_dual_partial",
            p.path.display()
        );
        assert!(
            src.contains("orig_lane * 2u"),
            "{}: dual_partial must start k16 at orig_lane*2",
            p.path.display()
        );
    }
}

/// 2026-09-25: Split the declaration starting at `sig` into (parameter list, body). The body is
/// brace-matched, so nested blocks are kept and the next function is not included.
fn fn_signature_and_body<'a>(src: &'a str, sig: &str) -> (&'a str, &'a str) {
    let start = src
        .find(sig)
        .unwrap_or_else(|| panic!("signature `{sig}` not found"));
    let open = start
        + src[start..]
            .find('{')
            .expect("no body brace after signature");
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return (&src[start..open], &src[open..=open + i]);
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after `{sig}`");
}

fn fn_body<'a>(src: &'a str, sig: &str) -> &'a str {
    fn_signature_and_body(src, sig).1
}

/// 2026-09-25: (file, partial signature, `__constant__` table it must not index, callers that
/// must pass it a shared-memory copy)
const DECODE_PARTIALS: &[(&str, &str, &str, &[(&str, &str)])] = &[
    (
        "w4a16_gemv.cu",
        "__device__ __forceinline__ float w4a16_gemv_partial(",
        "E2M1_LUT",
        &[("w4a16_gemv", "s_lut"), ("w4a16_gemv_sw", "warp_lut")],
    ),
    (
        "w4a16_gemv_fused.cu",
        "__device__ __forceinline__ float w4a16_dual_partial(",
        "E2M1_LUT_FUSED_W4",
        &[
            ("w4a16_gemv_dual", "s_lut"),
            ("w4a16_gemv_dual_sw", "warp_lut"),
        ],
    ),
    (
        "w4a16_gemv_fused.cu",
        "__device__ __forceinline__ float w4a16_silu_partial(",
        "E2M1_LUT_FUSED_W4",
        &[("w4a16_gemv_silu_input_sw", "warp_lut")],
    ),
];

/// 2026-09-25: Every M=1 decode GEMV partial in `DECODE_PARTIALS` dequantizes through a `lut`
/// parameter, and each listed caller passes it a copy of the E2M1 table staged in shared
/// memory, never the `__constant__` table. The index is a data-dependent weight nibble, and
/// constant memory serializes a warp's request across distinct addresses; shared memory
/// serves the 16 entries from 16 banks. The staged copy holds the same FP32 values, so this
/// does not change results. Indexing `E2M1_LUT[byte_val ...]` in a partial fails this test.
#[test]
fn decode_gemv_partials_index_a_shared_staged_lut() {
    for &(file, sig, table, callers) in DECODE_PARTIALS {
        let partial = sig
            .strip_suffix('(')
            .and_then(|sig| sig.split_whitespace().last())
            .expect("partial signature ends in a function name");
        let paths = named_cu(file);
        assert!(paths.len() >= 3, "{file}: expected 3 backend copies");
        for path in paths {
            let src = fs::read_to_string(&path.source).unwrap();
            let where_ = format!("{}::{sig}", path.path.display());
            let (params, body) = fn_signature_and_body(&src, sig);
            assert!(
                !body.contains(&format!("{table}[byte_val")),
                "{where_}: data-dependent index into __constant__ {table} \
                     serializes the warp — take the staged `lut` instead"
            );
            assert!(
                body.contains("lut[byte_val"),
                "{where_}: must dequant through the staged `lut` parameter"
            );
            assert!(
                params.contains("const float* __restrict__ lut"),
                "{where_}: must accept the staged table as `const float* __restrict__ lut`"
            );
            for &(caller, expected_lut) in callers {
                let cb = fn_body(&src, &format!("void {caller}("));
                assert!(
                    cb.contains("__shared__ float s_lut"),
                    "{}::{caller}: must stage the E2M1 table in shared memory",
                    path.path.display()
                );
                let calls: Vec<_> = cb
                    .lines()
                    .filter(|line| line.contains(&format!("{partial}(")))
                    .collect();
                assert!(
                    !calls.is_empty(),
                    "{}::{caller}: no {partial} call",
                    path.path.display()
                );
                for call in calls {
                    assert!(
                        call.contains(&format!(", {expected_lut})")),
                        "{}::{caller}: `{partial}` must receive {expected_lut}, got `{}`",
                        path.path.display(),
                        call.trim()
                    );
                    assert!(
                        !call.contains(table),
                        "{}::{caller}: `{partial}` received constant-memory {table}",
                        path.path.display()
                    );
                }
            }
        }
    }
}

/// 2026-09-25: The single-warp kernels stage the LUT per warp and contain no block barrier.
///
/// 1. `w4a16_gemv_sw`, `w4a16_gemv_dual_sw` and `w4a16_gemv_silu_input_sw` return early on
///    `n >= N`, where `n = blockIdx.x * N_PER_BLOCK_SW + threadIdx.x / 32`. That is uniform per
///    warp but not per block, so a `__syncthreads()` after it would be a divergent barrier
///    (undefined behaviour, not a compile error). The staging helpers publish with
///    `__syncwarp()`, and the kernels contain no `__syncthreads()`.
/// 2. Each warp owns one 16-float row, so the array is sized by `N_PER_BLOCK_SW`, not a
///    literal 8 (512 B per block at 8).
///
/// Replacing `__syncwarp()` with `__syncthreads()`, or writing `s_lut[8][16]`, fails this test.
#[test]
fn sw_gemv_stages_the_lut_per_warp_without_a_block_barrier() {
    let want_rows = "__shared__ float s_lut[N_PER_BLOCK_SW][16]";
    for (file, kernels, helper) in [
        (
            "w4a16_gemv.cu",
            &["w4a16_gemv_sw"][..],
            "stage_e2m1_lut_warp",
        ),
        (
            "w4a16_gemv_fused.cu",
            &["w4a16_gemv_dual_sw", "w4a16_gemv_silu_input_sw"][..],
            "stage_e2m1_lut_fused_warp",
        ),
    ] {
        for path in named_cu(file) {
            let src = fs::read_to_string(&path.source).unwrap();
            let hb = fn_body(&src, &format!("void {helper}("));
            assert!(
                hb.contains("__syncwarp()"),
                "{}::{helper}: warp-scoped staging must publish with __syncwarp()",
                path.path.display()
            );
            for k in kernels {
                let kb = fn_body(&src, &format!("void {k}("));
                assert!(
                    kb.contains(want_rows),
                    "{}::{k}: per-warp LUT rows must be sized by N_PER_BLOCK_SW",
                    path.path.display()
                );
                assert!(
                    kb.contains(&format!("{helper}(s_lut[local_out], lane)")),
                    "{}::{k}: must stage its own warp row",
                    path.path.display()
                );
                assert!(
                    !kb.contains("__syncthreads()"),
                    "{}::{k}: block barrier after a warp-uniform early return is \
                         divergent UB — and it would undo the barrier-free reduction",
                    path.path.display()
                );
            }
        }
    }
}

/// 2026-09-25: The attention decode files listed below contain no `w4a16_gemv(` call; they use
/// `nvfp4_decode_gemv` (`qwen3_attention/helpers.rs`), which picks the single-warp kernel and
/// its grid. A direct call would run the 64-thread kernel even when the single-warp one is
/// enabled.
#[test]
fn attention_decode_does_not_call_base_w4a16_gemv() {
    let attn = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/layers/qwen3_attention");
    let mut offenders = Vec::new();
    for rel in [
        "decode/attention_forward.rs",
        "decode/attention_forward/lora.rs",
        "decode/attention_forward/q_proj.rs",
        "decode/attention_forward/rope.rs",
        "decode/attention_forward_v4.rs",
        "decode/attention_forward_v4/comp_append.rs",
        "decode/attention_forward_v4/derotate.rs",
        "decode/attention_forward_v4/proj.rs",
        "decode/attention_forward_oproj.rs",
        "decode/attention_forward_mla.rs",
        "decode/attention_forward_kv.rs",
        "trait_impl/multi_seq/qkv.rs",
        "trait_impl/multi_seq/qkv/batch.rs",
        "trait_impl/multi_seq/qkv/post.rs",
        "trait_impl/multi_seq/qkv/rows.rs",
        "trait_impl/multi_seq/attn.rs",
        "trait_impl/multi_seq/attn/o_proj.rs",
        "trait_impl/multi_seq/mla.rs",
        "trait_impl/multi_seq/mla/decode_one.rs",
    ] {
        let src = fs::read_to_string(attn.join(rel)).unwrap();
        if src.contains("w4a16_gemv(") {
            offenders.push(rel);
        }
    }
    assert!(
        offenders.is_empty(),
        "use nvfp4_decode_gemv (N/8 grid) not ops::w4a16_gemv: {offenders:?}"
    );
}
