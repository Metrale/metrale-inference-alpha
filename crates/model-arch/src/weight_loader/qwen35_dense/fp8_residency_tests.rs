// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `DenseFp8Plan::resolve`, the byte functions and the
//! summary line. Each route is given as a `GemmDispatch` value and plain
//! flags, never through the process environment.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants: none beyond the types.

use super::*;
use metrale_model_layers::layers::ops::GemmDispatch;

/// 2026-09-25: FP8 overlays on both the FFN and attention, the default
/// `GemmDispatch` (block-scaled prefill on), the W8A8 kernels present, and no
/// other lever.
fn h100_default() -> DenseFp8Inputs {
    DenseFp8Inputs {
        ffn_fp8: true,
        attn_fp8: true,
        keep_nvfp4: false,
        dispatch: GemmDispatch::defaults(),
        w8a8_kernels: true,
        attn_w4a4: false,
        attn_prefill_q_t: false,
    }
}

/// 2026-09-25: The twin set `h100_default` resolves to.
const KV_ONLY: Fp8TwinSet = Fp8TwinSet {
    q: false,
    k: true,
    v: true,
    o: false,
};

#[test]
fn default_native_fp8_route_builds_no_nvfp4_and_only_the_kv_fp8_twins() {
    let p = DenseFp8Plan::resolve(h100_default());
    assert_eq!(
        p,
        DenseFp8Plan {
            ffn_nvfp4: false,
            attn_nvfp4: false,
            attn_fp8_twins: KV_ONLY,
        },
        "the dense FFN and the attention NVFP4 fallbacks are unreachable; K/V \
         FP8 twins are NOT (cache_skip_qkv.rs has no W8A8 arm)"
    );
}

#[test]
fn the_kv_fp8_twins_survive_every_default_route_variation() {
    // 2026-09-25: The first-chunk chain (`prefill/cache_skip_qkv.rs`) reads
    // the K and V twins whenever its cuBLASLt W8A8 arm declines, and none of
    // these inputs selects that arm.
    for dispatch in [
        GemmDispatch::defaults(),
        GemmDispatch {
            fp8_blockscaled_prefill: false,
            ..GemmDispatch::defaults()
        },
    ] {
        for w8a8_kernels in [true, false] {
            for attn_prefill_q_t in [true, false] {
                let p = DenseFp8Plan::resolve(DenseFp8Inputs {
                    dispatch,
                    w8a8_kernels,
                    attn_prefill_q_t,
                    ..h100_default()
                });
                assert!(p.attn_fp8_twins.k && p.attn_fp8_twins.v, "{p:?}");
            }
        }
    }
}

#[test]
fn a_non_fp8_layer_keeps_every_nvfp4_copy_and_gets_no_fp8_twins() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        ffn_fp8: false,
        attn_fp8: false,
        ..h100_default()
    });
    assert!(p.ffn_nvfp4, "no FP8 overlay -> NVFP4 is the only weight");
    assert!(p.attn_nvfp4);
    assert_eq!(
        p.attn_fp8_twins,
        Fp8TwinSet::NONE,
        "there are no FP8 weights to transpose on a non-FP8 layer"
    );
}

#[test]
fn the_ffn_and_attention_overlays_are_decided_independently() {
    // 2026-09-25: The loader tests `proj_is_native_fp8` on `mlp.gate_proj` for
    // the FFN and on `q_proj` for attention, so the two overlays can differ.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        attn_fp8: false,
        ..h100_default()
    });
    assert!(!p.ffn_nvfp4);
    assert!(p.attn_nvfp4);

    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        ffn_fp8: false,
        ..h100_default()
    });
    assert!(p.ffn_nvfp4);
    assert!(!p.attn_nvfp4);
}

#[test]
fn single_scale_prefill_brings_the_q_and_o_fp8_twins_back() {
    // 2026-09-25: `METRALE_FP8_SINGLE_SCALE=1` clears `fp8_blockscaled_prefill`,
    // so the W8A8 arms of `paged_qkv.rs` and `paged_oproj.rs` decline and the
    // transposed W8A16 arms read the Q and O twins.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        dispatch: GemmDispatch {
            fp8_blockscaled_prefill: false,
            ..GemmDispatch::defaults()
        },
        ..h100_default()
    });
    assert_eq!(p.attn_fp8_twins, Fp8TwinSet::ALL);
    assert!(
        !p.ffn_nvfp4,
        "the dense FFN still never reads NVFP4: `w8_gemm!` binds its \
         transposed operand to a literal None on every rung"
    );
}

#[test]
fn a_target_without_the_w8a8_kernels_keeps_every_fp8_twin() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        w8a8_kernels: false,
        ..h100_default()
    });
    assert_eq!(p.attn_fp8_twins, Fp8TwinSet::ALL);
}

#[test]
fn the_q_transpose_lever_keeps_only_the_q_twin() {
    // 2026-09-25: With `METRALE_ATTN_PREFILL_Q_T=1`, `cache_skip_qkv.rs` reads
    // the Q twin on the first-chunk chain.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        attn_prefill_q_t: true,
        ..h100_default()
    });
    assert_eq!(
        p.attn_fp8_twins,
        Fp8TwinSet {
            q: true,
            k: true,
            v: true,
            o: false,
        }
    );
}

#[test]
fn every_cutlass_nvfp4_attention_lever_keeps_the_nvfp4_attention_copies() {
    for set in [
        |d: &mut GemmDispatch| d.cutlass_nvfp4_gemm = true,
        |d: &mut GemmDispatch| d.cutlass_nvfp4_attn_q = true,
        |d: &mut GemmDispatch| d.cutlass_nvfp4_attn_kv = true,
        |d: &mut GemmDispatch| d.cutlass_nvfp4_attn_o = true,
    ] {
        let mut dispatch = GemmDispatch::defaults();
        set(&mut dispatch);
        let p = DenseFp8Plan::resolve(DenseFp8Inputs {
            dispatch,
            ..h100_default()
        });
        assert!(
            p.attn_nvfp4,
            "a CUTLASS NVFP4 attention lever reads the transposed NVFP4 twin: {dispatch:?}"
        );
    }
}

#[test]
fn the_umbrella_nvfp4_flag_builds_no_fp8_twins() {
    // 2026-09-25: `transpose_fp8_for_prefill_selected` builds nothing under
    // this flag, and the plan agrees.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        dispatch: GemmDispatch {
            cutlass_nvfp4_gemm: true,
            ..GemmDispatch::defaults()
        },
        ..h100_default()
    });
    assert_eq!(p.attn_fp8_twins, Fp8TwinSet::NONE);
    assert!(p.attn_nvfp4);
}

#[test]
fn attn_w4a4_keeps_the_nvfp4_o_proj() {
    // 2026-09-25: The W4A4 arm of `prefill/paged_oproj.rs` checks no weight
    // type and reads `self.attn.o_proj`, the NVFP4 o_proj.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        attn_w4a4: true,
        ..h100_default()
    });
    assert!(p.attn_nvfp4);
}

#[test]
fn an_ssm_only_lever_does_not_resurrect_the_attention_copies() {
    // 2026-09-25: `resolve` does not read `cutlass_nvfp4_ssm_out`.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        dispatch: GemmDispatch {
            cutlass_nvfp4_ssm_out: true,
            ..GemmDispatch::defaults()
        },
        ..h100_default()
    });
    assert!(!p.attn_nvfp4);
}

#[test]
fn attn_nvfp4_does_not_depend_on_the_w8a8_kernels() {
    // 2026-09-25: `RouteEnv::attn_nvfp4` runs before the layer exists and
    // passes `w8a8_kernels = true`, which is sound only while this holds.
    for w8a8_kernels in [true, false] {
        for attn_fp8 in [true, false] {
            let a = DenseFp8Plan::resolve(DenseFp8Inputs {
                attn_fp8,
                w8a8_kernels,
                ..h100_default()
            });
            let b = DenseFp8Plan::resolve(DenseFp8Inputs {
                attn_fp8,
                w8a8_kernels: !w8a8_kernels,
                ..h100_default()
            });
            assert_eq!(a.attn_nvfp4, b.attn_nvfp4);
        }
    }
}

#[test]
fn the_escape_hatch_restores_the_pre_915_loader() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        keep_nvfp4: true,
        ..h100_default()
    });
    assert_eq!(
        p,
        DenseFp8Plan {
            ffn_nvfp4: true,
            attn_nvfp4: true,
            attn_fp8_twins: Fp8TwinSet::ALL,
        },
        "METRALE_DENSE_FP8_KEEP_NVFP4 must reproduce the old footprint exactly, \
         or it is useless for bisecting a suspected gap in this table"
    );
}

#[test]
fn the_escape_hatch_cannot_invent_fp8_twins_on_a_non_fp8_layer() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        keep_nvfp4: true,
        attn_fp8: false,
        ..h100_default()
    });
    assert_eq!(
        p.attn_fp8_twins,
        Fp8TwinSet::NONE,
        "there is no FP8 weight to transpose"
    );
}

/// 2026-09-25: The byte functions at hidden 5120 and intermediate 17408
/// (`hidden_dim` and `intermediate_size` in `kernels/gb10/qwen3.8-27b/MODEL.toml`),
/// with the attention widths passed below.
#[test]
fn the_h100_ledger_rows_reproduce_from_the_model_shapes() {
    const H: usize = 5120;
    const INTER: usize = 17408;
    const MIB: usize = 1024 * 1024;

    let ffn = dense_ffn_nvfp4_bytes(H, INTER);
    assert_eq!(ffn / MIB, 286, "127.5 MiB of NVFP4 per layer, twice over");
    assert_eq!(
        64 * ffn / MIB,
        2 * (8160 + 1020),
        "8,160 MB of the 8,960 packed row and 1,020 of the 1,120 scale row"
    );

    let attn = attn_nvfp4_bytes(10240, 2560, 5120, H);
    let paired = 2 * (nvfp4_bytes(10240, H) + 2 * nvfp4_bytes(2560, H) + nvfp4_bytes(H, 5120));
    assert_eq!(16 * paired / MIB, 2 * (800 + 100));
    assert!(attn > paired, "the fused twin is extra");

    let twins = attn_fp8_twin_bytes(Fp8TwinSet::ALL, 10240, 2560, 5120, H);
    assert_eq!(16 * (twins / MIB), 1600, "100 MiB of FP8 twins per layer");

    let declined =
        64 * ffn + 16 * attn + 16 * (twins - attn_fp8_twin_bytes(KV_ONLY, 10240, 2560, 5120, H));
    assert!(
        (22.5..23.5).contains(&(declined as f64 / 1e9)),
        "expected ~23.1 GB of the sweep's 28.01 GB not to be built, got {} GB",
        declined as f64 / 1e9
    );
}

#[test]
fn the_summary_line_names_the_twins_it_built() {
    let mut r = DerivedResidency::default();
    r.keep(3_840 * 1024 * 1024);
    r.skip(20_000 * 1024 * 1024);
    r.free(1_024 * 1024 * 1024);
    r.twins.ssm_fp8_concat = true;
    let line = r.summary(28_747 * 1024 * 1024);
    assert!(
        line.starts_with("native FP8 dense residency: weights "),
        "{line}"
    );
    assert!(line.contains("(twins: ssm-qkvz-fp8)"), "{line}");
    assert!(line.contains("not built "), "{line}");
}

#[test]
fn no_twins_reads_as_none_not_as_an_empty_list() {
    assert_eq!(TwinsBuilt::default().describe(), "none");
    assert_eq!(
        TwinsBuilt {
            ffn_nvfp4: true,
            attn_fp8: true,
            ..TwinsBuilt::default()
        }
        .describe(),
        "ffn-nvfp4+t, attn-fp8-t"
    );
}
