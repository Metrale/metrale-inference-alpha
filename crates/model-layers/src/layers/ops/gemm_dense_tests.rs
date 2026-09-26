// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Source-contract tests for the W4A16 GEMM launchers. They read
//! `kernels/**.cu` and `gemm_dense.rs` as text and check the launcher/kernel
//! contract on the CPU; they compile and launch nothing.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! CI builds with `METRALE_SKIP_BUILD=1` (`.github/workflows/ci.yml`), which
//! produces no PTX, and `crates/kernels/tests/kernel_arity.rs`, which reads
//! parameter counts from the PTX, is `#[ignore]`d by default. These tests
//! check the same contract from the source.

#[path = "gemm_dense_tests_util.rs"]
mod util;

use util::{cu_files, indexed_exprs, kernel_sig_body, multiplies_by_bare_n};

/// 2026-09-25: Kernels whose signature has a 9th parameter `ldb`: the row
/// stride of the transposed `B_packed[K/2, N]`, which may exceed `N`. The
/// padded lm_head twin (`transpose_concat_for_gemm_padded` in `impl_a1.rs`) is
/// built at `align_up(vocab, 128)`, and its launchers pass that stride.
const LDB_KERNELS: &[&str] = &[
    "w4a16_gemm_t",
    "w4a16_gemm_t_p3",
    "w4a16_gemm_t_m128_bf16_v2",
];

/// 2026-09-25: Every copy of an [`LDB_KERNELS`] kernel declares `ldb`. `known`
/// lists the copies allowed to lack it and is empty, so a new copy without
/// `ldb` fails here.
#[test]
fn w4a16_gemm_t_ldb_drift_is_exactly_the_known_set() {
    let files = cu_files();

    let known: std::collections::BTreeSet<&str> = [
        // 2026-09-25: Empty: every copy declares `ldb`.
    ]
    .into_iter()
    .collect();

    let mut stale = std::collections::BTreeSet::new();
    let mut seen = 0usize;
    for p in &files {
        let src = std::fs::read_to_string(&p.source).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, _)) = kernel_sig_body(&src, name) else {
                continue;
            };
            seen += 1;
            if !sig.contains("ldb") {
                stale.insert(format!("{}::{name}", util::rel(&p.path)));
            }
        }
    }
    assert!(
        seen > 20,
        "only {seen} ldb-family kernels found — tree moved?"
    );
    let stale: std::collections::BTreeSet<&str> = stale.iter().map(String::as_str).collect();

    let newly: Vec<&&str> = stale.difference(&known).collect();
    assert!(
        newly.is_empty(),
        "NEW ldb-family copies without `ldb` — they will stride B by N: {newly:#?}"
    );
    let fixed: Vec<&&str> = known.difference(&stale).collect();
    assert!(
        fixed.is_empty(),
        "these were ported — delete them from the pinned list: {fixed:#?}"
    );
}

/// 2026-09-25: A copy can declare `ldb` and still stride B by `N` in its body,
/// which the test above does not see. This test requires every `B_packed` and
/// `B_scale` index to use `LDB` and none to multiply by a bare `N`.
///
/// Both operands are checked because they are separate allocations with the
/// same row pitch (`transpose_impl` allocates `stride * half_k` and
/// `stride * num_groups`); striding only one by `LDB` would scale the right
/// nibbles by the wrong block.
#[test]
fn ldb_kernels_actually_use_the_parameter() {
    let mut checked = 0usize;
    for p in &cu_files() {
        let src = std::fs::read_to_string(&p.source).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, body)) = kernel_sig_body(&src, name) else {
                continue;
            };
            if !sig.contains("ldb") {
                continue;
            }
            let where_ = format!("{}::{name}", util::rel(&p.path));
            let code = util::strip_line_comments(&body);
            assert!(
                code.contains("const unsigned int LDB = ldb;"),
                "{where_} must derive the body stride LDB from the launcher-supplied ldb"
            );
            for arr in ["B_packed", "B_scale"] {
                let idx = indexed_exprs(&body, arr);
                assert!(
                    !idx.is_empty(),
                    "{where_}: no `{arr}[...]` index found — the extractor or the \
                     kernel changed shape; this guard must be re-read, not deleted"
                );
                for e in &idx {
                    assert!(
                        e.contains("LDB"),
                        "{where_}: `{arr}[{e}]` does not stride by LDB"
                    );
                    assert!(
                        !multiplies_by_bare_n(e),
                        "{where_}: `{arr}[{e}]` still multiplies by N — a padded \
                         twin (ldb > N) will read sheared rows, silently"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 40, "only {checked} operand indexes checked");
}

/// 2026-09-25: The scalar and tile dialects bound B differently.
///
/// Scalar (`*/common/w4a16_gemm.cu`): B loads are single bytes, so the column
/// guard stays `gn < N`. `N` bounds the valid columns; only the stride became
/// `LDB`.
///
/// Tile (the per-model copies): B loads are 16-byte `cp.async` chunks, so the
/// load bound is `LDB`: a chunk that straddles `N` is inside the padded row.
///
/// Alignment: row `r` starts at `r * LDB`, so every row is 16-byte aligned
/// only if `LDB % 16 == 0`; `padded_twin_stride_is_16_byte_aligned` pins that.
#[test]
fn ldb_kernels_keep_their_dialect_specific_bounds() {
    let (mut scalar, mut tile) = (0usize, 0usize);
    // 2026-09-25: Scalar paths whose file the directory holds itself, rather
    // than reaching it through `[sources] use` or `[hardware] inherits`.
    let mut scalar_forked = 0usize;
    for p in &cu_files() {
        let src = std::fs::read_to_string(&p.source).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, body)) = kernel_sig_body(&src, name) else {
                continue;
            };
            if !sig.contains("ldb") {
                continue;
            }
            let where_ = format!("{}::{name}", util::rel(&p.path));
            // 2026-09-25: Classify by path, not by the guard text. Keying the
            // dialect off `gn < N` would let a widened guard reclassify the
            // kernel as a tile copy and pass.
            if util::rel(&p.path).contains("/common/") {
                scalar += 1;
                if p.held {
                    scalar_forked += 1;
                }
                assert!(
                    body.contains("gn < N"),
                    "{where_}: the scalar column guard `gn < N` is gone. N is the \
                     valid-COLUMN bound; only the STRIDE became LDB."
                );
                assert!(
                    !body.contains("gn < LDB"),
                    "{where_}: the scalar column guard was widened from N to LDB. \
                     N is the valid-COLUMN bound; only the STRIDE changed."
                );
            } else {
                tile += 1;
                assert!(
                    body.contains("< LDB"),
                    "{where_}: tile dialect but no load bound by LDB — 16-byte \
                     cp.async chunks past N are inside the padded row and must \
                     still be fetched"
                );
            }
        }
    }
    // 2026-09-25: Six scalar paths, two files. hopper and b200 inherit gb10
    // (`HARDWARE.toml`), and strix and b300 list `../gb10/common/w4a16_gemm.cu`
    // in their `KERNEL.toml`, so those four compile gb10's file; strix-hip holds
    // its own. A new sharing target raises only the path count, while a new
    // forked copy is a file that needs the `ldb` change by hand, so both counts
    // are asserted.
    assert_eq!(
        scalar, 6,
        "the shared `common/` scalar paths moved (gb10, strix, strix-hip, \
         hopper, b200, b300) — update this count with the tree"
    );
    assert_eq!(
        scalar_forked, 2,
        "expected exactly 2 FORKED `common/` scalar copies (gb10 and \
         strix-hip); a 3rd is a new backend that needs the same hand port, \
         where a sharing one would have inherited it"
    );
    assert!(tile > 20, "only {tile} tile copies found — tree moved?");
}

/// 2026-09-25: The `_ldb` launchers pass 9 arguments, and every
/// [`LDB_KERNELS`] copy declares 9 parameters.
#[test]
fn ldb_launcher_arg_count_matches_kernel_param_count() {
    let launchers = include_str!("gemm_dense.rs");
    for fname in ["w4a16_gemm_n128_ldb", "w4a16_gemm_n128_m128_bf16_ldb"] {
        let args = util::launcher_arg_count(launchers, fname)
            .unwrap_or_else(|| panic!("launcher `{fname}` not found in gemm_dense.rs"));
        assert_eq!(
            args, 9,
            "`{fname}` packs {args} kernel args; the ldb family compiles 9 params"
        );
    }

    let mut checked = 0usize;
    for p in &cu_files() {
        let src = std::fs::read_to_string(&p.source).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, _)) = kernel_sig_body(&src, name) else {
                continue;
            };
            let n = util::param_count(&sig);
            assert_eq!(
                n,
                9,
                "{}::{name} compiles {n} params; the launcher packs 9",
                util::rel(&p.path)
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} kernels checked");
}

/// 2026-09-25: `w4a16_gemm_n128` forwards `n` as the stride: in the packed
/// case the transposed rows are exactly N apart.
#[test]
fn non_ldb_wrapper_forwards_n_as_the_stride() {
    let src = include_str!("gemm_dense.rs");
    let body = util::fn_body(src, "w4a16_gemm_n128")
        .expect("`w4a16_gemm_n128` wrapper not found in gemm_dense.rs");
    let call = body
        .lines()
        .find(|l| l.contains("w4a16_gemm_n128_ldb("))
        .unwrap_or_else(|| panic!("wrapper no longer delegates to the _ldb launcher:\n{body}"));
    let args: Vec<String> = util::call_args(call)
        .into_iter()
        .map(|s| s.trim().to_string())
        .collect();
    assert_eq!(
        args.len(),
        10,
        "unexpected delegation shape `{call}` (9 kernel args + stream)"
    );
    assert_eq!(
        args[8], "n",
        "the wrapper forwards `{}` as ldb, not `n` — the packed case is DEFINED \
         by rows being exactly N apart",
        args[8]
    );
}

/// 2026-09-26: The tile kernels' 16-byte `cp.async` B loads need a 16-byte
/// aligned row pitch. The align argument at the one call site that builds the
/// padded twin (`impl_a1/spec_buffers.rs`) sets it. All five files of the model
/// build are scanned, so a second site in any of them is caught.
#[test]
fn padded_twin_stride_is_16_byte_aligned() {
    let src = [
        include_str!("../../../../model-engine/src/model/impl_a1.rs"),
        include_str!("../../../../model-engine/src/model/impl_a1/spec_buffers.rs"),
        include_str!("../../../../model-engine/src/model/impl_a1/comm_setup.rs"),
        include_str!("../../../../model-engine/src/model/impl_a1/kernels.rs"),
        include_str!("../../../../model-engine/src/model/impl_a1/ssm_setup.rs"),
    ]
    .concat();
    let mut sites = 0usize;
    let mut from = 0usize;
    while let Some(rel) = src[from..].find("transpose_concat_for_gemm_padded(") {
        let at = from + rel;
        let args = util::call_args(&src[at..]);
        from = at + 1;
        sites += 1;
        let align = args
            .last()
            .map(|s| s.trim().trim_end_matches(',').to_string())
            .unwrap_or_default();
        let n: usize = align.parse().unwrap_or_else(|_| {
            panic!(
                "`transpose_concat_for_gemm_padded` align is `{align}`, not a literal. \
                 A non-literal align cannot be checked here and the 16-byte cp.async \
                 invariant it carries is load-bearing — extend this guard, do not delete it."
            )
        });
        assert!(
            n.is_multiple_of(16),
            "padded twin align={n} is not a multiple of 16: rows land at r*align, \
             so the tile kernels' 16-byte cp.async B loads will fault (CUDA 716)"
        );
    }
    assert_eq!(
        sites, 1,
        "expected exactly one padded-twin construction site in impl_a1.rs + impl_a1/spec_buffers.rs"
    );
}
