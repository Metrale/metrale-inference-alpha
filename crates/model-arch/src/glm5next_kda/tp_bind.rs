// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Applies a `super::tp::KdaTpPlan`: slices each KDA tensor to this rank before the
//! binder sees it.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - [`KdaShardedSource`] presents each tensor whose name maps to a plan entry (`plan_name`) as
//!   this rank's bytes with a local shape, and passes every other name through unchanged.
//! - [`shard_bytes`] refuses a tensor whose size disagrees with the plan's full shape.
//!
//! The slicing happens upstream of `super::binding::bind_kda_weights`, which validates the local
//! shapes against the local config, so TP=1 and TP=2 run the same binder code.
//!
//! Row slicing (`HeadRows`, `ChannelRows`) is one contiguous byte range. Column slicing
//! (`ChannelCols`, only `o_proj`) is a strided gather: every row keeps its own middle slice. A
//! row slice of `o_proj` has the same byte count at `tp_size = 2` and the wrong contents, which is
//! why the tests compare values.

use anyhow::{Result, bail};

use super::binding::{KdaTensorSource, RawTensor};
use super::tp::{KdaShard, KdaTensorPlan, KdaTpPlan};

/// 2026-09-25: Plan name for a checkpoint tensor name: `self_attn.q_proj.weight` -> `q_proj`.
/// `A_log` and `dt_bias` have no `.weight` suffix, so the suffix is stripped only when present.
fn plan_name(checkpoint_name: &str) -> &str {
    let n = checkpoint_name
        .strip_prefix("self_attn.")
        .unwrap_or(checkpoint_name);
    n.strip_suffix(".weight").unwrap_or(n)
}

/// 2026-09-25: This rank's bytes for one tensor, given its plan entry. `full` is the whole on-disk
/// tensor; a size mismatch is an error. Returns a new buffer because a column slice is not
/// contiguous.
pub fn shard_bytes(plan: &KdaTensorPlan, full: &[u8]) -> Result<Vec<u8>> {
    let e = plan.elem_bytes;
    let expect = plan.full_rows * plan.full_row_elems * e;
    if full.len() != expect {
        bail!(
            "{}: {} B on disk, the plan's full shape [{}, {}] x {e} B implies {expect} B",
            plan.name,
            full.len(),
            plan.full_rows,
            plan.full_row_elems
        );
    }
    Ok(match plan.kind {
        KdaShard::Replicated => full.to_vec(),
        // 2026-09-25: Contiguous: this rank's rows are adjacent on disk.
        KdaShard::HeadRows | KdaShard::ChannelRows => {
            let row = plan.full_row_elems * e;
            let start = plan.src_row_offset * row;
            full[start..start + plan.local_rows * row].to_vec()
        }
        // 2026-09-25: Strided: every row keeps its `[src_col_offset, +local_row_elems)` slice.
        KdaShard::ChannelCols => {
            let row = plan.full_row_elems * e;
            let lo = plan.src_col_offset * e;
            let width = plan.local_row_elems * e;
            let mut out = Vec::with_capacity(plan.local_rows * width);
            for r in 0..plan.local_rows {
                let base = r * row + lo;
                out.extend_from_slice(&full[base..base + width]);
            }
            out
        }
    })
}

/// 2026-09-25: The local shape the binder should see, keeping the on-disk rank rather than the
/// plan's flat `[rows, row_elems]` view: the binder validates the conv tensors as rank 3.
fn local_shape(plan: &KdaTensorPlan, disk_shape: &[usize]) -> Vec<usize> {
    let mut s = disk_shape.to_vec();
    match plan.kind {
        KdaShard::Replicated => {}
        KdaShard::HeadRows | KdaShard::ChannelRows => {
            if let Some(first) = s.first_mut() {
                *first = plan.local_rows;
            }
        }
        KdaShard::ChannelCols => {
            if let Some(last) = s.last_mut() {
                *last = plan.local_row_elems;
            }
        }
    }
    s
}

/// 2026-09-25: A [`KdaTensorSource`] that yields this rank's slice of every planned KDA tensor.
/// The loader wraps each layer's source in it and passes it to `bind_kda_weights`.
pub struct KdaShardedSource<'a> {
    inner: &'a dyn KdaTensorSource,
    plan: &'a KdaTpPlan,
    /// 2026-09-25: Sliced bytes, built in `new`: `KdaTensorSource::get` returns a borrow, so the
    /// buffers must outlive the call.
    sliced: std::collections::BTreeMap<String, (super::binding::KdaDtype, Vec<usize>, Vec<u8>)>,
}

impl<'a> KdaShardedSource<'a> {
    pub fn new(inner: &'a dyn KdaTensorSource, plan: &'a KdaTpPlan) -> Result<Self> {
        let mut sliced = std::collections::BTreeMap::new();
        for name in inner.names() {
            // 2026-09-25: Names without a plan entry are not sliced; `get` passes them through.
            let Some(p) = plan.get(plan_name(&name)) else {
                continue;
            };
            let Some(t) = inner.get(&name) else { continue };
            sliced.insert(
                name.clone(),
                (t.dtype, local_shape(p, &t.shape), shard_bytes(p, t.bytes)?),
            );
        }
        Ok(Self {
            inner,
            plan,
            sliced,
        })
    }

    pub fn plan(&self) -> &KdaTpPlan {
        self.plan
    }
}

impl KdaTensorSource for KdaShardedSource<'_> {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        match self.sliced.get(name) {
            Some((dtype, shape, bytes)) => Some(RawTensor {
                dtype: *dtype,
                shape: shape.clone(),
                bytes,
            }),
            // 2026-09-25: Not a planned KDA tensor: pass it through, so the binder's refusal of
            // an unrecognised `self_attn` tensor still fires.
            None => self.inner.get(name),
        }
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }
}

#[cfg(test)]
mod tests {
    use super::super::binding::KdaDtype;
    use super::*;

    /// 2026-09-25: The `glm-5.3-flash` KDA geometry at TP=2 (hidden 4096, 64 heads of 128, conv
    /// kernel 4, gate rank 128).
    fn plan(rank: usize) -> KdaTpPlan {
        KdaTpPlan::new(rank, 2, 4096, 128, 64, 4, 128).unwrap()
    }

    #[test]
    fn plan_names_strip_the_prefix_but_not_a_missing_suffix() {
        assert_eq!(plan_name("self_attn.q_proj.weight"), "q_proj");
        assert_eq!(plan_name("self_attn.q_conv1d.weight"), "q_conv1d");
        assert_eq!(plan_name("self_attn.A_log"), "A_log");
        assert_eq!(plan_name("self_attn.dt_bias"), "dt_bias");
        // 2026-09-25: Every plan entry must be reachable from some checkpoint name.
        for t in &plan(0).tensors {
            let candidates = [
                format!("self_attn.{}.weight", t.name),
                format!("self_attn.{}", t.name),
            ];
            assert!(
                candidates.iter().any(|c| plan_name(c) == t.name),
                "plan entry {} is unreachable from a checkpoint name",
                t.name
            );
        }
    }

    /// 2026-09-25: Row slicing is contiguous; the two ranks partition the tensor exactly.
    #[test]
    fn channel_rows_partition_the_tensor() {
        // 2026-09-25: A small geometry the plan accepts: it requires
        // `2 * local_heads * head_dim % 256 == 0`, which heads = 4, head_dim = 64 at tp = 2 meets.
        let p = KdaTpPlan::new(0, 2, 4, 64, 4, 2, 4).unwrap();
        let q = p.get("q_proj").unwrap().clone();
        assert_eq!(q.kind, KdaShard::ChannelRows);
        // 2026-09-25: `[full_ch = 256, hidden = 4]` BF16, each byte equal to its row index.
        let full: Vec<u8> = (0..=255u8).flat_map(|r| [r; 8]).collect();

        let r0 = shard_bytes(&q, &full).unwrap();
        let mut q1 = q.clone();
        q1.src_row_offset = q.local_rows;
        let r1 = shard_bytes(&q1, &full).unwrap();

        assert_eq!(
            r0.len() + r1.len(),
            full.len(),
            "the two ranks partition it"
        );
        let mut rejoined = r0.clone();
        rejoined.extend_from_slice(&r1);
        assert_eq!(rejoined, full, "and rejoin to the original, in order");
        assert!(r0.iter().all(|b| *b < 128), "rank 0 keeps the low rows");
        assert!(r1.iter().all(|b| *b >= 128), "rank 1 keeps the high rows");
    }

    /// 2026-09-25: `o_proj` is column-sliced. A row slice of the same tensor has the same length
    /// and different contents, so only a value check separates them.
    #[test]
    fn o_proj_is_column_sliced_not_row_sliced() {
        let p = KdaTpPlan::new(1, 2, 4, 64, 4, 2, 4).unwrap();
        let o = p.get("o_proj").unwrap();
        assert_eq!(o.kind, KdaShard::ChannelCols);
        assert_eq!(o.full_rows, 4, "hidden");
        assert_eq!(o.full_row_elems, 256, "heads * head_dim");
        assert_eq!(o.local_rows, 4, "every row is kept");
        assert_eq!(o.local_row_elems, 128, "half the input dim");

        // 2026-09-25: `[4, 256]` BF16; each element's low byte is its column index.
        let full: Vec<u8> = (0..4)
            .flat_map(|_| (0..=255u8).flat_map(|c| [c, 0]))
            .collect();
        let got = shard_bytes(o, &full).unwrap();

        // 2026-09-25: Rank 1 keeps columns 128..256 of every row.
        let want: Vec<u8> = (0..4)
            .flat_map(|_| (128..=255u8).flat_map(|c| [c, 0]))
            .collect();
        assert_eq!(got, want);

        // 2026-09-25: A row slice of the same byte count is rows 2..4: same length, different
        // bytes.
        let row_sliced = &full[full.len() / 2..];
        assert_eq!(row_sliced.len(), got.len());
        assert_ne!(row_sliced, got.as_slice());
    }

    /// 2026-09-25: `o_norm` (`[head_dim]`) and the `_a` down-projections are copied unchanged to
    /// every rank.
    #[test]
    fn o_norm_and_the_low_rank_down_projections_are_replicated() {
        let p = plan(1);
        for n in ["o_norm", "f_a_proj", "g_a_proj"] {
            let t = p.get(n).unwrap();
            assert_eq!(t.kind, KdaShard::Replicated, "{n}");
            let full = vec![0xABu8; t.full_bytes()];
            assert_eq!(shard_bytes(t, &full).unwrap(), full, "{n} must be verbatim");
        }
    }

    /// 2026-09-25: `A_log` is per head (`[64]`) and `dt_bias` per channel (`[8192]`), so rank 1's
    /// offsets differ by a factor of `head_dim`.
    #[test]
    fn a_log_and_dt_bias_shard_at_different_granularity() {
        let p = plan(1);
        let a = p.get("A_log").unwrap();
        let d = p.get("dt_bias").unwrap();
        assert_eq!((a.full_rows, a.local_rows), (64, 32), "A_log is per-head");
        assert_eq!(
            (d.full_rows, d.local_rows),
            (8192, 4096),
            "dt_bias is per-channel"
        );
        assert_eq!(d.src_row_offset, a.src_row_offset * 128);
    }

    /// 2026-09-25: At `tp_size = 1` every tensor's shard is the whole tensor, and no all-reduce is
    /// needed.
    #[test]
    fn tp1_is_the_identity() {
        let p = KdaTpPlan::new(0, 1, 4096, 128, 64, 4, 128).unwrap();
        for t in &p.tensors {
            assert_eq!(t.local_bytes(), t.full_bytes(), "{}", t.name);
            let full = vec![0x5Au8; t.full_bytes()];
            assert_eq!(shard_bytes(t, &full).unwrap(), full, "{}", t.name);
        }
        assert!(!p.needs_output_all_reduce());
    }

    /// 2026-09-25: A tensor whose on-disk size disagrees with the plan is refused, not truncated.
    #[test]
    fn a_size_mismatch_is_refused() {
        let p = plan(0);
        let q = p.get("q_proj").unwrap();
        let err = shard_bytes(q, &vec![0u8; q.full_bytes() - 2]).unwrap_err();
        assert!(err.to_string().contains("on disk"), "{err}");
    }

    /// 2026-09-25: The local shape keeps the on-disk rank (rank 3 for the conv tensors).
    #[test]
    fn local_shape_preserves_the_on_disk_rank() {
        let p = plan(1);
        let c = p.get("q_conv1d").unwrap();
        assert_eq!(local_shape(c, &[8192, 1, 4]), vec![4096, 1, 4]);
        let o = p.get("o_proj").unwrap();
        assert_eq!(local_shape(o, &[4096, 8192]), vec![4096, 4096]);
        let n = p.get("o_norm").unwrap();
        assert_eq!(local_shape(n, &[128]), vec![128]);
    }

    /// 2026-09-25: Through the adapter, a planned tensor is this rank's slice with a local shape,
    /// and an unplanned name passes through unchanged.
    #[test]
    fn the_adapter_presents_local_slices_and_passes_the_rest_through() {
        struct Src(Vec<(String, KdaDtype, Vec<usize>, Vec<u8>)>);
        impl KdaTensorSource for Src {
            fn get(&self, name: &str) -> Option<RawTensor<'_>> {
                self.0
                    .iter()
                    .find(|(n, ..)| n == name)
                    .map(|(_, d, s, b)| RawTensor {
                        dtype: *d,
                        shape: s.clone(),
                        bytes: b,
                    })
            }
            fn names(&self) -> Vec<String> {
                self.0.iter().map(|(n, ..)| n.clone()).collect()
            }
        }
        let p = KdaTpPlan::new(1, 2, 4, 64, 4, 2, 4).unwrap();
        let q = p.get("q_proj").unwrap();
        let src = Src(vec![
            (
                "self_attn.q_proj.weight".into(),
                KdaDtype::Bf16,
                vec![256, 4],
                (0..=255u8).flat_map(|r| [r; 8]).collect(),
            ),
            (
                "input_layernorm.weight".into(),
                KdaDtype::Bf16,
                vec![4],
                vec![0xEE; 8],
            ),
        ]);
        let sh = KdaShardedSource::new(&src, &p).unwrap();

        let t = sh.get("self_attn.q_proj.weight").unwrap();
        assert_eq!(t.shape, vec![128, 4], "local rows, full row width");
        assert_eq!(t.bytes.len(), q.local_bytes());
        assert!(
            t.bytes.iter().all(|b| *b >= 128),
            "rank 1 keeps the high rows"
        );

        let n = sh.get("input_layernorm.weight").unwrap();
        assert_eq!(n.bytes, vec![0xEE; 8]);
        assert_eq!(sh.names().len(), 2, "the census still sees everything");
    }
}
