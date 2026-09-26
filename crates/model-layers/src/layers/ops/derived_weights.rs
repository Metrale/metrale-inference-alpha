// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-model memo of GPU re-encodings of a model's weights, keyed
//! by the source weight's device pointer.
//!
//! The derivations are: FP8 dequantized to BF16, block-scaled FP8
//! re-quantized to row-wise FP8, NVFP4 bytes transposed for CUTLASS, and FP8
//! packed to CUTLASS NVFP4. The key is a raw device pointer, which another
//! model's allocator can hand out again; because each model owns its memo, a
//! reused address cannot hit an entry made for a different model's weight.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Each `Derivation` has its own map, so a key never returns another kind's
//!   value.
//! - A key keeps the first value stored for it; a later insert is dropped.
//! - `release` empties every map, and frees only the values, never the keys.

use std::collections::HashMap;
// 2026-09-25: parking_lot's mutex does not poison, so a panic elsewhere cannot
// block teardown.
use parking_lot::Mutex;

/// 2026-09-25: Per-model memo of derived weight encodings, one map per
/// [`Derivation`].
#[derive(Default)]
pub struct DerivedWeights {
    /// 2026-09-25: FP8 weight ptr → `(row-wise FP8 ptr, per-row scale ptr)`.
    rowwise_fp8: Mutex<HashMap<u64, (u64, u64)>>,
    /// 2026-09-25: FP8 weight ptr → BF16 ptr.
    bf16: Mutex<HashMap<u64, u64>>,
    /// 2026-09-25: NVFP4 weight ptr → `[N,K/2]` transposed bytes ptr.
    cutlass_nvfp4_t: Mutex<HashMap<u64, u64>>,
    /// 2026-09-25: FP8 weight ptr → `(CUTLASS NVFP4 ptr, scale ptr)`.
    cutlass_nvfp4_from_fp8: Mutex<HashMap<u64, (u64, u64)>>,
}

/// 2026-09-25: Which derivation a lookup is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivation {
    RowwiseFp8,
    Bf16,
    CutlassNvfp4Transposed,
    CutlassNvfp4FromFp8,
}

impl DerivedWeights {
    pub fn new() -> Self {
        Self::default()
    }

    /// 2026-09-25: Memo read for a single-pointer derivation, for callers that
    /// allocate and launch between the lookup and [`Self::insert_ptr`]. A
    /// pair-valued `kind` returns `None`.
    pub fn get_ptr(&self, kind: Derivation, key: u64) -> Option<u64> {
        let map = match kind {
            Derivation::Bf16 => &self.bf16,
            Derivation::CutlassNvfp4Transposed => &self.cutlass_nvfp4_t,
            _ => return None,
        };
        map.lock().get(&key).copied()
    }

    pub fn insert_ptr(&self, kind: Derivation, key: u64, value: u64) {
        let map = match kind {
            Derivation::Bf16 => &self.bf16,
            Derivation::CutlassNvfp4Transposed => &self.cutlass_nvfp4_t,
            _ => return,
        };
        map.lock().entry(key).or_insert(value);
    }

    pub fn get_pair(&self, kind: Derivation, key: u64) -> Option<(u64, u64)> {
        let map = match kind {
            Derivation::RowwiseFp8 => &self.rowwise_fp8,
            Derivation::CutlassNvfp4FromFp8 => &self.cutlass_nvfp4_from_fp8,
            _ => return None,
        };
        map.lock().get(&key).copied()
    }

    pub fn insert_pair(&self, kind: Derivation, key: u64, value: (u64, u64)) {
        let map = match kind {
            Derivation::RowwiseFp8 => &self.rowwise_fp8,
            Derivation::CutlassNvfp4FromFp8 => &self.cutlass_nvfp4_from_fp8,
            _ => return,
        };
        map.lock().entry(key).or_insert(value);
    }

    /// 2026-09-25: Look up a single-pointer derivation, building it on a miss.
    /// A failed build stores nothing.
    ///
    /// `build` runs outside the lock. If two threads race on one key, both
    /// build, the first stored value is kept and returned to both, and the
    /// loser's value is not freed.
    pub fn get_or_build_ptr(
        &self,
        kind: Derivation,
        key: u64,
        build: impl FnOnce() -> anyhow::Result<u64>,
    ) -> anyhow::Result<u64> {
        let map = match kind {
            Derivation::Bf16 => &self.bf16,
            Derivation::CutlassNvfp4Transposed => &self.cutlass_nvfp4_t,
            _ => unreachable!("pair-valued derivation routed to get_or_build_ptr"),
        };
        if let Some(&hit) = map.lock().get(&key) {
            return Ok(hit);
        }
        let built = build()?;
        Ok(*map.lock().entry(key).or_insert(built))
    }

    /// 2026-09-25: [`Self::get_or_build_ptr`] for a pair-valued derivation.
    pub fn get_or_build_pair(
        &self,
        kind: Derivation,
        key: u64,
        build: impl FnOnce() -> anyhow::Result<(u64, u64)>,
    ) -> anyhow::Result<(u64, u64)> {
        let map = match kind {
            Derivation::RowwiseFp8 => &self.rowwise_fp8,
            Derivation::CutlassNvfp4FromFp8 => &self.cutlass_nvfp4_from_fp8,
            _ => unreachable!("single-valued derivation routed to get_or_build_pair"),
        };
        if let Some(&hit) = map.lock().get(&key) {
            return Ok(hit);
        }
        let built = build()?;
        Ok(*map.lock().entry(key).or_insert(built))
    }

    /// 2026-09-25: Total memoized entries across the four maps.
    pub fn len(&self) -> usize {
        self.rowwise_fp8.lock().len()
            + self.bf16.lock().len()
            + self.cutlass_nvfp4_t.lock().len()
            + self.cutlass_nvfp4_from_fp8.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for DerivedWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DerivedWeights")
            .field("entries", &self.len())
            .finish()
    }
}

/// 2026-09-25: Release every derived allocation.
///
/// The keys are the source weights' pointers, which the weight store owns and
/// frees; only the values are freed here. Every free is attempted, and the
/// first error is returned.
impl metrale_core::scope::ModelResource<dyn metrale_gpu_runtime::gpu::GpuBackend>
    for DerivedWeights
{
    fn label(&self) -> &'static str {
        "derived weights"
    }

    fn release(&mut self, gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend) -> anyhow::Result<()> {
        let mut owned: Vec<u64> = Vec::new();
        // 2026-09-25: Drain, so a later lookup cannot return a freed pointer.
        for (_, (a, b)) in self.rowwise_fp8.lock().drain() {
            owned.push(a);
            owned.push(b);
        }
        owned.extend(self.bf16.lock().drain().map(|(_, v)| v));
        owned.extend(self.cutlass_nvfp4_t.lock().drain().map(|(_, v)| v));
        for (_, (a, b)) in self.cutlass_nvfp4_from_fp8.lock().drain() {
            owned.push(a);
            owned.push(b);
        }
        let mut first_error = None;
        for raw in owned {
            if let Err(e) = gpu.free(metrale_gpu_runtime::gpu::DevicePtr(raw))
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_core::scope::ModelResource;
    use metrale_gpu_runtime::gpu::GpuBackend;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_fresh_cache_is_empty() {
        assert!(DerivedWeights::new().is_empty());
    }

    #[test]
    fn a_derivation_is_built_once_per_key() {
        let d = DerivedWeights::new();
        let builds = AtomicUsize::new(0);
        let build_for = |ptr: u64| {
            d.get_or_build_ptr(Derivation::Bf16, ptr, || {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok(ptr + 1000)
            })
            .unwrap()
        };
        assert_eq!(build_for(10), 1010);
        assert_eq!(build_for(10), 1010);
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(build_for(20), 1020);
        assert_eq!(builds.load(Ordering::Relaxed), 2);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn the_derivations_do_not_share_a_keyspace() {
        // 2026-09-25: One source pointer, two encodings, two separate entries.
        let d = DerivedWeights::new();
        let bf16 = d
            .get_or_build_ptr(Derivation::Bf16, 0x1000, || Ok(0xB16))
            .unwrap();
        let nvfp4 = d
            .get_or_build_ptr(Derivation::CutlassNvfp4Transposed, 0x1000, || Ok(0x4444))
            .unwrap();
        assert_eq!(bf16, 0xB16);
        assert_eq!(nvfp4, 0x4444);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn a_failed_build_is_not_memoized() {
        let d = DerivedWeights::new();
        assert!(
            d.get_or_build_ptr(Derivation::Bf16, 7, || anyhow::bail!("oom"))
                .is_err()
        );
        assert!(d.is_empty(), "a failure must not poison the key");
        assert_eq!(
            d.get_or_build_ptr(Derivation::Bf16, 7, || Ok(99)).unwrap(),
            99
        );
    }

    #[test]
    fn two_models_memoize_independently_even_on_a_recycled_pointer() {
        // 2026-09-25: Model A memoizes a derivation for an address, A is
        // dropped, and model B sees the same address for a different weight.
        let a = DerivedWeights::new();
        let recycled = 0x7f00_0000u64;
        assert_eq!(
            a.get_or_build_ptr(Derivation::Bf16, recycled, || Ok(0xAAAA))
                .unwrap(),
            0xAAAA
        );
        drop(a);

        let b = DerivedWeights::new();
        assert_eq!(
            b.get_or_build_ptr(Derivation::Bf16, recycled, || Ok(0xBBBB))
                .unwrap(),
            0xBBBB,
            "the same address must resolve to the NEW model's derivation"
        );
    }

    #[test]
    fn pair_derivations_round_trip_without_sharing_a_keyspace() {
        let d = DerivedWeights::new();
        let got = d
            .get_or_build_pair(Derivation::RowwiseFp8, 5, || Ok((11, 22)))
            .unwrap();
        assert_eq!(got, (11, 22));
        assert_eq!(
            d.get_or_build_pair(Derivation::CutlassNvfp4FromFp8, 5, || Ok((33, 44)))
                .unwrap(),
            (33, 44)
        );
        assert_eq!(
            d.get_or_build_pair(Derivation::RowwiseFp8, 5, || Ok((99, 99)))
                .unwrap(),
            (11, 22),
            "cached, not rebuilt"
        );
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn release_frees_only_derived_values_and_drains_every_map() {
        let gpu = MockGpuBackend::new();
        let mut d = DerivedWeights::new();
        let keys: Vec<u64> = (0..4).map(|_| gpu.alloc(1).unwrap().0).collect();
        let values: Vec<u64> = (0..6).map(|_| gpu.alloc(1).unwrap().0).collect();

        d.insert_pair(Derivation::RowwiseFp8, keys[0], (values[0], values[1]));
        d.insert_ptr(Derivation::Bf16, keys[1], values[2]);
        d.insert_ptr(Derivation::CutlassNvfp4Transposed, keys[2], values[3]);
        d.insert_pair(
            Derivation::CutlassNvfp4FromFp8,
            keys[3],
            (values[4], values[5]),
        );
        assert_eq!(gpu.alloc_count(), 10);
        assert_eq!(d.len(), 4);

        d.release(&gpu).unwrap();

        assert!(d.is_empty(), "freed pointers must not remain memoized");
        assert_eq!(gpu.alloc_count(), 4, "source-weight keys remain GPU-owned");
        for key in keys {
            gpu.free(metrale_gpu_runtime::gpu::DevicePtr(key)).unwrap();
        }
    }
}
