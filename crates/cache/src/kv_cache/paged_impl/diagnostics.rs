// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `PagedKvCache` BF16 fingerprint diagnostics: per-layer region
//! checksums and per-block sums of squares, read back from the device.
//!
//! Owner: cache.
//! Invariants: diagnostic only; nothing here writes the cache. Events keep
//! the parent module's log target.

use super::PagedKvCache;

impl PagedKvCache {
    /// 2026-09-25: Sum, sum of squares and sum of absolute values of a
    /// little-endian BF16 buffer, for the two fingerprint diagnostics below.
    /// A BF16 value is the top 16 bits of an f32.
    fn bf16_reductions(buf: &[u8]) -> (f64, f64, f64) {
        let (mut sum, mut ssq, mut sabs) = (0f64, 0f64, 0f64);
        for c in buf.chunks_exact(2) {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let v = f32::from_bits((bits as u32) << 16) as f64;
            sum += v;
            ssq += v * v;
            sabs += v.abs();
        }
        (sum, ssq, sabs)
    }

    /// 2026-09-25: Diagnostic. Synchronises `stream`, then logs a K and V
    /// (sum, ssq, sabs) line per BF16 layer for each of two regions of
    /// `blocks`: `[0, boundary_idx)` ("prefix") and the rest ("suffix").
    /// Per-layer lines keep a local divergence from cancelling in a global
    /// sum. Non-BF16 layers are skipped, with a warning when layer 0 is one;
    /// a block whose copy fails is skipped. The model engine calls it only
    /// under `METRALE_SSM_SAVE_DUMP`.
    pub fn debug_kv_checksum_per_layer(
        &self,
        blocks: &[u32],
        boundary_idx: usize,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        stream: u64,
        tag: &str,
    ) {
        gpu.synchronize(stream).ok();
        let boundary = boundary_idx.min(blocks.len());
        let regions: [(&str, &[u32]); 2] = [
            ("prefix", &blocks[..boundary]),
            ("suffix", &blocks[boundary..]),
        ];
        for (li, layer) in self.layers.iter().enumerate() {
            if layer.dtype != super::KvCacheDtype::Bf16 {
                if li == 0 {
                    tracing::warn!(
                        target: "metrale_cache::kv_cache::paged_impl",
                        "METRALE_KV_CKSUM[{tag}] layer 0 dtype={:?} != bf16 — probe \
                         only decodes BF16; skipping",
                        layer.dtype
                    );
                }
                continue;
            }
            // 2026-09-25: A BF16 layer is symmetric, so the K stride is also
            // the V stride.
            let nbytes = layer.k_block_stride;
            for (rname, rblocks) in &regions {
                let (mut k_sum, mut k_ssq, mut k_sabs) = (0f64, 0f64, 0f64);
                let (mut v_sum, mut v_ssq, mut v_sabs) = (0f64, 0f64, 0f64);
                for &blk in *rblocks {
                    let mut kb = vec![0u8; nbytes];
                    let mut vb = vec![0u8; nbytes];
                    if gpu.copy_d2h(self.k_cache_ptr(li, blk), &mut kb).is_err()
                        || gpu.copy_d2h(self.v_cache_ptr(li, blk), &mut vb).is_err()
                    {
                        continue;
                    }
                    let (ks, kq, ka) = Self::bf16_reductions(&kb);
                    let (vs, vq, va) = Self::bf16_reductions(&vb);
                    k_sum += ks;
                    k_ssq += kq;
                    k_sabs += ka;
                    v_sum += vs;
                    v_ssq += vq;
                    v_sabs += va;
                }
                tracing::warn!(
                    target: "metrale_cache::kv_cache::paged_impl",
                    "METRALE_KV_CKSUM[{tag}] L{li} {rname} nblk={} \
                     k_sum={k_sum:.4} k_ssq={k_ssq:.4} k_sabs={k_sabs:.4} \
                     v_sum={v_sum:.4} v_ssq={v_ssq:.4} v_sabs={v_sabs:.4}",
                    rblocks.len(),
                );
            }
        }
    }

    /// 2026-09-25: Diagnostic. Synchronises `stream`, then logs one line per
    /// entry of `blocks`, in order: logical index, physical block, K and V sum
    /// of squares for layer `layer_idx`. Unlike a region sum this shows a
    /// wrong block-to-position mapping. Does nothing for a non-BF16 layer.
    pub fn debug_kv_per_block(
        &self,
        layer_idx: usize,
        blocks: &[u32],
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        stream: u64,
        tag: &str,
    ) {
        gpu.synchronize(stream).ok();
        let layer = &self.layers[layer_idx];
        if layer.dtype != super::KvCacheDtype::Bf16 {
            return;
        }
        // 2026-09-25: A BF16 layer is symmetric, so the K stride is also the V
        // stride.
        let nbytes = layer.k_block_stride;
        for (li, &blk) in blocks.iter().enumerate() {
            let mut kb = vec![0u8; nbytes];
            let mut vb = vec![0u8; nbytes];
            if gpu
                .copy_d2h(self.k_cache_ptr(layer_idx, blk), &mut kb)
                .is_err()
                || gpu
                    .copy_d2h(self.v_cache_ptr(layer_idx, blk), &mut vb)
                    .is_err()
            {
                continue;
            }
            let (_, k_ssq, _) = Self::bf16_reductions(&kb);
            let (_, v_ssq, _) = Self::bf16_reductions(&vb);
            tracing::warn!(
                target: "metrale_cache::kv_cache::paged_impl",
                "METRALE_KVBLK[{tag}] L{layer_idx} logical={li} phys={blk} \
                 k_ssq={k_ssq:.4} v_ssq={v_ssq:.4}"
            );
        }
    }
}
