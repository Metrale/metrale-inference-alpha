// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: High-speed swap for an attention layer: whether it is engaged, and the per-layer
//! offload of new KV blocks to the disk store as BF16.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - When `high_speed_swap_offload_new_blocks` is engaged, finds `disk_block_ids` non-empty and
//!   succeeds, it sets this layer's `disk_last_offloaded_per_layer` entry to
//!   `disk_block_ids.len()`. On an error the entry is left as it was.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_cache::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in super::super) fn high_speed_swap_engaged(
        &self,
        kv_cache: &metrale_cache::kv_cache::PagedKvCache,
    ) -> bool {
        if kv_cache.config().cache_blocks_per_seq.is_none() {
            return false;
        }
        if !metrale_storage::local_installed() {
            return false;
        }
        // 2026-09-25: Engaged only for these dtypes, each of which has a host-side dequant to BF16
        // below. A turbo cache holds WHT(K) and WHT(V); `attention_forward` applies WHT(Q) and
        // iWHT(out) around the orchestrator call.
        matches!(
            kv_cache.dtype_for_layer(self.attn_layer_idx),
            KvCacheDtype::Bf16
                | KvCacheDtype::Fp8
                | KvCacheDtype::Nvfp4
                | KvCacheDtype::Turbo3
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo8,
        )
    }

    /// 2026-09-25: Offload this layer's K/V to disk for every block in `disk_block_ids` it has not
    /// offloaded yet, plus the block before them, which may have gained slots since. A no-op when
    /// high-speed swap is not engaged. Called from decode (`attention_forward`) and prefill
    /// (`prefill_inner.rs`) after their K/V writes.
    pub(in super::super) fn high_speed_swap_offload_new_blocks(
        &self,
        kv_cache: &mut PagedKvCache,
        block_table: &Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        nkv: u32,
        hd: u32,
        bs: usize,
    ) -> Result<()> {
        if !self.high_speed_swap_engaged(kv_cache) {
            return Ok(());
        }
        let layer_u32 = self.attn_layer_idx as u32;
        let block_floats = bs * (nkv as usize) * (hd as usize);

        // 2026-09-25: The caller's block allocation (`ensure_blocks_through_decode` / `_prefill` in
        // model-engine `block_mgmt.rs`) grows `disk_block_ids`; this only checks it in debug
        // builds.
        debug_assert!(
            disk_block_ids.len() >= block_table.len(),
            "Phase 6.3 invariant: alloc helper must keep disk_block_ids ≥ block_table.len() \
             (got disk={} bt={})",
            disk_block_ids.len(),
            block_table.len()
        );

        // 2026-09-25: This layer's offloaded count lags `disk_block_ids.len()` by the blocks
        // allocated since it last ran. Each missing block is in the HBM cache at
        // `block_table[bt_idx]`, with `bt_idx = logical_pos - window_start`; read it back and push
        // it to disk.
        if disk_last_offloaded_per_layer.len() <= self.attn_layer_idx {
            disk_last_offloaded_per_layer.resize(self.attn_layer_idx + 1, 0);
        }
        let last = disk_last_offloaded_per_layer[self.attn_layer_idx] as usize;
        let total = disk_block_ids.len();
        if total == 0 {
            return Ok(());
        }
        // 2026-09-25: The HBM-resident `block_table` covers logical blocks
        // `[total - block_table.len(), total)`.
        let window_start = total.saturating_sub(block_table.len());
        // 2026-09-25: Re-offload the block before `last` as well as `last..total`, because a block
        // can gain slots after it was offloaded:
        // - in decode with no new block (`last == total`), the active block `total - 1` gains one
        //   slot per step;
        // - after a prefill chunk that ended mid-block, the next chunk fills that block's tail.
        //   Attention streams the history from disk (`attend_layer_on_stream`), so a stale copy
        //   there would be read as the block's contents.
        let start = last.saturating_sub(1).min(total - 1);
        for logical_pos in start..total {
            if logical_pos < window_start {
                // 2026-09-25: The sliding window already moved past `logical_pos` before this layer
                // offloaded it, so its K/V is no longer in HBM. Refuse rather than write a block
                // that was never copied.
                anyhow::bail!(
                    "high-speed-swap: layer {} block {} was evicted before this layer offloaded \
                     it (issue #31). \n\
                     Diagnostic state: attn_layer_idx={}, logical_pos={}, \
                     window_start={}, total=disk_block_ids.len()={}, \
                     block_table.len()={}, this_layer.last_offloaded={}, \
                     all_layer_cursors={:?}.\n\
                     This means the sliding-window eviction loop advanced past disk slot \
                     {} before attention layer {} could push its K/V there. The slide-before-alloc \
                     invariant in block_mgmt.rs (every attention layer must offload before any \
                     slide) is debug-asserted only — release builds skip it.\n\
                     Workaround: raise --high-speed-swap-cache-blocks-per-seq so \
                     `cap × block_size` ≥ your largest prompt, OR drop --high-speed-swap \
                     entirely if KV fits HBM at this batch/quant.",
                    self.attn_layer_idx,
                    logical_pos,
                    self.attn_layer_idx,
                    logical_pos,
                    window_start,
                    total,
                    block_table.len(),
                    last,
                    disk_last_offloaded_per_layer,
                    logical_pos,
                    self.attn_layer_idx,
                );
            }
            let bt_idx = logical_pos - window_start;
            let phys_blk = block_table[bt_idx];
            let disk_id = disk_block_ids[logical_pos];
            let k_block_dev = kv_cache.k_cache_ptr(self.attn_layer_idx, phys_blk).0;
            let v_block_dev = kv_cache.v_cache_ptr(self.attn_layer_idx, phys_blk).0;
            let mut k_host = vec![half::bf16::from_f32(0.0); block_floats];
            let mut v_host = vec![half::bf16::from_f32(0.0); block_floats];
            // 2026-09-25: By layer dtype: BF16 copies the bytes directly; the others read raw bytes
            // and dequantize to BF16 on the host before the disk write.
            let layer_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
            let layer_block_bytes = kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx);
            let bs_us = bs;
            let nkv_us = nkv as usize;
            let hd_us = hd as usize;
            match layer_dtype {
                KvCacheDtype::Bf16
                | KvCacheDtype::Bf16KTurbo4V
                | KvCacheDtype::Bf16KTurbo3V
                | KvCacheDtype::Bf16KTurbo2V => {
                    // 2026-09-25: `copy_d2h_on_stream` orders the copy after the cache write on
                    // `stream`. The bytes go straight into the BF16 host vector, so the device
                    // block stride must equal the host span; debug builds assert it.
                    debug_assert_eq!(
                        layer_block_bytes,
                        block_floats * 2,
                        "BF16 KV block stride must equal bs*nkv*hd*2 (layer {})",
                        self.attn_layer_idx
                    );
                    // 2026-09-25: SAFETY: `k_host` and `v_host` are initialised `vec![bf16;
                    // block_floats]`, and `size_of::<half::bf16>() == 2`, so `block_floats * 2`
                    // bytes is exactly each allocation. The two `&mut` byte slices cover disjoint
                    // allocations and are the only live references to them during each call.
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(k_block_dev),
                        unsafe {
                            std::slice::from_raw_parts_mut(
                                k_host.as_mut_ptr() as *mut u8,
                                block_floats * 2,
                            )
                        },
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(v_block_dev),
                        unsafe {
                            std::slice::from_raw_parts_mut(
                                v_host.as_mut_ptr() as *mut u8,
                                block_floats * 2,
                            )
                        },
                        stream,
                    )?;
                }
                KvCacheDtype::Fp8
                | KvCacheDtype::Fp8KTurbo4V
                | KvCacheDtype::Fp8KTurbo3V
                | KvCacheDtype::Fp8KTurbo2V => {
                    let mut k_raw = vec![0u8; block_floats];
                    let mut v_raw = vec![0u8; block_floats];
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    let (k_scale, v_scale) = self.effective_fp8_scales();
                    dequant_fp8_to_bf16(&k_raw, k_scale, &mut k_host);
                    dequant_fp8_to_bf16(&v_raw, v_scale, &mut v_host);
                }
                KvCacheDtype::Nvfp4
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo4KTurbo3V
                | KvCacheDtype::Turbo4KTurbo8V => {
                    let mut k_raw = vec![0u8; layer_block_bytes];
                    let mut v_raw = vec![0u8; layer_block_bytes];
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    let lut = if layer_dtype == KvCacheDtype::Nvfp4 {
                        &NVFP4_E2M1_LUT
                    } else {
                        &TURBO4_LUT
                    };
                    dequant_4bit_block_to_bf16(&k_raw, bs_us, nkv_us, hd_us, lut, &mut k_host);
                    dequant_4bit_block_to_bf16(&v_raw, bs_us, nkv_us, hd_us, lut, &mut v_host);
                }
                KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V | KvCacheDtype::Turbo2 => {
                    let mut k_raw = vec![0u8; layer_block_bytes];
                    let mut v_raw = vec![0u8; layer_block_bytes];
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    dequant_turbo3_block_to_bf16(&k_raw, bs_us, nkv_us, hd_us, &mut k_host);
                    dequant_turbo3_block_to_bf16(&v_raw, bs_us, nkv_us, hd_us, &mut v_host);
                }
                KvCacheDtype::Turbo8 => {
                    let mut k_raw = vec![0u8; layer_block_bytes];
                    let mut v_raw = vec![0u8; layer_block_bytes];
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        metrale_gpu_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    dequant_turbo8_block_to_bf16(&k_raw, bs_us, nkv_us, hd_us, &mut k_host);
                    dequant_turbo8_block_to_bf16(&v_raw, bs_us, nkv_us, hd_us, &mut v_host);
                }
            }
            metrale_storage::with_local(|hss| {
                match layer_dtype {
                    KvCacheDtype::Bf16
                    | KvCacheDtype::Bf16KTurbo4V
                    | KvCacheDtype::Bf16KTurbo3V
                    | KvCacheDtype::Bf16KTurbo2V => hss.offload_block_on_stream(
                        stream,
                        layer_u32,
                        disk_id,
                        k_block_dev,
                        &k_host,
                        &v_host,
                    ),
                    // 2026-09-25: Other dtypes skip the predictor projection, which reads the
                    // device block as BF16.
                    _ => hss.offload_block_no_predict_on_stream(
                        stream, layer_u32, disk_id, &k_host, &v_host,
                    ),
                }
            })
            .expect("local_installed checked in high_speed_swap_engaged")?;
        }
        disk_last_offloaded_per_layer[self.attn_layer_idx] = total as u32;
        Ok(())
    }
}
