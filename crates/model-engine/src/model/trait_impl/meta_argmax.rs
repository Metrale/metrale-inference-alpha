// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device argmax over logits rows, for one row (`argmax_on_device_dispatch`) or n rows (`argmax_batch_dispatch`).
//!
//! Owner: model-engine.
//! Invariants:
//! - Both functions ignore their `_stream` argument and launch on `self.gpu.default_stream()`.
//! - The u32 results are staged in the first `4 * n` bytes of `buffers.scratch()` and copied to the host before returning.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn argmax_on_device_dispatch(
        &self,
        logits_ptr: DevicePtr,
        _stream: u64,
    ) -> Result<u32> {
        let stream = self.gpu.default_stream();
        let out_ptr = self.buffers.scratch();
        // 2026-09-25: `argmax_fp32` runs only when `use_fp32_logits` is set and the
        // pointer is `logits_fp32_buf`; otherwise the buffer is BF16 and
        // `argmax_bf16` runs. Both kernels take (ptr, ptr, u32), so the launch
        // helper is shared and only the handle differs.
        let is_fp32 = self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0;
        let kernel = if is_fp32 {
            self.argmax_logits_kernel
        } else {
            self.argmax_kernel
        };
        ops::argmax_bf16(
            self.gpu.as_ref(),
            kernel,
            logits_ptr,
            out_ptr,
            self.config.vocab_size as u32,
            stream,
        )?;
        let mut buf = [0u8; 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let gpu_token = u32::from_le_bytes(buf);

        Ok(gpu_token)
    }

    pub(super) fn argmax_batch_dispatch(
        &self,
        logits_ptr: DevicePtr,
        n: usize,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let v = self.config.vocab_size;
        let bf16 = 2usize;
        let out_ptr = self.buffers.scratch();
        // 2026-09-25: One launch with one block per row. The single-row `argmax_bf16`
        // is a one-CTA reduction (grid [1, 1, 1]), so n calls on one stream
        // serialise n single-SM scans. The batched kernel runs the same per-row
        // body, so ties resolve to the same index. The per-row loop runs when the
        // kernel set lacks `argmax_bf16_batch` or `METRALE_NO_ARGMAX_BATCH=1`.
        fn argmax_batch_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("METRALE_NO_ARGMAX_BATCH").ok().as_deref() != Some("1")
            })
        }
        if self.argmax_batch_kernel.0 != 0 && argmax_batch_enabled() {
            ops::argmax_bf16_batch(
                self.gpu.as_ref(),
                self.argmax_batch_kernel,
                logits_ptr,
                out_ptr,
                v as u32,
                n as u32,
                v as u32,
                stream,
            )?;
        } else {
            for i in 0..n {
                let logits_i = logits_ptr.offset(i * v * bf16);
                let out_i = out_ptr.offset(i * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    v as u32,
                    stream,
                )?;
            }
        }
        let mut buf = vec![0u8; n * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push(u32::from_le_bytes([
                buf[i * 4],
                buf[i * 4 + 1],
                buf[i * 4 + 2],
                buf[i * 4 + 3],
            ]));
        }
        Ok(results)
    }
}
