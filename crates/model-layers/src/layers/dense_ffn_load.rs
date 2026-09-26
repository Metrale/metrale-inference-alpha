// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Load-time copies of the NVFP4 weights for the Q4_K, NVFP4 MMQ and int8 prefill
//! arms: the `ensure_*_weight` builders, and the steps of `finalize_q4k_load` and
//! `finalize_nvfp4_mmq_load` (in `dense_ffn.rs`) that follow their environment reads.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - Each `ensure_*_weight` builds its copy at most once per `OnceLock` cell and returns the
//!   copy stored in the cell.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::{DenseFfnLayer, Fp4MmqWeight, Int8Weight, Q4kWeight};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

impl DenseFfnLayer {
    /// 2026-09-26: The gate and up Q4_K copies `finalize_q4k_load` builds.
    pub(super) fn build_q4k_gate_up(
        &self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        self.ensure_q4k_weight(
            &self.q4k_gate,
            gpu,
            &self.weights.gate_proj,
            inter,
            h,
            stream,
        )?;
        self.ensure_q4k_weight(&self.q4k_up, gpu, &self.weights.up_proj, inter, h, stream)?;
        Ok(())
    }

    /// 2026-09-26: The rest of `finalize_q4k_load` once the gate and up copies exist: the down copy
    /// (int8 when `down_faith2`, else Q4_K), a sync of `stream`, and the freeing of the
    /// transposed `_t` copies.
    pub(super) fn finish_q4k_load(
        &mut self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
        down_faith2: bool,
    ) -> Result<()> {
        if down_faith2 {
            self.ensure_int8_weight(
                &self.int8_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        } else {
            self.ensure_q4k_weight(
                &self.q4k_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let mut freed = 0usize;
        for wt in [
            &mut self.weights.gate_proj_t,
            &mut self.weights.up_proj_t,
            &mut self.weights.down_proj_t,
        ] {
            if let Some(w) = wt.as_ref()
                && !w.weight.is_null()
            {
                gpu.free(w.weight)?;
                gpu.free(w.weight_scale)?;
                freed += 1;
            }
            *wt = None;
        }
        if freed > 0 {
            // 2026-09-25: Latched per backend (`OpCache::once`): a model loaded onto
            // another backend logs again.
            if gpu.op_cache().once("log:ffn_mmq_freed_twins") {
                tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "[metrale] METRALE_FFN_MMQ: freed transposed FFN `_t` copies (dead under Q4_K prefill) — Q4_K weights net to ~0 vs NVFP4 baseline"
                );
            }
        }
        Ok(())
    }

    /// 2026-09-26: The gate and up `block_nvfp4` repacks `finalize_nvfp4_mmq_load` builds.
    pub(super) fn build_nvfp4_mmq_gate_up(
        &self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        self.ensure_nvfp4_mmq_weight(
            &self.fp4mmq_gate,
            gpu,
            &self.weights.gate_proj,
            inter,
            h,
            stream,
        )?;
        self.ensure_nvfp4_mmq_weight(
            &self.fp4mmq_up,
            gpu,
            &self.weights.up_proj,
            inter,
            h,
            stream,
        )?;
        Ok(())
    }

    /// 2026-09-26: The rest of `finalize_nvfp4_mmq_load` once the gate and up repacks exist: the down
    /// repack when `down_mmq`, a sync of `stream`, and the freeing of the repacked projections'
    /// transposed `_t` copies.
    pub(super) fn finish_nvfp4_mmq_load(
        &mut self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
        down_mmq: bool,
    ) -> Result<()> {
        if down_mmq {
            self.ensure_nvfp4_mmq_weight(
                &self.fp4mmq_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        // 2026-09-25: Free the `_t` copies of the repacked projections.
        let mut down_t = if down_mmq {
            Some(&mut self.weights.down_proj_t)
        } else {
            None
        };
        let mut freed = 0usize;
        for wt in [&mut self.weights.gate_proj_t, &mut self.weights.up_proj_t]
            .into_iter()
            .chain(down_t.take())
        {
            if let Some(w) = wt.as_ref()
                && !w.weight.is_null()
            {
                gpu.free(w.weight)?;
                gpu.free(w.weight_scale)?;
                freed += 1;
            }
            *wt = None;
        }
        if freed > 0 {
            // 2026-09-25: Latched per backend (`OpCache::once`): a model loaded onto
            // another backend logs again.
            if gpu.op_cache().once("log:ffn_fp4mmq_freed_twins") {
                tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "[metrale] METRALE_FFN_NVFP4_MMQ: freed gate/up `_t` copies (dead under FP4-MMQ prefill) — block_nvfp4 copies net to ~0 vs NVFP4 baseline"
                );
            }
        }
        Ok(())
    }

    /// 2026-09-25: Returns the `block_nvfp4` repack of `src` (`[n, k]`), building it on the first
    /// call and keeping it in `cell`. If another thread filled `cell` first, this call's buffer is
    /// freed after a stream sync.
    pub(super) fn ensure_nvfp4_mmq_weight(
        &self,
        cell: &std::sync::OnceLock<Fp4MmqWeight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Fp4MmqWeight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        let w = gpu.alloc(ops::nvfp4_mmq_weight_bytes(n, k))?;
        ops::nvfp4_mmq_repack(
            gpu,
            self.nvfp4_repack_k,
            src.weight,
            src.weight_scale,
            w,
            n,
            k,
            stream,
        )?;
        let built = Fp4MmqWeight { w };
        if let Err(dup) = cell.set(built) {
            gpu.synchronize(stream)?;
            let _ = gpu.free(dup.w);
        }
        Ok(*cell.get().expect("fp4mmq weight cell set above"))
    }

    /// 2026-09-25: Returns the int8 copy of `src` (`[n, k]`), building it on the first call with
    /// `requant_w_nvfp4_int8` from the non-transposed NVFP4 weight and keeping it in `cell`.
    /// The requant is queued on `stream` and not waited for; readers on the same stream are
    /// ordered after it.
    pub(super) fn ensure_int8_weight(
        &self,
        cell: &std::sync::OnceLock<Int8Weight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Int8Weight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        let (nn, kk) = (n as usize, k as usize);
        let w_i8 = gpu.alloc(nn * kk)?;
        let w_scale = gpu.alloc(nn * (kk / 32) * 4)?;
        ops::requant_w_nvfp4_int8(
            gpu,
            self.requant_w_int8_k,
            src.weight,
            src.weight_scale,
            src.weight_scale_2,
            w_i8,
            w_scale,
            n,
            k,
            stream,
        )?;
        let built = Int8Weight { w_i8, w_scale };
        // 2026-09-25: Another thread filled `cell` first: free this call's buffers.
        if let Err(dup) = cell.set(built) {
            let _ = gpu.free(dup.w_i8);
            let _ = gpu.free(dup.w_scale);
        }
        Ok(*cell.get().expect("int8 weight cell set above"))
    }

    /// 2026-09-25: Returns the Q4_K copy of `src` (non-transposed NVFP4, `[n, k]`), building it on
    /// the first call (dequant to a temporary BF16 buffer, then quantize to `block_q4_K`) and
    /// keeping it in `cell`.
    pub(super) fn ensure_q4k_weight(
        &self,
        cell: &std::sync::OnceLock<Q4kWeight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Q4kWeight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        let bf16_tmp = gpu.alloc((n as usize) * (k as usize) * 2)?;
        ops::dequant_nvfp4_to_bf16(
            gpu,
            self.dequant_nvfp4_bf16_k,
            src.weight,
            src.weight_scale,
            bf16_tmp,
            src.weight_scale_2,
            n,
            k,
            stream,
        )?;
        let w_q4k = gpu.alloc(ops::q4k_weight_bytes(n, k))?;
        ops::quantize_weight_q4k(gpu, self.q4k_quant_w_k, bf16_tmp, w_q4k, n, k, stream)?;
        // 2026-09-25: The quantize on `stream` reads `bf16_tmp`; sync before freeing it.
        gpu.synchronize(stream)?;
        let _ = gpu.free(bf16_tmp);
        let built = Q4kWeight { w_q4k };
        if let Err(dup) = cell.set(built) {
            let _ = gpu.free(dup.w_q4k);
        }
        Ok(*cell.get().expect("q4k weight cell set above"))
    }
}
