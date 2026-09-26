// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The KV-cache write, dispatched on the layer's KV dtype. Each WHT-rotated (turbo)
//! side is rotated with `wht_bf16` before the write, unless the weights were pre-rotated at load;
//! plain FP8 goes to `write_kv_cache_fp8`.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

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
    pub(in super::super) fn write_kv_cache(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
        graph_capture: bool,
    ) -> Result<()> {
        match self.kv_dtype {
            KvCacheDtype::Nvfp4 => ops::reshape_and_cache_nvfp4(
                gpu,
                self.reshape_cache_k,
                k,
                v,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                stream,
            ),
            KvCacheDtype::Turbo4
            | KvCacheDtype::Turbo3
            | KvCacheDtype::Turbo8
            | KvCacheDtype::Turbo2 => {
                // 2026-09-25: WHT on K and V before the write. K and V are `[num_tokens,
                // num_kv_heads, head_dim]` BF16 and the kernel runs one CTA per head row, so the
                // grid covers every (token, kv_head) pair; a grid of `num_kv_heads` alone would
                // rotate only the first token's heads.
                let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use metrale_gpu_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(k)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                    // 2026-09-25: InnerQ post-WHT scale on K. While `d_innerq_calibrating` is 1,
                    // block 0 also accumulates K² statistics; with both device flags 0 the kernel
                    // does nothing.
                    if self.innerq_apply_k_k.0 != 0 && head_dim == 128 {
                        KernelLaunch::new(gpu, self.innerq_apply_k_k)
                            .grid([total_heads, 1, 1])
                            .block([32, 1, 1])
                            .arg_ptr(k)
                            .arg_u32(head_dim)
                            .launch(stream)?;
                    }
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo8 => kv_cache.turbo8_data_bytes() as u64,
                    KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V => {
                        kv_cache.turbo3_data_bytes() as u64
                    }
                    KvCacheDtype::Turbo2 => kv_cache.turbo2_data_bytes() as u64,
                    // 2026-09-25: Turbo4 uses the NVFP4 data size.
                    _ => kv_cache.nvfp4_data_bytes() as u64,
                };
                ops::reshape_and_cache_nvfp4(
                    gpu,
                    self.reshape_cache_k,
                    k,
                    v,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    slot,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    key_stride,
                    value_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo3V => {
                // 2026-09-25: K as BF16, V as turbo3, in one write kernel with separate K and V
                // block strides. Only V is rotated.
                let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use metrale_gpu_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                ops::reshape_and_cache_bf16k_turbo3v(
                    gpu,
                    self.reshape_cache_k,
                    k,
                    v,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    slot,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    key_stride,
                    value_stride,
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo3_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo4V => {
                // 2026-09-25: K as BF16, V as turbo4, in one write kernel with separate K and V
                // block strides. Only V is rotated.
                let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use metrale_gpu_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                ops::reshape_and_cache_bf16k_turbo4v(
                    gpu,
                    self.reshape_cache_k,
                    k,
                    v,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    slot,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    key_stride,
                    value_stride,
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo2V => {
                // 2026-09-25: K as BF16, V as turbo2. Only V is rotated.
                let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use metrale_gpu_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                ops::reshape_and_cache_bf16k_turbo2v(
                    gpu,
                    self.reshape_cache_k,
                    k,
                    v,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    slot,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    key_stride,
                    value_stride,
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo2_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V
            | KvCacheDtype::Turbo3KTurbo8V => {
                // 2026-09-25: K and V are different turbo dtypes. Both are rotated, and the InnerQ
                // K scale runs as in the symmetric turbo arm.
                let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use metrale_gpu_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(k)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                    if self.innerq_apply_k_k.0 != 0 && head_dim == 128 {
                        KernelLaunch::new(gpu, self.innerq_apply_k_k)
                            .grid([total_heads, 1, 1])
                            .block([32, 1, 1])
                            .arg_ptr(k)
                            .arg_u32(head_dim)
                            .launch(stream)?;
                    }
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                let k_block_stride =
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Turbo4KTurbo3V => ops::reshape_and_cache_turbo4k_turbo3v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Turbo4KTurbo8V => ops::reshape_and_cache_turbo4k_turbo8v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Turbo3KTurbo8V => ops::reshape_and_cache_turbo3k_turbo8v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Fp8KTurbo3V | KvCacheDtype::Fp8KTurbo4V | KvCacheDtype::Fp8KTurbo2V => {
                // 2026-09-25: K as FP8 with the per-tensor `k_scale`, V as turbo2, 3 or 4, in one
                // write kernel. Only V is rotated.
                let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use metrale_gpu_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                let (k_scale, _v_scale) = self.effective_fp8_scales();
                let k_block_stride =
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Fp8KTurbo3V => ops::reshape_and_cache_fp8k_turbo3v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_scale,
                        k_block_stride,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo4V => ops::reshape_and_cache_fp8k_turbo4v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_scale,
                        k_block_stride,
                        v_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo2V => ops::reshape_and_cache_fp8k_turbo2v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_scale,
                        k_block_stride,
                        v_block_stride,
                        kv_cache.turbo2_data_bytes() as u64,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Bf16 => ops::reshape_and_cache(
                gpu,
                self.reshape_cache_k,
                k,
                v,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
                kv_cache.cache_stride() as u64,
                stream,
            ),
            // 2026-09-25: The remaining dtype, FP8 (`write_kv_cache_fp8.rs`).
            _ => self.write_kv_cache_fp8(
                gpu,
                k,
                v,
                kv_cache,
                slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
                stream,
                graph_capture,
            ),
        }
    }
}
