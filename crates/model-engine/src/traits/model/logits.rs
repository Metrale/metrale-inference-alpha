// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelLogits`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-09-26: Reading a forward pass's output: the logits buffers, device argmax and the post-norm
/// hidden state.
pub trait ModelLogits {
    fn vocab_size(&self) -> usize;

    /// 2026-09-25: Copy one row of logits to the host; `dst` is `vocab_size * 2` bytes for BF16
    /// logits.
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()>;

    /// 2026-09-25: Whether `logits_ptr` holds FP32 logits (`vocab * 4` bytes per row). Default
    /// `false`; `TransformerModel` returns `false` too, because its `use_fp32_logits` is fixed to
    /// `false` in `TransformerModel::new`.
    fn logits_ptr_is_fp32(&self, _logits_ptr: DevicePtr) -> bool {
        false
    }

    /// 2026-09-25: The device logits buffer, where a verify pass leaves its `[k, vocab]` rows.
    fn logits_buffer_ptr(&self) -> DevicePtr;

    /// 2026-09-25: Argmax of one logits row, computed on the device.
    fn argmax_on_device(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32>;

    /// 2026-09-25: Argmax of each of `n` logits rows on the device.
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>>;

    /// 2026-09-25: The post-final-norm hidden state of the last decode step.
    fn hidden_after_norm(&self) -> DevicePtr;

    /// 2026-09-25: Whether single-token decode writes FP32 logits, read from
    /// [`Self::decode_logits_ptr`] at 4 bytes per element. Default `false`; `TransformerModel`
    /// returns its `use_fp32_logits`, which is always `false`.
    fn decode_logits_fp32(&self) -> bool {
        false
    }

    /// 2026-09-25: The buffer single-token decode last wrote its logits to: FP32 when
    /// [`Self::decode_logits_fp32`] is true, else BF16. The default panics, so a model must
    /// override it; `TransformerModel` does.
    fn decode_logits_ptr(&self) -> DevicePtr {
        unreachable!(
            "Model::decode_logits_ptr() must be overridden alongside \
             decode_logits_fp32() — default cannot return a valid pointer."
        )
    }
}
