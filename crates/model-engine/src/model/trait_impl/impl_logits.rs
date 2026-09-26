// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelLogits for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::model::types::TransformerModel;
use crate::traits::ModelLogits;

impl ModelLogits for TransformerModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size_dispatch()
    }

    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.copy_logits_to_host_dispatch(logits_ptr, dst)
    }

    fn logits_ptr_is_fp32(&self, logits_ptr: DevicePtr) -> bool {
        self.logits_ptr_is_fp32_dispatch(logits_ptr)
    }

    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr_dispatch()
    }

    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        self.argmax_on_device_dispatch(logits_ptr, _stream)
    }

    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        self.argmax_batch_dispatch(logits_ptr, n, _stream)
    }

    fn hidden_after_norm(&self) -> DevicePtr {
        self.hidden_after_norm_dispatch()
    }

    fn decode_logits_fp32(&self) -> bool {
        self.decode_logits_fp32_dispatch()
    }

    fn decode_logits_ptr(&self) -> DevicePtr {
        self.decode_logits_ptr_dispatch()
    }
}
