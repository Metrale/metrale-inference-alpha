// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The final-norm step in front of every lm_head projection.
//!
//! Owner: model-engine.
//! Invariants:
//! - `TransformerModel` reads `final_norm` only in `final_norm_apply`, so a
//!   checkpoint without a final norm (`final_norm_identity`) gets the identity
//!   copy at every norm-then-lm-head site.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::TransformerModel;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: Apply the final norm to `num_tokens` BF16 rows of
    /// `hidden_size`. When the checkpoint has no final norm
    /// (`final_norm_identity`, set by the qwen4_exp parser) this is a plain
    /// copy: `rms_norm` with a ones weight is not an identity, it still divides
    /// each row by its RMS.
    pub(super) fn final_norm_apply(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if self.config.final_norm_identity {
            return self.gpu.copy_d2d_async(
                input,
                output,
                num_tokens as usize * hidden_size as usize * 2,
                stream,
            );
        }
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            input,
            &self.final_norm,
            output,
            num_tokens,
            hidden_size,
            eps,
            stream,
        )
    }
}
