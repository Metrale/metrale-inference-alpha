// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: mHC (hyper-connection highway) and PLE attachment for the GDN
//! layer, and the refusal used by the decode paths that have no highway form.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - With `hc` set, `TransformerLayer::decode`, `decode_multi_seq` and
//!   `prefill` run the hc forward paths (`trait_decode_hc.rs`,
//!   `trait_decode_multi_seq/hc.rs`, `trait_prefill_hc.rs`), and
//!   `decode_batched` and `decode_verify_multi` return an error.

use super::Qwen3SsmLayer;
impl Qwen3SsmLayer {
    /// 2026-09-25: Attach the mHC highway weights; the entry points then route
    /// as the module header states.
    pub fn set_hc_weights(&mut self, hc: crate::layers::qwen3_attention::HcWeights) {
        self.hc = Some(hc);
    }

    /// 2026-09-25: Attach the PLE n-gram injection. The hc prefill and
    /// single-sequence decode paths run it before this layer's `hc_pre_site`.
    pub fn set_ple(&mut self, ple: crate::layers::ple::PleLayer) {
        self.ple = Some(ple);
    }

    /// 2026-09-25: Error when the mHC highway is attached. `decode_batched` and
    /// `decode_verify_multi` call it first: they take the caller's `residual`
    /// buffer, while the hc paths carry the residual on the highway and take
    /// none.
    pub(crate) fn refuse_batched_under_hc(&self, path: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.hc.is_none(),
            "qwen3_ssm::{path}: the mHC highway has no batched GDN path yet. \
             This model serves at concurrency 1; the batched paths maintain \
             their own residual, which the highway replaces, so running them \
             would count every block output twice. Metrale Engine #753 item B."
        );
        Ok(())
    }
}
