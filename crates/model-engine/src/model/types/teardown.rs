// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `TransformerModel` teardown: `adopt_weight_store` and `release_pools`.
//!
//! Owner: model-engine.
//! Invariants:
//! - `release_pools` attempts every release even after an earlier one fails, and returns the first error.

use metrale_gpu_runtime::gpu::GpuBackend;

use super::TransformerModel;

impl TransformerModel {
    /// 2026-09-25: Hand the model the ledger of its own weights, for teardown.
    pub fn adopt_weight_store(&mut self, store: metrale_model_weights::weights::WeightStore) {
        self.weight_store = Some(store);
    }

    /// 2026-09-25: Release every GPU resource this model owns, in this order:
    /// derived weights, SSM snapshots, the SSM state pool, the KV cache, the
    /// buffer arena, the weight store, then whatever the backend still tracks
    /// (`GpuBackend::sweep_unreleased`). Every step is attempted even after an
    /// earlier one fails; the first error is returned, with its label as
    /// context.
    pub(in crate::model) fn release_pools(&mut self) -> anyhow::Result<()> {
        use metrale_core::scope::ModelResource;

        let gpu: &dyn GpuBackend = self.gpu.as_ref();
        let mut first_error: Option<anyhow::Error> = None;
        let mut attempt = |label: &'static str, r: anyhow::Result<()>| {
            if let Err(e) = r
                && first_error.is_none()
            {
                first_error = Some(e.context(label));
            }
        };

        attempt("derived weights", self.derived.release(gpu));
        attempt("ssm snapshots", self.ssm_snapshots.release(gpu));
        // 2026-09-25: The pool is Arc'd because sequences hold slots. Another
        // live handle means a sequence was not released before teardown; that
        // is reported as an error rather than forced.
        match std::sync::Arc::get_mut(&mut self.ssm_pool) {
            Some(pool) => attempt("ssm state pool", pool.release(gpu)),
            None => attempt(
                "ssm state pool",
                Err(anyhow::anyhow!(
                    "{} handle(s) still hold the SSM pool — a sequence was not \
                     released before teardown",
                    std::sync::Arc::strong_count(&self.ssm_pool) - 1
                )),
            ),
        }
        attempt("kv cache", self.kv_cache.lock().release(gpu));
        attempt("buffer arena", self.buffers.release(gpu));
        // 2026-09-25: The weight store goes after every owner above, because
        // the layers hold pointers into it.
        if let Some(mut store) = self.weight_store.take() {
            attempt("weight store", store.release(gpu));
        }
        // 2026-09-25: Last: allocations no owner above released, such as
        // weights the loaders fused into layer-owned buffers.
        let swept = gpu.sweep_unreleased();
        if swept > 0 {
            tracing::warn!(target: "metrale_model_engine::model::types", "teardown swept {swept} allocation(s) that no ModelResource \
                 released — they are reclaimed, but each one is memory whose \
                 owner is unaccounted for"
            );
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
