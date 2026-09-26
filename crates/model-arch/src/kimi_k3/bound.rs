// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `K3BoundLayer`, the Kimi K3 decoder layer the engine binds, and `K3HostShared`, the state its layers share.
//!
//! Decode runs on the host between a copy of the hidden row out and a copy back
//! (`host_decode.rs`). The KDA mixer uses CUDA `kda_decode` with per-sequence
//! device state unless `K3_CUDA_KDA=0`, the MLA mixer uses CUDA `mla_decode`
//! unless `K3_CUDA_MLA=0`, and packed LatentMoE experts use
//! `moe_w4a16_grouped_gemm_ptrtable_e8m0`.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants:
//! - `decode_graph_unsupported`, `decode_multi_seq_unsupported`,
//!   `decode_verify_multi_unsupported` and `decode_rollback_unsupported` are
//!   all true.
//! - Layer state is always a [`K3CpuFallbackState`]; the layer never uses the
//!   SSM pool.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::kimi_k3_host::{
    AttnResStream, K3CpuLayer, K3Graph, K3LayerSpec, KdaConfig, LatentMoeConfig, LayerCache,
    MixerKind, MlaConfig,
};
use metrale_model_weights::weights::WeightDtype;
use parking_lot::Mutex;

use super::kda_cuda::K3KdaDecodeKernels;
use super::mla_cuda::K3MlaDecodeKernels;
use super::moe_cuda::K3MoeGemmKernels;
use super::state::K3CpuFallbackState;
use metrale_model_layers::layer::{ForwardContext, LayerState, TransformerLayer};
use metrale_model_layers::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
    LayerWriteOnAccept,
};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight};

/// 2026-09-25: Checkpoint name, dtype and element count of one resident weight.
/// The first decode uses them to copy the weight to host FP32.
#[derive(Clone, Debug)]
pub struct WeightMeta {
    pub name: String,
    pub dtype: WeightDtype,
    pub numel: usize,
}

/// 2026-09-25: One per loaded model; every decoder layer holds the same `Arc`.
pub struct K3HostShared {
    pub config: ModelConfig,
    pub graph: K3Graph,
    pub kda: KdaConfig,
    pub mla: MlaConfig,
    pub moe: LatentMoeConfig,
    pub output_res_proj: DenseWeight,
    pub output_res_norm: DenseWeight,
    pub output_res_proj_meta: (WeightDtype, usize),
    pub output_res_norm_meta: (WeightDtype, usize),
    pub output_host: OnceLock<(Vec<f32>, Vec<f32>)>,
    /// 2026-09-25: Resolved on the first CUDA KDA decode, then reused.
    pub kda_kernels: OnceLock<K3KdaDecodeKernels>,
    /// 2026-09-25: Resolved on the first CUDA MLA decode, then reused.
    pub mla_kernels: OnceLock<K3MlaDecodeKernels>,
    /// 2026-09-25: Resolved at load for a checkpoint with packed experts, then
    /// reused.
    pub moe_kernels: OnceLock<K3MoeGemmKernels>,
    /// 2026-09-25: One AttnRes stream per token in flight, carried across layers.
    /// Keyed by the token's `residual` pointer (`hidden` when that is null):
    /// prefill runs every token through one layer before the next, and each
    /// token has its own residual row, so each token keeps its own stream.
    pub attnres: Mutex<HashMap<DevicePtr, AttnResStream>>,
}

/// 2026-09-25: One decoder layer. `weights` must be BF16 or FP32: the host
/// bind refuses any other dtype.
pub struct K3BoundLayer {
    pub index: usize,
    pub spec: K3LayerSpec,
    pub weights: Vec<DenseWeight>,
    pub weight_meta: Vec<WeightMeta>,
    /// 2026-09-25: Routed experts stored as `{prefix}.weight_packed`, bound as
    /// MXFP4 E8M0 through `quantized_mxfp4_e8m0_pair`. Empty when the
    /// checkpoint stores its experts unpacked.
    pub mxfp4_experts: Vec<(String, QuantizedWeight)>,
    pub host: OnceLock<K3CpuLayer>,
    pub shared: Arc<K3HostShared>,
}

impl TransformerLayer for K3BoundLayer {
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_host(
            hidden,
            residual,
            state,
            seq_len,
            ctx,
            stream,
            metrale_model_weights::kimi_k3_host::cuda_kda_enabled(),
            metrale_model_weights::kimi_k3_host::cuda_mla_enabled(),
        )
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let cache = match self.spec.mixer {
            MixerKind::Kda => LayerCache::Kda(metrale_model_weights::kimi_k3_host::KdaState::new(
                &self.shared.kda,
            )),
            MixerKind::Mla => {
                LayerCache::Mla(metrale_model_weights::kimi_k3_host::MlaKv::default())
            }
        };
        Ok(Box::new(K3CpuFallbackState::new(cache)))
    }

    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        state
            .as_any_mut()
            .downcast_mut::<K3CpuFallbackState>()
            .context("K3 release_state: expected K3CpuFallbackState")?
            .release(gpu)
    }
}

impl LayerCapabilities for K3BoundLayer {
    fn decode_graph_unsupported(&self) -> bool {
        // 2026-09-25: Decode copies to and from the host and synchronises, which
        // a CUDA graph cannot capture.
        true
    }

    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }

    fn decode_verify_multi_unsupported(&self) -> bool {
        // 2026-09-25: Each call advances one sequence's host AttnRes stream and
        // mixer state.
        true
    }

    fn decode_rollback_unsupported(&self) -> bool {
        // 2026-09-25: The mixer state is not in the paged KV cache, so lowering its
        // cursor would not rewind it. Only `restore_aux` sets it back.
        true
    }

    fn uses_ssm_pool(&self) -> bool {
        // 2026-09-25: Mixer state lives in `K3CpuFallbackState` (`alloc_state`).
        false
    }
}

impl LayerWeightSetup for K3BoundLayer {}
impl LayerWriteOnAccept for K3BoundLayer {}
impl LayerGraphHooks for K3BoundLayer {}

impl LayerAuxState for K3BoundLayer {
    fn has_aux_state(&self) -> bool {
        true
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let st = state
            .as_any()
            .downcast_ref::<K3CpuFallbackState>()
            .context("K3 snapshot_aux: expected K3CpuFallbackState")?;
        Ok(Some(st.snapshot(gpu, stream)?))
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<K3CpuFallbackState>()
            .context("K3 restore_aux: expected K3CpuFallbackState")?;
        st.restore(gpu, blob, stream)
    }
}

impl LayerSplitPrefill for K3BoundLayer {}
