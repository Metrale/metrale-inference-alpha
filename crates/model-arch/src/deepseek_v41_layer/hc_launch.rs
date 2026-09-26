// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The hyper-connection launches of one DeepSeek-V4.1 block.
//!
//! Per site (attention, FFN): `hc_v41_mixes_dot` (one block per token and
//! mix), then `hc_v41_finish_collapse`, which does the site's Sinkhorn finish
//! and the block's collapse in one launch over ceil(H / 256) blocks a token.
//! The two are independent: the collapse reads the previous site's `pre`, the
//! finish writes this site's. `hc_post` and the final collapse use the same
//! ceil(H / 256)-block grid, one column a thread.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::DeepSeekV41Layer;
use metrale_model_layers::layers::qwen3_attention::HcSiteWeights;

const HC_BLOCK: u32 = 256;

impl DeepSeekV41Layer {
    /// 2026-09-25: Site mixes (`pre_out`, `post_s`, `comb_s` from `streams`)
    /// and the block's collapse (`y = pre_in . streams`) in two launches: the
    /// dots over (m, mix_hc) blocks, then `hc_v41_finish_collapse`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mixes_collapse(
        &self,
        gpu: &dyn GpuBackend,
        site: &HcSiteWeights,
        streams: DevicePtr,
        pre_out: DevicePtr,
        pre_in: DevicePtr,
        y: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let mix_hc = (2 + rt.hc_mult) * rt.hc_mult;
        KernelLaunch::new(gpu, self.k_mixes_dot)
            .grid([m as u32, mix_hc as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(site.hc_fn)
            .arg_ptr(rt.mixes_s)
            .arg_u32(rt.hidden as u32)
            .arg_u32(rt.hc_mult as u32)
            .arg_f32(rt.norm_eps)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_finish_collapse)
            .grid([(rt.hidden as u32).div_ceil(HC_BLOCK), m as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(rt.mixes_s)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(pre_out)
            .arg_ptr(rt.post_s)
            .arg_ptr(rt.comb_s)
            .arg_ptr(streams)
            .arg_ptr(pre_in)
            .arg_ptr(y)
            .arg_u32(rt.hidden as u32)
            .arg_u32(rt.hc_mult as u32)
            .arg_u32(rt.sinkhorn_iters as u32)
            .arg_f32(rt.hc_eps)
            .launch(stream)
    }

    /// 2026-09-25: `y = pre . streams` alone (the final collapse of the last block).
    pub(super) fn collapse(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        pre: DevicePtr,
        y: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_collapse_wide)
            .grid([(self.rt.hidden as u32).div_ceil(HC_BLOCK), m as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(pre)
            .arg_ptr(y)
            .arg_u32(self.rt.hidden as u32)
            .arg_u32(self.rt.hc_mult as u32)
            .launch(stream)
    }

    /// 2026-09-25: `streams[j] = post[j] * block_out + sum_i comb[i][j] *
    /// streams[i]`, in place (`hc_v41_post_wide`), one column a thread.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_post(
        &self,
        gpu: &dyn GpuBackend,
        block_out: DevicePtr,
        streams: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_post_wide)
            .grid([(self.rt.hidden as u32).div_ceil(HC_BLOCK), m as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(block_out)
            .arg_ptr(streams)
            .arg_ptr(post)
            .arg_ptr(comb)
            .arg_ptr(streams)
            .arg_u32(self.rt.hidden as u32)
            .arg_u32(self.rt.hc_mult as u32)
            .launch(stream)
    }
}
