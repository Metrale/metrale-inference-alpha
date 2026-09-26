// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The MoE router: the logits on the GPU, the top-k selection on
//! the CPU (`route`, `route_launch`, `route_select`), and the next layer's
//! router on this layer's input for prefetching (`predict_launch`,
//! `predict_select`, `route_predict`).
//!
//! Owner: model-arch (DeepSeek-V4.1 MoE).
//! Invariants:
//! - Every selection in this module is `route_from_logits` on logits read
//!   back from the device.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{MoeV41, MoeV41LayerWeights, RouterWeights, route_from_logits};

impl MoeV41 {
    /// 2026-09-25: Router logits on the GPU (f32 out), the selection on the
    /// CPU. Drains `stream`.
    pub fn route(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        self.route_launch(gpu, w, x, m, stream)?;
        self.route_select(gpu, w, m, stream)
    }

    /// 2026-09-25: The router GEMM/GEMV into `self.logits`: device work only,
    /// so a CUDA graph can hold it (segment A of the step graph ends with it).
    /// Errors when `m` is outside `1..=max_tokens`.
    pub fn route_launch(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "moe_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        // 2026-09-25: At m <= 8 a GEMV. By default the products form
        // (`moe_v41_router_gemv_f32out_products`, one gate row a block: the
        // 256 threads write the bf16 x bf16 products into shared memory and
        // thread 0 adds them in k order); `METRALE_DS41_ROUTER_STAGED=0`
        // selects the direct form, one thread per logit. Above 8 rows the
        // tiled `dense_gemm_bf16_f32out`.
        let (kernel, grid, block, smem) = if m <= 8 {
            if router_staged() {
                (
                    self.k.router_gemv_staged,
                    [c.n_routed as u32, m as u32, 1],
                    [256, 1, 1],
                    c.dim as u32 * 4,
                )
            } else {
                (
                    self.k.router_gemv,
                    [(c.n_routed as u32).div_ceil(64), m as u32, 1],
                    [64, 1, 1],
                    0,
                )
            }
        } else {
            (
                self.k.gemm_f32out,
                [(c.n_routed as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
                [16, 16, 1],
                0,
            )
        };
        KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block(block)
            .shared_mem(smem)
            .arg_ptr(x)
            .arg_ptr(w.gate_w)
            .arg_ptr(self.logits)
            .arg_u32(m as u32)
            .arg_u32(c.n_routed as u32)
            .arg_u32(c.dim as u32)
            .launch(stream)
    }

    /// 2026-09-25: The selection: drain the stream, download the logits
    /// `route_launch` wrote, pick the top-k on the CPU with
    /// `route_from_logits`. Errors when the host bias length is not
    /// `n_routed`.
    pub fn route_select(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        m: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        let mut bytes = vec![0u8; m * c.n_routed * 4];
        gpu.copy_d2h_on_stream(self.logits, &mut bytes, stream)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        ensure!(
            w.gate_bias.len() == c.n_routed,
            "gate bias has {} entries for {} experts",
            w.gate_bias.len(),
            c.n_routed
        );
        Ok(route_from_logits(&logits, m, &w.gate_bias, c))
    }
}

impl MoeV41 {
    /// 2026-09-25: The next layer's router GEMV on this layer's single-token
    /// input, into `pred_logits`. Device work only; `predict_select` reads it
    /// back.
    pub fn predict_launch(
        &self,
        gpu: &dyn GpuBackend,
        next: &RouterWeights,
        x: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (kernel, grid, block, smem) = if router_staged() {
            (
                self.k.router_gemv_staged,
                [c.n_routed as u32, 1, 1],
                [256, 1, 1],
                c.dim as u32 * 4,
            )
        } else {
            (
                self.k.router_gemv,
                [(c.n_routed as u32).div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
            )
        };
        KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block(block)
            .shared_mem(smem)
            .arg_ptr(x)
            .arg_ptr(next.gate_w)
            .arg_ptr(self.pred_logits)
            .arg_u32(1)
            .arg_u32(c.n_routed as u32)
            .arg_u32(c.dim as u32)
            .launch(stream)
    }

    /// 2026-09-25: The `k` best experts of the predicted layer by
    /// `score + bias`, as `(layer, expert)` cache keys, from the logits
    /// `predict_launch` wrote. Drains `stream`.
    pub fn predict_select(
        &self,
        gpu: &dyn GpuBackend,
        next: &RouterWeights,
        k: usize,
        stream: u64,
    ) -> Result<Vec<(u32, u32)>> {
        let c = &self.cfg;
        let mut bytes = vec![0u8; c.n_routed * 4];
        gpu.copy_d2h_on_stream(self.pred_logits, &mut bytes, stream)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut wide = c.clone();
        wide.topk = k.min(c.n_routed);
        let (_, ids) = route_from_logits(&logits, 1, &next.gate_bias, &wide);
        Ok(ids.into_iter().map(|e| (next.layer, e as u32)).collect())
    }

    /// 2026-09-25: The router of this layer on `x` (one token), the `k` best
    /// experts by `score + bias` in `route_from_logits` order. Drains the
    /// stream; overwrites `self.logits`.
    pub fn route_predict(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<Vec<usize>> {
        let c = &self.cfg;
        self.route_launch(gpu, w, x, 1, stream)?;
        let mut bytes = vec![0u8; c.n_routed * 4];
        gpu.copy_d2h_on_stream(self.logits, &mut bytes, stream)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut wide = c.clone();
        wide.topk = k.min(c.n_routed);
        Ok(route_from_logits(&logits, 1, &w.gate_bias, &wide).1)
    }
}

fn router_staged() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("METRALE_DS41_ROUTER_STAGED").is_ok_and(|v| v == "0"))
}
