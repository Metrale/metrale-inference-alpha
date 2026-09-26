// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: `MoeV41::forward`, one layer's MoE for `m` tokens (the routed
//! experts from the expert cache plus the shared expert), and the `launch_n`
//! elementwise launcher.
//!
//! Owner: model-arch (DeepSeek-V4.1 MoE).
//! Invariants:
//! - `forward` clears `acc` on `stream` before any routed or shared row is
//!   added to it.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_weights::weights::expert_stream::{ExpertLru, ExpertSource};

use super::{MoeV41, MoeV41LayerWeights, MoeV41Timing, RouterWeights, prefetch_k, shared_side};
use metrale_model_layers::layers::ops::{
    self, Q2K_MMQ_SMEM, Q3K_MMQ_SMEM, Q8_1_BLOCK_BYTES, kquant_mmq_gemm, kquant_mmvq_w,
    kquant_q8_1_rows,
};

/// 2026-09-25: `METRALE_DS41_M1_LOOP=1`, read once: a single token runs
/// through the per-expert group loop of `forward` (one-row groups on the
/// decode GEMV) instead of the pointer-table arm (`routed_m1`).
fn m1_loop() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_DS41_M1_LOOP").is_ok_and(|v| v == "1"))
}

/// 2026-09-25: `METRALE_DS41_PREFILL_GEMV=1`, read once: expert groups of
/// more than eight rows run on the decode GEMV in chunks of at most eight rows
/// instead of the MMQ arm (see `forward`).
fn prefill_gemv() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_DS41_PREFILL_GEMV").is_ok_and(|v| v == "1"))
}

impl MoeV41 {
    pub(super) fn launch_n(
        &self,
        gpu: &dyn GpuBackend,
        k: KernelHandle,
        n: usize,
        stream: u64,
        f: impl FnOnce(KernelLaunch) -> KernelLaunch,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        f(KernelLaunch::new(gpu, k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1]))
        .launch(stream)
    }

    /// 2026-09-25: One layer's MoE for `m` tokens: routed experts from the
    /// cache, plus the shared expert. Returns the bf16 `[m, dim]` output (the
    /// `out` buffer) and the routing weights and indices, `[m, topk]` each.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<S: ExpertSource + ?Sized>(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        lru: &mut ExpertLru,
        src: &S,
        x: DevicePtr,
        m: usize,
        reader_threads: usize,
        next: Option<&RouterWeights>,
        stream: u64,
    ) -> Result<(DevicePtr, Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        let t0 = std::time::Instant::now();
        let m1_arm = m == 1 && !m1_loop();
        // 2026-09-25: On the single-token arm, unless `METRALE_DS41_SHARED_SIDE=0`,
        // the shared expert runs on `self.side` while this stream runs the
        // router and the host selects and fetches; this stream waits on
        // `ev_out` before the shared tail.
        let side = m1_arm && shared_side();
        // 2026-09-25: The token's q8_1 rows, written once into `a_q8` and read
        // by the routed experts and the shared w1 / w3 projections (the side
        // stream after `ev_in`).
        if m1_arm {
            self.ffn_input_q8_m1(gpu, x, stream)?;
        }
        if side {
            gpu.record_event(self.ev_in, stream)?;
        }
        self.route_launch(gpu, w, x, m, stream)?;
        if side {
            gpu.stream_wait_event(self.side, self.ev_in)?;
            self.shared_expert_body(gpu, w, x, 1, self.a_q8, self.sh_q8, true, self.side)?;
            gpu.record_event(self.ev_out, self.side)?;
        }
        // 2026-09-25: `METRALE_DS41_DEVICE_ROUTE=1` on a layer that
        // `device_route_ok` accepts: the single-token selection, plan and
        // pointer table are computed on the device (`device_route.rs`).
        let dev_route = m1_arm && self.device_route_ok(w);
        let predict = next.filter(|_| m == 1 && !dev_route && prefetch_k() > 0 && lru.has_pool());
        if let Some(nw) = predict {
            self.predict_launch(gpu, nw, x, stream)?;
        }
        let (weights, indices, plan, slots, before, after, t1) = if dev_route {
            let before = lru.stats();
            let (weights, indices) =
                self.route_device_m1(gpu, w, lru, src, reader_threads, stream)?;
            let after = lru.stats();
            let t1 = std::time::Instant::now();
            (weights, indices, Vec::new(), Vec::new(), before, after, t1)
        } else {
            let (weights, indices) = self.route_select(gpu, w, m, stream)?;
            let predicted = match predict {
                Some(nw) => self.predict_select(gpu, nw, prefetch_k(), stream)?,
                None => Vec::new(),
            };
            let t1 = std::time::Instant::now();
            // 2026-09-25: The batch's experts, fetched once; `predicted` (the
            // next layer's likely experts) is non-empty only when prefetching.
            lru.begin_token();
            let before = lru.stats();
            let keys: Vec<(u32, u32)> = indices.iter().map(|&e| (w.layer, e as u32)).collect();
            let slots = lru.fetch_many_on(gpu, stream, src, &keys, &predicted, reader_threads)?;
            let after = lru.stats();
            let (rows_host, w_host, plan) = self.plan(&indices, &weights, m);
            gpu.copy_h2d_async(&rows_host, self.rows_dev, stream)?;
            gpu.copy_h2d_async(&w_host, self.weight_dev, stream)?;
            (weights, indices, plan, slots, before, after, t1)
        };
        let t2 = std::time::Instant::now();
        gpu.memset_async(self.acc, 0, m * c.dim * 4, stream)?;
        if m1_arm {
            // 2026-09-25: The single-token arm: every selected expert reads the
            // same q8_1 token row, gate and up run as one launch over the
            // pointer table and down as another, the routing weight is folded
            // in at the SwiGLU, and the expert rows are summed into `acc` in
            // plan order (`routed_m1`).
            let ne = if dev_route {
                c.topk
            } else {
                self.upload_expert_table(gpu, &plan, &slots, stream)?
            };
            self.routed_m1(gpu, x, ne, true, stream)?;
        }
        // 2026-09-25: Groups of at most eight rows take the decode GEMV (q8_1
        // blocks of 32). Larger groups take the MMQ arm (D2S6 activations,
        // block 64, for the Q2_K gate / up; D4, block 32, for the Q3_K down),
        // or with `METRALE_DS41_PREFILL_GEMV=1` the decode GEMV in chunks of
        // at most eight rows.
        let prefill_gemv = prefill_gemv();
        for &(a0, off, r) in plan.iter().filter(|_| !m1_arm) {
            let slot = slots[a0];
            let rows_ptr = DevicePtr(self.rows_dev.0 + (off * 4) as u64);
            let w_ptr = DevicePtr(self.weight_dev.0 + (off * 4) as u64);
            KernelLaunch::new(gpu, self.k.gather)
                .grid([r as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(rows_ptr)
                .arg_ptr(self.a_rows)
                .arg_u32(c.dim as u32)
                .launch(stream)?;
            if prefill_gemv && r > 8 {
                // 2026-09-25: Chunks of at most 8 rows, each an offset view
                // into the group's buffers.
                let q8_row = (c.dim / 32) * Q8_1_BLOCK_BYTES;
                let mut r0 = 0usize;
                while r0 < r {
                    let rr = (r - r0).min(8);
                    let a = DevicePtr(self.a_rows.0 + (r0 * c.dim * 2) as u64);
                    let q = DevicePtr(self.a_q8.0 + (r0 * q8_row) as u64);
                    let go = DevicePtr(self.gate_out.0 + (r0 * c.inter * 2) as u64);
                    let uo = DevicePtr(self.up_out.0 + (r0 * c.inter * 2) as u64);
                    kquant_q8_1_rows(gpu, self.k.q8_rows, a, q, rr as u32, c.dim as u32, stream)?;
                    kquant_mmvq_w(
                        gpu,
                        self.k.mmvq_q2k,
                        slot.gate,
                        q,
                        go,
                        c.inter as u32,
                        c.dim as u32,
                        rr as u32,
                        stream,
                    )?;
                    kquant_mmvq_w(
                        gpu,
                        self.k.mmvq_q2k,
                        slot.up,
                        q,
                        uo,
                        c.inter as u32,
                        c.dim as u32,
                        rr as u32,
                        stream,
                    )?;
                    r0 += rr;
                }
            } else if r <= 8 {
                kquant_q8_1_rows(
                    gpu,
                    self.k.q8_rows,
                    self.a_rows,
                    self.a_q8,
                    r as u32,
                    c.dim as u32,
                    stream,
                )?;
                kquant_mmvq_w(
                    gpu,
                    self.k.mmvq_q2k,
                    slot.gate,
                    self.a_q8,
                    self.gate_out,
                    c.inter as u32,
                    c.dim as u32,
                    r as u32,
                    stream,
                )?;
                kquant_mmvq_w(
                    gpu,
                    self.k.mmvq_q2k,
                    slot.up,
                    self.a_q8,
                    self.up_out,
                    c.inter as u32,
                    c.dim as u32,
                    r as u32,
                    stream,
                )?;
            } else {
                ops::quantize_act_q8_1(
                    gpu,
                    self.k.quant_d2s6,
                    self.a_rows,
                    self.a_q8,
                    r as u32,
                    c.dim as u32,
                    stream,
                )?;
                kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    slot.gate,
                    self.gate_out,
                    r as u32,
                    c.inter as u32,
                    c.dim as u32,
                    Q2K_MMQ_SMEM,
                    stream,
                )?;
                kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    slot.up,
                    self.up_out,
                    r as u32,
                    c.inter as u32,
                    c.dim as u32,
                    Q2K_MMQ_SMEM,
                    stream,
                )?;
            }
            self.launch_n(gpu, self.k.swiglu, r * c.inter, stream, |l| {
                l.arg_ptr(self.gate_out)
                    .arg_ptr(self.up_out)
                    .arg_ptr(w_ptr)
                    .arg_ptr(self.h)
                    .arg_u32(r as u32)
                    .arg_u32(c.inter as u32)
                    .arg_f32(c.swiglu_limit)
            })?;
            if prefill_gemv && r > 8 {
                let q8_row = (c.inter / 32) * Q8_1_BLOCK_BYTES;
                let mut r0 = 0usize;
                while r0 < r {
                    let rr = (r - r0).min(8);
                    let h = DevicePtr(self.h.0 + (r0 * c.inter * 2) as u64);
                    let q = DevicePtr(self.h_q8.0 + (r0 * q8_row) as u64);
                    let d = DevicePtr(self.down_out.0 + (r0 * c.dim * 2) as u64);
                    kquant_q8_1_rows(gpu, self.k.q8_rows, h, q, rr as u32, c.inter as u32, stream)?;
                    kquant_mmvq_w(
                        gpu,
                        self.k.mmvq_q3k,
                        slot.down,
                        q,
                        d,
                        c.dim as u32,
                        c.inter as u32,
                        rr as u32,
                        stream,
                    )?;
                    r0 += rr;
                }
            } else if r <= 8 {
                kquant_q8_1_rows(
                    gpu,
                    self.k.q8_rows,
                    self.h,
                    self.h_q8,
                    r as u32,
                    c.inter as u32,
                    stream,
                )?;
                kquant_mmvq_w(
                    gpu,
                    self.k.mmvq_q3k,
                    slot.down,
                    self.h_q8,
                    self.down_out,
                    c.dim as u32,
                    c.inter as u32,
                    r as u32,
                    stream,
                )?;
            } else {
                ops::quantize_act_q8_1(
                    gpu,
                    self.k.quant_d4,
                    self.h,
                    self.h_q8,
                    r as u32,
                    c.inter as u32,
                    stream,
                )?;
                kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q3k_nc,
                    self.k.mmq_q3k_wc,
                    self.h_q8,
                    slot.down,
                    self.down_out,
                    r as u32,
                    c.dim as u32,
                    c.inter as u32,
                    Q3K_MMQ_SMEM,
                    stream,
                )?;
            }
            KernelLaunch::new(gpu, self.k.scatter_add)
                .grid([r as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.acc)
                .arg_ptr(self.down_out)
                .arg_ptr(rows_ptr)
                .arg_u32(c.dim as u32)
                .launch(stream)?;
        }
        if side {
            gpu.stream_wait_event(stream, self.ev_out)?;
            self.shared_expert_tail(gpu, m, stream)?;
        } else {
            self.shared_expert(gpu, w, x, m, m1_arm, stream)?;
        }
        if self.timing_sync {
            // 2026-09-25: `METRALE_DS41_DIAG=1` (`init.rs`): the sync makes
            // `compute_ms` include the GPU time.
            gpu.synchronize(stream)?;
        }
        self.last.set(MoeV41Timing {
            route_ms: (t1 - t0).as_secs_f64() * 1e3,
            fetch_ms: (t2 - t1).as_secs_f64() * 1e3,
            compute_ms: t2.elapsed().as_secs_f64() * 1e3,
            hits: after.hits - before.hits,
            misses: after.misses - before.misses,
            bytes_read: after.bytes_read - before.bytes_read,
        });
        Ok((self.out, weights, indices))
    }

    /// 2026-09-25: The shared expert added into `acc`, then `acc` cast into
    /// `out`. Device work only.
    pub(super) fn shared_expert(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        x_pre: bool,
        stream: u64,
    ) -> Result<()> {
        self.shared_expert_body(gpu, w, x, m, self.a_q8, self.h_q8, x_pre, stream)?;
        self.shared_expert_tail(gpu, m, stream)
    }

    /// 2026-09-25: `acc += sd`, then `out = bf16(acc)`.
    pub(super) fn shared_expert_tail(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        self.launch_n(gpu, self.k.accumulate, m * c.dim, stream, |l| {
            l.arg_ptr(self.acc)
                .arg_ptr(self.sd)
                .arg_u32((m * c.dim) as u32)
        })?;
        self.launch_n(gpu, self.k.finish, m * c.dim, stream, |l| {
            l.arg_ptr(self.acc)
                .arg_ptr(self.out)
                .arg_u32((m * c.dim) as u32)
        })
    }
}
