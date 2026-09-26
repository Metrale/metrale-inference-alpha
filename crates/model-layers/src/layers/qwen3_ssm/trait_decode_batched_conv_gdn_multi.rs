// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cross-sequence batched conv + WY for one run of a batched MTP verify
//! (`decode_batched_conv_gdn_multi`): the n sequences of a run share the verify width k
//! and are covered by two launches, `gdn_verify_fused_conv_kn_batched` and a
//! pointer-table WY kernel (wy2, wy3, wy4, the write-on-accept wy4 twin, or the wyN table
//! twin for k = 5..=16).
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - The conv and WY launches run only when n >= 2, the handles and WY tables exist, and
//!   the checks on the actual state pointers pass: conv states on consecutive slots, k
//!   conv intermediates `conv_bytes` apart and at least k-1 h intermediates per sequence,
//!   and per-sequence conv intermediate regions evenly spaced without overlap. Otherwise
//!   it returns `Ok(false)` and the caller runs each sequence alone.
//!
//! The conv launch writes all k conv intermediates, including index k-1, which the
//! per-row path skips. The WY tables are staged by the model
//! (`TransformerModel::upload_verify_wy_tables`) before the verify runs.
//! `METRALE_NO_VERIFY_GDN_BATCH` set to any value, `0` included, turns this path off.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::{LayerState, VERIFY_WY_TABLE_STRIDE_BYTES};
use crate::layers::ops;

/// 2026-09-25: Counts of batched calls that launched and of pointer-check declines
/// (`gdn_multi_decline`).
static BATCHED_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 2026-09-25: Bit `k` is set once the engaged (declined) line for verify width `k` has
/// been logged, so each width logs its first outcome of each kind.
static ENGAGED_KMASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static DECLINED_KMASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// 2026-09-25: Set bit `k & 31` in `mask`; true if this call set it.
fn first_for_k(mask: &std::sync::atomic::AtomicU32, k: usize) -> bool {
    let bit = 1u32 << (k & 31);
    (mask.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit) == 0
}

/// 2026-09-25: Every 2048 calls, log the engaged and declined counts at INFO. Only when
/// `METRALE_MTP_ACCEPT_DEBUG` is set (`speculative::mtp_accept_debug`), which is checked
/// first.
fn record_multi_rate(n: usize, kk: usize) {
    use std::sync::atomic::Ordering;
    const PERIOD: u64 = 2048;
    static SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !crate::speculative::mtp_accept_debug() {
        return;
    }
    if SINCE.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        SINCE.store(0, Ordering::Relaxed);
        let ok = BATCHED_OK.load(Ordering::Relaxed);
        let fb = FALLBACK.load(Ordering::Relaxed);
        tracing::info!(
            "batched-verify GDN conv+WY [last n={n} k={kk}]: engaged={ok} declined={fb} \
             declined_frac={:.3} (a declined call is {} launches/layer, not 2)",
            fb as f64 / (ok + fb).max(1) as f64,
            n * (kk + kk - 1),
        );
    }
}

/// 2026-09-25: `METRALE_NO_VERIFY_GDN_BATCH` set to any value, `0` included, disables
/// the batched path. Read once per process.
fn verify_gdn_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_VERIFY_GDN_BATCH").is_none())
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Try the batched conv + WY for one run of a `GdnStates::Multi` verify.
    /// `args` holds the run's buffers, offset to its first row, with
    /// `args.num_tokens = k`; `states` are the run's n sequences in batch order;
    /// `wy_tables` is this layer's table slice for the run (NULL declines).
    ///
    /// `Ok(false)` means nothing was launched for the run (apart from the write-on-accept
    /// clear) and the caller must run each sequence alone. Under `--exact-verify` it
    /// returns `decode_batched_conv_gdn_multi_exact`'s result instead. The pointer checks
    /// are host arithmetic.
    pub(super) fn decode_batched_conv_gdn_multi(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        wy_tables: DevicePtr,
        ctx: &crate::layer::ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<bool> {
        let n = states.len();
        let kk = args.num_tokens;
        // 2026-09-25: With write-on-accept requested, clear this layer's engaged word
        // before any decline, so the fold after the verdict sees 0 unless this call ran
        // the write-on-accept twin (woa.rs).
        if ctx.gdn_write_on_accept {
            self.woa_clear_at_entry(ctx.gpu, args.stream)?;
        }
        // 2026-09-25: `--exact-verify`: the strided exact arm. Its `Ok(false)` sends each
        // sequence to `decode_batched_conv_gdn`, which runs the per-row exact arm under the
        // same predicate.
        if super::verify_exact_enabled() {
            return self.decode_batched_conv_gdn_multi_exact(states, ctx, args);
        }
        // 2026-09-25: The WY handle for width k: the `wy2_kernel` / `wy3_kernel` /
        // `wy4_kernel` selectors, or the wyN table twin for 5..=16. An unresolved kernel is
        // a zero handle, which declines below.
        let wy_k = match kk {
            2 => self.wy2_kernel(args.kd, args.vd, n),
            3 => self.wy3_kernel(args.kd, args.vd, n),
            4 => self.wy4_kernel(),
            5..=16 => match self.wyn_table_kernel(kk, ctx.levers.gdn_wyn) {
                Some(h) => h,
                None => return Ok(false),
            },
            _ => return Ok(false),
        };
        // 2026-09-25: With an FP16 h-state a zero handle is an error here, not a decline.
        self.require_wy_f16(kk, wy_k)?;
        if n < 2
            || !verify_gdn_batch_enabled()
            || self.gdn_verify_fused_conv_kn_batched_k.0 == 0
            || wy_k.0 == 0
            || wy_tables.is_null()
        {
            return Ok(false);
        }

        let conv_bytes = self.conv_state_bytes;
        // 2026-09-25: Layout checks on the actual state pointers.
        let mut conv_base = DevicePtr::NULL;
        let mut inter_base = DevicePtr::NULL;
        let mut inter_seq_stride = 0u64;
        for i in 0..n {
            let Some(st) = states[i].as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(self.gdn_multi_decline(n, kk));
            };
            // 2026-09-25: The batched conv writes conv intermediates 0..k-1, `conv_bytes`
            // apart, so all k must exist there. The WY tables read h intermediates
            // 0..k-2; `upload_verify_wy_tables` checks the same count.
            if st.conv_state_intermediates.len() < kk || st.h_state_intermediates.len() < kk - 1 {
                return Ok(self.gdn_multi_decline(n, kk));
            }
            let i0 = st.conv_state_intermediates[0];
            for t in 1..kk {
                if st.conv_state_intermediates[t].0 != i0.0 + (t * conv_bytes) as u64 {
                    return Ok(self.gdn_multi_decline(n, kk));
                }
            }
            if i == 0 {
                conv_base = st.conv_state;
                inter_base = i0;
            } else {
                if st.conv_state.0 != conv_base.0 + (i * conv_bytes) as u64 {
                    return Ok(self.gdn_multi_decline(n, kk));
                }
                if i == 1 {
                    inter_seq_stride = i0.0.wrapping_sub(inter_base.0);
                    // 2026-09-25: The next sequence's region must not overlap this one's k
                    // intermediates.
                    if inter_seq_stride < (kk * conv_bytes) as u64 {
                        return Ok(self.gdn_multi_decline(n, kk));
                    }
                } else if i0.0 != inter_base.0 + (i as u64) * inter_seq_stride {
                    return Ok(self.gdn_multi_decline(n, kk));
                }
            }
        }

        let ConvGdnArgs {
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            qkvz_size,
            conv_dim,
            key_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
            ..
        } = *args;

        // 2026-09-25: One launch: conv1d + L2 norm for n sequences × k rows, writing every
        // conv intermediate.
        ops::gdn_verify_fused_conv_kn_batched(
            ctx.gpu,
            self.gdn_verify_fused_conv_kn_batched_k,
            conv_base,
            deinterleaved,
            &self.ssm.conv1d,
            conv_out_buf,
            inter_base,
            kk as u32,
            conv_dim as u32,
            d_conv as u32,
            qk_ch,
            kd as u32,
            qkvz_size as u32,
            conv_dim as u32,
            (conv_bytes / 4) as u32,
            1e-6,
            n as u32,
            (conv_bytes / 4) as u32,
            (kk * qkvz_size) as u32,
            (kk * conv_dim) as u32,
            (inter_seq_stride / 4) as u32,
            stream,
        )?;

        // 2026-09-25: One WY launch over the n sequences. Rows are sequence-major
        // (`b * k + t`); the state arguments are pointer tables (h, Hi0, ...),
        // `VERIFY_WY_TABLE_STRIDE_BYTES` apart. wy2 reads h and Hi0, wy3 adds Hi1, wy4
        // adds Hi2.
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gate_ptr = gates_buf;
        let beta_ptr = gates_buf.offset(nv * fp32);
        let hi = |t: usize| wy_tables.offset(t * VERIFY_WY_TABLE_STRIDE_BYTES);
        // 2026-09-25: Write-on-accept: only when the caller requests it
        // (`ctx.gdn_write_on_accept`) and `woa::woa_decision` agrees, the K=4 twin runs
        // instead and the model folds the accepted rows after the verdict.
        let woa_now = self.woa_now(ctx.gdn_write_on_accept, kk, n);
        if woa_now {
            self.woa_launch(
                ctx.gpu,
                wy_tables,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                n,
                conv_dim,
                stream,
            )?;
        } else {
            match kk {
                2 => ops::gdn_decode_wy2(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    true,
                    stream,
                )?,
                3 => ops::gdn_decode_wy3(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    hi(2),
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    true,
                    stream,
                )?,
                4 => ops::gdn_decode_wy4(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    hi(2),
                    hi(3),
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    true,
                    stream,
                )?,
                // 2026-09-25: k = 5..=16: the wyN table twin gets slab 0 (h) and slab 1
                // (Hi0), and finds Hi_t `VERIFY_WY_TABLE_SEQS` entries after Hi_{t-1}.
                _ => ops::gdn_decode_wyn_table(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    crate::layer::VERIFY_WY_TABLE_SEQS as u32,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    stream,
                )?,
            }
        }

        let ok = BATCHED_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        record_multi_rate(n, kk);
        if first_for_k(&ENGAGED_KMASK, kk) {
            tracing::info!(
                "batched-verify GDN conv+WY ENGAGED (n={n}, k={kk}): per-layer {} launches \
                 -> 2 (batched conv kernel + table-form wy{kk}); count logged at debug",
                n * (kk + kk - 1),
            );
        }
        if ok.is_multiple_of(1024) {
            tracing::debug!("batched-verify GDN conv+WY engaged x{ok}");
        }
        Ok(true)
    }

    /// 2026-09-25: Count a pointer-check decline and log the first one per width.
    /// Always returns `false`.
    fn gdn_multi_decline(&self, n: usize, kk: usize) -> bool {
        let n_fb = FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        record_multi_rate(n, kk);
        if first_for_k(&DECLINED_KMASK, kk) {
            tracing::info!(
                "batched-verify GDN conv+WY DECLINED (n={n}, k={kk}): batch is not on \
                 consecutive ssm-pool slots (or intermediates/tables unavailable) — running \
                 the per-sequence loop. Slots fragment as sequences finish, so this can \
                 recur; count logged at debug."
            );
        }
        tracing::debug!("batched-verify GDN conv+WY fallback #{n_fb} (n={n}, k={kk})");
        false
    }
}
