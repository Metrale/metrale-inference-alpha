// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The conv1d + L2 norm + GDN phase of `decode_batched_inner`, routed on
//! `GdnStates`: one sequence through `decode_batched_conv_gdn`, or a batched MTP verify per
//! run of equal `ks` through `decode_batched_conv_gdn_multi`, falling back per sequence.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - For `GdnStates::Multi`, `num_tokens == Σ ks` and `ks.len() == states.len()` are checked
//!   before any conv/GDN launch.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-26: Runs the conv/GDN body for `gdn` and returns the conv output buffer and
    /// the GDN output buffer, `(conv_out_buf, gdn_out_buf)`, that the gated norm reads.
    pub(super) fn batched_conv_gdn_route(
        &self,
        gdn: GdnStates<'_, '_>,
        ctx: &ForwardContext,
        d: &BatchedDims,
        deinterleaved: DevicePtr,
        gates_buf: DevicePtr,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let BatchedDims {
            num_tokens,
            bf16,
            fp32,
            nk,
            kd,
            nv,
            vd,
            key_dim,
            value_dim,
            conv_dim,
            qk_ch,
            d_conv,
            qkvz_size,
            stream,
            ..
        } = *d;
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();
        // 2026-09-25: The pool's per-slot h pitch (see `ConvGdnArgs::h_bytes`), which is
        // `h_state_bytes` except under `--ssm-h-dtype f16-pool`.
        let h_bytes = self.h_slot_stride_bytes();
        let conv_bytes = self.conv_state_bytes;

        match gdn {
            GdnStates::Single(state) => {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
                // 2026-09-25: The conv/GDN body indexes K-1 h and K conv intermediates, so
                // a state without them returns an error here instead of panicking on the
                // index.
                if ssm_state.h_state_intermediates.len() + 1 < num_tokens
                    || ssm_state.conv_state_intermediates.len() < num_tokens
                {
                    anyhow::bail!(
                        "SSM MTP intermediate buffers not allocated (need K-1 h + K conv; \
                         h_state_intermediates.len()={}, \
                         conv_state_intermediates.len()={}, num_tokens={}). \
                         If this is an EP=2 worker, the head node is sending MTP verify commands \
                         but the worker was started without `--speculative` (and matching \
                         `--mtp-quantization`/`--num-drafts`). Add those flags to the worker invocation.",
                        ssm_state.h_state_intermediates.len(),
                        ssm_state.conv_state_intermediates.len(),
                        num_tokens,
                    );
                }

                let args = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                    num_tokens,
                    deinterleaved,
                    gates_buf,
                    conv_out_buf,
                    gdn_out_buf,
                    normed_out: conv_out_buf,
                    h_bytes,
                    conv_bytes,
                    qkvz_size,
                    conv_dim,
                    key_dim,
                    value_dim,
                    d_conv,
                    qk_ch,
                    nk,
                    nv,
                    kd,
                    vd,
                    bf16,
                    fp32,
                    stream,
                };
                self.decode_batched_conv_gdn(ssm_state, ctx, &args)?;
            }
            GdnStates::Multi {
                states,
                ks,
                wy_tables,
            } => {
                // 2026-09-25: Batched MTP verify. Each run of sequences with equal `ks` first
                // tries `decode_batched_conv_gdn_multi` (one batched conv launch and one
                // table-form WY launch). When it declines, each sequence of the run goes
                // through `decode_batched_conv_gdn` with its buffers offset to its first row:
                // deinterleaved rows `qkvz_size`, conv rows `conv_dim`, gate rows `nv * 2`
                // FP32 and GDN rows `value_dim` apart.
                anyhow::ensure!(
                    !states.is_empty()
                        && ks.len() == states.len()
                        && num_tokens == ks.iter().sum::<usize>(),
                    "decode_batched_inner Multi: num_tokens {} != Σ ks {:?} (n {})",
                    num_tokens,
                    ks,
                    states.len(),
                );
                // 2026-09-25: First row of each sequence (the prefix sum of `ks`).
                let mut off: Vec<usize> = Vec::with_capacity(states.len());
                let mut acc = 0usize;
                for &k in ks.iter() {
                    off.push(acc);
                    acc += k;
                }
                // 2026-09-25: One batched attempt per maximal run of adjacent sequences with
                // equal `ks`; a uniform batch is one run.
                let mut g0 = 0usize;
                while g0 < states.len() {
                    let kk = ks[g0];
                    let mut g1 = g0 + 1;
                    while g1 < states.len() && ks[g1] == kk {
                        g1 += 1;
                    }
                    let row0 = off[g0];
                    let run_args = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                        num_tokens: kk,
                        deinterleaved: deinterleaved.offset(row0 * qkvz_size * bf16),
                        gates_buf: gates_buf.offset(row0 * nv * 2 * fp32),
                        conv_out_buf: conv_out_buf.offset(row0 * conv_dim * bf16),
                        gdn_out_buf: gdn_out_buf.offset(row0 * value_dim * bf16),
                        // 2026-09-25: Normed rows are `value_dim` apart from row 0 of the
                        // out_proj input, not `conv_dim` like the conv rows.
                        normed_out: conv_out_buf.offset(row0 * value_dim * bf16),
                        h_bytes,
                        conv_bytes,
                        qkvz_size,
                        conv_dim,
                        key_dim,
                        value_dim,
                        d_conv,
                        qk_ch,
                        nk,
                        nv,
                        kd,
                        vd,
                        bf16,
                        fp32,
                        stream,
                    };
                    // 2026-09-25: Table entries are per sequence in batch order, 8 bytes
                    // each, so a run starts at entry `g0`.
                    let run_tables = if wy_tables.is_null() {
                        wy_tables
                    } else {
                        wy_tables.offset(g0 * 8)
                    };
                    let batched = self.decode_batched_conv_gdn_multi(
                        &mut states[g0..g1],
                        run_tables,
                        ctx,
                        &run_args,
                    )?;
                    if !batched {
                        for i in g0..g1 {
                            let ssm_state = states[i]
                                .as_any_mut()
                                .downcast_mut::<SsmLayerState>()
                                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
                            if ssm_state.h_state_intermediates.len() + 1 < kk
                                || ssm_state.conv_state_intermediates.len() < kk
                            {
                                anyhow::bail!(
                                    "SSM MTP intermediate buffers not allocated for batched \
                                 verify (seq {i}: h={}, conv={}, k={kk})",
                                    ssm_state.h_state_intermediates.len(),
                                    ssm_state.conv_state_intermediates.len(),
                                );
                            }
                            let r = off[i];
                            let args_i = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                                num_tokens: kk,
                                deinterleaved: deinterleaved.offset(r * qkvz_size * bf16),
                                gates_buf: gates_buf.offset(r * nv * 2 * fp32),
                                conv_out_buf: conv_out_buf.offset(r * conv_dim * bf16),
                                gdn_out_buf: gdn_out_buf.offset(r * value_dim * bf16),
                                normed_out: conv_out_buf.offset(r * value_dim * bf16),
                                h_bytes,
                                conv_bytes,
                                qkvz_size,
                                conv_dim,
                                key_dim,
                                value_dim,
                                d_conv,
                                qk_ch,
                                nk,
                                nv,
                                kd,
                                vd,
                                bf16,
                                fp32,
                                stream,
                            };
                            self.decode_batched_conv_gdn(ssm_state, ctx, &args_i)?;
                        }
                    }
                    g0 = g1;
                }
            }
        }
        Ok((conv_out_buf, gdn_out_buf))
    }
}
