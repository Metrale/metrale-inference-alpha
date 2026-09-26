// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Write-on-accept for the batched K=4 GDN verify
//! (`gated_delta_rule_wy4_woa` plus `gated_delta_rule_wy4_fold`): the
//! layer-side state, the engage decision as a pure function, the one-time
//! stash binding and the post-verdict fold.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - The twin runs only when `woa_decision` holds, which requires the
//!   caller's request.
//!
//! ## The engaged word
//!
//! The batched verify replays as a CUDA graph, so the record of which kernel
//! ran for a layer is a device word written inside the graph: the woa twin
//! sets it to 1. A batched verify that requests write-on-accept clears it
//! first, inside the same capture and before this layer's GDN, so the word
//! describes that launch only. A step that ran the twin and never folded
//! cannot leak its stash into a later fold: the next requesting verify
//! clears the word before its GDN runs.
//!
//! ## Who asks
//!
//! The caller asks per verify (`ForwardContext::gdn_write_on_accept`, from
//! `VerifyBatchedOpts::write_on_accept`); only the DFlash batched step sets
//! it. The MTP batched K-row verify uses the default `false`, commits
//! through its own verdict path without a fold, and gets the parent wy4,
//! which writes every intermediate.
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use std::sync::atomic::Ordering;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layers::ops;

/// 2026-09-25: The three kernel handles, each `KernelHandle(0)` when the
/// module is absent (`try_kernel`); the decision below needs all three.
pub(super) struct WoaKernels {
    pub woa_k: KernelHandle,
    pub fold_k: KernelHandle,
    pub clear_k: KernelHandle,
}

pub(super) fn woa_kernels(gpu: &dyn GpuBackend) -> WoaKernels {
    let m = "gated_delta_rule_wy4_woa";
    WoaKernels {
        woa_k: crate::layers::try_kernel(gpu, m, "gated_delta_rule_wy4_woa"),
        fold_k: crate::layers::try_kernel(gpu, m, "gated_delta_rule_wy4_fold"),
        clear_k: crate::layers::try_kernel(gpu, m, "gated_delta_rule_wy4_flag_clear"),
    }
}

/// 2026-09-25: Per-sequence stash width in floats:
/// `vn[4][nv][vd] | g[4][nv] | sk[4][nk][kd]`, the layout of the kernel's
/// `WOA_VN`/`WOA_G`/`WOA_SK` macros. 131,840 bytes per sequence at 48
/// v-heads, 16 k-heads and 128-dim heads.
pub(super) const fn stash_seq_floats(nk: usize, nv: usize, kd: usize, vd: usize) -> usize {
    4 * (nv * vd + nv + nk * kd)
}

/// 2026-09-25: Everything the engage decision depends on, so tests can pin it
/// without a GPU. `kd`/`vd` must be 128: the kernels are built for 128 x 128
/// and leave that check to the host.
#[derive(Clone, Copy, Debug)]
pub(super) struct WoaRequest {
    /// 2026-09-25: `ForwardContext::gdn_write_on_accept`: the caller will fold.
    pub requested: bool,
    pub kk: usize,
    /// 2026-09-25: `METRALE_GDN_WOA=1` (opt-in, `gdn_flags::gdn_woa_enabled`).
    pub enabled: bool,
    /// 2026-09-25: FP16 h-state (`--ssm-h-dtype f16`); the twins read FP32.
    pub h_f16: bool,
    pub kd: usize,
    pub vd: usize,
    pub kernels_linked: bool,
    /// 2026-09-25: Sequences covered by the bound stash; 0 when none is bound.
    pub bound_seqs: usize,
    pub n: usize,
}

/// 2026-09-25: The one place that decides whether the K=4 write-on-accept
/// twin runs.
pub(super) const fn woa_decision(r: WoaRequest) -> bool {
    r.requested
        && r.kk == 4
        && r.enabled
        && !r.h_f16
        && r.kd == 128
        && r.vd == 128
        && r.kernels_linked
        && r.bound_seqs > 0
        && r.n <= r.bound_seqs
}

impl Qwen3SsmLayer {
    fn woa_flag(&self) -> DevicePtr {
        DevicePtr(self.woa_flag.load(Ordering::Acquire))
    }
    fn woa_stash(&self) -> DevicePtr {
        DevicePtr(self.woa_stash.load(Ordering::Acquire))
    }
    pub(super) fn woa_kernels_linked(&self) -> bool {
        self.gdn_wy4_woa_k.0 != 0 && self.gdn_wy4_fold_k.0 != 0 && self.gdn_wy4_clear_k.0 != 0
    }

    /// 2026-09-26: `LayerWriteOnAccept::gdn_woa_stash_seq_floats`: the stash
    /// width, or `None` when the kernels are missing, write-on-accept is off,
    /// or a head dim is not 128.
    pub(super) fn woa_stash_seq_floats_impl(&self) -> Option<usize> {
        let [nk, nv, kd, vd] = self.woa_dims;
        (self.woa_kernels_linked() && super::gdn_flags::gdn_woa_enabled() && kd == 128 && vd == 128)
            .then(|| stash_seq_floats(nk, nv, kd, vd))
    }

    /// 2026-09-26: `LayerWriteOnAccept::gdn_woa_bind`: record the flag word, the
    /// stash and the sequence count it covers.
    pub(super) fn woa_bind_impl(&self, flag: DevicePtr, stash: DevicePtr, seqs: usize) {
        self.woa_flag.store(flag.0, Ordering::Release);
        self.woa_stash.store(stash.0, Ordering::Release);
        self.woa_seqs.store(seqs, Ordering::Release);
    }

    pub(super) fn woa_now(&self, requested: bool, kk: usize, n: usize) -> bool {
        let [_, _, kd, vd] = self.woa_dims;
        woa_decision(WoaRequest {
            requested,
            kk,
            enabled: super::gdn_flags::gdn_woa_enabled(),
            h_f16: super::ssm_h_fp16_enabled(),
            kd,
            vd,
            kernels_linked: self.woa_kernels_linked(),
            bound_seqs: if self.woa_stash().is_null() {
                0
            } else {
                self.woa_seqs.load(Ordering::Acquire)
            },
            n,
        })
    }

    /// 2026-09-25: Clear the engaged word at the top of a batched verify that
    /// requested write-on-accept: one small launch per layer per step, inside
    /// the same capture. A no-op when no flag is bound or the clear kernel is
    /// missing.
    pub(super) fn woa_clear_at_entry(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let flag = self.woa_flag();
        if flag.is_null() || self.gdn_wy4_clear_k.0 == 0 {
            return Ok(());
        }
        ops::gdn_wy4_flag_clear(gpu, self.gdn_wy4_clear_k, flag, stream)
    }

    /// 2026-09-25: Launch the K=4 write-on-accept twin. A launch failure is
    /// returned as an error, not a fallback: the launch path (gpu-runtime
    /// `registry.rs`) raises the dynamic shared memory limit for any launch
    /// above 48 KB, so a refusal means a build or device mismatch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn woa_launch(
        &self,
        gpu: &dyn GpuBackend,
        wy_tables: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gate_ptr: DevicePtr,
        beta_ptr: DevicePtr,
        gdn_out_buf: DevicePtr,
        n: usize,
        conv_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let [nk, nv, kd, vd] = self.woa_dims;
        ops::gdn_decode_wy4_woa(
            gpu,
            self.gdn_wy4_woa_k,
            wy_tables,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            self.woa_stash(),
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            (nv * 2) as u32,
            stash_seq_floats(nk, nv, kd, vd) as u32,
            self.woa_flag(),
            stream,
        )?;
        static WOA_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !WOA_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "batched-verify GDN WRITE-ON-ACCEPT ENGAGED (n={n}, k=4): wy4_woa + post-verdict fold \
                 (state read once, written once per step)"
            );
        }
        Ok(())
    }

    /// 2026-09-26: `LayerWriteOnAccept::gdn_fold_accepted`. Returns `Ok(false)`,
    /// launching nothing, when write-on-accept is off, the fold kernel, flag
    /// or stash is missing, `h_table` is NULL, or `n` exceeds the bound
    /// sequences. Otherwise the fold kernel reads the engaged word: 1 applies
    /// the stashed rows 0..na to H; 0 performs the parent kernel's
    /// partial-accept restore from the Hi tables. The word is not cleared
    /// here; the next requesting verify clears it.
    pub(super) fn fold_accepted_impl(
        &self,
        gpu: &dyn GpuBackend,
        h_table: DevicePtr,
        na_tab: DevicePtr,
        k_rows: usize,
        n: usize,
        stream: u64,
    ) -> Result<bool> {
        let flag = self.woa_flag();
        let stash = self.woa_stash();
        if !super::gdn_flags::gdn_woa_enabled()
            || self.gdn_wy4_fold_k.0 == 0
            || flag.is_null()
            || stash.is_null()
            || h_table.is_null()
            || n > self.woa_seqs.load(Ordering::Acquire)
        {
            return Ok(false);
        }
        let [nk, nv, kd, vd] = self.woa_dims;
        let hi_tables = h_table.offset(crate::layer::VERIFY_WY_TABLE_STRIDE_BYTES);
        ops::gdn_wy4_fold(
            gpu,
            self.gdn_wy4_fold_k,
            h_table,
            stash,
            na_tab,
            hi_tables,
            crate::layer::VERIFY_WY_TABLE_SEQS as u32,
            flag,
            k_rows as u32,
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            stash_seq_floats(nk, nv, kd, vd) as u32,
            stream,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{WoaRequest, stash_seq_floats, woa_decision};

    const GO: WoaRequest = WoaRequest {
        requested: true,
        kk: 4,
        enabled: true,
        h_f16: false,
        kd: 128,
        vd: 128,
        kernels_linked: true,
        bound_seqs: 16,
        n: 16,
    };

    /// 2026-09-25: The DFlash batched step at K=4 with everything linked and
    /// bound.
    #[test]
    fn dflash_k4_request_engages() {
        assert!(woa_decision(GO));
        assert!(woa_decision(WoaRequest { n: 2, ..GO }));
    }

    /// 2026-09-25: A caller that did not ask never gets the twin, whatever
    /// else holds; the MTP batched K-row verify is such a caller.
    #[test]
    fn unrequested_never_engages() {
        for n in 2..=4 {
            assert!(!woa_decision(WoaRequest {
                requested: false,
                n,
                ..GO
            }));
        }
    }

    /// 2026-09-25: Each other condition declines on its own: width,
    /// `METRALE_GDN_WOA` off, FP16 h-state, head dims, missing kernels, no
    /// bound stash, and a batch wider than the bound stash.
    #[test]
    fn each_gate_declines_alone() {
        assert!(!woa_decision(WoaRequest { kk: 3, ..GO }));
        assert!(!woa_decision(WoaRequest { kk: 5, ..GO }));
        assert!(!woa_decision(WoaRequest {
            enabled: false,
            ..GO
        }));
        assert!(!woa_decision(WoaRequest { h_f16: true, ..GO }));
        assert!(!woa_decision(WoaRequest { kd: 64, ..GO }));
        assert!(!woa_decision(WoaRequest { vd: 256, ..GO }));
        assert!(!woa_decision(WoaRequest {
            kernels_linked: false,
            ..GO
        }));
        assert!(!woa_decision(WoaRequest {
            bound_seqs: 0,
            ..GO
        }));
        assert!(!woa_decision(WoaRequest {
            bound_seqs: 8,
            n: 9,
            ..GO
        }));
    }

    /// 2026-09-25: The stash width matches the kernel's layout macros:
    /// `vn[4][nv][vd] | g[4][nv] | sk[4][nk][kd]`.
    #[test]
    fn stash_width_matches_kernel_layout() {
        assert_eq!(
            stash_seq_floats(16, 48, 128, 128),
            4 * (48 * 128 + 48 + 16 * 128)
        );
        assert_eq!(stash_seq_floats(16, 48, 128, 128) * 4, 131_840);
    }
}
