// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The weight overlays a loader installs on a `DenseFfnLayer` (FP8, fused FP8
//! gate+up, BF16, packed Q2, LoRA) and the LoRA delta launches the forward paths call.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `set_lora_weights` refuses an adapter while an FP8, BF16 or packed-Q2 overlay is
//!   installed, or when the layer cannot take the split-SiLU decode path.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{
    DenseFfnLayer, DenseFfnWeightsBf16, DenseFfnWeightsFp8, DenseFfnWeightsQ2, FfnActivation,
};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, PackedQ2Weight};

impl DenseFfnLayer {
    /// 2026-09-25: Install native block-scaled FP8 dense MLP weights. The FP8 kernel handles are
    /// optional lookups (zero when absent), so the caller installs these only on a target that
    /// carries the FP8 kernels.
    pub fn set_fp8_weights(&mut self, gate: Fp8Weight, up: Fp8Weight, down: Fp8Weight) {
        self.fp8_weights = Some(DenseFfnWeightsFp8 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// 2026-09-25: Install the `[2*inter, hidden]` FP8 gate+up weight that `gateup_fused_plan` can
    /// select. Contract: the `gate` and `up` passed to [`Self::set_fp8_weights`] are views into
    /// `fused` (`gate.weight == fused.weight`, `up.weight == fused.weight + inter*hidden`, and
    /// likewise for the scale grid). Only the first equality is checked, by `debug_assert!`.
    /// Otherwise the fused and unfused arms of one layer compute from different bytes.
    pub fn set_fp8_gate_up_fused(&mut self, fused: Fp8Weight) {
        debug_assert!(
            self.fp8_weights
                .as_ref()
                .is_some_and(|w| w.gate_proj.weight == fused.weight),
            "the fused gate+up weight must be the buffer gate_proj is a view into"
        );
        self.fp8_gate_up_fused = Some(fused);
    }

    /// 2026-09-25: Install the LoRA overlay for gate/up/down. Errors when an FP8, BF16 or packed-Q2
    /// overlay is installed (their branches return before the NVFP4 code that applies the
    /// deltas), or when the layer cannot take the split-SiLU decode path.
    pub fn set_lora_weights(&mut self, w: ops::lora_delta::LoraFfnWeights) -> Result<()> {
        anyhow::ensure!(
            self.fp8_weights.is_none() && self.bf16_weights.is_none(),
            "LoRA v0 supports only the NVFP4 dense-FFN path (FP8/BF16 weight \
             overlays installed on this layer)"
        );
        anyhow::ensure!(
            self.q2_weights.is_none(),
            "LoRA v0 supports only the NVFP4 dense-FFN path (packed-Q2 weights \
             installed on this layer)"
        );
        // 2026-09-25: The down delta needs `silu(gate)*up` materialised, which only the split-SiLU
        // decode path does; `forward` takes that path whenever an adapter is installed.
        anyhow::ensure!(
            self.activation == FfnActivation::SiLU && self.act_mul.0 != 0 && self.w4a16_gemv.0 != 0,
            "LoRA v0 needs the split-SiLU decode path (SiLU activation + \
             act_mul + w4a16_gemv kernels); this layer resolved activation \
             {:?}, act_mul={}, w4a16_gemv={}",
            self.activation,
            self.act_mul.0,
            self.w4a16_gemv.0,
        );
        self.lora = Some(w);
        Ok(())
    }

    /// 2026-09-25: `gate_out += ΔW_gate · x` and `up_out += ΔW_up · x` for `m` rows. Call after the
    /// gate/up projections and before the activation. No launches when the layer has no
    /// adapter or `METRALE_LORA_NO_FFN=1`.
    pub(super) fn apply_lora_gate_up(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if ops::lora_delta::lora_no_ffn() {
            return Ok(());
        }
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        for (pair, base) in [(&lw.gate, gate_out), (&lw.up, up_out)] {
            if let Some(pair) = pair.as_ref() {
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    pair,
                    input,
                    base,
                    m,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: `output += ΔW_down · act`, where `act` is the `silu(gate)*up` the down
    /// projection read (every caller passes `expert_gate_out`, which the activation overwrites in
    /// place). No launches in the same cases as `apply_lora_gate_up`.
    pub(super) fn apply_lora_down(
        &self,
        ctx: &ForwardContext,
        act: DevicePtr,
        output: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if ops::lora_delta::lora_no_ffn() {
            return Ok(());
        }
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        let Some(ref pair) = lw.down else {
            return Ok(());
        };
        ops::lora_delta::apply_lora_delta(
            ctx.gpu,
            &lw.kernels,
            pair,
            act,
            output,
            m,
            ctx.buffers.lora_xa(),
            ctx.buffers.lora_delta(),
            stream,
        )
    }

    /// 2026-09-25: Install packed Q2_0 dense MLP weights and look up the Q2_0 MMQ kernels. A
    /// missing `q2_0_gemv_vec`, `q2_0_gemv_vec_batchm` or dequant kernel makes the path that needs
    /// it return an error.
    pub fn set_q2_weights(
        &mut self,
        gate: PackedQ2Weight,
        up: PackedQ2Weight,
        down: PackedQ2Weight,
        gpu: &dyn GpuBackend,
    ) {
        self.q2_weights = Some(DenseFfnWeightsQ2 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
        self.q2_0_mmq_nc_k = super::try_kernel(gpu, "q2_0_mmq", "metrale_q2_0_mmq128_nc");
        self.q2_0_mmq_wc_k = super::try_kernel(gpu, "q2_0_mmq", "metrale_q2_0_mmq128_wc");
    }

    /// 2026-09-25: Install BF16 dense MLP weights. The BF16 kernel handles are optional lookups, so
    /// the caller installs these only on a target that carries them. `forward_k2`, `forward_k3`
    /// and `forward_km` then hand the layer to `forward_prefill`, so no NVFP4 kernel reads its
    /// NVFP4 weights.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }
}
