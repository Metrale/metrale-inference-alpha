// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Nemotron-H Mamba-2 SSM layer (no FFN), implementing
//! `TransformerLayer`.
//!
//! Decode, per token (`trait_impl.rs`):
//!   1. RMS norm, the layer input copied to `residual`
//!   2. in_proj GEMV into one row `[z (d_inner) | xBC (d_xbc) | dt (num_heads)]`
//!   3. causal conv1d update on xBC, with bias and a fused SiLU
//!   4. xBC split into x, B, C
//!   5. Mamba-2 SSM decode (state update and output y)
//!   6. gated RMS norm of y with gate z, in the order of the resolved
//!      `gated_rms_norm` kernel: the gb10 Nemotron trees' kernels gate first,
//!      `y * silu(z)`, then normalise each `d_inner / n_groups` slice
//!   7. out_proj GEMV
//!   8. residual add into `hidden`
//!
//! Prefill runs the same steps over all tokens (`prefill.rs`, projection arms
//! in `prefill_proj.rs`).
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, NemotronSsmWeights, QuantizedWeight,
};

mod prefill;
mod prefill_proj;
mod trait_impl;

#[allow(dead_code)]
pub struct NemotronMamba2Layer {
    input_norm: DenseWeight,
    ssm: NemotronSsmWeights,
    // 2026-09-25: The checkpoint's own block-scaled FP8 projections. Without
    // them an FP8 checkpoint's projections are dequantized to BF16 and
    // requantized to NVFP4 at load.
    in_proj_fp8: Option<Fp8Weight>,
    out_proj_fp8: Option<Fp8Weight>,
    // 2026-09-25: Whether prefill may use the native FP8 weights above. False
    // in the `METRALE_NEMOTRON_NATIVE_FP8_SSM=decode` mode, where the native
    // weights are installed for decode only and the requantized NVFP4 copies
    // are still built and used by prefill. Prefill keys off this flag, not off
    // `in_proj_fp8.is_some()`.
    native_fp8_prefill: bool,
    // 2026-09-25: Transposed NVFP4 projections for the `w4a16_gemm_t` prefill arms.
    in_proj_t: Option<QuantizedWeight>,
    out_proj_t: Option<QuantizedWeight>,
    // 2026-09-25: Pre-dequantized FP8 E4M3 copies of the two SSM projections,
    // [N, K], for the prefill `fp8_gemm_t_k` / `fp8_fp8_gemm_t_k` arms.
    in_proj_pd_fp8: Option<DevicePtr>,
    out_proj_pd_fp8: Option<DevicePtr>,
    // 2026-09-25: The checkpoint's own BF16 projections, for a layer the
    // checkpoint left unquantized; used instead of requantizing them to NVFP4.
    in_proj_bf16: Option<DenseWeight>,
    out_proj_bf16: Option<DenseWeight>,
    rms_norm_residual_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// 2026-09-25: Single-warp `w4a16_gemv_sw`; `KernelHandle(0)` when the
    /// kernel is missing, and then decode uses the base GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    w8a16_gemv_k: KernelHandle,
    conv1d_update_k: KernelHandle,
    mamba2_ssm_k: KernelHandle,
    gated_rms_norm_k: KernelHandle,
    residual_add_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    // 2026-09-25: Native FP8 block-scaled prefill GEMM (paired with in_proj_fp8/out_proj_fp8).
    w8a16_gemm_k: KernelHandle,
    w8a16_gemm_pipelined_k: KernelHandle,
    w4a16_gemm_t_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    fp8_gemm_t_k: KernelHandle,
    fp8_fp8_gemm_t_k: KernelHandle,
    // 2026-09-25: Native-BF16 projection kernels (paired with in_proj_bf16/out_proj_bf16).
    dense_gemm_bf16_k: KernelHandle,
    dense_gemv_bf16_k: KernelHandle,
    bf16_to_fp8_k: KernelHandle,
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
    conv1d_prefill_k: KernelHandle,
    conv1d_prefill_tp_k: KernelHandle,
    mamba2_ssm_prefill_k: KernelHandle,
    mamba2_ssm_prefill_persistent_k: KernelHandle,
    // 2026-09-25: SSD chunked prefill scan (`ops::ssm_ssd`, chunks of `SSD_L` tokens).
    ssd_cumsum_k: KernelHandle,
    ssd_bmm_k: KernelHandle,
    ssd_scan_k: KernelHandle,
    d_inner: usize,
    d_xbc: usize,
    in_proj_size: usize,
    num_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
    d_conv: usize,
    h_state_bytes: usize,
    conv_state_bytes: usize,
    layer_idx: usize,
}

impl NemotronMamba2Layer {
    pub fn new(
        input_norm: DenseWeight,
        ssm: NemotronSsmWeights,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
        layer_idx: usize,
    ) -> Result<Self> {
        let num_heads = config.mamba_num_heads;
        let head_dim = config.mamba_head_dim;
        let state_size = config.ssm_state_size;
        let n_groups = config.n_groups;
        let d_conv = config.linear_conv_kernel_dim;
        let d_inner = config.mamba2_d_inner();
        let d_xbc = config.mamba2_d_xbc();
        let in_proj_size = config.mamba2_in_proj_size();

        Ok(Self {
            input_norm,
            ssm,
            in_proj_fp8: None,
            out_proj_fp8: None,
            native_fp8_prefill: false,
            in_proj_t: None,
            out_proj_t: None,
            in_proj_pd_fp8: None,
            out_proj_pd_fp8: None,
            in_proj_bf16: None,
            out_proj_bf16: None,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_sw",
            ),
            w8a16_gemv_k: metrale_model_layers::layers::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            conv1d_update_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            mamba2_ssm_k: gpu.kernel("mamba2_ssm", "mamba2_ssm_decode")?,
            gated_rms_norm_k: gpu.kernel("norm", "gated_rms_norm")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w8a16_gemm_k: metrale_model_layers::layers::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            w4a16_gemm_t_k: metrale_model_layers::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            w4a16_gemm_t_m128_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128",
            ),
            fp8_gemm_t_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16",
                "fp8_gemm_t_m128_mfast",
            ),
            fp8_fp8_gemm_t_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16",
                "fp8_fp8_gemm_t_m128_mfast",
            ),
            dense_gemm_bf16_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            dense_gemv_bf16_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemv",
                "dense_gemv_bf16",
            ),
            bf16_to_fp8_k: metrale_model_layers::layers::try_kernel(gpu, "w4a16", "bf16_to_fp8"),
            w4a4_gemm_k: metrale_model_layers::layers::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_bf16_to_nvfp4",
            ),
            conv1d_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv1d_prefill_tp_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp",
            ),
            mamba2_ssm_prefill_k: gpu.kernel("mamba2_ssm", "mamba2_ssm_prefill")?,
            ssd_cumsum_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "mamba2_ssd_chunk",
                "mamba2_ssd_cumsum",
            ),
            ssd_bmm_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "mamba2_ssd_chunk",
                "mamba2_ssd_bmm",
            ),
            ssd_scan_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "mamba2_ssd_chunk",
                "mamba2_ssd_scan",
            ),
            mamba2_ssm_prefill_persistent_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "mamba2_ssm",
                "mamba2_ssm_prefill_persistent",
            ),
            d_inner,
            d_xbc,
            in_proj_size,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            d_conv,
            h_state_bytes: num_heads * head_dim * state_size * 4,
            conv_state_bytes: d_xbc * d_conv * 4,
            layer_idx,
        })
    }

    /// 2026-09-25: Install the checkpoint's native FP8 projections. Decode then
    /// uses `w8a16_gemv`, and when `prefill` is true the prefill GEMMs use
    /// `w8a16_gemm_pipelined` (or `w8a16_gemm`) ahead of the NVFP4 / W4A4 arms.
    ///
    /// Inputs must be tagged `WeightQuantFormat::Fp8BlockScaled`: the w8a16
    /// kernels index `block_scale[n_block * k_blocks + k_block]` with 128-wide
    /// blocks, so a per-row `[N]` scale or a scalar `weight_scale` would be read
    /// past its end. A wrong tag panics (`expect`); a missing `w8a16_gemv`, or
    /// with `prefill` a missing prefill GEMM, is an error here at load.
    ///
    /// When `prefill` is false (`METRALE_NEMOTRON_NATIVE_FP8_SSM=decode`) only
    /// `w8a16_gemv` reads these weights and prefill uses the NVFP4 /
    /// pre-dequantized copies, which the loader builds in that mode.
    pub fn set_fp8_weights(
        &mut self,
        in_proj: Option<Fp8Weight>,
        out_proj: Option<Fp8Weight>,
        prefill: bool,
    ) -> Result<()> {
        use metrale_model_layers::weight_map::WeightQuantFormat;
        if let Some(ref w) = in_proj {
            w.scale_format.expect(
                WeightQuantFormat::Fp8BlockScaled,
                "nemotron mamba2 in_proj (w8a16 expects [ceil(N/128),ceil(K/128)] FP32 block scales)",
            );
        }
        if let Some(ref w) = out_proj {
            w.scale_format.expect(
                WeightQuantFormat::Fp8BlockScaled,
                "nemotron mamba2 out_proj (w8a16 expects [ceil(N/128),ceil(K/128)] FP32 block scales)",
            );
        }
        anyhow::ensure!(
            self.w8a16_gemv_k.0 != 0,
            "native FP8 SSM requires the w8a16_gemv kernel (decode)"
        );
        anyhow::ensure!(
            !prefill || self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0,
            "native FP8 SSM requires w8a16_gemm[_pipelined] (prefill)"
        );
        self.in_proj_fp8 = in_proj;
        self.out_proj_fp8 = out_proj;
        self.native_fp8_prefill = prefill;
        Ok(())
    }

    /// 2026-09-25: The layer's Mamba-2 mixer weights.
    pub fn ssm_weights(&self) -> &NemotronSsmWeights {
        &self.ssm
    }

    /// 2026-09-25: Install transposed NVFP4 projections: prefill then uses
    /// `w4a16_gemm_t` (or, above 128 tokens and when it resolved,
    /// `w4a16_gemm_t_m128`) instead of the plain `w4a16_gemm`, unless an
    /// earlier arm in `prefill_proj.rs` applies.
    pub fn set_prefill_weights(
        &mut self,
        in_proj_t: Option<QuantizedWeight>,
        out_proj_t: Option<QuantizedWeight>,
    ) {
        self.in_proj_t = in_proj_t;
        self.out_proj_t = out_proj_t;
    }

    /// 2026-09-25: Install the checkpoint's own BF16 projections; decode and
    /// prefill then use the dense BF16 kernels ahead of every other arm. Valid
    /// only when both projections are BF16 in the checkpoint and the dense
    /// kernels resolved; the caller checks that (`bf16_native_ready`).
    pub fn set_bf16_weights(&mut self, in_proj: DenseWeight, out_proj: DenseWeight) {
        self.in_proj_bf16 = Some(in_proj);
        self.out_proj_bf16 = Some(out_proj);
    }

    /// 2026-09-25: Whether this layer can run natively BF16 (weights installed
    /// and both dense kernels present).
    pub fn bf16_native_ready(&self) -> bool {
        self.in_proj_bf16.is_some()
            && self.out_proj_bf16.is_some()
            && self.dense_gemm_bf16_k.0 != 0
            && self.dense_gemv_bf16_k.0 != 0
    }

    /// 2026-09-25: Install pre-dequantized FP8 E4M3 copies of in_proj /
    /// out_proj for prefill; their arms come after the native and W4A4 arms
    /// and before the transposed NVFP4 ones (`prefill_proj.rs`).
    pub fn set_fp8_prefill_weights(&mut self, in_proj: DevicePtr, out_proj: DevicePtr) {
        self.in_proj_pd_fp8 = Some(in_proj);
        self.out_proj_pd_fp8 = Some(out_proj);
    }

    /// 2026-09-25: Conv1d update with the layer's conv1d bias.
    ///
    /// Kernel: `causal_conv1d_update(conv_state, input, weight, bias, output,
    ///          batch, dim, d_conv)`
    fn conv1d_update_biased(
        &self,
        gpu: &dyn GpuBackend,
        conv_state: DevicePtr,
        input: DevicePtr,
        output: DevicePtr,
        d_inner: u32,
        d_conv: u32,
        batch_size: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.conv1d_update_k)
            .grid([div_ceil(d_inner, 256), batch_size, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_ptr(self.ssm.conv1d_weight.weight)
            .arg_ptr(self.ssm.conv1d_bias.weight)
            .arg_ptr(output)
            .arg_u32(batch_size)
            .arg_u32(d_inner)
            .arg_u32(d_conv)
            .launch(stream)
    }

    /// 2026-09-25: Launch the Mamba-2 SSM decode kernel.
    ///
    /// Grid: (num_heads, batch, 1)  Block: (state_size, 1, 1)
    #[allow(clippy::too_many_arguments)]
    fn ssm_decode(
        &self,
        gpu: &dyn GpuBackend,
        h_state: DevicePtr,
        x: DevicePtr,
        b_proj: DevicePtr,
        c_proj: DevicePtr,
        dt_raw: DevicePtr,
        output: DevicePtr,
        batch_size: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.mamba2_ssm_k)
            .grid([self.num_heads as u32, batch_size, 1])
            .block([self.state_size as u32, 1, 1])
            .arg_ptr(h_state)
            .arg_ptr(x)
            .arg_ptr(b_proj)
            .arg_ptr(c_proj)
            .arg_ptr(dt_raw)
            .arg_ptr(self.ssm.a_log.weight)
            .arg_ptr(self.ssm.d_param.weight)
            .arg_ptr(self.ssm.dt_bias.weight)
            .arg_ptr(output)
            .arg_u32(batch_size)
            .arg_u32(self.num_heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_u32(self.state_size as u32)
            .arg_u32(self.n_groups as u32)
            .arg_f32(1e-9)
            .arg_f32(1e9)
            .launch(stream)
    }
}
