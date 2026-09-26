// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load-time MoE setup: the pre-expert norm and GeGLU setters, the
//! routed/shared expert transposes for the prefill kernels (full, gate+up only,
//! unified, hybrid), and the CUTLASS grouped host tables.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: Install the pre-expert norm: the router reads the un-normed
    /// input and the experts read it normed with this weight (`forward`,
    /// `forward_prefill`). The Gemma-4 loader passes pre_feedforward_layernorm_2.
    pub fn set_pre_expert_norm(&mut self, norm: crate::weight_map::DenseWeight) {
        self.pre_expert_norm = Some(norm);
    }

    /// 2026-09-25: Switch the expert activation kernel `moe_act_mul` to GeGLU
    /// (`gelu_mul`) and set `gelu_activation`, which only disables the fused
    /// SiLU+FP8-quant kernel in `forward_prefill_fp8`. Fused decode kernels are
    /// not steered away by it.
    pub fn set_gelu_activation(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.moe_act_mul = gpu.kernel("gelu", "gelu_mul")?;
        self.gelu_activation = true;
        Ok(())
    }

    /// 2026-09-25: Transpose every routed expert's gate, up and down `[N, K/2]`
    /// weights and scales to `[K/2, N]` (the `*_ptrs_t` tables), and the shared
    /// expert's when it has one. The originals are kept, so the expert weights
    /// are held twice.
    pub fn transpose_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, true)
    }

    /// 2026-09-25: As `transpose_for_prefill`, for gate and up only:
    /// `down_ptrs_t` stays `None`, so prefill runs the fused transposed gate/up
    /// kernel and the untransposed grouped down GEMM.
    pub fn transpose_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, false)
    }

    pub(super) fn transpose_for_prefill_impl(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        include_down: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        let num_experts = self.weights.experts.len();
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        let mut down_t = Vec::with_capacity(num_experts);

        // 2026-09-25: The scale block is 32 for E8M0 (native MXFP4) routed experts
        // and 16 for NVFP4; the scale transpose must use the matching size.
        let routed_gs =
            if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                32
            } else {
                16
            };
        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
                if include_down {
                    down_t.push(QuantizedWeight::null());
                }
            } else {
                gate_t.push(
                    expert
                        .gate_proj
                        .transpose_for_gemm_gs(gpu, inter, h, routed_gs)?,
                );
                up_t.push(
                    expert
                        .up_proj
                        .transpose_for_gemm_gs(gpu, inter, h, routed_gs)?,
                );
                if include_down {
                    down_t.push(
                        expert
                            .down_proj
                            .transpose_for_gemm_gs(gpu, h, inter, routed_gs)?,
                    );
                }
            }
        }

        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        if include_down {
            self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        }

        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            if include_down {
                self.shared_down_t =
                    Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                        gpu,
                        h,
                        shared_inter,
                    )?);
            }
        }

        Ok(())
    }

    /// 2026-09-25: Unified layout: transpose gate and up for every expert, free
    /// their untransposed copies, then the same for down, so only one half is
    /// ever held twice. Afterwards the `[N, K/2]` kernels cannot run.
    ///
    /// The caller sets `METRALE_UNIFIED_MOE_LAYOUT=1` (without the hybrid
    /// variable) so `use_t_layout_for_decode()` sends decode to the `_t`
    /// kernels, calls this instead of `transpose_for_prefill` /
    /// `transpose_gate_up_for_prefill`, and does not build the CUTLASS grouped
    /// tables afterwards (they point at the freed weights).
    pub fn transpose_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, false)
    }

    /// 2026-09-25: Hybrid layout: the same build as `transpose_for_prefill_unified`
    /// but the untransposed originals are kept, so decode and verify stay on the
    /// `[N, K/2]` kernels (`use_t_layout_for_decode()` is false in hybrid mode).
    /// The expert weights are held twice; the caller must check that they fit.
    pub fn transpose_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, true)
    }

    /// 2026-09-25: Transpose gate and up (routed and shared), then down. With
    /// `keep_originals` false, each half's untransposed weights are freed right
    /// after it is transposed; with true, nothing is freed. An error leaves the
    /// layer partly transposed.
    pub(super) fn transpose_for_prefill_unified_inner(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        keep_originals: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let _num_experts = self.weights.experts.len();

        let routed_gs =
            if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                32
            } else {
                16
            };
        let gate_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.gate_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.gate_proj
                }
            })
            .collect();
        let up_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.gate_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.up_proj
                }
            })
            .collect();
        let gate_t = self.transpose_experts_gpu(gpu, &gate_src, inter, h, routed_gs)?;
        let up_t = self.transpose_experts_gpu(gpu, &up_src, inter, h, routed_gs)?;
        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
        }

        if !keep_originals {
            // 2026-09-25: The `gate_ptrs`/`up_ptrs` tables keep the freed
            // addresses. Decode avoids them under `use_t_layout_for_decode()`;
            // the CUTLASS grouped tables would still read them.
            for expert in &mut self.weights.experts {
                if !expert.gate_proj.weight.is_null() {
                    gpu.free(expert.gate_proj.weight)?;
                    gpu.free(expert.gate_proj.weight_scale)?;
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                if !expert.up_proj.weight.is_null() {
                    gpu.free(expert.up_proj.weight)?;
                    gpu.free(expert.up_proj.weight_scale)?;
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !self.weights.shared_expert.gate_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.gate_proj.weight)?;
                gpu.free(self.weights.shared_expert.gate_proj.weight_scale)?;
                self.weights.shared_expert.gate_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.gate_proj.weight_scale = DevicePtr::NULL;
                gpu.free(self.weights.shared_expert.up_proj.weight)?;
                gpu.free(self.weights.shared_expert.up_proj.weight_scale)?;
                self.weights.shared_expert.up_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL;
            }
        }

        let down_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.down_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.down_proj
                }
            })
            .collect();
        let down_t = self.transpose_experts_gpu(gpu, &down_src, h, inter, routed_gs)?;
        self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        if !keep_originals {
            for expert in &mut self.weights.experts {
                if !expert.down_proj.weight.is_null() {
                    gpu.free(expert.down_proj.weight)?;
                    gpu.free(expert.down_proj.weight_scale)?;
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !self.weights.shared_expert.down_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.down_proj.weight)?;
                gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
                self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        }

        Ok(())
    }

    /// 2026-09-25: Transpose one projection of every routed expert on the GPU with
    /// `moe_transpose_u8_batched`, into one packed slab and one scale slab.
    ///
    /// `src` gives each expert's untransposed `[n, k/2]` packed bytes and
    /// `[n, k/group_size]` scales. The returned weights point into the slabs and
    /// copy `weight_scale_2`, `input_scale` and `weight_scale_2_vec` from the
    /// source. A NULL source gets a NULL slot, which the kernel skips.
    #[allow(clippy::too_many_arguments)]
    fn transpose_experts_gpu(
        &self,
        gpu: &dyn GpuBackend,
        src: &[QuantizedWeight],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<QuantizedWeight>> {
        let num_experts = src.len();
        let packed_each = n * (k / 2);
        let scale_each = n * (k / group_size);
        anyhow::ensure!(
            packed_each > 0 && scale_each > 0,
            "transpose_experts_gpu: zero-sized projection (n={n} k={k} gs={group_size})"
        );

        let packed_slab = gpu.alloc(num_experts * packed_each)?;
        let scale_slab = gpu.alloc(num_experts * scale_each)?;

        let mut out = Vec::with_capacity(num_experts);
        for (e, w) in src.iter().enumerate() {
            if w.is_null() {
                out.push(QuantizedWeight::null());
            } else {
                out.push(QuantizedWeight {
                    weight: packed_slab.offset(e * packed_each),
                    weight_scale: scale_slab.offset(e * scale_each),
                    weight_scale_2: w.weight_scale_2,
                    input_scale: w.input_scale,
                    weight_scale_2_vec: w.weight_scale_2_vec,
                });
            }
        }

        let src_tbl = build_ptr_table_from_qw(src, gpu)?;
        let dst_tbl = build_ptr_table_from_qw(&out, gpu)?;
        let stream = gpu.default_stream();
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.packed_ptrs,
            dst_tbl.packed_ptrs,
            n as u32,
            (k / 2) as u32,
            num_experts as u32,
            stream,
        )?;
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.scale_ptrs,
            dst_tbl.scale_ptrs,
            n as u32,
            (k / group_size) as u32,
            num_experts as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.free(src_tbl.packed_ptrs)?;
        gpu.free(src_tbl.scale_ptrs)?;
        gpu.free(src_tbl.scale2_vals)?;
        gpu.free(dst_tbl.packed_ptrs)?;
        gpu.free(dst_tbl.scale_ptrs)?;
        gpu.free(dst_tbl.scale2_vals)?;
        Ok(out)
    }

    /// 2026-09-25: Build the host tables of the CUTLASS grouped NVFP4 path
    /// (METRALE_HOLO_MOE_GROUPED_CUTLASS). Per expert, `pack_weight_sfb` swizzles
    /// the gate/up scales (and down's, when a down scale table exists) into the
    /// CUTLASS SFB layout, from the transposed `[K/16, N]` scales when
    /// `gate_ptrs_t`/`up_ptrs_t` exist and from the untransposed `[N, K/16]` ones
    /// otherwise. The snapshot pairs them with the untransposed packed pointers
    /// and scale2 values. A no-op when a gate or up scale table is NULL.
    pub fn build_cutlass_grouped_sfb(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let num = self.weights.experts.len();
        // 2026-09-25: Swizzled SFB size in bytes: round_up(N, 128) * round_up(K/16, 4).
        let sfb_len = |n: usize, k: usize| n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4;
        // 2026-09-25: Without transposed scales, read the untransposed [N, K/16]
        // ones with `src_n_major` set, rather than transposing only for this.
        let (gate_scale_dev, up_scale_dev, src_n_major) =
            match (self.gate_ptrs_t.as_ref(), self.up_ptrs_t.as_ref()) {
                (Some(g), Some(u)) => (g.scale_ptrs, u.scale_ptrs, false),
                _ => (self.gate_ptrs.scale_ptrs, self.up_ptrs.scale_ptrs, true),
            };
        if gate_scale_dev.is_null() || up_scale_dev.is_null() {
            return Ok(());
        }
        let down_scale_dev = match self.down_ptrs_t.as_ref() {
            Some(d) => Some(d.scale_ptrs),
            None if !self.down_ptrs.scale_ptrs.is_null() => Some(self.down_ptrs.scale_ptrs),
            None => None,
        };
        let mut owned: Vec<DevicePtr> = Vec::new();
        // 2026-09-25: `n`/`k` are the projection's GEMM dims (gate/up: inter, hidden;
        // down: hidden, inter). Returns the host vector of per-expert SFB
        // pointers, which goes into the snapshot below; a NULL scale pointer
        // leaves 0.
        let mut build_one = |scale_ptrs_dev: DevicePtr, n: usize, k: usize| -> Result<Vec<u64>> {
            let len = sfb_len(n, k);
            let sp = crate::layers::ops::read_expert_ptrs_u64(gpu, scale_ptrs_dev, num)?;
            let mut sfb_ptrs = vec![0u64; num];
            for (e, &sptr) in sp.iter().enumerate() {
                if sptr == 0 {
                    continue;
                }
                let sfb = gpu.alloc(len)?;
                metrale_gpu_runtime::cutlass::pack_weight_sfb(
                    sptr,
                    sfb.0,
                    n as u32,
                    k as u32,
                    src_n_major,
                    stream,
                )?;
                sfb_ptrs[e] = sfb.0;
                owned.push(sfb);
            }
            gpu.synchronize(stream)?;
            Ok(sfb_ptrs)
        };
        let gate_sfb = build_one(gate_scale_dev, inter, h)?;
        let up_sfb = build_one(up_scale_dev, inter, h)?;
        let down = match down_scale_dev {
            Some(ds) => Some((
                self.down_ptrs.packed_ptrs,
                build_one(ds, h, inter)?,
                self.down_ptrs.scale2_vals,
            )),
            None => None,
        };
        self.cutlass_grouped_host = Some(crate::layers::ops::MoeCutlassHostTables::snapshot(
            gpu,
            num,
            self.gate_ptrs.packed_ptrs,
            gate_sfb,
            self.gate_ptrs.scale2_vals,
            self.up_ptrs.packed_ptrs,
            up_sfb,
            self.up_ptrs.scale2_vals,
            down,
        )?);
        self._cutlass_sfb_owned = owned;
        tracing::info!(
            "CUTLASS grouped SFB: built {num} experts gate/up (N={inter} K={h}) + down (N={h} K={inter})"
        );
        Ok(())
    }
}
