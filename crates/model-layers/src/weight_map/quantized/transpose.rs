// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `QuantizedWeight` transposed-twin builders for the GEMM paths: plain, group-scaled, concatenated and padded.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - Every builder returns new buffers and leaves its inputs unchanged.
//! - A transpose runs on the `transpose_u8` kernel when it resolves and
//!   `METRALE_HOST_TRANSPOSE` is not `1`; otherwise on the host.

use super::*;

impl QuantizedWeight {
    /// 2026-09-25: Resolve the `transpose_u8` GPU kernel for the load-time transpose
    /// paths, or `None` to use the host byte-loop fallback. `None` when the
    /// target's kernel set lacks it, or when `METRALE_HOST_TRANSPOSE=1` forces
    /// the host path.
    fn host_transpose_kernel(
        gpu: &dyn GpuBackend,
    ) -> Option<metrale_gpu_runtime::gpu::KernelHandle> {
        if std::env::var("METRALE_HOST_TRANSPOSE").as_deref() == Ok("1") {
            return None;
        }
        let k = crate::layers::try_kernel(gpu, "transpose_u8", "transpose_u8");
        (k.0 != 0).then_some(k)
    }

    /// 2026-09-25: Transpose the packed weight `[N, K/2]` to `[K/2, N]` and the
    /// scales `[N, K/16]` to `[K/16, N]`, into new buffers.
    pub fn transpose_for_gemm(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
    ) -> Result<QuantizedWeight> {
        // 2026-09-25: 16 is the NVFP4 group; native-MXFP4 experts call
        // `transpose_for_gemm_gs` with 32.
        self.transpose_for_gemm_gs(gpu, n, k, 16)
    }

    /// 2026-09-25: `transpose_for_gemm` with an explicit scale block size. The scale
    /// tensor is `[N, K/group_size]`; the packed-weight transpose does not depend on it.
    pub fn transpose_for_gemm_gs(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<QuantizedWeight> {
        let half_k = k / 2;
        let num_groups = k / group_size;
        let packed_size = n * half_k;
        let scale_size = n * num_groups;

        // 2026-09-25: Both launches run on stream 0, which is synchronized before
        // the new buffers are returned.
        if let Some(tk) = Self::host_transpose_kernel(gpu) {
            let new_weight = gpu.alloc(packed_size)?;
            let new_scale = gpu.alloc(scale_size)?;
            crate::layers::ops::transpose_u8(
                gpu,
                tk,
                self.weight,
                new_weight,
                n as u32,
                half_k as u32,
                0,
            )?;
            crate::layers::ops::transpose_u8(
                gpu,
                tk,
                self.weight_scale,
                new_scale,
                n as u32,
                num_groups as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            return Ok(QuantizedWeight {
                weight: new_weight,
                weight_scale: new_scale,
                weight_scale_2: self.weight_scale_2,
                input_scale: self.input_scale,
                weight_scale_2_vec: self.weight_scale_2_vec,
            });
        }

        let mut buf = vec![0u8; packed_size];
        gpu.copy_d2h(self.weight, &mut buf)?;
        let mut t_buf = vec![0u8; packed_size];
        for i in 0..n {
            for j in 0..half_k {
                t_buf[j * n + i] = buf[i * half_k + j];
            }
        }
        let new_weight = gpu.alloc(packed_size)?;
        gpu.copy_h2d(&t_buf, new_weight)?;

        let mut sbuf = vec![0u8; scale_size];
        gpu.copy_d2h(self.weight_scale, &mut sbuf)?;
        let mut st_buf = vec![0u8; scale_size];
        for i in 0..n {
            for j in 0..num_groups {
                st_buf[j * n + i] = sbuf[i * num_groups + j];
            }
        }
        let new_scale = gpu.alloc(scale_size)?;
        gpu.copy_h2d(&st_buf, new_scale)?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: self.weight_scale_2,
            input_scale: self.input_scale,
            weight_scale_2_vec: self.weight_scale_2_vec,
        })
    }

    /// 2026-09-25: Transpose several weights `(weight, n)` sharing one K and
    /// concatenate them along N into one `[K/2, N_total]` twin, so one GEMM
    /// replaces several.
    ///
    /// The result carries the first part's `weight_scale_2`, `input_scale` and
    /// `weight_scale_2_vec`, so the caller must ensure every part shares them;
    /// nothing here checks. An empty `parts` is an error.
    pub fn transpose_concat_for_gemm(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
    ) -> Result<QuantizedWeight> {
        Self::transpose_concat_for_gemm_gs(gpu, parts, k, 16)
    }

    /// 2026-09-25: `transpose_concat_for_gemm_gs` with the output row stride
    /// padded to `align_up(n_total, align)` and the pad columns zeroed.
    ///
    /// Row `r` of the result starts at byte `r * stride`, and the tile GEMM reads
    /// B with 16-byte `cp.async`, which needs 16-byte aligned rows; an N such as
    /// an odd vocab size is not. Returns `(weight, stride)`; pass the stride to
    /// `w4a16_gemm_n128_ldb`.
    pub fn transpose_concat_for_gemm_padded(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
        group_size: usize,
        align: usize,
    ) -> Result<(QuantizedWeight, usize)> {
        let n_total: usize = parts.iter().map(|(_, n)| *n).sum();
        let stride = n_total.div_ceil(align) * align;
        Self::transpose_impl(gpu, parts, k, group_size, stride).map(|w| (w, stride))
    }

    /// 2026-09-25: `transpose_concat_for_gemm` with an explicit scale block size.
    pub fn transpose_concat_for_gemm_gs(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
        group_size: usize,
    ) -> Result<QuantizedWeight> {
        let n_total: usize = parts.iter().map(|(_, n)| *n).sum();
        Self::transpose_impl(gpu, parts, k, group_size, n_total)
    }

    /// 2026-09-25: The implementation behind both concat builders. `stride` is the
    /// row pitch of the output and must be at least `n_total` (a debug assertion
    /// only); columns `n_total..stride` are zero.
    fn transpose_impl(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
        group_size: usize,
        stride: usize,
    ) -> Result<QuantizedWeight> {
        let first = parts
            .first()
            .map(|(w, _)| *w)
            .context("transpose_concat_for_gemm: empty parts")?;
        let half_k = k / 2;
        let num_groups = k / group_size;
        let n_total: usize = parts.iter().map(|(_, n)| *n).sum();
        debug_assert!(
            stride >= n_total,
            "transpose_impl: stride {stride} < n_total {n_total}"
        );

        // 2026-09-25: Per part, one `transpose_u8` into a contiguous `[half_k, n]`
        // temp, then one pitched copy into the part's column window. The pad
        // columns are zeroed up front, as the host path's staging vector is.
        if let Some(tk) = Self::host_transpose_kernel(gpu) {
            let new_weight = gpu.alloc(stride * half_k)?;
            let new_scale = gpu.alloc(stride * num_groups)?;
            if stride > n_total {
                gpu.memset(new_weight, 0, stride * half_k)?;
                gpu.memset(new_scale, 0, stride * num_groups)?;
            }
            let mut temps: Vec<DevicePtr> = Vec::with_capacity(parts.len() * 2);
            let mut n_off = 0usize;
            for (w, n) in parts {
                let n = *n;
                let t_w = gpu.alloc(n * half_k)?;
                crate::layers::ops::transpose_u8(
                    gpu,
                    tk,
                    w.weight,
                    t_w,
                    n as u32,
                    half_k as u32,
                    0,
                )?;
                gpu.copy_d2d_2d_async(t_w, n, new_weight.offset(n_off), stride, n, half_k, 0)?;
                let t_s = gpu.alloc(n * num_groups)?;
                crate::layers::ops::transpose_u8(
                    gpu,
                    tk,
                    w.weight_scale,
                    t_s,
                    n as u32,
                    num_groups as u32,
                    0,
                )?;
                gpu.copy_d2d_2d_async(t_s, n, new_scale.offset(n_off), stride, n, num_groups, 0)?;
                temps.push(t_w);
                temps.push(t_s);
                n_off += n;
            }
            gpu.synchronize(0)?;
            for t in temps {
                gpu.free(t)?;
            }
            return Ok(QuantizedWeight {
                weight: new_weight,
                weight_scale: new_scale,
                weight_scale_2: first.weight_scale_2,
                input_scale: first.input_scale,
                weight_scale_2_vec: first.weight_scale_2_vec,
            });
        }

        let mut t_buf = vec![0u8; stride * half_k];
        let mut st_buf = vec![0u8; stride * num_groups];
        let mut n_off = 0usize;
        for (w, n) in parts {
            let n = *n;
            let mut buf = vec![0u8; n * half_k];
            gpu.copy_d2h(w.weight, &mut buf)?;
            for i in 0..n {
                for j in 0..half_k {
                    t_buf[j * stride + n_off + i] = buf[i * half_k + j];
                }
            }
            let mut sbuf = vec![0u8; n * num_groups];
            gpu.copy_d2h(w.weight_scale, &mut sbuf)?;
            for i in 0..n {
                for j in 0..num_groups {
                    st_buf[j * stride + n_off + i] = sbuf[i * num_groups + j];
                }
            }
            n_off += n;
        }

        let new_weight = gpu.alloc(t_buf.len())?;
        gpu.copy_h2d(&t_buf, new_weight)?;
        let new_scale = gpu.alloc(st_buf.len())?;
        gpu.copy_h2d(&st_buf, new_scale)?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: first.weight_scale_2,
            input_scale: first.input_scale,
            weight_scale_2_vec: first.weight_scale_2_vec,
        })
    }
}
