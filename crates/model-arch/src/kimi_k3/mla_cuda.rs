// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host launch of the Kimi K3 MLA decode kernels (PTX module [`MODULE`]) for one token.
//!
//! Two launches per token: [`ROPE_ENTRY`] rotates the rope slice of q and k
//! (it returns without rotating when `mla_use_nope` is set), then
//! [`SDPA_ENTRY`] attends over the cached KV and applies the sigmoid output
//! gate when `mla_use_output_gate` is set. The CPU reference is
//! [`mla_decode_token`](metrale_model_weights::kimi_k3_host::mla_decode_token).
//! Decode uses [`launch_k3_mla_decode_token_on_device`] unless `K3_CUDA_MLA=0`.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants:
//! - Every temporary device buffer a launch allocates gets a `free` call
//!   before it returns, on success and on error; a failed `free` is ignored.
//! - [`MlaDeviceKv::alloc`] and [`MlaDeviceKv::alloc_and_upload`], when they
//!   fail, call `free` on what they allocated, ignoring a failed `free`.

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_weights::kimi_k3_host::{MlaConfig, MlaKv};

/// 2026-09-25: Built from `kernels/gb10/kimi-k3/bf16/mla_decode.cu`.
pub const MODULE: &str = "mla_decode";
pub const ROPE_ENTRY: &str = "k3_mla_maybe_rope_f32";
pub const SDPA_ENTRY: &str = "k3_mla_sdpa_gate_f32";
const ROPE_BLOCK: u32 = 128;
const SDPA_BLOCK: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub struct K3MlaDecodeKernels {
    pub rope: KernelHandle,
    pub sdpa: KernelHandle,
}

impl K3MlaDecodeKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rope: gpu.kernel(MODULE, ROPE_ENTRY)?,
            sdpa: gpu.kernel(MODULE, SDPA_ENTRY)?,
        })
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn up(gpu: &dyn GpuBackend, v: &[f32], hold: &mut Vec<DevicePtr>) -> Result<DevicePtr> {
    let b = f32_bytes(v);
    let p = gpu.alloc(b.len().max(1))?;
    hold.push(p);
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: One sequence's MLA K and V on the device, allocated for `cap`
/// tokens. A decode token appends one row (K device to device after rope, V
/// from the host) and attention reads the buffers in place; `download` copies
/// the history to a host [`MlaKv`].
pub struct MlaDeviceKv {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub seq_len: usize,
    pub cap: usize,
    k_row: usize,
    v_row: usize,
}

impl MlaDeviceKv {
    pub fn k_row(&self) -> usize {
        self.k_row
    }

    pub fn v_row(&self) -> usize {
        self.v_row
    }

    pub fn alloc(gpu: &dyn GpuBackend, cap: usize, k_row: usize, v_row: usize) -> Result<Self> {
        ensure!(k_row > 0 && v_row > 0, "k3 mla: resident row rank");
        let cap = cap.max(1);
        let k = gpu.alloc((cap * k_row * 4).max(1))?;
        let v = match gpu.alloc((cap * v_row * 4).max(1)) {
            Ok(ptr) => ptr,
            Err(error) => {
                let _ = gpu.free(k);
                return Err(error);
            }
        };
        Ok(Self {
            k,
            v,
            seq_len: 0,
            cap,
            k_row,
            v_row,
        })
    }

    pub fn alloc_and_upload(
        gpu: &dyn GpuBackend,
        host: &MlaKv,
        cap: usize,
        k_row: usize,
        v_row: usize,
    ) -> Result<Self> {
        ensure!(
            host.k.len() == host.seq_len.saturating_mul(k_row)
                && host.v.len() == host.seq_len.saturating_mul(v_row),
            "k3 mla: host KV rank"
        );
        ensure!(
            cap >= host.seq_len.max(1),
            "k3 mla: device KV cap {cap} < seq {}",
            host.seq_len
        );
        let mut kv = Self::alloc(gpu, cap, k_row, v_row)?;
        if host.seq_len == 0 {
            return Ok(kv);
        }
        let upload = gpu
            .copy_h2d(&f32_bytes(&host.k), kv.k)
            .and_then(|()| gpu.copy_h2d(&f32_bytes(&host.v), kv.v));
        if let Err(error) = upload {
            let _ = kv.free(gpu);
            return Err(error);
        }
        kv.seq_len = host.seq_len;
        Ok(kv)
    }

    pub fn validate_cfg(&self, cfg: &MlaConfig) -> Result<()> {
        let k_row = cfg.heads * cfg.qk_head_dim();
        let v_row = cfg.heads * cfg.v_head_dim;
        ensure!(
            self.k_row == k_row && self.v_row == v_row,
            "k3 mla: resident KV rank"
        );
        Ok(())
    }

    pub fn download(&self, gpu: &dyn GpuBackend, host: &mut MlaKv) -> Result<()> {
        let n = self.seq_len;
        let mut kb = vec![0u8; n.saturating_mul(self.k_row) * 4];
        let mut vb = vec![0u8; n.saturating_mul(self.v_row) * 4];
        if n > 0 {
            gpu.copy_d2h(self.k, &mut kb)?;
            gpu.copy_d2h(self.v, &mut vb)?;
        }
        host.k = bytes_f32(&kb);
        host.v = bytes_f32(&vb);
        host.seq_len = n;
        Ok(())
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        let k = gpu.free(self.k);
        let v = gpu.free(self.v);
        k.and(v)
    }

    fn append_from_device(
        &mut self,
        gpu: &dyn GpuBackend,
        k_new: DevicePtr,
        v_host: &[f32],
    ) -> Result<()> {
        if self.seq_len >= self.cap {
            bail!("k3 mla: device KV cap {} full", self.cap);
        }
        ensure!(
            v_host.len() == self.v_row,
            "k3 mla: V row rank {} != {}",
            v_host.len(),
            self.v_row
        );
        let k_off = DevicePtr(self.k.0 + (self.seq_len * self.k_row * 4) as u64);
        gpu.copy_d2d(k_new, k_off, self.k_row * 4)?;
        let v_off = DevicePtr(self.v.0 + (self.seq_len * self.v_row * 4) as u64);
        gpu.copy_h2d(&f32_bytes(v_host), v_off)?;
        self.seq_len += 1;
        Ok(())
    }
}

/// 2026-09-25: One decode token against a host [`MlaKv`]: writes the roped
/// `q`/`k` back, appends to `kv`, and uploads the whole KV history.
#[allow(clippy::too_many_arguments)]
pub fn launch_k3_mla_decode_token(
    gpu: &dyn GpuBackend,
    kernels: &K3MlaDecodeKernels,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    g: &[f32],
    kv: &mut MlaKv,
    cfg: &MlaConfig,
    pos: usize,
    theta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    let (h, dq, dv) = (cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    if q.len() != h * dq || k.len() != h * dq {
        bail!("k3 mla: q/k rank");
    }
    if v.len() != h * dv || g.len() != h * dv {
        bail!("k3 mla: v/g rank");
    }
    if dq == 0 {
        bail!("k3 mla: dq must be > 0");
    }

    let mut hold = Vec::new();
    let run = (|| {
        let dq_ptr = up(gpu, q, &mut hold)?;
        let dk_new = up(gpu, k, &mut hold)?;
        KernelLaunch::new(gpu, kernels.rope)
            .grid([div_ceil(h as u32, ROPE_BLOCK), 1, 1])
            .block([ROPE_BLOCK, 1, 1])
            .arg_ptr(dq_ptr)
            .arg_ptr(dk_new)
            .arg_u32(h as u32)
            .arg_u32(cfg.qk_nope_head_dim as u32)
            .arg_u32(cfg.qk_rope_head_dim as u32)
            .arg_u32(pos as u32)
            .arg_f32(theta)
            .arg_u32(u32::from(cfg.mla_use_nope))
            .launch(stream)
            .context("k3_mla_maybe_rope_f32")?;
        gpu.synchronize(stream)?;

        let mut qb = vec![0u8; q.len() * 4];
        let mut kb = vec![0u8; k.len() * 4];
        gpu.copy_d2h(dq_ptr, &mut qb)?;
        gpu.copy_d2h(dk_new, &mut kb)?;
        q.copy_from_slice(&bytes_f32(&qb));
        k.copy_from_slice(&bytes_f32(&kb));
        kv.append(k, v);

        let dq2 = up(gpu, q, &mut hold)?;
        let dk = up(gpu, &kv.k, &mut hold)?;
        let dv_ptr = up(gpu, &kv.v, &mut hold)?;
        let dg = up(gpu, g, &mut hold)?;
        let dout = gpu.alloc((h * dv * 4).max(1))?;
        hold.push(dout);
        KernelLaunch::new(gpu, kernels.sdpa)
            .grid([h as u32, 1, 1])
            .block([SDPA_BLOCK, 1, 1])
            .arg_ptr(dq2)
            .arg_ptr(dk)
            .arg_ptr(dv_ptr)
            .arg_ptr(dg)
            .arg_ptr(dout)
            .arg_u32(kv.seq_len as u32)
            .arg_u32(h as u32)
            .arg_u32(dq as u32)
            .arg_u32(dv as u32)
            .arg_u32(u32::from(cfg.mla_use_output_gate))
            .launch(stream)
            .context("k3_mla_sdpa_gate_f32")?;
        gpu.synchronize(stream)?;

        let mut out_b = vec![0u8; h * dv * 4];
        gpu.copy_d2h(dout, &mut out_b)?;
        Ok(bytes_f32(&out_b))
    })();
    for p in hold {
        let _ = gpu.free(p);
    }
    run
}

/// 2026-09-25: One decode token against an [`MlaDeviceKv`]: rope stays on the
/// device, the append is one row, and only the output is copied back.
#[allow(clippy::too_many_arguments)]
pub fn launch_k3_mla_decode_token_on_device(
    gpu: &dyn GpuBackend,
    kernels: &K3MlaDecodeKernels,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    kv: &mut MlaDeviceKv,
    cfg: &MlaConfig,
    pos: usize,
    theta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    let (h, dq, dv) = (cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    if q.len() != h * dq || k.len() != h * dq || v.len() != h * dv || g.len() != h * dv {
        bail!("k3 mla: q/k/v/g rank");
    }
    let mut hold = Vec::new();
    let run = (|| {
        let dq_ptr = up(gpu, q, &mut hold)?;
        let dk_new = up(gpu, k, &mut hold)?;
        KernelLaunch::new(gpu, kernels.rope)
            .grid([div_ceil(h as u32, ROPE_BLOCK), 1, 1])
            .block([ROPE_BLOCK, 1, 1])
            .arg_ptr(dq_ptr)
            .arg_ptr(dk_new)
            .arg_u32(h as u32)
            .arg_u32(cfg.qk_nope_head_dim as u32)
            .arg_u32(cfg.qk_rope_head_dim as u32)
            .arg_u32(pos as u32)
            .arg_f32(theta)
            .arg_u32(u32::from(cfg.mla_use_nope))
            .launch(stream)
            .context("k3_mla_maybe_rope_f32")?;
        kv.append_from_device(gpu, dk_new, v)?;
        let dg = up(gpu, g, &mut hold)?;
        let dout = gpu.alloc((h * dv * 4).max(1))?;
        hold.push(dout);
        KernelLaunch::new(gpu, kernels.sdpa)
            .grid([h as u32, 1, 1])
            .block([SDPA_BLOCK, 1, 1])
            .arg_ptr(dq_ptr)
            .arg_ptr(kv.k)
            .arg_ptr(kv.v)
            .arg_ptr(dg)
            .arg_ptr(dout)
            .arg_u32(kv.seq_len as u32)
            .arg_u32(h as u32)
            .arg_u32(dq as u32)
            .arg_u32(dv as u32)
            .arg_u32(u32::from(cfg.mla_use_output_gate))
            .launch(stream)
            .context("k3_mla_sdpa_gate_f32")?;
        gpu.synchronize(stream)?;
        let mut out_b = vec![0u8; h * dv * 4];
        gpu.copy_d2h(dout, &mut out_b)?;
        Ok(bytes_f32(&out_b))
    })();
    for p in hold {
        let _ = gpu.free(p);
    }
    run
}

#[cfg(test)]
#[path = "mla_cuda_tests.rs"]
mod tests;
