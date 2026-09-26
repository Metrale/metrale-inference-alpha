// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The head_dim-128 FlashInfer ragged-prefill entry, with an optional
//! sliding window. A child of `flashinfer`, so it shares that module's
//! workspaces.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

#[cfg(metrale_flashinfer)]
use std::ffi::c_void;

#[cfg(metrale_flashinfer)]
use super::workspaces;

#[cfg(metrale_flashinfer)]
unsafe extern "C" {
    // 2026-09-25: The hd256 ABI plus `window_left` after `causal`: -1 for no
    // window, otherwise `sliding_window - 1`.
    #[allow(clippy::too_many_arguments)]
    fn metrale_fi_ragged_prefill_bf16_hd128(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        qo_indptr_d: *const i32,
        kv_indptr_d: *const i32,
        batch: u32,
        total_qo_rows: u32,
        total_kv_rows: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        sm_scale: f32,
        causal: i32,
        window_left: i32,
        float_ws: *mut c_void,
        float_ws_bytes: usize,
        int_ws: *mut c_void,
        int_ws_bytes: usize,
        pinned_int_ws: *mut c_void,
        pinned_int_ws_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// 2026-09-25: Ragged batched prefill attention, BF16, head_dim 128, GQA.
///
/// The contract of [`super::ragged_prefill_bf16_hd256`], plus `sliding_window`
/// in the in-tree convention (mask when `q - k >= w`, as in
/// `kernels/gb10/common/attn_prefill.cu`); `None` or `Some(0)` means no window.
/// It is passed to FlashInfer as `window_left = w - 1`.
#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd128(
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    qo_indptr_h: &[i32],
    kv_indptr_h: &[i32],
    qo_indptr_d: u64,
    kv_indptr_d: u64,
    batch: u32,
    total_qo_rows: u32,
    total_kv_rows: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sm_scale: f32,
    causal: bool,
    sliding_window: Option<u32>,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_flashinfer)]
    {
        if head_dim != 128 {
            bail!("ragged_prefill_bf16_hd128 is head_dim=128 only (got {head_dim})");
        }
        if qo_indptr_h.len() != (batch + 1) as usize || kv_indptr_h.len() != (batch + 1) as usize {
            bail!("indptr host slices must be batch+1 long");
        }
        let window_left: i32 = match sliding_window {
            Some(w) if w > 0 => (w - 1) as i32,
            _ => -1,
        };
        let ws = workspaces()?;
        let st = unsafe {
            metrale_fi_ragged_prefill_bf16_hd128(
                q as *const c_void,
                k as *const c_void,
                v as *const c_void,
                o as *mut c_void,
                qo_indptr_h.as_ptr(),
                kv_indptr_h.as_ptr(),
                qo_indptr_d as *const i32,
                kv_indptr_d as *const i32,
                batch,
                total_qo_rows,
                total_kv_rows,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                sm_scale,
                if causal { 1 } else { 0 },
                window_left,
                ws.float_ws as *mut c_void,
                ws.float_sz,
                ws.int_ws as *mut c_void,
                ws.int_sz,
                ws.pinned_int_ws as *mut c_void,
                ws.pinned_sz,
                stream as *mut c_void,
            )
        };
        if st != 0 {
            bail!(
                "FlashInfer ragged prefill hd128 failed: status {st} \
                 (batch={batch}, qo={total_qo_rows}, window_left={window_left})"
            );
        }
        Ok(())
    }
    #[cfg(not(metrale_flashinfer))]
    {
        let _ = (
            q,
            k,
            v,
            o,
            qo_indptr_h,
            kv_indptr_h,
            qo_indptr_d,
            kv_indptr_d,
            batch,
            total_qo_rows,
            total_kv_rows,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
            causal,
            sliding_window,
            stream,
        );
        bail!("FlashInfer support was not built; set FLASHINFER_HOME when building")
    }
}
