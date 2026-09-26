// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Patch-embed step: f32 pixels → BF16 → patch_embed GEMM → +pos_embed.
//!
//! Owner: model-layers (vision).
//! Invariants:
//! - No pixel upload is issued before `check_pixel_len` has accepted the
//!   slice's length and the end row.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{PATCH_DIM, VisionEncoder};

/// 2026-09-25: Check a host pixel buffer against the width and the row capacity the
/// encoder was built for, before its bytes are reinterpreted and copied.
///
/// 1. Width: the CPU preprocessor sizes `pixels` from the checkpoint's
///    `vision_config` (`3 × temporal_patch_size × patch_size²` per patch),
///    while the device buffer and GEMM use [`PATCH_DIM`]. A `patch_size: 14`
///    checkpoint gives 1176 floats per patch.
/// 2. Rows: `end_row`, the last device row this upload touches, must not
///    exceed `p_max`, the row capacity of `buf_f32` and every later buffer.
///    `patch_size: 4, temporal_patch_size: 32` also gives `PATCH_DIM` floats
///    per patch, so it passes the width check with many more patches. The
///    single-image path runs from `forward_oversized_fallback`, which bounds
///    only the merged rows.
fn check_pixel_len(pixels: &[f32], patches: usize, end_row: usize, p_max: usize) -> Result<()> {
    let want = patches
        .checked_mul(PATCH_DIM)
        .ok_or_else(|| anyhow::anyhow!("vision: patch count {patches} overflows"))?;
    anyhow::ensure!(
        pixels.len() == want,
        "vision: pixel buffer is {} floats for {patches} patches, but this encoder is built \
         for {PATCH_DIM} floats per patch ({want}). The checkpoint's vision_config \
         patch_size/temporal_patch_size do not match the compiled ViT.",
        pixels.len()
    );
    anyhow::ensure!(
        end_row <= p_max,
        "vision: this upload ends at patch row {end_row} but the encoder's buffers hold \
         {p_max} rows ({patches} patches in this image). The checkpoint's vision_config \
         yields a finer patch grid than the compiled ViT was allocated for."
    );
    Ok(())
}

impl VisionEncoder {
    /// 2026-09-25: Upload f32 pixels → convert to BF16 → patch embed GEMM → add pos_embed.
    pub(super) fn patch_embed(
        &self,
        pixels: &[f32],
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Single image at row 0, so the last row touched is `p`.
        check_pixel_len(pixels, p, p, self.p_max)?;
        let n_f32 = pixels.len();
        // 2026-09-25: SAFETY: `pixels` is a live `&[f32]`; the byte length is taken
        // from that same slice (`len() * 4`), so the view never leaves the
        // allocation. f32 has no padding or invalid bit patterns, and u8 has
        // alignment 1, so every byte of it is a valid `u8`. The view is
        // read-only and dies at the end of this function, before `pixels`.
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, n_f32 * 4) };
        gpu.copy_h2d_async(f32_bytes, self.scratch().buf_f32, stream)?;
        // 2026-09-25: f32 → bf16, into buf_wide[0..p*PATCH_DIM].
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_f32)
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        // 2026-09-25: buf_wide[p, K] @ patch_embed_w[hidden, K]^T + b → buf_h1[p, hidden].
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.scratch().buf_h1,
            p as u32,
            self.hidden_size as u32,
            PATCH_DIM as u32,
            stream,
        )?;
        // 2026-09-25: Add buf_pos_resampled, which `forward_oversized_fallback`
        // filled (interpolated, or the raw table under METRALE_VISION_POSINTERP=0).
        let n_pe = p * self.hidden_size;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_pe as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_pos_resampled)
            .arg_u32(n_pe as u32)
            .launch(stream)
    }

    /// 2026-09-25: Batched patch-embed over N images packed at row `p_off[i]`.
    /// Uploads each image's f32 pixels into `buf_f32` at its row offset, then
    /// runs one f32→bf16, one patch_embed GEMM (M=p_total), and one pos_embed
    /// add over the whole batch. `buf_pos_resampled` must already hold each
    /// image's per-row pos embed.
    pub(super) fn patch_embed_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        p_off: &[usize],
        p_total: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Upload each image's pixels into its row slice of buf_f32.
        // Each image lands at row `p_off[i]`, so `check_pixel_len` gets
        // `p_off[i] + gh*gw` as the end row.
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p_i = gh * gw;
            let end_row = p_off[i]
                .checked_add(p_i)
                .ok_or_else(|| anyhow::anyhow!("vision: patch row offset overflows"))?;
            check_pixel_len(pixels, p_i, end_row, self.p_max)?;
            // 2026-09-25: SAFETY: `pixels` is a live `&[f32]` and the byte length is
            // derived from that same slice, so the view stays inside its
            // allocation. `f32` has no invalid bit patterns and `u8` has
            // alignment 1, so the reinterpretation is valid for every byte.
            let f32_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
            };
            gpu.copy_h2d_async(
                f32_bytes,
                self.scratch().buf_f32.offset(p_off[i] * PATCH_DIM * 4),
                stream,
            )?;
        }
        let n_f32 = p_total * PATCH_DIM;
        // 2026-09-25: f32 → bf16, into buf_wide[0..p_total*PATCH_DIM].
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_f32)
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.scratch().buf_h1,
            p_total as u32,
            self.hidden_size as u32,
            PATCH_DIM as u32,
            stream,
        )?;
        // 2026-09-25: Add the per-image pos embed packed in buf_pos_resampled.
        let n_pe = p_total * self.hidden_size;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_pe as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_pos_resampled)
            .arg_u32(n_pe as u32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{PATCH_DIM, check_pixel_len};

    /// 2026-09-25: The Qwen3-VL geometry: patch_size 16, temporal_patch_size 2 →
    /// 3 × 2 × 16 × 16 = 1536 floats per patch.
    #[test]
    fn accepts_the_geometry_the_encoder_was_built_for() {
        assert_eq!(PATCH_DIM, 3 * 2 * 16 * 16);
        let pixels = vec![0.0f32; 64 * PATCH_DIM];
        assert!(check_pixel_len(&pixels, 64, 64, 6400).is_ok());
        // 2026-09-25: Zero patches with an empty slice is consistent.
        assert!(check_pixel_len(&[], 0, 0, 6400).is_ok());
    }

    /// 2026-09-25: A `patch_size: 14` checkpoint makes the CPU preprocessor emit
    /// 3 × 2 × 14 × 14 = 1176 floats per patch; the check refuses it.
    #[test]
    fn rejects_narrower_patch_dim_instead_of_reading_past_the_buffer() {
        let narrow = 3 * 2 * 14 * 14;
        assert!(narrow < PATCH_DIM, "this test must model an UNDER-run");
        let pixels = vec![0.0f32; 64 * narrow];
        let err = check_pixel_len(&pixels, 64, 64, 6400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("patch_size"), "{err}");
        assert!(err.contains(&format!("{}", 64 * narrow)), "{err}");
    }

    /// 2026-09-25: A wider per-patch width is refused too.
    #[test]
    fn rejects_wider_patch_dim() {
        let wide = 3 * 2 * 32 * 32;
        assert!(wide > PATCH_DIM);
        let pixels = vec![0.0f32; 4 * wide];
        assert!(check_pixel_len(&pixels, 4, 4, 6400).is_err());
    }

    /// 2026-09-25: A patch count large enough to wrap the multiply is an error, not a
    /// wrapped expected length.
    #[test]
    fn rejects_patch_count_that_overflows() {
        let err = check_pixel_len(&[], usize::MAX / 2, usize::MAX / 2, 6400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"), "{err}");
    }

    /// 2026-09-25: `patch_size: 4, temporal_patch_size: 32` gives 3×32×4×4 = 1536
    /// floats per patch, so the width check passes; the row check refuses a
    /// patch count past `p_max`.
    #[test]
    fn rejects_a_consistent_buffer_with_too_many_patches() {
        assert_eq!(
            3 * 32 * 4 * 4,
            PATCH_DIM,
            "the hostile geometry is width-consistent"
        );
        let p_max = 6400;
        assert!(check_pixel_len(&vec![0.0f32; p_max * PATCH_DIM], p_max, p_max, p_max).is_ok());
        let over = p_max + 1;
        let err = check_pixel_len(&vec![0.0f32; over * PATCH_DIM], over, over, p_max)
            .unwrap_err()
            .to_string();
        assert!(err.contains("6400 rows"), "{err}");
    }

    /// 2026-09-25: The batched path places each image at its own row offset, so the
    /// bound is the end row, not the per-image count.
    #[test]
    fn rejects_a_small_image_placed_past_the_end() {
        let p_max = 6400;
        let pixels = vec![0.0f32; 8 * PATCH_DIM];
        let err = check_pixel_len(&pixels, 8, 6399 + 8, p_max)
            .unwrap_err()
            .to_string();
        assert!(err.contains("6407"), "{err}");
        assert!(check_pixel_len(&pixels, 8, 100 + 8, p_max).is_ok());
    }
}
