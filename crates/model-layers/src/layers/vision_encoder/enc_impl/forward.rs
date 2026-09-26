// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `VisionEncoder::forward` / `forward_batched`: the image → token pipeline
//! (pos_embed → RoPE → patch embed → ViT blocks with DeepStack merger taps →
//! final merger).
//!
//! Owner: model-layers (vision).
//! Invariants:
//! - Rows are packed in image order: image `i`'s final-merger rows start at
//!   row `mp_off[i]` of `buf_out`.
//!
//! The batched form runs the block GEMMs once over Σpatches of N images;
//! stages that depend on one image's geometry (host pos/rope prep, attention,
//! mergers) loop per image.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::VisionEncoder;

/// 2026-09-25: Whether the packed merged rows of this batch fit `buf_out` (`p_max` rows).
///
/// All temporal groups of one video arrive as one media item and are encoded
/// as one batch, so a per-item cap upstream does not bound their sum.
/// Returns an error for inconsistent lengths, an overflowing offset, or an
/// end row past `p_max`. A free function, so the refusal is testable without
/// a GPU.
fn check_packed_rows(mp_i: &[usize], mp_off: &[usize], p_max: usize) -> Result<()> {
    anyhow::ensure!(
        mp_i.len() == mp_off.len(),
        "vision: {} merged row counts but {} offsets; the packed layout is inconsistent",
        mp_i.len(),
        mp_off.len()
    );
    let end = match (mp_off.last(), mp_i.last()) {
        (Some(off), Some(n)) => off
            .checked_add(*n)
            .ok_or_else(|| anyhow::anyhow!("vision: merged row offset {off} + {n} overflows"))?,
        _ => 0,
    };
    anyhow::ensure!(
        end <= p_max,
        "vision: this batch packs {end} merged rows into an output buffer of {p_max} rows. \
         A video arrives as one media item whose temporal groups encode as a single batch, \
         which defeats the scheduler's per-item cap. Send fewer frames, or allocate a \
         larger vision scratch."
    );
    Ok(())
}

impl VisionEncoder {
    /// 2026-09-25: Single-image forward through `forward_batched`. `pixels` is
    /// `[P, PATCH_DIM]` f32. Returns `(1 + deepstack_indexes.len()) * merged_p`.
    pub fn forward(
        &self,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<usize> {
        let images = [(pixels, grid_h, grid_w)];
        let per_image = self.forward_batched(&images, gpu, stream)?;
        let merged_p = per_image[0].2;
        Ok((1 + self.deepstack_indexes.len()) * merged_p)
    }

    /// 2026-09-25: Batched forward over N images. Ops that do not depend on image
    /// geometry (patch_embed and every block's GEMMs, norms, GELU and
    /// residuals) run once over M = Σpᵢ; host pos/rope prep, attention and the
    /// mergers loop per image.
    ///
    /// `buf_out` layout, rows of `out_hidden_size` BF16:
    /// - `0 .. Σmerged_p`: the final merger, packed in image order (the rows
    ///   the LLM reads);
    /// - from `(k+1)*Σmerged_p`: DeepStack merger `k`, packed the same way.
    ///
    /// When Σp <= p_max this path writes up to `(1 + deepstack count) *
    /// Σmerged_p` rows, which fits the `p_max`-row `buf_out` only while
    /// `1 + deepstack count <= spatial_merge_size²`; nothing checks it. When
    /// Σp > p_max it encodes image by image without DeepStack rows.
    ///
    /// Returns per-image `(post_h, post_w, merged_p)` in image order.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        // 2026-09-25: Allocate the ViT scratch on the first image (`forward`
        // delegates here).
        self.scratch_init(gpu)?;
        let sms2 = self.spatial_merge_size * self.spatial_merge_size;
        let sms = self.spatial_merge_size.max(1);
        let n_img = images.len();

        // 2026-09-25: Per-image pre-merge patch counts (p_i) and post-merge counts
        // (mp_i), with running row offsets into the shared buffers (p_off / mp_off).
        let mut p_i = Vec::with_capacity(n_img);
        let mut p_off = Vec::with_capacity(n_img);
        let mut mp_i = Vec::with_capacity(n_img);
        let mut mp_off = Vec::with_capacity(n_img);
        let (mut p_total, mut mp_total) = (0usize, 0usize);
        for (_px, gh, gw) in images.iter() {
            let p = gh * gw;
            let mp = p / sms2;
            p_off.push(p_total);
            mp_off.push(mp_total);
            p_i.push(p);
            mp_i.push(mp);
            p_total += p;
            mp_total += mp;
        }

        // 2026-09-25: Past p_max, encode image by image, packing the final-merger
        // rows of `buf_out` the same way.
        if p_total > self.p_max {
            return self.forward_oversized_fallback(images, &p_i, &mp_i, &mp_off, sms, gpu, stream);
        }

        let _sec0 = std::time::Instant::now();
        // 2026-09-25: 1. Per-image host prep, packed into the shared buffers at p_off[i].
        let pos_interp_on = std::env::var("METRALE_VISION_POSINTERP")
            .map(|v| v != "0")
            .unwrap_or(true);
        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            let pos_dst = self
                .scratch()
                .buf_pos_resampled
                .offset(p_off[i] * self.hidden_size * 2);
            if pos_interp_on {
                self.resample_pos_embed_into(*gh, *gw, pos_dst, gpu, stream)?;
            } else {
                self.gpu_copy_bf16(
                    gpu,
                    self.pos_embed,
                    pos_dst,
                    p * self.hidden_size * 2,
                    stream,
                )?;
            }
            let cos_dst = self
                .scratch()
                .buf_rope_cos
                .offset(p_off[i] * self.head_dim * 2);
            let sin_dst = self
                .scratch()
                .buf_rope_sin
                .offset(p_off[i] * self.head_dim * 2);
            self.build_rope_cossin_into(*gh, *gw, cos_dst, sin_dst, gpu, stream)?;
        }

        let timing = std::env::var("METRALE_VISION_TIMING").is_ok();
        if timing {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_SEC host_prep({n_img} imgs): {:.1}ms",
                _sec0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let _sec1 = std::time::Instant::now();
        // 2026-09-25: 2. Patch embed over M=Σp.
        self.patch_embed_batched(images, &p_off, p_total, gpu, stream)?;
        Self::maybe_dump_buf(
            gpu,
            self.scratch().buf_h1,
            p_total * self.hidden_size,
            "patch_embed",
            stream,
        )?;

        // 2026-09-25: 3. The blocks: batched ops once, attention and DeepStack
        // mergers per image.
        let n_h_bytes = p_total * self.hidden_size * 2;
        let mut deepstack_iter = self.deepstack_indexes.iter().enumerate();
        let mut next_ds = deepstack_iter.next(); // 2026-09-25: (merger index, &1-based block)
        for (block_idx, blk) in self.blocks.iter().enumerate() {
            self.vit_block_batched(blk, p_total, &p_i, &p_off, gpu, stream)?;
            Self::maybe_dump_buf(
                gpu,
                self.scratch().buf_h1,
                p_total * self.hidden_size,
                &format!("block{block_idx:02}"),
                stream,
            )?;
            if let Some((ds_idx, &ds_block)) = next_ds
                && block_idx + 1 == ds_block
            {
                // 2026-09-25: Copy buf_h1 → buf_h2 and merge from the copy:
                // `apply_merger` normalises its source in place, and buf_h1 feeds
                // the next block.
                self.gpu_copy_bf16(
                    gpu,
                    self.scratch().buf_h1,
                    self.scratch().buf_h2,
                    n_h_bytes,
                    stream,
                )?;
                let ds_region_base = (ds_idx + 1) * mp_total;
                for (i, (_px, gh, gw)) in images.iter().enumerate() {
                    let src = self
                        .scratch()
                        .buf_h2
                        .offset(p_off[i] * self.hidden_size * 2);
                    let out_rows = ds_region_base + mp_off[i];
                    let out_slice = self
                        .scratch()
                        .buf_out
                        .offset(out_rows * self.out_hidden_size * 2);
                    self.apply_merger(
                        &self.deepstack[ds_idx],
                        p_i[i],
                        *gh,
                        *gw,
                        src,
                        out_slice,
                        gpu,
                        stream,
                    )?;
                }
                next_ds = deepstack_iter.next();
            }
        }

        if timing {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_SEC patch+27blocks(M={p_total}): {:.1}ms",
                _sec1.elapsed().as_secs_f64() * 1000.0
            );
        }
        let _sec2 = std::time::Instant::now();
        // 2026-09-25: 4. Final merger per image → packed rows `0 .. Σmerged_p`.
        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let src = self
                .scratch()
                .buf_h1
                .offset(p_off[i] * self.hidden_size * 2);
            let out_slice = self
                .scratch()
                .buf_out
                .offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(&self.merger, p_i[i], *gh, *gw, src, out_slice, gpu, stream)?;
        }
        if timing {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_SEC mergers(final+{} ds): {:.1}ms",
                self.deepstack_indexes.len(),
                _sec2.elapsed().as_secs_f64() * 1000.0
            );
        }
        // 2026-09-25: Dump the final and DeepStack regions (METRALE_DUMP_VIT).
        let dump_rows = (1 + self.deepstack_indexes.len()) * mp_total;
        Self::maybe_dump_buf(
            gpu,
            self.scratch().buf_out,
            dump_rows * self.out_hidden_size,
            "final",
            stream,
        )?;

        Ok(images
            .iter()
            .map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / sms2))
            .collect())
    }

    /// 2026-09-25: Fallback for Σp > p_max: encode each image alone, writing its
    /// final-merger rows into the packed `buf_out` at `mp_off[i]`, after
    /// `check_packed_rows` has bounded them. No DeepStack rows are written.
    #[allow(clippy::too_many_arguments)]
    fn forward_oversized_fallback(
        &self,
        images: &[(&[f32], usize, usize)],
        p_i: &[usize],
        mp_i: &[usize],
        mp_off: &[usize],
        sms: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        check_packed_rows(mp_i, mp_off, self.p_max)?;
        let pos_interp_on = std::env::var("METRALE_VISION_POSINTERP")
            .map(|v| v != "0")
            .unwrap_or(true);
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            if pos_interp_on {
                self.resample_pos_embed(*gh, *gw, gpu, stream)?;
            } else {
                self.gpu_copy_bf16(
                    gpu,
                    self.pos_embed,
                    self.scratch().buf_pos_resampled,
                    p * self.hidden_size * 2,
                    stream,
                )?;
            }
            self.build_rope_cossin(*gh, *gw, gpu, stream)?;
            self.patch_embed(pixels, p, gpu, stream)?;
            for blk in self.blocks.iter() {
                self.vit_block(blk, p, gpu, stream)?;
            }
            let out_slice = self
                .scratch()
                .buf_out
                .offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(
                &self.merger,
                p,
                *gh,
                *gw,
                self.scratch().buf_h1,
                out_slice,
                gpu,
                stream,
            )?;
        }
        Ok(images
            .iter()
            .map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / (sms * sms)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::check_packed_rows;

    /// 2026-09-25: One video of 19 temporal groups at 30x34 patches merges 2x2 to
    /// 255 rows per group, so the packed write ends at row 4845. Against a
    /// 1024-row buffer it is refused, and the error names both numbers.
    #[test]
    fn refuses_the_video_batch_that_poisoned_the_cuda_context() {
        let groups = 19;
        let merged_per_group = (30 / 2) * (34 / 2);
        let mp_i = vec![merged_per_group; groups];
        let mp_off: Vec<usize> = (0..groups).map(|g| g * merged_per_group).collect();
        assert_eq!(mp_off.last().unwrap() + mp_i.last().unwrap(), 4845);

        let err = check_packed_rows(&mp_i, &mp_off, 1024)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("4845"),
            "must name the rows it would have written: {err}"
        );
        assert!(err.contains("1024"), "must name the capacity: {err}");
    }

    /// 2026-09-25: A batch that fits passes, including one that fills the buffer
    /// exactly; one row more fails.
    #[test]
    fn admits_a_batch_that_fits() {
        assert!(check_packed_rows(&[100, 200], &[0, 100], 1024).is_ok());
        assert!(check_packed_rows(&[24], &[1000], 1024).is_ok());
        assert!(check_packed_rows(&[25], &[1000], 1024).is_err());
    }

    /// 2026-09-25: An empty batch passes; counts and offsets of different lengths
    /// are an error, not a panic.
    #[test]
    fn handles_empty_and_mismatched_layouts_without_panicking() {
        assert!(check_packed_rows(&[], &[], 1024).is_ok());
        let err = check_packed_rows(&[], &[0], 1024).unwrap_err().to_string();
        assert!(err.contains("inconsistent"), "{err}");
    }

    /// 2026-09-25: `usize` addition on request-controlled geometry must not wrap
    /// into a passing comparison.
    #[test]
    fn an_overflowing_offset_is_an_error_not_a_wrap() {
        let err = check_packed_rows(&[2], &[usize::MAX - 1], 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflows"), "{err}");
    }
}
