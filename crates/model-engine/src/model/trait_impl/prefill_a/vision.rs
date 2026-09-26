// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Vision-encoder dispatch for prefill, for one request or for every request of a tick in one call.
//!
//! Both functions stage the packed patch embeddings, `vision_embed_patches`
//! and `vision_image_grids` that the chunk-0 splice and MRoPE read.
//!
//! Owner: model-engine.
//! Invariants:
//! - `vision_image_grids` holds one `(t, h, w)` entry per item, in item order.
//! - Without a vision encoder both functions return before touching any state.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    pub(in crate::model) fn prepare_vision_embed_dispatch(
        &self,
        images: &[metrale_model_layers::VisionItem],
    ) -> Result<()> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        // 2026-09-25: One `forward_batched` call over every image of the request.
        // Each item is flattened into one encoder row-block per temporal group,
        // and `item_groups` records each item's group count so the grids below
        // carry its temporal extent. The encoder returns results in input order,
        // which is the order of the pad-token splice.
        let mut img_refs: Vec<(&[f32], usize, usize)> = Vec::new();
        let mut item_groups: Vec<usize> = Vec::with_capacity(images.len());
        for it in images {
            item_groups.push(it.t_len());
            for g in &it.groups {
                img_refs.push((g.as_slice(), it.grid_h, it.grid_w));
            }
        }
        let _vt0 = std::time::Instant::now();
        let per_image = ve.forward_batched(&img_refs, self.gpu.as_ref(), stream)?;
        if std::env::var("METRALE_VISION_TIMING").is_ok() {
            self.gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_TIMING self-encode {} imgs: {:.1}ms",
                images.len(),
                _vt0.elapsed().as_secs_f64() * 1000.0
            );
        }
        // 2026-09-25: Fold the per-group output back into one grid per item,
        // `(t_len, h, w)`; a still image yields `(1, h, w)`.
        let post_merge_grids: Vec<(usize, usize, usize)> = {
            let mut out = Vec::with_capacity(item_groups.len());
            let mut row = 0usize;
            for t_len in &item_groups {
                let (h, w, _) = per_image[row];
                out.push((*t_len, h, w));
                row += t_len;
            }
            out
        };
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        *self.vision_embed_patches.lock() = total_merged;
        *self.vision_image_grids.lock() = post_merge_grids;
        tracing::info!(
            "Vision encoder (batched): {} images, {} merged patches encoded",
            images.len(),
            total_merged
        );
        Ok(())
    }

    /// 2026-09-25: Encode the images of every request in one `forward_batched`
    /// call. `per_request[i]` holds request i's images. Fills the shared packed
    /// output and `vision_image_grids` in request-then-item order, and returns
    /// one `(patch_row_offset, grid_index_offset, num_items, patch_row_count)`
    /// per request. The scheduler passes each request's offsets to
    /// `set_vision_slice_base` before its chunk-0 prefill.
    pub(in crate::model) fn prepare_vision_embed_batched_dispatch(
        &self,
        per_request: &[Vec<metrale_model_layers::VisionItem>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(Vec::new()),
        };
        let stream = self.gpu.default_stream();
        let mut flat: Vec<(&[f32], usize, usize)> = Vec::new();
        let mut req_bounds: Vec<(usize, usize)> = Vec::with_capacity(per_request.len());
        let mut per_req_groups: Vec<Vec<usize>> = Vec::with_capacity(per_request.len());
        for imgs in per_request {
            let start = flat.len();
            let mut groups = Vec::with_capacity(imgs.len());
            for it in imgs {
                groups.push(it.t_len());
                for g in &it.groups {
                    flat.push((g.as_slice(), it.grid_h, it.grid_w));
                }
            }
            // 2026-09-25: Bounds count encoder rows (temporal groups), the unit
            // that `per_image` is indexed by.
            req_bounds.push((start, flat.len() - start));
            per_req_groups.push(groups);
        }
        let per_image = ve.forward_batched(&flat, self.gpu.as_ref(), stream)?;
        let grids: Vec<(usize, usize, usize)> = {
            let mut out = Vec::new();
            let mut row = 0usize;
            for groups in &per_req_groups {
                for t_len in groups {
                    let (h, w, _) = per_image[row];
                    out.push((*t_len, h, w));
                    row += t_len;
                }
            }
            out
        };
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        *self.vision_embed_patches.lock() = total_merged;
        *self.vision_image_grids.lock() = grids;
        // 2026-09-25: Per-request descriptors. Row offsets accumulate the merged
        // rows of earlier requests; grid offsets and counts are in items, because
        // `vision_image_grids` has one entry per item.
        let mut out = Vec::with_capacity(per_request.len());
        let mut row_cursor = 0usize;
        let mut grid_cursor = 0usize;
        for ((enc_start, n_rows), groups) in req_bounds.iter().zip(&per_req_groups) {
            let row_count: usize = per_image[*enc_start..*enc_start + *n_rows]
                .iter()
                .map(|(_, _, mp)| *mp)
                .sum();
            out.push((row_cursor, grid_cursor, groups.len(), row_count));
            row_cursor += row_count;
            grid_cursor += groups.len();
        }
        tracing::info!(
            "Vision encoder (co-dispatch): {} requests, {} images, {} merged patches",
            per_request.len(),
            flat.len(),
            total_merged
        );
        Ok(out)
    }
}
