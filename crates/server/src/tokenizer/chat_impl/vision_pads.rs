// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Vision placeholder token ids, and the expansion of one placeholder per image
//! or video into one per vision-encoder output row.
//!
//! Owner: server (tokenizer).
//! Invariants:
//! - `fan_out_pads` never fails: a placeholder beyond `pad_counts` stays a single token, and
//!   counts beyond the last placeholder are unused.

use super::super::ChatTokenizer;

impl ChatTokenizer {
    /// 2026-09-26: The image placeholder token id.
    ///
    /// `declared` is the `image_token_id` from the checkpoint's config (top level or under
    /// `vision_config`) and wins when non-zero. Otherwise the id is that of `<|image_pad|>`
    /// when the string encodes to exactly one token, else `None`. With `None`,
    /// `expand_vision_pads` leaves image placeholders unexpanded and reports nothing.
    pub fn image_pad_token_id(&self, declared: u32) -> Option<u32> {
        if declared != 0 {
            return Some(declared);
        }
        self.encode("<|image_pad|>")
            .ok()
            .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
    }

    /// 2026-09-26: The video placeholder token id, with the same rule as
    /// `image_pad_token_id`: `declared` when non-zero, else `<|video_pad|>` when it encodes
    /// to one token, else `None`.
    pub fn video_pad_token_id(&self, declared: u32) -> Option<u32> {
        if declared != 0 {
            return Some(declared);
        }
        self.encode("<|video_pad|>")
            .ok()
            .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
    }

    /// 2026-09-26: Expand the image and video placeholders of a rendered token sequence, the
    /// i-th placeholder to `pad_counts[i]` copies (at least one), so the prompt holds one
    /// placeholder per vision-encoder output row. The Qwen3.6 templates in
    /// test_data/chat_templates write one placeholder per image or video. Returns `tokens`
    /// unchanged when every count is at most 1; see `fan_out_pads` for mismatched counts.
    pub fn expand_vision_pads(
        &self,
        tokens: Vec<u32>,
        pad_counts: &[usize],
        declared: (u32, u32),
    ) -> Vec<u32> {
        if pad_counts.is_empty() || pad_counts.iter().all(|&c| c <= 1) {
            return tokens;
        }
        fan_out_pads(
            tokens,
            pad_counts,
            self.image_pad_token_id(declared.0),
            self.video_pad_token_id(declared.1),
        )
    }
}

/// 2026-09-26: The expansion over explicit ids, testable without a tokenizer.
///
/// `pad_counts[i]` is the number of tokens the i-th media item, image or video, produces.
/// Placeholders take counts left to right, one each; a placeholder with no count left stays
/// one token. Returns `tokens` unchanged when both ids are `None`.
pub(crate) fn fan_out_pads(
    tokens: Vec<u32>,
    pad_counts: &[usize],
    image_pad: Option<u32>,
    video_pad: Option<u32>,
) -> Vec<u32> {
    if image_pad.is_none() && video_pad.is_none() {
        return tokens;
    }
    let extra: usize = pad_counts.iter().map(|c| c.saturating_sub(1)).sum();
    let mut out = Vec::with_capacity(tokens.len() + extra);
    let mut img_idx = 0usize;
    for t in tokens {
        if Some(t) == image_pad || Some(t) == video_pad {
            let count = pad_counts.get(img_idx).copied().unwrap_or(1).max(1);
            for _ in 0..count {
                out.push(t);
            }
            img_idx += 1;
        } else {
            out.push(t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::fan_out_pads;

    /// 2026-09-26: 154854 is the `image_token_id` in the GLM-5.3 config fixture
    /// (crates/model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json). With that id
    /// the placeholder expands; with no id nothing changes.
    #[test]
    fn a_declared_id_expands_where_a_literal_probe_would_not_have() {
        let glm_image = 154_854u32;
        let tokens = vec![1, 154_830, glm_image, 154_831, 2];
        let out = fan_out_pads(tokens.clone(), &[256], Some(glm_image), None);
        assert_eq!(out.len(), tokens.len() + 255);
        assert_eq!(out.iter().filter(|&&t| t == glm_image).count(), 256);
        assert_eq!(out[1], 154_830);
        assert_eq!(*out.last().unwrap(), 2);
        assert_eq!(out[out.len() - 2], 154_831);

        assert_eq!(fan_out_pads(tokens.clone(), &[256], None, None), tokens);
    }

    /// 2026-09-26: Images and videos take counts from one left-to-right cursor.
    #[test]
    fn counts_are_consumed_in_order_across_both_modalities() {
        let (img, vid) = (100u32, 101u32);
        let out = fan_out_pads(vec![img, 7, vid, img], &[2, 3, 1], Some(img), Some(vid));
        assert_eq!(out, vec![img, img, 7, vid, vid, vid, img]);
    }

    /// 2026-09-26: More placeholders than counts: the surplus stays one token each.
    #[test]
    fn a_short_count_list_does_not_reuse_the_last_count() {
        let img = 100u32;
        let out = fan_out_pads(vec![img, img], &[3], Some(img), None);
        assert_eq!(out, vec![img, img, img, img]);
    }
}
