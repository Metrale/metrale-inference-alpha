// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool retrieval: rank declared tools by embedding similarity to a query
//! and keep the top K.
//!
//! Owner: server.
//! Invariants:
//! - `retrieve_top_k` returns at most `k` names, in non-increasing score order.
//! - `filter_tools_by_name` keeps the input order.
//!
//! The embedding model is supplied through [`Embedder`]. A tool is anchored on
//! its example queries, or on its description when it has none. Nothing
//! outside this module uses it.

use std::collections::HashSet;

/// 2026-09-26: The embedding model. All vectors must share one dimension: `cosine`
/// returns 0.0 for a length mismatch.
pub trait Embedder {
    /// 2026-09-26: Encode `s` to an L2-normalised vector.
    fn embed(&self, s: &str) -> Vec<f32>;
}

/// 2026-09-26: One declared tool with its retrieval anchor.
#[derive(Debug, Clone)]
pub struct ToolAnchor {
    /// 2026-09-26: Tool name as it appears in the schema.
    pub name: String,
    /// 2026-09-26: Embedding of the tool's example queries joined with " • ", or of
    /// its description when it has none.
    pub embedding: Vec<f32>,
}

impl ToolAnchor {
    /// 2026-09-26: Build an anchor by embedding the tool's example queries, natural
    /// sentences such as "list the files in a directory". An empty list falls back
    /// to the description.
    pub fn build<E: Embedder>(
        name: &str,
        example_queries: &[&str],
        description_fallback: &str,
        embedder: &E,
    ) -> Self {
        let text = if example_queries.is_empty() {
            description_fallback.to_string()
        } else {
            example_queries.join(" • ")
        };
        Self {
            name: name.to_string(),
            embedding: embedder.embed(&text),
        }
    }
}

/// 2026-09-26: Dot product, which is the cosine similarity for L2-normalised
/// vectors; 0.0 when the lengths differ or the vectors are empty.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 2026-09-26: The `k` tool names most similar to `query_embedding`, in descending
/// similarity order; all of them when `k >= anchors.len()`.
pub fn retrieve_top_k(query_embedding: &[f32], anchors: &[ToolAnchor], k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = anchors
        .iter()
        .map(|a| (cosine(query_embedding, &a.embedding), a.name.as_str()))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(_, name)| name.to_string())
        .collect()
}

/// 2026-09-26: Keep only the tools whose names appear in `keep`, in their input order,
/// so the same kept set renders identically across turns.
pub fn filter_tools_by_name<T, F>(tools: Vec<T>, keep: &[String], get_name: F) -> Vec<T>
where
    F: Fn(&T) -> &str,
{
    let keep_set: HashSet<&str> = keep.iter().map(|s| s.as_str()).collect();
    tools
        .into_iter()
        .filter(|t| keep_set.contains(get_name(t)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-26: Synthetic embedder: sum the input bytes into an 8-dim vector, then
    /// L2-normalise.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn embed(&self, s: &str) -> Vec<f32> {
            let mut v = [0f32; 8];
            for (i, b) in s.bytes().enumerate() {
                v[i % 8] += b as f32;
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            v.iter().map(|x| x / norm).collect()
        }
    }

    #[test]
    fn cosine_normalised_vectors_in_range() {
        let a = StubEmbedder.embed("hello world");
        let b = StubEmbedder.embed("hello there");
        let s = cosine(&a, &b);
        assert!((-1.0..=1.0 + 1e-5).contains(&s));
    }

    #[test]
    fn retrieve_top_k_returns_in_descending_similarity_order() {
        // 2026-09-26: The stub embedder is a byte sum, not a semantic model, so the
        // test checks only that returned names are in non-increasing score order.
        let anchors = vec![
            ToolAnchor::build("read_file", &["read a file"], "", &StubEmbedder),
            ToolAnchor::build("list_dir", &["list a directory"], "", &StubEmbedder),
            ToolAnchor::build("write_file", &["write a file"], "", &StubEmbedder),
        ];
        let q = StubEmbedder.embed("read a file please");
        let top = retrieve_top_k(&q, &anchors, 3);
        assert_eq!(top.len(), 3);
        let scored: Vec<f32> = top
            .iter()
            .map(|name| {
                let a = anchors.iter().find(|a| a.name == *name).unwrap();
                cosine(&q, &a.embedding)
            })
            .collect();
        for w in scored.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-5,
                "scores must be sorted descending: {:?}",
                scored
            );
        }
    }

    #[test]
    fn retrieve_top_k_clamped_to_anchor_count() {
        let anchors = vec![ToolAnchor::build(
            "only",
            &["the only tool"],
            "",
            &StubEmbedder,
        )];
        let q = StubEmbedder.embed("anything");
        let top = retrieve_top_k(&q, &anchors, 100);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0], "only");
    }

    #[test]
    fn filter_tools_preserves_original_order() {
        // 2026-09-26: Kept tools stay in request order, whatever the order of `keep`.
        #[derive(Debug, Clone, PartialEq)]
        struct Tool(&'static str);
        let tools = vec![Tool("a"), Tool("b"), Tool("c"), Tool("d")];
        let keep = vec!["c".to_string(), "a".to_string()];
        let filtered = filter_tools_by_name(tools, &keep, |t| t.0);
        assert_eq!(filtered, vec![Tool("a"), Tool("c")]);
    }

    #[test]
    fn filter_tools_empty_keep_filters_everything() {
        #[derive(Debug, Clone, PartialEq)]
        struct Tool(&'static str);
        let tools = vec![Tool("a"), Tool("b")];
        let keep: Vec<String> = vec![];
        let filtered = filter_tools_by_name(tools, &keep, |t| t.0);
        assert!(filtered.is_empty());
    }

    #[test]
    fn empty_example_queries_fall_back_to_description() {
        let a = ToolAnchor::build("x", &[], "fallback description", &StubEmbedder);
        assert_eq!(a.embedding.len(), 8);
    }
}
