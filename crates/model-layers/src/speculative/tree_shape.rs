// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Spine-plus-hedge draft trees for tree speculative decoding: the
//! tree shape, its verify-row layout and the accepted path.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - `TreeShape::parse` returns only shapes that pass `validate`: every depth
//!   holds 1..=`MAX_RANK` nodes, and the tree holds at most `MAX_NODES`.
//! - `TreeDraft::rows` puts the root at row 0, the spine at rows `1..=L` in
//!   depth order, then the hedges in the order of `hedges`.
//!
//! A tree is a spine (the drafter's top-1 chain) plus hedge leaves: lower-ranked
//! siblings of spine nodes, with no children. So every node's parent is the
//! root or a spine node. A shape is a comma-separated list of per-depth node
//! counts, e.g. `"1,2,2,2"`: depth d holds one spine node and `count_d - 1`
//! hedges, and the verify width is the node count plus the root row.

use anyhow::{Result, bail};

/// 2026-09-25: Largest node count `validate` accepts at one depth: the spine
/// node plus up to three hedges.
pub const MAX_RANK: usize = 4;
/// 2026-09-25: Largest node count `validate` accepts for the whole tree,
/// excluding the root row.
pub const MAX_NODES: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeShape {
    /// 2026-09-25: `counts[d-1]` is the number of nodes at depth d; 1 means
    /// the spine node only.
    pub counts: Vec<u8>,
}

impl TreeShape {
    pub fn parse(s: &str) -> Result<Self> {
        let counts: Vec<u8> = s
            .split(',')
            .map(|t| t.trim().parse::<u8>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("METRALE_TREE_SHAPE parse: {e}"))?;
        let shape = Self { counts };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(&self) -> Result<()> {
        if self.counts.is_empty() {
            bail!("tree shape: empty");
        }
        if self.counts.iter().any(|&c| c == 0 || c as usize > MAX_RANK) {
            bail!("tree shape: per-depth count must be 1..={MAX_RANK}");
        }
        if self.nodes() > MAX_NODES {
            bail!("tree shape: {} nodes > max {MAX_NODES}", self.nodes());
        }
        Ok(())
    }

    pub fn spine_len(&self) -> usize {
        self.counts.len()
    }

    /// 2026-09-25: Total tree nodes (spine and hedges), excluding the root row.
    pub fn nodes(&self) -> usize {
        self.counts.iter().map(|&c| c as usize).sum()
    }

    /// 2026-09-25: Verify width: the root row plus the nodes.
    pub fn verify_width(&self) -> usize {
        self.nodes() + 1
    }

    /// 2026-09-25: The counts packed 4 bits per depth. Distinct for distinct
    /// valid shapes: every count is in 1..=4, so no nibble is zero.
    pub fn shape_id(&self) -> u64 {
        self.counts
            .iter()
            .fold(0u64, |acc, &c| (acc << 4) | (c as u64))
    }

    /// 2026-09-25: True when no depth holds a hedge.
    pub fn is_chain(&self) -> bool {
        self.counts.iter().all(|&c| c == 1)
    }
}

/// 2026-09-25: One hedge leaf: the drafter's rank-`rank` candidate at `depth`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HedgeNode {
    pub depth: usize,
    /// 2026-09-25: Candidate rank. The spine holds rank 1, so a hedge's rank
    /// is 2 or more.
    pub rank: usize,
    pub token: u32,
}

/// 2026-09-25: A proposed draft tree for one verify step.
#[derive(Debug, Clone)]
pub struct TreeDraft {
    pub shape: TreeShape,
    /// 2026-09-25: `spine[d-1]` is the top-1 draft at depth d.
    pub spine: Vec<u32>,
    /// 2026-09-25: Hedge leaves; `rows` lays them out in this order, and
    /// callers keep it sorted by (depth, rank).
    pub hedges: Vec<HedgeNode>,
}

/// 2026-09-25: One verify row. Row 0 is the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeRow {
    pub token: u32,
    pub depth: usize,
    pub parent_row: usize,
}

impl TreeDraft {
    /// 2026-09-25: Verify rows `[root, spine..., hedges...]`. `root_token` is
    /// the last committed token. A depth-d node's parent is row d-1, the
    /// spine node (or the root) one depth up.
    pub fn rows(&self, root_token: u32) -> Vec<TreeRow> {
        let l = self.spine.len();
        let mut rows = Vec::with_capacity(1 + l + self.hedges.len());
        rows.push(TreeRow {
            token: root_token,
            depth: 0,
            parent_row: 0,
        });
        for (i, &t) in self.spine.iter().enumerate() {
            rows.push(TreeRow {
                token: t,
                depth: i + 1,
                parent_row: i,
            });
        }
        for h in &self.hedges {
            rows.push(TreeRow {
                token: h.token,
                depth: h.depth,
                parent_row: h.depth - 1,
            });
        }
        rows
    }

    /// 2026-09-25: The accepted rows, root to leaf, given `v[row]`, the
    /// target's next token after each row. Returns them with the bonus token
    /// `v` at the last accepted row (row 0 when none is accepted).
    ///
    /// Each step accepts the child whose token equals `v[parent]`, so the
    /// accepted tokens are the ones greedy decoding of the target emits. The
    /// spine child is checked first, then the hedges in order, and the first
    /// match wins. Accepting a hedge ends the path, since a hedge is a leaf.
    ///
    /// Panics when `rows` or `v` is shorter than the row layout of this draft.
    pub fn accept_path(&self, rows: &[TreeRow], v: &[u32]) -> (Vec<usize>, u32) {
        let l = self.spine.len();
        let mut path = Vec::with_capacity(l);
        let mut cur = 0usize;
        for d in 1..=l {
            let want = v[cur];
            // 2026-09-25: `cur` is a spine row at depth d-1 (or the root); its
            // children are the spine row at depth d and every hedge at depth d.
            let spine_row = d;
            let mut next = None;
            if rows[spine_row].token == want {
                next = Some(spine_row);
            } else {
                for (i, h) in self.hedges.iter().enumerate() {
                    if h.depth == d && h.token == want {
                        next = Some(1 + l + i);
                        break;
                    }
                }
            }
            match next {
                Some(r) => {
                    path.push(r);
                    cur = r;
                    if r > l {
                        break;
                    }
                }
                None => break,
            }
        }
        (path, v[cur])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(shape: &str, spine: &[u32], hedges: &[(usize, usize, u32)]) -> TreeDraft {
        TreeDraft {
            shape: TreeShape::parse(shape).unwrap(),
            spine: spine.to_vec(),
            hedges: hedges
                .iter()
                .map(|&(depth, rank, token)| HedgeNode { depth, rank, token })
                .collect(),
        }
    }

    #[test]
    fn parse_and_validate() {
        let s = TreeShape::parse("1,2,2,2").unwrap();
        assert_eq!(s.spine_len(), 4);
        assert_eq!(s.nodes(), 7);
        assert_eq!(s.verify_width(), 8);
        assert!(!s.is_chain());
        assert!(TreeShape::parse("1,1").unwrap().is_chain());
        assert!(TreeShape::parse("").is_err());
        assert!(TreeShape::parse("1,0").is_err());
        assert!(TreeShape::parse("5").is_err());
        assert!(TreeShape::parse("2,2,2,2").is_err());
        assert_ne!(
            TreeShape::parse("1,2").unwrap().shape_id(),
            TreeShape::parse("2,1").unwrap().shape_id()
        );
    }

    #[test]
    fn row_layout_spine_prefix() {
        let t = draft("2,2", &[10, 20], &[(1, 2, 11), (2, 2, 21)]);
        let rows = t.rows(5);
        assert_eq!(rows.len(), 5);
        assert_eq!(
            rows[0],
            TreeRow {
                token: 5,
                depth: 0,
                parent_row: 0
            }
        );
        assert_eq!(
            rows[1],
            TreeRow {
                token: 10,
                depth: 1,
                parent_row: 0
            }
        );
        assert_eq!(
            rows[2],
            TreeRow {
                token: 20,
                depth: 2,
                parent_row: 1
            }
        );
        assert_eq!(
            rows[3],
            TreeRow {
                token: 11,
                depth: 1,
                parent_row: 0
            }
        );
        assert_eq!(
            rows[4],
            TreeRow {
                token: 21,
                depth: 2,
                parent_row: 1
            }
        );
    }

    #[test]
    fn accept_full_spine() {
        let t = draft("1,1", &[10, 20], &[]);
        let rows = t.rows(5);
        let (path, bonus) = t.accept_path(&rows, &[10, 20, 99]);
        assert_eq!(path, vec![1, 2]);
        assert_eq!(bonus, 99);
    }

    #[test]
    fn accept_hedge_rescue_is_terminal() {
        // 2026-09-25: The hedge's bonus equals the depth-2 spine token, so a
        // path that continued below the hedge would accept row 2 as well.
        let t = draft("2,1", &[10, 20], &[(1, 2, 11)]);
        let rows = t.rows(5);
        let (path, bonus) = t.accept_path(&rows, &[11, 55, 66, 20]);
        assert_eq!(path, vec![3]);
        assert_eq!(bonus, 20);
    }

    #[test]
    fn accept_hedge_after_spine_prefix_is_terminal() {
        // 2026-09-25: The first spine token is accepted before a depth-2 hedge
        // matches, so the hedge lookup runs after `cur` has advanced.
        let t = draft("1,2,1", &[10, 20, 30], &[(2, 2, 21)]);
        let rows = t.rows(5);
        let (path, bonus) = t.accept_path(&rows, &[10, 21, 66, 77, 88]);
        assert_eq!(path, vec![1, 4]);
        assert_eq!(bonus, 88);
    }

    #[test]
    fn accept_reject_all_gives_root_bonus() {
        let t = draft("2,1", &[10, 20], &[(1, 2, 11)]);
        let rows = t.rows(5);
        let (path, bonus) = t.accept_path(&rows, &[42, 0, 0, 0]);
        assert!(path.is_empty());
        assert_eq!(bonus, 42);
    }
}
