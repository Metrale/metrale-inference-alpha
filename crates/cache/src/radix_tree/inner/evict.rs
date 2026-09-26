// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LRU eviction and entry counting for [`RadixTreeInner`]. A
//! child module of `inner`, so it can reach the private node fields.
//!
//! Owner: cache.
//! Invariants: eviction removes only leaves with `ref_count <= 1`, never a
//! root.

use super::RadixTreeInner;

impl RadixTreeInner {
    /// 2026-09-25: Remove least-recently-accessed leaves holding only the
    /// cache's ref (`ref_count <= 1`) until at least `num_blocks` blocks are
    /// returned or none is left. A leaf's `partial_suffix` block is returned
    /// with it, so the result can hold `num_blocks + 1`. Returns the blocks and
    /// their parallel disk ids (`u32::MAX` for none; `RadixTree::evict` drops
    /// those).
    pub(in crate::radix_tree) fn evict(&mut self, num_blocks: usize) -> (Vec<u32>, Vec<u32>) {
        let mut freed_phys = Vec::new();
        let mut freed_disk = Vec::new();
        if num_blocks == 0 {
            return (freed_phys, freed_disk);
        }

        loop {
            if freed_phys.len() >= num_blocks {
                break;
            }

            let mut best: Option<(usize, u64)> = None;
            for (id, node) in self.nodes.iter().enumerate() {
                // 2026-09-25: Roots and freed nodes have no parent.
                if node.parent.is_none() {
                    continue;
                }
                if node.ref_count <= 1 && node.children.is_empty() && node.block_idx != u32::MAX {
                    match best {
                        None => best = Some((id, node.last_access)),
                        Some((_, best_access)) if node.last_access < best_access => {
                            best = Some((id, node.last_access));
                        }
                        _ => {}
                    }
                }
            }

            match best {
                Some((node_id, _)) => {
                    let block = self.nodes[node_id].block_idx;
                    let disk = self.nodes[node_id].disk_block_id;
                    freed_phys.push(block);
                    freed_disk.push(disk);

                    if let Some((_, partial_block, partial_disk)) =
                        self.nodes[node_id].partial_suffix.take()
                    {
                        freed_phys.push(partial_block);
                        freed_disk.push(partial_disk);
                    }

                    if let Some(parent_id) = self.nodes[node_id].parent
                        && let Some(key) = self.nodes[node_id].parent_key.clone()
                    {
                        self.nodes[parent_id].children.remove(&key);
                    }

                    self.nodes[node_id].block_idx = u32::MAX;
                    self.nodes[node_id].disk_block_id = u32::MAX;
                    self.nodes[node_id].children.clear();
                    self.nodes[node_id].parent = None;
                    self.nodes[node_id].parent_key = None;
                    self.nodes[node_id].partial_suffix = None;
                    self.free_nodes.push(node_id);
                }
                None => break,
            }
        }

        (freed_phys, freed_disk)
    }

    pub(in crate::radix_tree) fn num_entries(&self) -> usize {
        // 2026-09-25: Roots and freed nodes carry `block_idx == u32::MAX`.
        self.nodes
            .iter()
            .filter(|n| n.block_idx != u32::MAX)
            .count()
    }
}
