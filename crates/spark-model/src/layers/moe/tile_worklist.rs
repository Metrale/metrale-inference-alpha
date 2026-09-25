// SPDX-License-Identifier: AGPL-3.0-only

//! Optional Hopper work-list builder. Same ABI, ordering and arena extents.

use super::*;

fn ordered_builder_eligible(experts: u32, n_tiles: u32, m_tile: u32, kernel: bool) -> bool {
    kernel && (1..=256).contains(&experts) && (1..=64).contains(&n_tiles) && m_tile == 128
}

impl MoeLayer {
    pub(super) fn tile_worklist_kernel(
        &self,
        experts: u32,
        n_tiles: u32,
        m_tile: u32,
        ctx: &ForwardContext,
    ) -> KernelHandle {
        if ordered_builder_eligible(
            experts,
            n_tiles,
            m_tile,
            self.moe_build_tile_worklist_ordered_k.0 != 0,
        ) {
            if ctx.stats.once("log:moe_ordered_worklist") {
                tracing::info!("MoE prefill: Hopper parallel ordered work-list builder selected");
            }
            self.moe_build_tile_worklist_ordered_k
        } else {
            self.moe_build_tile_worklist_k
        }
    }
}

#[cfg(test)]
#[path = "tile_worklist_tests.rs"]
mod tests;
