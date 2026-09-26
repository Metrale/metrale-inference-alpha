// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelAdapters for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::ModelAdapters;

impl ModelAdapters for TransformerModel {
    fn set_active_lora(&mut self, name: &str) -> Result<()> {
        self.rotate_lora_to(name)
    }

    fn adapter_id_for(&self, slot: i32) -> u64 {
        self.adapter_id_for_slot(slot)
    }

    fn acquire_adapter_slot(&self, slot: i32) -> i32 {
        TransformerModel::acquire_adapter_slot(self, slot)
    }

    fn release_adapter_slot(&self, resolved: i32) {
        TransformerModel::release_adapter_slot(self, resolved)
    }

    fn swap_lora_from_disk(
        &mut self,
        dir: &std::path::Path,
        name: &str,
        slot: usize,
    ) -> Result<()> {
        // 2026-09-25: Disk staging reads files and needs no RDMA, unlike the
        // unix-only peer path. It is cuda-gated because it loads into a device
        // weight store.
        #[cfg(feature = "cuda")]
        {
            self.swap_lora_slot_from_disk(dir, name, slot)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dir, name, slot);
            anyhow::bail!("LoRA disk swap requires the cuda feature")
        }
    }

    fn promote_lora_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        name: &str,
        peft: metrale_config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        #[cfg(all(feature = "cuda", unix))]
        {
            self.promote_lora_slot_from_peer(peer_addr, adapter_id, name, peft)
        }
        #[cfg(not(all(feature = "cuda", unix)))]
        {
            let _ = (peer_addr, adapter_id, name, peft);
            anyhow::bail!("LoRA peer promotion stages over RDMA (rdma-core); unix-only")
        }
    }

    fn promote_lora_from_disk(
        &mut self,
        dir: &std::path::Path,
        name: &str,
    ) -> Result<(usize, Option<String>)> {
        #[cfg(feature = "cuda")]
        {
            self.promote_lora_slot_from_disk(dir, name)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dir, name);
            anyhow::bail!("LoRA disk promotion requires the cuda feature")
        }
    }
}
