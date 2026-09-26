// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelAdapters`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

/// 2026-09-26: LoRA adapters: rotation, slot refs, disk swap and promotion.
pub trait ModelAdapters {
    /// 2026-09-25: Make the resident LoRA adapter `name` the active one. Takes `&mut self`, so it
    /// runs only while nothing else holds the model. Default: an error.
    fn set_active_lora(&mut self, _name: &str) -> Result<()> {
        bail!("this model does not support LoRA adapter rotation")
    }

    /// 2026-09-25: The prefix-cache adapter id for a LoRA slot selector: `>= 0` names a resident
    /// slot, `-1` the active adapter. Default `0`, the base model's id.
    fn adapter_id_for(&self, _slot: i32) -> u64 {
        0
    }

    /// 2026-09-25: Take a ref on the LoRA slot a sequence will use (called at prefill), resolving
    /// `-1` to the active adapter as [`Self::adapter_id_for`] does. Returns the resolved slot,
    /// which the caller must pass to [`Self::release_adapter_slot`], or `-1` when no ref was
    /// taken. Default `-1`.
    fn acquire_adapter_slot(&self, _slot: i32) -> i32 {
        -1
    }

    /// 2026-09-25: Release a ref taken by [`Self::acquire_adapter_slot`], by the slot it returned.
    /// Default: no-op.
    fn release_adapter_slot(&self, _resolved: i32) {}

    /// 2026-09-25: Load the LoRA adapter at `dir` into pool slot `slot`. Takes `&mut self`, so it
    /// runs only while nothing else holds the model. Default: an error.
    fn swap_lora_from_disk(
        &mut self,
        _dir: &std::path::Path,
        _name: &str,
        _slot: usize,
    ) -> Result<()> {
        bail!("this model does not support LoRA disk swap")
    }

    /// 2026-09-25: Fetch the adapter `name` (staged on `peer_addr` as `adapter_id`) from the peer
    /// into a pool slot and make it active; returns `(slot, evicted adapter name)`. `peft`
    /// supplies the rank, alpha and scaling. Default: an error.
    fn promote_lora_from_peer(
        &mut self,
        _peer_addr: &str,
        _adapter_id: &str,
        _name: &str,
        _peft: metrale_config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        bail!("this model does not support LoRA peer promotion")
    }

    /// 2026-09-25: Load the adapter `name` from `adapter_dir` into a pool slot and make it active;
    /// returns `(slot, evicted adapter name)`. Default: an error.
    fn promote_lora_from_disk(
        &mut self,
        _adapter_dir: &std::path::Path,
        _name: &str,
    ) -> Result<(usize, Option<String>)> {
        bail!("this model does not support LoRA disk promotion")
    }
}
