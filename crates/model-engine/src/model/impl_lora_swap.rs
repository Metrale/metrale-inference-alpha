// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Runtime LoRA slot swap and promotion, from an RDMA peer or from disk.
//!
//! Owner: model-engine.
//! Invariants:
//! - A swap refused for a busy slot, a disarmed rotation or a missing pool has not
//!   touched the slot's bytes or identity.

use anyhow::{Context, Result};

use metrale_gpu_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: Stage the adapter `adapter_name` (held by the peer at
    /// `peer_addr` as `adapter_id`) into pool `slot` in place and make it that
    /// slot's resident adapter. Call it only at a scheduler-quiescent point.
    ///
    /// Refuses when no pool is loaded, rotation is not armed
    /// (`lora_rotatable`), `slot >= max_loras`, or the slot has in-flight
    /// sequences. Otherwise the slot region is zeroed, the tensors land over
    /// RDMA, the slot's pairs are rebuilt with the new r/scale, its tables are
    /// refreshed and its generation bumped. If the slot is active, its pairs
    /// are re-installed and the LoRA graph caches destroyed.
    #[cfg(feature = "cuda")]
    // 2026-09-25: Peer staging lands the tensors over RDMA, so this is
    // unix-only. The `_from_disk` functions below are plain file I/O.
    #[cfg(unix)]
    pub fn swap_lora_slot_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        adapter_name: &str,
        slot: usize,
        peft: metrale_config::PeftAdapterConfig,
    ) -> Result<()> {
        use metrale_model_layers::lora::rdma_stage;

        let (pool, max_rank, max_loras) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: no adapter pool loaded"))?;
            (lw.pool, lw.max_rank, lw.max_loras)
        };
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA RDMA swap needs rotation armed (set $METRALE_LORA_PEER or \
                 METRALE_LORA_ROTATE=1 so decode runs eager)"
            );
        }
        if slot >= max_loras {
            anyhow::bail!("LoRA RDMA swap: slot {slot} >= max_loras {max_loras}");
        }
        // 2026-09-25: Refuse a busy slot before the memset below, so a refused
        // swap leaves the slot's bytes and identity untouched.
        {
            let busy = self.lora.as_ref().unwrap().slot_ref_count(slot);
            if busy > 0 {
                anyhow::bail!(
                    "LoRA RDMA swap REFUSED: slot {slot} has {busy} in-flight \
                     sequence(s) (ref_count>0); cannot replace an adapter mid-decode"
                );
            }
        }

        let manifest = rdma_stage::fetch_adapter_manifest(peer_addr, adapter_id)?;
        let targets =
            rdma_stage::build_land_targets(&manifest, &self.config, pool, slot, max_rank)?;

        // 2026-09-25: Zero the slot region before landing: a reused slot still
        // holds the previous adapter's bytes.
        let slot_bytes = rdma_stage::slot_bytes(&self.config, max_rank);
        let slot_base = DevicePtr(pool.0 + (slot * slot_bytes) as u64);
        self.gpu.memset(slot_base, 0, slot_bytes)?;
        let loader = metrale_model_weights::weight_lora_rdma::RdmaLoraLoader::new(
            peer_addr.to_string(),
            adapter_id.to_string(),
        );
        loader.stage_into_slot(self.gpu.as_ref(), &targets)?;

        let layers =
            rdma_stage::rebuild_slot_layers(&targets, &self.config, &peft, pool, slot, max_rank)?;
        // 2026-09-25: Refresh this slot's a/b pointer tables and scale table from
        // the new adapter's module coverage, so the slot keeps no route entry
        // from the adapter it replaced. `pack_store_into_slot` does the same for
        // the disk swap.
        self.lora.as_ref().unwrap().refresh_slot_tables(
            slot,
            &layers,
            peft.scaling(),
            self.gpu.as_ref(),
        )?;
        {
            let lw = self.lora.as_mut().unwrap();
            let s = lw
                .slots
                .get_mut(slot)
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: slot {slot} not resident"))?;
            s.name = adapter_name.to_string();
            s.adapter_config = peft;
            s.layers = layers;
            // 2026-09-25: Bump the generation so the slot yields a new adapter
            // id, and a same-name request cannot hit prefix-cache entries keyed
            // to the old contents.
            s.generation = s.generation.wrapping_add(1);
        }

        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let installed_layers = self.lora.as_ref().unwrap().slots[slot].layers.clone();
            let tables = self.lora.as_ref().unwrap().tables.clone();
            let scale_table = self.lora.as_ref().unwrap().scale_table;
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&installed_layers, kernels, &tables, scale_table)?;
            self.lora.as_mut().unwrap().name = adapter_name.to_string();
            self.destroy_lora_decode_graphs();
        }
        tracing::info!(
            "LoRA RDMA swap: '{adapter_name}' landed in slot {slot} \
             ({} targets, active_slot={active})",
            targets.len()
        );
        Ok(())
    }

    /// 2026-09-25: Promote the adapter `adapter_name` from the peer into a
    /// cache-region pool slot and make it active; returns `(slot,
    /// evicted_name)`. Call it only at a scheduler-quiescent point.
    ///
    /// The victim (`select_victim_slot`) is a never-filled placeholder first,
    /// else the least recently used idle (`ref_count == 0`) cache slot. With
    /// every cache slot busy it fails with `POOL_FULL`; a busy slot is never
    /// evicted. [`Self::swap_lora_slot_from_peer`] re-checks the ref count and
    /// bumps the slot generation.
    #[cfg(feature = "cuda")]
    // 2026-09-25: Peer staging lands the tensors over RDMA, so this is
    // unix-only. The `_from_disk` functions below are plain file I/O.
    #[cfg(unix)]
    pub fn promote_lora_slot_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        adapter_name: &str,
        peft: metrale_config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        let (slot, evicted) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA promote: no adapter pool loaded"))?;
            let views = lw.cache_slot_views();
            let slot =
                metrale_model_layers::lora::select_victim_slot(&views).map_err(|e| match e {
                    metrale_model_layers::lora::VictimError::PoolFull => anyhow::anyhow!(
                        "POOL_FULL: all {} cache slot(s) are busy (ref_count>0); retry",
                        views.len()
                    ),
                })?;
            // 2026-09-25: The name the victim held, if any, returned as `evicted_name`.
            let evicted = lw
                .slots
                .get(slot)
                .map(|s| s.name.clone())
                .filter(|n| !n.is_empty());
            (slot, evicted)
        };

        self.swap_lora_slot_from_peer(peer_addr, adapter_id, adapter_name, slot, peft)?;

        // 2026-09-25: Make the promoted slot active. If it already was,
        // `swap_lora_slot_from_peer` has re-installed it; otherwise install its
        // pairs here.
        let already_active = self.lora.as_ref().unwrap().active == slot;
        if !already_active {
            let (layers, tables, scale_table) = {
                let lw = self.lora.as_mut().unwrap();
                lw.active = slot;
                lw.name = lw.slots[slot].name.clone();
                lw.adapter_config = lw.slots[slot].adapter_config.clone();
                (
                    lw.slots[slot].layers.clone(),
                    lw.tables.clone(),
                    lw.scale_table,
                )
            };
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.destroy_lora_decode_graphs();
        }
        // 2026-09-25: Mark the promoted slot most recently used, so a
        // back-to-back promote of a different adapter picks an older victim
        // before the request that triggered this one has taken its ref.
        self.lora.as_ref().unwrap().touch_slot(slot);
        tracing::info!(
            "LoRA promote: '{adapter_name}' hot in cache slot {slot} \
             (evicted={:?}), now active",
            evicted
        );
        Ok((slot, evicted))
    }

    /// 2026-09-25: [`Self::promote_lora_slot_from_peer`] with the adapter
    /// loaded from `adapter_dir` (named `name`) instead of a peer: same victim
    /// choice, same make-active step, returns `(slot, evicted_name)`.
    /// [`Self::swap_lora_slot_from_disk`] reads the directory's
    /// `adapter_config.json`, refuses when rotation is not armed, re-checks
    /// the ref count and bumps the slot generation. Call it only at a
    /// scheduler-quiescent point.
    pub fn promote_lora_slot_from_disk(
        &mut self,
        adapter_dir: &std::path::Path,
        name: &str,
    ) -> Result<(usize, Option<String>)> {
        let (slot, evicted) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA disk promote: no adapter pool loaded"))?;
            let views = lw.cache_slot_views();
            let slot =
                metrale_model_layers::lora::select_victim_slot(&views).map_err(|e| match e {
                    metrale_model_layers::lora::VictimError::PoolFull => anyhow::anyhow!(
                        "POOL_FULL: all {} cache slot(s) are busy (ref_count>0); retry",
                        views.len()
                    ),
                })?;
            // 2026-09-25: The name the victim held, if any, returned as `evicted_name`.
            let evicted = lw
                .slots
                .get(slot)
                .map(|s| s.name.clone())
                .filter(|n| !n.is_empty());
            (slot, evicted)
        };

        self.swap_lora_slot_from_disk(adapter_dir, name, slot)?;

        // 2026-09-25: Make the promoted slot active. If it already was,
        // `swap_lora_slot_from_disk` has re-installed it; otherwise install its
        // pairs here.
        let already_active = self.lora.as_ref().unwrap().active == slot;
        if !already_active {
            let (layers, tables, scale_table) = {
                let lw = self.lora.as_mut().unwrap();
                lw.active = slot;
                lw.name = lw.slots[slot].name.clone();
                lw.adapter_config = lw.slots[slot].adapter_config.clone();
                (
                    lw.slots[slot].layers.clone(),
                    lw.tables.clone(),
                    lw.scale_table,
                )
            };
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.destroy_lora_decode_graphs();
        }
        // 2026-09-25: Mark the promoted slot most recently used, so a
        // back-to-back promote of a different adapter picks an older victim
        // before the request that triggered this one has taken its ref.
        self.lora.as_ref().unwrap().touch_slot(slot);
        tracing::info!(
            "LoRA disk-promote: '{name}' hot in cache slot {slot} \
             (evicted={:?}), now active",
            evicted
        );
        Ok((slot, evicted))
    }

    /// 2026-09-25: Load the adapter at `adapter_dir` into pool `slot` in place
    /// and make it that slot's resident adapter, using the directory's own
    /// `adapter_config.json`. If the slot is active, its pairs are re-installed
    /// and the LoRA graph caches destroyed. Call it only at a
    /// scheduler-quiescent point.
    ///
    /// Refuses when rotation is not armed (`lora_rotatable`) or the slot has
    /// in-flight sequences. `pack_store_into_slot` also refuses adapters with
    /// router/expert deltas or token-overlay tensors.
    pub fn swap_lora_slot_from_disk(
        &mut self,
        adapter_dir: &std::path::Path,
        name: &str,
        slot: usize,
    ) -> Result<()> {
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA disk swap needs rotation armed (set METRALE_LORA_ROTATE=1 so \
                 decode runs eager); a single startup adapter with no rotation env \
                 is baked into the decode graph and a re-point would replay stale"
            );
        }
        // 2026-09-25: Refuse a busy slot before the disk load.
        // `pack_store_into_slot` checks again before its memset.
        if let Some(lw) = self.lora.as_ref() {
            let busy = lw.slot_ref_count(slot);
            if busy > 0 {
                anyhow::bail!(
                    "LoRA disk swap REFUSED: slot {slot} has {busy} in-flight \
                     sequence(s) (ref_count>0); cannot replace an adapter mid-decode"
                );
            }
        }
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let peft = metrale_config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("parse {}", cfg_path.display()))?;
        let store = metrale_model_weights::weights::adapter::load_adapter_safetensors(
            adapter_dir,
            self.gpu.as_ref(),
            0,
        )
        .context("load LoRA adapter weights for disk swap")?;
        let layers = {
            let lw = self
                .lora
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("LoRA disk swap: no adapter pool loaded"))?;
            metrale_model_layers::lora::pack_store_into_slot(
                lw,
                slot,
                name,
                &store,
                &peft,
                &self.config,
                self.gpu.as_ref(),
            )?
        };
        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let tables = self.lora.as_ref().unwrap().tables.clone();
            let scale_table = self.lora.as_ref().unwrap().scale_table;
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.lora.as_mut().unwrap().name = name.to_string();
            self.destroy_lora_decode_graphs();
        }
        tracing::info!(
            "LoRA disk swap: '{name}' packed into slot {slot} (r={}, active_slot={active})",
            peft.r
        );
        Ok(())
    }
}
