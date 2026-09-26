// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA per-layer install walk, graph-cache drain and runtime rotation
//! (`install_lora_layers`, `destroy_lora_decode_graphs`, `rotate_lora_to`).
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

use metrale_gpu_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: Install one slot's per-layer pairs onto the layer structs;
    /// the startup install, rotation and every swap path use it. `layers` is
    /// indexed by global layer index. Returns the number of layers installed.
    pub(super) fn install_lora_layers(
        &mut self,
        layers: &[Option<metrale_model_layers::lora::LoraLayerWeights>],
        kernels: ops::lora_delta::LoraKernels,
        tables: &std::collections::BTreeMap<
            (usize, metrale_model_layers::lora::LoraModule),
            (
                metrale_gpu_runtime::gpu::DevicePtr,
                metrale_gpu_runtime::gpu::DevicePtr,
            ),
        >,
        scale_table: metrale_gpu_runtime::gpu::DevicePtr,
    ) -> Result<usize> {
        use metrale_model_layers::lora::LoraModule;
        // 2026-09-25: Per-module route: the pool's a/b tables for (layer,
        // module) plus this slot's pair dimensions. `None` when the slot has no
        // pair for the module or the pool has no table for it.
        let mk_route = |layer_idx: usize,
                        module: LoraModule,
                        pair: &Option<ops::lora_delta::LoraPair>|
         -> Option<ops::lora_delta::LoraRoute> {
            let p = pair.as_ref()?;
            let (a_table, b_table) = *tables.get(&(layer_idx, module))?;
            Some(ops::lora_delta::LoraRoute {
                a_table,
                b_table,
                scale_table,
                k_in: p.k_in,
                n_out: p.n_out,
                max_rank: p.max_rank,
            })
        };
        let mut installed = 0usize;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let Some(layer_weights) = layers.get(idx).and_then(|o| o.as_ref()) else {
                continue;
            };
            let has_moe = layer_weights.router.is_some()
                || layer_weights
                    .experts
                    .as_ref()
                    .is_some_and(|e| !e.is_empty());
            // 2026-09-25: Built before the downcast because both layer kinds
            // install the dense FFN delta; building it once keeps them from
            // disagreeing.
            let ffn_weights = if layer_weights.gate_proj.is_some()
                || layer_weights.up_proj.is_some()
                || layer_weights.down_proj.is_some()
            {
                Some(ops::lora_delta::LoraFfnWeights {
                    gate: layer_weights.gate_proj,
                    up: layer_weights.up_proj,
                    down: layer_weights.down_proj,
                    kernels,
                })
            } else {
                None
            };
            let any = layer
                .as_any_mut()
                .ok_or_else(|| anyhow::anyhow!("LoRA: adapted layer {idx} is not downcastable"))?;
            // 2026-09-25: A full-attention layer takes attention, dense-FFN and
            // MoE deltas. A linear-attention layer takes dense-FFN, GDN
            // `out_proj` and MoE deltas; `classify_key` rejects attention
            // projections on it.
            if let Some(attn) =
                any.downcast_mut::<metrale_model_layers::layers::Qwen3AttentionLayer>()
            {
                let attn_weights = ops::lora_delta::LoraAttnWeights {
                    layer_idx: idx,
                    q: layer_weights.q_proj,
                    k: layer_weights.k_proj,
                    v: layer_weights.v_proj,
                    o: layer_weights.o_proj,
                    kernels,
                    q_route: mk_route(idx, LoraModule::QProj, &layer_weights.q_proj),
                    k_route: mk_route(idx, LoraModule::KProj, &layer_weights.k_proj),
                    v_route: mk_route(idx, LoraModule::VProj, &layer_weights.v_proj),
                    o_route: mk_route(idx, LoraModule::OProj, &layer_weights.o_proj),
                };
                attn.set_lora_weights(attn_weights, ffn_weights)?;
                if has_moe {
                    attn.set_moe_lora_weights(
                        layer_weights.router,
                        layer_weights.experts.clone().unwrap_or_default(),
                        kernels,
                        self.gpu.as_ref(),
                    )?;
                }
            } else if let Some(ssm) =
                any.downcast_mut::<metrale_model_layers::layers::Qwen3SsmLayer>()
            {
                // 2026-09-25: `classify_key` rejects q/k/v/o on this layer
                // kind, so one reaching here is a loader bug.
                let has_attn_proj = layer_weights.q_proj.is_some()
                    || layer_weights.k_proj.is_some()
                    || layer_weights.v_proj.is_some()
                    || layer_weights.o_proj.is_some();
                if has_attn_proj {
                    anyhow::bail!(
                        "LoRA: attention-projection delta on linear-attention layer {idx} — \
                         classify should have rejected this (that layer has no q/k/v/o)"
                    );
                }
                if let Some(ffn) = ffn_weights {
                    ssm.set_ffn_lora_weights(ffn).with_context(|| {
                        format!("LoRA: installing dense-FFN delta on linear-attention layer {idx}")
                    })?;
                }
                // 2026-09-25: GDN out_proj: only linear-attention layers have one.
                if let Some(pair) = layer_weights.out_proj {
                    ssm.set_out_proj_lora(pair, kernels);
                }
                if has_moe {
                    ssm.set_moe_lora_weights(
                        layer_weights.router,
                        layer_weights.experts.clone().unwrap_or_default(),
                        kernels,
                        self.gpu.as_ref(),
                    )?;
                }
            } else {
                anyhow::bail!(
                    "LoRA: adapted layer {idx} is neither a Qwen3AttentionLayer nor a \
                     Qwen3SsmLayer (loader/adapter layer-type mismatch)"
                );
            }
            installed += 1;
        }
        Ok(installed)
    }

    /// 2026-09-25: After a rotate or swap re-points the installed LoRA pairs,
    /// drain the decode, batched-decode, K=2/3/4 verify, K=γ verify and fused
    /// decode+verify graph caches and destroy each graph, so none replays the
    /// old pair pointers. `GraphHandle` has no `Drop`, so clearing a map alone
    /// would leak the graphs. A failed destroy is logged. The batched-verify
    /// cache (`verify_batched_graphs`) is not drained here.
    pub(super) fn destroy_lora_decode_graphs(&self) {
        let drain = |name: &str, graphs: Vec<metrale_gpu_runtime::gpu::GraphHandle>| {
            for g in graphs {
                if g.0 != 0
                    && let Err(e) = self.gpu.destroy_graph(g)
                {
                    tracing::warn!("LoRA graph clear: destroy {name}: {e:#}");
                }
            }
        };
        drain(
            "decode_graph",
            self.decode_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "batch_decode_graph",
            self.batch_decode_graphs
                .lock()
                .0
                .drain()
                .map(|(_, (g, _))| g)
                .collect(),
        );
        drain(
            "verify2_graph",
            self.verify2_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "verify3_graph",
            self.verify3_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "verify4_graph",
            self.verify4_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "verify_kgamma_graph",
            self.verify_kgamma_graph
                .lock()
                .drain()
                .map(|(_, g)| g)
                .collect(),
        );
        drain(
            "fused_graph",
            self.fused_graph.lock().drain().map(|(_, g)| g).collect(),
        );
    }

    /// 2026-09-25: Make the resident adapter `name` the active one: re-install
    /// its per-layer pairs with `install_lora_layers`, then
    /// `destroy_lora_decode_graphs`. Call it only at a scheduler-quiescent
    /// point.
    ///
    /// Refuses, changing nothing, when no adapter is loaded, `name` is not
    /// resident, rotation is not armed (`lora_rotatable`), or the current
    /// active slot has in-flight sequences. If the install walk fails, the
    /// active slot has already switched.
    pub fn rotate_lora_to(&mut self, name: &str) -> Result<()> {
        let slot = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA rotation: no adapter loaded"))?;
            lw.slot_of(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "LoRA rotation: adapter '{name}' is not resident (have [{}])",
                    lw.adapter_names().join(", ")
                )
            })?
        };
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA rotation not armed (single adapter, METRALE_LORA_ROTATE unset); \
                 set METRALE_LORA_ROTATE=1 (forces eager decode) to rotate at runtime"
            );
        }
        // 2026-09-25: Rotation re-installs the new slot's pairs onto the layer
        // structs, so a sequence still decoding on the old active adapter
        // would continue with the new delta. Refuse while the active slot has
        // in-flight sequences.
        {
            let lw = self.lora.as_ref().unwrap();
            let cur = lw.active;
            if lw.slot_ref_count(cur) > 0 {
                anyhow::bail!(
                    "LoRA rotation refused: active slot {cur} has in-flight \
                     sequences (ref_count>0); rotate at a quiescent point"
                );
            }
        }
        let (layers, active_name, r, tables, scale_table) = {
            let lw = self.lora.as_mut().unwrap();
            lw.active = slot;
            lw.name = lw.slots[slot].name.clone();
            lw.adapter_config = lw.slots[slot].adapter_config.clone();
            (
                lw.slots[slot].layers.clone(),
                lw.name.clone(),
                lw.adapter_config.r,
                lw.tables.clone(),
                lw.scale_table,
            )
        };
        let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
        let installed = self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
        self.destroy_lora_decode_graphs();
        tracing::info!(
            "LoRA rotation → slot {slot} '{active_name}' (r={r}) re-installed on \
             {installed} layers"
        );
        Ok(())
    }
}
