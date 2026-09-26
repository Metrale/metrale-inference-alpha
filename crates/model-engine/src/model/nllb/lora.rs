// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: PEFT LoRA for the served NLLB model, applied with the shared
//! [`apply_lora_delta`] kernels ([`LoraKernels`], [`LoraPair`]).
//!
//! After each adapted projection, `y += scale·(x·Aᵀ)·Bᵀ`, keyed by the base weight path
//! (e.g. `model.decoder.layers.0.encoder_attn.q_proj`), which is the prefix
//! `linear`/`linear1` pass. `NllbLora` owns the adapter's `WeightStore`, so the
//! `LoraPair` pointers stay valid for as long as it lives.
//!
//! Owner: model-engine.
//! Invariants:
//! - Every loaded module has both A and B, 2-D, with matching rank.
//! - `apply` refuses more than `max_rows` rows.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layers::ops::lora_delta::{LoraKernels, LoraPair, apply_lora_delta};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: The PEFT config fields read for the LoRA scaling (`r`,
/// `lora_alpha`, `use_rslora`). `target_modules` is not read: the safetensors
/// A/B keys decide which modules are adapted.
#[derive(serde::Deserialize)]
struct PeftCfg {
    r: usize,
    lora_alpha: f64,
    #[serde(default)]
    use_rslora: bool,
}

pub(super) struct NllbLora {
    /// 2026-09-25: Owns the adapter A/B device buffers referenced by every `LoraPair`.
    _store: WeightStore,
    kernels: LoraKernels,
    /// 2026-09-25: Keyed by base-module path (the PEFT key without
    /// `base_model.model.` and `.lora_A.weight`/`.lora_B.weight`).
    pairs: HashMap<String, LoraPair>,
    xa: DevicePtr,
    delta: DevicePtr,
    max_rows: usize,
}

impl NllbLora {
    /// 2026-09-25: Load a PEFT adapter directory (`adapter_config.json` and its
    /// safetensors) for NLLB. `max_rows` is the most rows one `apply` call
    /// accepts; the model passes `cache_rows`.
    pub(super) fn load(dir: &Path, gpu: &dyn GpuBackend, max_rows: usize) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("adapter_config.json"))
            .with_context(|| format!("reading {}/adapter_config.json", dir.display()))?;
        let peft: PeftCfg =
            serde_json::from_str(&raw).context("parsing NLLB adapter_config.json")?;
        if peft.r == 0 {
            bail!("NLLB adapter: r must be > 0");
        }
        let scale = if peft.use_rslora {
            peft.lora_alpha as f32 / (peft.r as f32).sqrt()
        } else {
            peft.lora_alpha as f32 / peft.r as f32
        };
        let store = metrale_model_weights::weights::adapter::load_adapter_safetensors(dir, gpu, 0)
            .context("loading NLLB adapter safetensors")?;

        let mut a_map: HashMap<String, (Vec<usize>, DevicePtr)> = HashMap::new();
        let mut b_map: HashMap<String, (Vec<usize>, DevicePtr)> = HashMap::new();
        for name in store.names() {
            let t = store.get(name)?;
            if let Some(m) = strip_lora_key(name, ".lora_A.weight") {
                a_map.insert(m, (t.shape.clone(), t.ptr));
            } else if let Some(m) = strip_lora_key(name, ".lora_B.weight") {
                b_map.insert(m, (t.shape.clone(), t.ptr));
            }
        }

        let mut pairs = HashMap::new();
        let (mut max_rank, mut max_n_out) = (0u32, 0usize);
        for (module, (a_shape, a_ptr)) in a_map {
            let (b_shape, b_ptr) = b_map
                .remove(&module)
                .ok_or_else(|| anyhow!("NLLB adapter: lora_A without lora_B for '{module}'"))?;
            if a_shape.len() != 2 || b_shape.len() != 2 {
                bail!("NLLB adapter: '{module}' A/B must be 2-D");
            }
            // 2026-09-25: A: [rank, k_in], B: [n_out, rank].
            let (rank, k_in) = (a_shape[0] as u32, a_shape[1] as u32);
            let n_out = b_shape[0] as u32;
            if b_shape[1] as u32 != rank {
                bail!(
                    "NLLB adapter: '{module}' rank mismatch A={rank} B={}",
                    b_shape[1]
                );
            }
            max_rank = max_rank.max(rank);
            max_n_out = max_n_out.max(n_out as usize);
            pairs.insert(
                module,
                LoraPair {
                    a: DenseWeight { weight: a_ptr },
                    b: DenseWeight { weight: b_ptr },
                    rank,
                    k_in,
                    n_out,
                    scale,
                    // 2026-09-25: One adapter, so no rank padding: the kernels run at its rank.
                    max_rank: rank,
                },
            );
        }
        if let Some((leftover, _)) = b_map.into_iter().next() {
            bail!("NLLB adapter: lora_B without lora_A for '{leftover}'");
        }
        if pairs.is_empty() {
            bail!(
                "NLLB adapter '{}' contained no lora_A/lora_B tensors",
                dir.display()
            );
        }

        let kernels = LoraKernels::new(gpu)?;
        let xa = gpu.alloc(max_rows * max_rank as usize * 2)?;
        let delta = gpu.alloc(max_rows * max_n_out * 2)?;
        tracing::info!(
            "NLLB LoRA loaded: {} modules, r_max={max_rank}, scale={scale:.4} from {}",
            pairs.len(),
            dir.display()
        );
        Ok(Self {
            _store: store,
            kernels,
            pairs,
            xa,
            delta,
            max_rows,
        })
    }

    /// 2026-09-25: Add the LoRA residual for base-module `prefix` in place onto
    /// `base_out` (`[m, n_out]` bf16), reading input `x` (`[m, k_in]` bf16).
    /// No-op if this module is not adapted; fails when `m > max_rows`.
    pub(super) fn apply(
        &self,
        gpu: &dyn GpuBackend,
        prefix: &str,
        x: DevicePtr,
        base_out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(pair) = self.pairs.get(prefix) {
            if m as usize > self.max_rows {
                bail!("nllb lora: m={m} exceeds scratch rows {}", self.max_rows);
            }
            apply_lora_delta(
                gpu,
                &self.kernels,
                pair,
                x,
                base_out,
                m,
                self.xa,
                self.delta,
                stream,
            )?;
        }
        Ok(())
    }
}

/// 2026-09-25: Recover the base-module path from a PEFT key: strip a trailing
/// `suffix` and a leading `base_model.model.` wrapper. `None` if `suffix` is absent.
fn strip_lora_key(name: &str, suffix: &str) -> Option<String> {
    let base = name.strip_suffix(suffix)?;
    Some(
        base.strip_prefix("base_model.model.")
            .unwrap_or(base)
            .to_string(),
    )
}
