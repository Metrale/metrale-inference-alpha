// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash text-model skeleton: topology, residual/norm/mHC wiring, structural weight contract and state plan, as data.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A skeleton from `from_config` has one `layers` entry per `layer_types` entry, each with
//!   `hyper_connection` set and `is_mtp` clear. The MTP layer, when present, is held in `mtp`,
//!   at index `num_hidden_layers`, with no hyper-connection.
//!
//! It executes nothing: there is no forward pass here, and the MLP site is a named step
//! (`ResidualStep::Mlp`) with its kind recorded.
//!
//! Facts about the reference checkpoint `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`, checked
//! against its fixtures by `crates/model-engine/tests/glm5next_skeleton.rs`:
//! * The structural (non-MLP) surface is 1,047 tensors in three signatures: 34 KDA layers
//!   x 23, 11 DSA layers x 22, the MTP layer x 20, plus 3 non-layer tensors.
//! * The MTP layer has no hyper-connection: all 270 `hc_*` tensors are on layers 0..=44.
//! * The final collapse is an unweighted mean: the checkpoint has no `hc_head` tensor.
//! * Layers 0..=2 carry a dense MLP (`first_k_dense_replace = 3`); the other 42 text layers
//!   and the MTP layer route to experts.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use metrale_config::{LayerType, ModelConfig};

mod budget;
pub use budget::{StateBudget, StructuralAccounting};

/// 2026-09-25: Which mixer a layer runs. Narrower than [`LayerType`]: `from_config` refuses
/// the kinds GLM-5.3 does not have rather than carrying them as unreachable arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// 2026-09-25: Kimi Delta Attention: recurrent state, no KV cache.
    Kda,
    /// 2026-09-25: Sparse MLA behind a top-k indexer; attends a paged KV cache.
    Dsa,
}

/// 2026-09-25: Which MLP a layer runs. Recorded, not executed, by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mlp {
    Dense,
    RoutedMoe,
}

/// 2026-09-25: One decoder layer's structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletonLayer {
    pub index: usize,
    pub mixer: Mixer,
    pub mlp: Mlp,
    /// 2026-09-25: False only for the MTP layer.
    pub hyper_connection: bool,
    /// 2026-09-25: True only for the MTP layer, which is not part of the text stack.
    pub is_mtp: bool,
}

/// 2026-09-25: One step of a layer's residual path, in execution order.
///
/// At each of its two sites a text layer runs `hc_pre` → norm → sublayer → `hc_post`; the
/// residual streams entering `hc_pre` are the ones `hc_post` mixes through `comb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualStep {
    /// 2026-09-25: Snapshot the `hc_mult` streams as the residual for this site's `hc_post`.
    SaveResidual,
    /// 2026-09-25: `hc_pre`: collapse the streams to one sequence, emit `post`/`comb`.
    HcPre(Site),
    /// 2026-09-25: RMSNorm on the collapsed sequence, named by its weight.
    Norm(&'static str),
    /// 2026-09-25: The attention mixer.
    Mixer,
    /// 2026-09-25: The MLP site.
    Mlp,
    /// 2026-09-25: `hc_post`: `out[j] = post[j]*block_out + Σ_i comb[i][j]*residual[i]`.
    HcPost(Site),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    Attn,
    Ffn,
}

/// 2026-09-25: What the model does after the last text layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalStep {
    /// 2026-09-25: Unweighted mean over the `hc_mult` streams; it has no parameters.
    HyperHeadMean,
    Norm(&'static str),
    LmHead,
}

/// 2026-09-25: Which per-sequence state a layer needs. Each layer needs exactly one kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    /// 2026-09-25: Recurrent `[heads, head_dim, head_dim]` fp32 state plus a bf16
    /// causal-conv window.
    KdaRecurrent,
    /// 2026-09-25: Paged KV blocks plus the indexer's own key state.
    SparseKv,
}

#[derive(Debug, Clone)]
pub struct Glm5NextTextSkeleton {
    pub hidden_size: usize,
    pub hc_mult: usize,
    /// 2026-09-25: Text stack, indices `0..num_hidden_layers`. Never includes the MTP layer.
    pub layers: Vec<SkeletonLayer>,
    /// 2026-09-25: The MTP layer, index `num_hidden_layers`. Held apart from `layers` so a loop
    /// over `layers` covers only the text stack.
    pub mtp: Option<SkeletonLayer>,
}

const NON_LAYER_TENSORS: [&str; 3] = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "lm_head.weight",
];

/// 2026-09-25: The `self_attn` tensors of a KDA block.
const KDA_ATTN: [&str; 15] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.b_proj.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.g_a_proj.weight",
    "self_attn.g_b_proj.weight",
    "self_attn.q_conv1d.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.o_norm.weight",
    "self_attn.o_proj.weight",
];

/// 2026-09-25: The `self_attn` tensors of a DSA block, the MTP layer's included.
///
/// `indexer.k_norm` is a LayerNorm and carries a bias, the only `.bias` in this structural
/// contract; a binder that takes only `.weight` drops it silently.
const DSA_ATTN: [&str; 14] = [
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.index_kpool_compress_ape",
    "self_attn.indexer.index_kpool_compress_gate",
];

const LAYER_NORMS: [&str; 2] = ["input_layernorm.weight", "post_attention_layernorm.weight"];

const HC_PARAMS: [&str; 6] = [
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
];

/// 2026-09-25: Tensors the MTP layer carries besides its DSA block and layer norms. Its mixer
/// is an ordinary DSA block, so MTP needs no attention implementation of its own.
const MTP_HEAD: [&str; 4] = [
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
];

fn qualify(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

impl Glm5NextTextSkeleton {
    /// 2026-09-25: Derive the topology from a parsed config. Every kind is read, never
    /// defaulted. Errors on a `model_type` other than `glm5_next`, a `layer_types` length other
    /// than `num_hidden_layers`, `hc_mult == 0`, a layer type other than linear or sparse
    /// attention, and more than one MTP layer.
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        if cfg.model_type != "glm5_next" {
            bail!(
                "Glm5NextTextSkeleton built from a {:?} config",
                cfg.model_type
            );
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "layer_types has {} entries, num_hidden_layers is {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }
        if cfg.hc_mult == 0 {
            bail!("glm5_next skeleton needs hc_mult > 0; got 0 (mHC is not optional here)");
        }
        let dense: BTreeSet<usize> = cfg.mlp_only_layers.iter().copied().collect();

        let mixer_of = |t: LayerType, i: usize| -> Result<Mixer> {
            Ok(match t {
                LayerType::LinearAttention => Mixer::Kda,
                LayerType::SparseAttention => Mixer::Dsa,
                other => {
                    bail!("layer {i}: GLM-5.3-Flash has no {other:?} layers; refusing to bind one")
                }
            })
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, t) in cfg.layer_types.iter().enumerate() {
            layers.push(SkeletonLayer {
                index: i,
                mixer: mixer_of(*t, i)?,
                mlp: if dense.contains(&i) {
                    Mlp::Dense
                } else {
                    Mlp::RoutedMoe
                },
                hyper_connection: true,
                is_mtp: false,
            });
        }

        // 2026-09-25: The MTP layer sits past the text stack (`ModelConfig::layer_type_at`
        // resolves it through `mtp_layer_types`) and is never appended to `layers`.
        let mtp = match cfg.mtp_layer_types.len() {
            0 => None,
            1 => {
                let i = cfg.num_hidden_layers;
                Some(SkeletonLayer {
                    index: i,
                    mixer: mixer_of(cfg.mtp_layer_types[0], i)?,
                    mlp: Mlp::RoutedMoe,
                    // 2026-09-25: The checkpoint has no `hc_*` tensor on the MTP layer.
                    hyper_connection: false,
                    is_mtp: true,
                })
            }
            n => bail!("glm5_next skeleton expects 0 or 1 MTP layers, config declares {n}"),
        };

        Ok(Self {
            hidden_size: cfg.hidden_size,
            hc_mult: cfg.hc_mult,
            layers,
            mtp,
        })
    }

    /// 2026-09-25: Text stack plus the MTP layer, in checkpoint index order.
    pub fn all_layers(&self) -> Vec<SkeletonLayer> {
        let mut v = self.layers.clone();
        v.extend(self.mtp);
        v
    }

    /// 2026-09-25: The structural (non-MLP, non-MoE) tensors this layer needs, fully qualified.
    pub fn structural_tensors(&self, l: &SkeletonLayer) -> Vec<String> {
        let mut v: Vec<String> = match l.mixer {
            Mixer::Kda => KDA_ATTN.iter().map(|t| qualify(l.index, t)).collect(),
            Mixer::Dsa => DSA_ATTN.iter().map(|t| qualify(l.index, t)).collect(),
        };
        v.extend(LAYER_NORMS.iter().map(|t| qualify(l.index, t)));
        if l.hyper_connection {
            v.extend(HC_PARAMS.iter().map(|t| qualify(l.index, t)));
        }
        if l.is_mtp {
            v.extend(MTP_HEAD.iter().map(|t| qualify(l.index, t)));
        }
        v
    }

    /// 2026-09-25: Every structural tensor the whole skeleton needs, including the non-layer
    /// ones.
    pub fn structural_tensor_set(&self) -> BTreeSet<String> {
        let mut s: BTreeSet<String> = NON_LAYER_TENSORS.iter().map(|t| t.to_string()).collect();
        for l in self.all_layers() {
            s.extend(self.structural_tensors(&l));
        }
        s
    }

    /// 2026-09-25: The residual/norm/mHC wiring for one layer, in execution order.
    ///
    /// A layer without a hyper-connection (the MTP layer) gets the ordinary
    /// `x = x + sublayer(norm(x))` shape, with no mHC steps.
    pub fn residual_plan(&self, l: &SkeletonLayer) -> Vec<ResidualStep> {
        let mut p = Vec::new();
        for (site, norm, sub) in [
            (Site::Attn, LAYER_NORMS[0], ResidualStep::Mixer),
            (Site::Ffn, LAYER_NORMS[1], ResidualStep::Mlp),
        ] {
            if l.hyper_connection {
                p.push(ResidualStep::SaveResidual);
                p.push(ResidualStep::HcPre(site));
                p.push(ResidualStep::Norm(norm));
                p.push(sub);
                p.push(ResidualStep::HcPost(site));
            } else {
                p.push(ResidualStep::SaveResidual);
                p.push(ResidualStep::Norm(norm));
                p.push(sub);
            }
        }
        p
    }

    /// 2026-09-25: What runs after the last text layer.
    pub fn final_plan(&self) -> [FinalStep; 3] {
        [
            FinalStep::HyperHeadMean,
            FinalStep::Norm("model.language_model.norm.weight"),
            FinalStep::LmHead,
        ]
    }

    /// 2026-09-25: Per-layer state kind, for every layer including the MTP layer.
    pub fn state_plan(&self) -> BTreeMap<usize, StateKind> {
        self.all_layers()
            .iter()
            .map(|l| {
                (
                    l.index,
                    match l.mixer {
                        Mixer::Kda => StateKind::KdaRecurrent,
                        Mixer::Dsa => StateKind::SparseKv,
                    },
                )
            })
            .collect()
    }

    /// 2026-09-25: Layers whose recurrent state is carried between steps.
    pub fn kda_state_layers(&self) -> Vec<usize> {
        self.state_plan()
            .into_iter()
            .filter(|(_, k)| *k == StateKind::KdaRecurrent)
            .map(|(i, _)| i)
            .collect()
    }

    /// 2026-09-25: Layers that consume paged KV blocks, the MTP layer included.
    pub fn kv_cache_layers(&self) -> Vec<usize> {
        self.state_plan()
            .into_iter()
            .filter(|(_, k)| *k == StateKind::SparseKv)
            .map(|(i, _)| i)
            .collect()
    }

    /// 2026-09-25: The per-sequence state budget, from `cfg`'s geometry. `kda_recurrent` is
    /// sized at 4 bytes (fp32) per element.
    pub fn state_budget(&self, cfg: &ModelConfig, num_spec: usize) -> StateBudget {
        let kda_layers = self.kda_state_layers().len();
        let kv_layers = self.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count();
        let heads = cfg.linear_num_value_heads.max(1);
        let hd = cfg.linear_value_head_dim.max(1);
        let conv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        StateBudget {
            kda_recurrent: kda_layers * heads * hd * hd * 4,
            kda_conv: kda_layers * conv_dim * (cfg.linear_conv_kernel_dim - 1 + num_spec) * 2,
            dsa_kv_per_token: kv_layers * cfg.kv_lora_rank * 2,
            // 2026-09-25: Two buffers of `index_head_dim` BF16 (`k_normed` and the compress
            // `gate`) plus the 1 B validity flag, as `Glm5NextDsaState::alloc` allocates.
            dsa_indexer_per_token: kv_layers * (cfg.index_head_dim * 2 * 2 + 1),
            mhc_highway_per_token: self.hc_mult * self.hidden_size * 4,
            moe_routing_per_token: cfg.num_experts * 4 + cfg.num_experts_per_tok * 8,
        }
    }

    /// 2026-09-25: Account the skeleton's structural contract against a checkpoint's tensor
    /// names.
    ///
    /// `available` is the full name list. A required name that is absent goes to `missing`.
    /// Any other name goes to `deferred` when it contains `.mlp.` or starts with
    /// `model.visual.`/`model.vision`, and to `unexpected` otherwise.
    pub fn account(&self, available: &BTreeSet<String>) -> StructuralAccounting {
        let required = self.structural_tensor_set();
        let mut missing = Vec::new();
        for r in &required {
            if !available.contains(r) {
                missing.push(r.clone());
            }
        }
        let mut unexpected = Vec::new();
        let mut deferred = 0usize;
        for a in available {
            if required.contains(a) {
                continue;
            }
            let is_mlp = a.contains(".mlp.");
            let is_vision = a.starts_with("model.visual.") || a.starts_with("model.vision");
            if is_mlp || is_vision {
                deferred += 1;
            } else {
                unexpected.push(a.clone());
            }
        }
        StructuralAccounting {
            required: required.len(),
            bound: required.len() - missing.len(),
            missing,
            unexpected,
            deferred,
        }
    }
}
