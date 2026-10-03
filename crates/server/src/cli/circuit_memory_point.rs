// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: One `met circuit memory` model: the checkpoint at the serve's formats on a device,
//! the serve's settings and the engine's counts, and the one evaluator ([`Point::eval_with`]) the
//! forward report, the inverse queries and the ledger validation share. Split from
//! `circuit_memory.rs`.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here sizes a term: `metrale_circuit::memory` does, from the counts gathered here.
//! - A boot's own pool sizes ([`BootPool`]) replace the demand-sized KV only in the ledger
//!   validation; the CLI always sizes KV from the workload.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_circuit::hardware::{self, CheckpointSource, CircuitSource, ModelSpec, Registry};
use metrale_circuit::ir::Circuit;
use metrale_circuit::memory::{
    self, ActivationRun, CacheInputs, CopyRule, DriverTerms, LookupInputs, MemoryInputs,
    MemoryReport, budget,
};
use metrale_circuit::state::{KvInputs, StateDtype, StateInputs, VerifyInputs};
use metrale_circuit::venn::{Repo, Run};
use metrale_circuit::{Mode, QuantMetadata, ServePrecision, resolve_checkpoint};

use super::CircuitMemoryArgs;
use super::circuit_hw::{CheckpointTexts, FsTree};
use super::circuit_memory_serve::{Behavior, EngineFacts, engine_facts, serve_args};
use crate::cli::ServeArgs;
use metrale_model_layers::layers::MoeExpertTables;

/// 2026-10-02: A booted serve's pool sizes: the main KV blocks and the MTP head's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BootPool {
    pub kv_blocks: u64,
    pub draft_kv_blocks: u64,
}

/// 2026-10-02: The CLI's what-ifs beyond the serve's own flags: `--slots`, `--tree-nodes`,
/// `--capture-rows`, `--prompt-lookup`. A serve has none ([`Query::SERVE`]); the expert-table
/// decision evaluates under that, so the CLI and `met serve` decide alike.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Query {
    pub slots: Option<String>,
    pub tree_nodes: Option<u64>,
    pub capture_rows: u64,
    pub prompt_lookup: bool,
}

impl Query {
    /// 2026-10-02: What a serve runs: its own slots, no token tree, drafter capture or lookup.
    pub(crate) const SERVE: Query = Query {
        slots: None,
        tree_nodes: None,
        capture_rows: 0,
        prompt_lookup: false,
    };

    fn of(a: &CircuitMemoryArgs) -> Self {
        Query {
            slots: a.slots.clone(),
            tree_nodes: a.tree_nodes,
            capture_rows: a.capture_rows,
            prompt_lookup: a.prompt_lookup,
        }
    }
}

/// 2026-10-02: Everything fixed across the evaluations of one command.
pub(crate) struct Point<'a> {
    pub(crate) a: &'a CircuitMemoryArgs,
    /// 2026-10-02: The what-ifs every evaluation applies.
    pub(crate) query: Query,
    tree: &'a FsTree,
    reg: &'a Registry,
    pub(crate) args: ServeArgs,
    pub(crate) device: hardware::Device,
    pub(crate) config: metrale_config::ModelConfig,
    pub(crate) behavior: Behavior,
    pub(crate) served: Circuit,
    pub(crate) declared: Circuit,
    pub(crate) model: hardware::ModelUnderPlan,
    pub(crate) copies: Vec<CopyRule>,
    pub(crate) settings: BTreeMap<String, String>,
    pub(crate) driver: DriverTerms,
    pub(crate) budget_bytes: u64,
    pub(crate) sm_count: u64,
    pub(crate) outside: Option<u64>,
    /// 2026-10-02: Fused plans by (mode, rows) and the class's families, reused across the
    /// inverse queries' evaluations.
    plans: std::cell::RefCell<BTreeMap<(String, u64), metrale_circuit::FusionPlan>>,
    families: std::cell::RefCell<Option<metrale_circuit::venn::Families>>,
    /// 2026-10-02: The MoE expert-table decision `settings` carries; `None` when the plan has
    /// no such tables.
    pub(crate) tables: Option<super::circuit_memory_tables::TablesDecision>,
}

fn dtype(s: &str) -> Result<StateDtype> {
    StateDtype::parse(s).with_context(|| format!("KV dtype `{s}` has no element size"))
}

/// 2026-10-02: The formats the serve's flags choose for the projections the engine itself
/// decides (the head and the draft head), over `tier`. Mirrors INSTANCES.toml's `engine` lists:
/// `--lm-head-dtype`, or under `--weight-quantization nvfp4` an NVFP4 head
/// (lm_head_setup.rs:101); the draft head runs NVFP4 when the target head is not
/// (lm_head_setup.rs:119); `--mtp-quantization` for the rest of the draft head.
pub(crate) fn engine_formats(
    args: &ServeArgs,
    tier: &str,
) -> Vec<(String, metrale_circuit::LinearFormats)> {
    use metrale_circuit::{Format, LinearFormats};
    let lf = |w: Format| LinearFormats {
        weight: w,
        activation: Format::Bf16,
    };
    let nv = Format::Nvfp4 { group: 16 };
    let head = match (args.lm_head_dtype.as_str(), tier) {
        ("bf16", _) => Some(Format::Bf16),
        ("nvfp4", _) | ("default", "nvfp4") => Some(nv),
        _ => None,
    };
    let mut out = Vec::new();
    if let Some(h) = head {
        out.push(("lm_head".to_string(), lf(h)));
    }
    if args.speculative_proposer_requested() && head != Some(nv) {
        out.push(("mtp.lm_head".to_string(), lf(nv)));
    }
    match args.mtp_quantization.as_str() {
        "nvfp4" => out.push(("mtp.*".to_string(), lf(nv))),
        "bf16" => out.push(("mtp.*".to_string(), lf(Format::Bf16))),
        _ => {}
    }
    out
}

impl Point<'_> {
    /// 2026-10-02: The sequence slots at `c` sequences: `--slots`, `auto` (exactly `c`), or the
    /// serve's `--max-batch-size`.
    pub(crate) fn slots(&self, c: u64) -> Result<usize> {
        use metrale_model_engine::factory::SlotRequest;
        Ok(
            match (self.query.slots.as_deref(), self.args.max_batch_size) {
                (None, SlotRequest::Count(n)) => n,
                // 2026-10-02: `--max-batch-size auto` balances slots against KV at boot; at a given
                // concurrency the model sizes exactly that many, as `--slots auto` does.
                (None, SlotRequest::Auto) | (Some("auto"), _) => c as usize,
                (Some(n), _) => n.parse().with_context(|| format!("--slots {n}"))?,
            },
        )
    }

    /// 2026-10-02: Why `c` sequences of `isl + osl` tokens cannot be served at all (more
    /// sequences than slots, a length past `--max-seq-len`), or `None`.
    pub(crate) fn refusal(&self, c: u64, isl: u64, osl: u64) -> Result<Option<String>> {
        let slots = self.slots(c)?;
        if c as usize > slots {
            return Ok(Some(format!(
                "{c} sequences need {c} slots; the serve has {slots}"
            )));
        }
        let tokens = isl + osl;
        Ok((tokens as usize > self.args.max_seq_len).then(|| {
            format!(
                "ISL + OSL = {tokens} exceeds the serve's --max-seq-len {}",
                self.args.max_seq_len
            )
        }))
    }

    /// 2026-10-02: The model at `c` sequences of `isl + osl` tokens.
    pub(crate) fn eval(&self, c: u64, isl: u64, osl: u64) -> Result<(MemoryReport, EngineFacts)> {
        self.eval_with(c, isl, osl, None)
    }

    /// 2026-10-02: [`Self::eval`], with a booted serve's pools in place of the demand's KV.
    pub(crate) fn eval_with(
        &self,
        c: u64,
        isl: u64,
        osl: u64,
        boot: Option<BootPool>,
    ) -> Result<(MemoryReport, EngineFacts)> {
        if let Some(why) = self.refusal(c, isl, osl)? {
            bail!("{why}");
        }
        let slots = self.slots(c)?;
        let tokens = isl + osl;
        let tree = self.query.tree_nodes.map(|n| n as usize);
        let f = engine_facts(&self.args, &self.config, &self.behavior, slots, tree)?;
        let bs = self.args.block_size as u64;
        // 2026-10-02: A token tree's nodes write their KV before the verify accepts any.
        let written = tokens + self.query.tree_nodes.unwrap_or(0);
        let kv_blocks = match boot {
            Some(b) => b.kv_blocks,
            None => memory::kv_blocks_for(c, written, bs).context("KV blocks")?,
        };
        let has_draft = self.served.states.iter().any(|s| {
            s.section == metrale_circuit::Section::Draft
                && s.kind == metrale_circuit::state::StateKind::PagedKv
        });
        // 2026-10-02: The MTP head's own pool, 16-token blocks, as kv_budget.rs:175-204 and
        // mtp_head/new.rs:229-258 size it, at this demand's length.
        let p = tokens / 16 + 1;
        let draft_kv = (f.spec && has_draft).then(|| KvInputs {
            blocks: boot.map_or((p * f.mtp_max_seqs as u64).min(kv_blocks.max(p)), |b| {
                b.draft_kv_blocks
            }),
            block_size: 16,
        });
        let h = if f.h_f16_pool { "f16" } else { "f32" };
        let states = StateInputs {
            formats: BTreeMap::from([
                ("kv_cache_dtype".to_string(), dtype(&f.kv_dtype)?),
                ("ssm_h_storage".to_string(), dtype(h)?),
            ]),
            slots: f.pool.slots as u64,
            verify: f.pool.verify.as_ref().map(|v| VerifyInputs {
                h_steps: v.h_steps.iter().map(|&x| x as u64).collect(),
                conv_steps: v.conv_steps as u64,
            }),
            kv: Some(KvInputs {
                blocks: kv_blocks,
                block_size: bs,
            }),
            draft_kv,
        };
        // 2026-10-02: An MTP head under NVFP4 keeps an FP8 pool; a dense-FFN head runs BF16
        // whatever the flag (kv_budget.rs:175-204).
        let draft_kv_dtype = match (
            self.args.mtp_quantization.as_str(),
            self.served.arch.as_str(),
        ) {
            ("nvfp4", "qwen3_6_moe") => StateDtype::Fp8,
            _ => StateDtype::Bf16,
        };
        let caches = CacheInputs {
            prefix_snapshot_slots: f.marconi_slots as u64,
            ring: (f.ring_slots as u64, slots as u64),
            carry: f.carry,
            verify_table_rows: f.verify_rows,
            capture_rows: self.query.capture_rows,
            lookup: self.query.prompt_lookup.then_some(LookupInputs {
                sequences: c,
                history_tokens: tokens,
            }),
            tree: self.query.tree_nodes.map(|n| (c, n)),
            drafts: self.query.tree_nodes.map(|n| (c, n)),
        };
        let decode = if c == 1 { Mode::Decode } else { Mode::MultiSeq };
        let k = f.num_drafts as u64 + 1;
        let mut runs: Vec<(String, Mode, u64)> = vec![(format!("decode C={c}"), decode, c)];
        if f.spec {
            runs.push((
                format!("verify {c}x{k} (multi_seq rows)"),
                Mode::MultiSeq,
                c * k,
            ));
        }
        let chunk = (f.max_batch_tokens.saturating_sub(slots) as u64)
            .min(isl)
            .max(1);
        runs.push((
            format!("prefill T={chunk} (multi_seq rows)"),
            Mode::MultiSeq,
            chunk,
        ));
        let mut plans = Vec::new();
        for (label, mode, rows) in runs {
            plans.push((label, rows, self.plan(mode, rows)?));
        }
        let activation: Vec<ActivationRun<'_>> = plans
            .iter()
            .map(|(label, rows, plan)| ActivationRun {
                label,
                rows: *rows,
                plan,
            })
            .collect();
        let fams = self
            .families
            .borrow()
            .clone()
            .context("no activation run")?;
        let r = memory::evaluate(&MemoryInputs {
            served: &self.served,
            declared: &self.declared,
            copies: &self.copies,
            settings: &self.settings,
            states: &states,
            draft_kv_dtype: Some(draft_kv_dtype),
            caches: &caches,
            runs: &activation,
            families: Some((&fams, self.sm_count)),
            legacy_arena: Some(f.legacy_arena),
            driver: self.driver,
            budget_bytes: self.budget_bytes,
            chunk_slack: 0,
        })?;
        Ok((r.with_outside(self.outside.unwrap_or(0)), f))
    }

    /// 2026-10-02: The fused plan of `mode` at `rows` on the device (cached).
    pub(crate) fn plan(&self, mode: Mode, rows: u64) -> Result<metrale_circuit::FusionPlan> {
        let key = (mode.name().to_string(), rows);
        if let Some(p) = self.plans.borrow().get(&key) {
            return Ok(p.clone());
        }
        let one = hardware::plan_one(
            self.reg,
            &self.a.hardware,
            self.tree,
            &self.model,
            Run { mode, rows },
        )?;
        self.families
            .borrow_mut()
            .get_or_insert_with(|| one.resolved.families.clone());
        self.plans
            .borrow_mut()
            .insert(key, one.planned.plan.clone());
        Ok(one.planned.plan)
    }

    /// 2026-10-02: The target KV pool the budget leaves once everything else in `r` is placed
    /// (the engine gives the pool what remains): `(blocks, tokens, blocks the workload of `r`
    /// needs)`. The MTP head's pool stays at its size in `r`.
    pub(crate) fn kv_pool(&self, r: &MemoryReport) -> (u64, u64, u64) {
        let main = |t: &&metrale_circuit::state::StateTerm| {
            t.holding == metrale_circuit::state::Holding::Blocks
                && self
                    .served
                    .states
                    .iter()
                    .any(|d| d.id == t.state && d.section == metrale_circuit::Section::Main)
        };
        let terms: Vec<_> = r.states.terms.iter().filter(main).collect();
        let bytes: u64 = terms.iter().map(|t| t.bytes).sum();
        let bs = self.args.block_size as u64;
        let blocks = terms.first().map_or(0, |t| t.units / bs.max(1));
        let per_block = bytes.checked_div(blocks).unwrap_or(0);
        let rest = r.totals.device.saturating_sub(bytes);
        let fit = r
            .budget_bytes
            .saturating_sub(rest)
            .checked_div(per_block)
            .unwrap_or(0);
        (fit, fit * bs, blocks)
    }

    /// 2026-10-02: Whether `c` sequences of `isl + osl` fit the budget.
    pub(crate) fn fits(&self, c: u64, isl: u64, osl: u64) -> Result<bool, budget::BudgetError> {
        let q = |e: anyhow::Error| budget::BudgetError::Query(format!("{e:#}"));
        if self.refusal(c, isl, osl).map_err(q)?.is_some() {
            return Ok(false);
        }
        Ok(self.eval(c, isl, osl).map_err(q)?.0.headroom() >= 0)
    }
}

/// 2026-10-02: The model `a` names (its checkpoint's `texts`), on its device, from the
/// repository at `root`, under the serve flags `a` renders (its recipe and overrides).
pub(crate) fn prepare<'a>(
    a: &'a CircuitMemoryArgs,
    texts: &CheckpointTexts,
    root: &Path,
    tree: &'a FsTree,
    reg: &'a Registry,
) -> Result<Point<'a>> {
    let args = serve_args(root, &texts.id, a.recipe.as_deref(), &a.serve)?;
    prepare_with(a, args, texts, tree, reg)
}

/// 2026-10-02: [`prepare`] under resolved serve flags `args` (what `met serve` parsed). The MoE
/// expert-table decision ([`super::circuit_memory_tables::decide`]) is made here, once, and
/// every evaluation of the point uses it.
pub(crate) fn prepare_with<'a>(
    a: &'a CircuitMemoryArgs,
    args: ServeArgs,
    texts: &CheckpointTexts,
    tree: &'a FsTree,
    reg: &'a Registry,
) -> Result<Point<'a>> {
    let device = reg
        .devices
        .iter()
        .find(|d| d.id == a.hardware)
        .context("device")?
        .clone();
    let hw_text = tree
        .read(&format!("kernels/{}/HARDWARE.toml", device.class))
        .map_err(anyhow::Error::msg)?;
    let driver = DriverTerms::parse(&device.class, &hw_text)?;
    let config_json = texts.config.clone().with_context(|| {
        format!(
            "{}: no config.json (local cache, directory or --allow-network)",
            texts.id
        )
    })?;
    let config = metrale_config::parse_config(&config_json)?;
    let quant = QuantMetadata {
        hf_quant_config: texts.hf_quant.as_deref(),
    };
    let tier = args.weight_quantization.0.name().to_string();
    let served = resolve_checkpoint(
        &config_json,
        quant,
        &ServePrecision::Policy {
            tier: tier.clone(),
            caps: Vec::new(),
            engine: engine_formats(&args, &tier),
        },
    )?;
    let declared = resolve_checkpoint(&config_json, quant, &ServePrecision::Declared)?.circuit;
    let mut model = CheckpointSource { tree }.model(&ModelSpec {
        checkpoint: &texts.id,
        config_json: Some(&config_json),
        hf_quant: texts.hf_quant.as_deref(),
        precision: hardware::PrecisionChoice::Declared,
    })?;
    let behavior =
        super::circuit_memory_weights::behavior(tree, &device.class, &model.kernel_model)?;
    let facts = engine_facts(
        &args,
        &config,
        &behavior,
        args.max_batch_size.ceiling(),
        None,
    )?;
    model.circuit = served.circuit.clone();
    model.policy = hardware::model_checkpoint::derive_policy(&served.circuit, served.kv_cache)?;
    let set = &mut model.policy.settings;
    set.insert("kv_cache_dtype".into(), facts.kv_dtype.clone());
    set.insert(
        "ssm_h_dtype".into(),
        if facts.h_f16_pool { "f16" } else { "f32" }.into(),
    );
    let head = set.get("lm_head_dtype").cloned().unwrap_or_default();
    let settings = BTreeMap::from([
        (
            "speculative".to_string(),
            if facts.spec { "on" } else { "off" }.to_string(),
        ),
        ("weight_quantization".to_string(), tier.clone()),
        ("lm_head_dtype".to_string(), head),
        (
            "expert_quantization".to_string(),
            args.expert_quantization.0.name().to_string(),
        ),
        (
            "latent_moe".to_string(),
            if config.moe_latent_size > 0 {
                "on"
            } else {
                "off"
            }
            .to_string(),
        ),
        (
            super::circuit_memory_tables::SETTING.to_string(),
            MoeExpertTables::Build.name().to_string(),
        ),
    ]);
    let copies = memory::parse_copies(
        &tree
            .read("kernels/circuits/COPIES.toml")
            .map_err(anyhow::Error::msg)?,
    )?;
    let outside = super::circuit_memory_weights::outside_bytes(texts, &served.circuit)?;
    let mut p = Point {
        a,
        query: Query::of(a),
        tree,
        reg,
        budget_bytes: budget::util_budget(device.memory_bytes as u64, args.gpu_memory_utilization),
        sm_count: u64::from(device.sms),
        args,
        config,
        behavior,
        served: served.circuit,
        declared,
        model,
        copies,
        settings,
        driver,
        outside,
        plans: Default::default(),
        families: Default::default(),
        device,
        tables: None,
    };
    p.tables = super::circuit_memory_tables::decide(&mut p)?;
    Ok(p)
}
