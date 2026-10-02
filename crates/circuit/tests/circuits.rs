// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The two checked-in circuits instantiate as their checkpoints are built, and the
//! rule set is consistent with the kernel tree, the golden plans and the audit.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::{Circuit, Format, Instance, LayerKind, Mode, Numerics, Section};

fn instance(recipe: &str) -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == recipe)
        .expect("instance")
}

fn dense() -> (Instance, Circuit) {
    let i = instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth");
    let c = common::load(&i).circuit;
    (i, c)
}

fn moe() -> (Instance, Circuit) {
    let i = instance("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head");
    let c = common::load(&i).circuit;
    (i, c)
}

fn weight(c: &Circuit, id: &str) -> Option<Format> {
    c.nodes[c.node(id).unwrap_or_else(|| panic!("no node {id}"))].weight
}

fn blocks_by_template(c: &Circuit, section: Section) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for b in c.blocks.iter().filter(|b| b.section == section) {
        *out.entry(b.template.clone()).or_insert(0) += 1;
    }
    out
}

#[test]
fn dense_circuit_has_the_checkpoint_layers_formats_and_draft_head() {
    let (_, c) = dense();
    assert_eq!(c.layer_kinds.len(), 64);
    let attn: Vec<usize> = (0..64)
        .filter(|&i| c.layer_kinds[i] == LayerKind::FullAttention)
        .collect();
    assert_eq!(attn, (0..16).map(|k| 4 * k + 3).collect::<Vec<_>>());
    assert_eq!(
        blocks_by_template(&c, Section::Main),
        BTreeMap::from([
            ("attn".into(), 16),
            ("dense_ffn".into(), 64),
            ("embed".into(), 1),
            ("gdn".into(), 48),
            ("head".into(), 1)
        ])
    );
    assert_eq!(
        blocks_by_template(&c, Section::Draft),
        BTreeMap::from([
            ("attn".into(), 1),
            ("dense_ffn".into(), 1),
            ("mtp_in".into(), 1),
            ("mtp_out".into(), 1)
        ])
    );
    let nvfp4 = Some(Format::Nvfp4 { group: 16 });
    assert_eq!(weight(&c, "l0.gdn.qkvz"), nvfp4);
    assert_eq!(
        weight(&c, "l3.attn.q"),
        nvfp4,
        "FP8 per-channel q is requantized to NVFP4"
    );
    assert_eq!(
        weight(&c, "l60.dense_ffn.down"),
        nvfp4,
        "the FP8 MLPs of 56-63 too"
    );
    assert_eq!(weight(&c, "l0.gdn.ba"), Some(Format::Bf16));
    assert_eq!(weight(&c, "head.lm_head"), Some(Format::Bf16));
    assert_eq!(weight(&c, "draft.attn.q"), Some(Format::Bf16));
    assert_eq!(weight(&c, "draft.mtp_out.lm_head"), nvfp4);
    let binding = &c.nodes[c.node("draft.dense_ffn.gate_up").unwrap()].binding;
    assert_eq!(
        binding,
        &["mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj"]
    );
    let qkvz = c.edge("l0.gdn.qkvz").unwrap();
    assert_eq!(c.edges[qkvz].dim_value, 16384);
    assert_eq!(c.edges[c.edge("l0.gdn.conv").unwrap()].format, Format::F32);
}

#[test]
fn each_layer_output_edge_is_the_next_layer_input_edge() {
    let (_, c) = dense();
    for i in 0..63 {
        let out = c.edge(&format!("l{i}.dense_ffn.y")).unwrap();
        let next = c.blocks.iter().find(|b| b.layer == Some(i + 1)).unwrap();
        assert_eq!(next.stream_in, Some(out), "layer {}", i + 1);
        let readers: BTreeSet<&str> = c.edges[out]
            .consumers
            .iter()
            .map(|&n| c.nodes[n].local.as_str())
            .collect();
        assert_eq!(readers, BTreeSet::from(["input_norm", "add"]), "layer {i}");
    }
    let last = c.edge("l63.dense_ffn.y").unwrap();
    let readers: Vec<&str> = c.edges[last]
        .consumers
        .iter()
        .map(|&n| c.nodes[n].id.as_str())
        .collect();
    assert_eq!(readers, ["head.final_norm", "draft.mtp_in.hidden_norm"]);
}

#[test]
fn moe_circuit_has_the_checkpoint_layers_formats_and_experts() {
    let (inst, c) = moe();
    assert_eq!(c.layer_kinds.len(), 40);
    assert_eq!(
        c.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count(),
        10
    );
    assert_eq!(blocks_by_template(&c, Section::Main)["moe_ffn"], 40);
    let block = Some(Format::parse("fp8/block128x128").unwrap());
    assert_eq!(weight(&c, "l0.gdn.qkvz"), block);
    assert_eq!(weight(&c, "l7.attn.o"), block);
    assert_eq!(weight(&c, "l0.moe_ffn.experts_gate_up"), block);
    assert_eq!(weight(&c, "l0.moe_ffn.router"), Some(Format::Bf16));
    assert_eq!(weight(&c, "l0.moe_ffn.shared_gate"), Some(Format::Bf16));
    assert_eq!(
        weight(&c, "draft.moe_ffn.experts_down"),
        block,
        "the MTP experts stay FP8"
    );
    assert_eq!(weight(&c, "draft.attn.k"), Some(Format::Bf16));
    let egu = c.edge("l0.moe_ffn.egu").unwrap();
    assert_eq!(c.edges[egu].rows.text(), "n*top_k");
    let mut dims = c.dims.clone();
    dims.insert("n".into(), 4);
    assert_eq!(c.edges[egu].rows.eval(&dims), Ok(32));
    assert_eq!(c.edges[egu].dim_value, 1024);
    assert_eq!(
        c.edges[c.edge("l0.moe_ffn.eact").unwrap()].format,
        Format::F32
    );
    assert_eq!(inst.shape.dims["experts"], 256);
}

#[test]
fn every_rule_kernel_is_compiled_by_a_golden_target() {
    let mut modules = Vec::new();
    let mut rules = Vec::new();
    for inst in common::instances().iter().filter(|i| i.golden) {
        modules.push(common::target_modules(inst));
        rules = common::load(inst).rules;
    }
    for r in &rules {
        for k in &r.kernels {
            assert!(
                modules.iter().any(|m| common::present(m, k)),
                "rule `{}` names {k}, which no golden target compiles",
                r.id
            );
        }
    }
}

#[test]
fn every_reference_rule_is_used_by_a_golden_plan_and_every_lever_moves_one() {
    let plans = common::golden_plans();
    let used: BTreeSet<String> = plans
        .iter()
        .flat_map(|(_, text)| {
            text.split(" rule=")
                .skip(1)
                .map(|t| t.split_whitespace().next().unwrap_or_default().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    // 2026-09-28: The legacy plans: every golden instance, mode and row count fused with the
    // bit-identical kernels removed. A reference rule a bit-identical fusion supersedes (the
    // per-row GDN residual add at a layer end) is still today's routing and the fallback when
    // the fused kernel is absent, so it must be selected there.
    let mut legacy_used = BTreeSet::new();
    for inst in common::instances().iter().filter(|i| i.golden) {
        let loaded = common::load(inst);
        let mut avail = common::available(inst, &loaded.rules);
        for r in &loaded.rules {
            if matches!(r.numerics, Numerics::BitIdentical { .. }) {
                for k in &r.kernels {
                    avail.kernels.remove(k);
                }
            }
        }
        for (&mode, rows) in &inst.plans {
            for &n in rows {
                let plan = metrale_circuit::fuse(
                    &loaded.circuit,
                    &loaded.rules,
                    &avail,
                    &inst.policy,
                    mode,
                    n,
                )
                .expect("legacy plan");
                legacy_used.extend(plan.groups.into_iter().map(|g| g.rule));
            }
        }
        for table in &inst.verify_batch {
            let plan = metrale_circuit::fuse_table(
                &loaded.circuit,
                &loaded.rules,
                &avail,
                &inst.policy,
                table,
            )
            .expect("legacy verify_batch plan");
            legacy_used.extend(plan.groups.into_iter().map(|g| g.rule));
        }
    }
    let inst = &common::instances()[0];
    let rules = common::load(inst).rules;
    // 2026-09-28: Ops some golden circuit has. A bit-identical rule may wait for a circuit
    // that needs it (an RmsNorm -> ActQuant fusion before any circuit quantizes activations),
    // but only when one of its ops is absent everywhere; otherwise it is dead.
    let ops: BTreeSet<metrale_circuit::OpKind> = common::instances()
        .iter()
        .filter(|i| i.golden)
        .flat_map(|i| common::load(i).circuit.nodes.into_iter().map(|n| n.op))
        .collect();
    // 2026-09-30: A bit-identical rule may also wait behind a policy setting no golden instance
    // turns on (`rms_norm_act_quant`: its emitter is not in the executor yet).
    let gated_off = |r: &metrale_circuit::Rule| {
        !r.when.is_empty()
            && common::instances().iter().filter(|i| i.golden).all(|i| {
                r.when
                    .iter()
                    .any(|(k, v)| i.policy.settings.get(k) != Some(v))
            })
    };
    for r in &rules {
        match &r.numerics {
            Numerics::Differs { .. } => assert!(
                !used.contains(&r.id),
                "`{}` selected without its lever",
                r.id
            ),
            Numerics::BitIdentical { .. } if !used.contains(&r.id) => assert!(
                gated_off(r)
                    || r.pattern
                        .iter()
                        .any(|p| !ops.contains(&p.op) && p.roles.is_empty()),
                "bit-identical rule `{}` matches circuit ops but no golden plan selects it",
                r.id
            ),
            _ => assert!(
                used.contains(&r.id) || legacy_used.contains(&r.id),
                "rule `{}` is used by no golden plan and no legacy plan",
                r.id
            ),
        }
    }
    let (d, loaded) = (
        instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth"),
        common::load(&instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth")),
    );
    let avail = common::available(&d, &loaded.rules);
    let fams = common::families(&d);
    for (lever, mode, rows) in [
        ("gdn_fused_norm", Mode::Decode, 1),
        ("decode_fused_silu", Mode::Decode, 1),
        ("gdn_fused_verify", Mode::Verify, 2),
        ("w4a4_downcast", Mode::Verify, 4),
    ] {
        let base = metrale_circuit::render_plan(&d, &loaded, &avail, mode, rows, &fams).unwrap();
        let mut on = d.clone();
        on.policy.opt_in_levers.insert(lever.to_string());
        let with = metrale_circuit::render_plan(&on, &loaded, &avail, mode, rows, &fams).unwrap();
        let body = |t: &str| {
            t.lines()
                .filter(|l| l.starts_with('g'))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert_ne!(body(&base), body(&with), "lever {lever} changes no group");
        assert!(with.contains(&format!("differs({lever})")), "{lever}");
    }
}

#[test]
fn the_kv_write_runs_before_attention_reads_it() {
    for (name, text) in common::golden_plans() {
        let mut pending: BTreeMap<String, usize> = BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            let Some(block) = line.split(' ').nth(1) else {
                continue;
            };
            if line.contains("[kv_write]") {
                pending.insert(block.to_string(), i);
            } else if line.contains("[attend]") {
                assert!(
                    pending.contains_key(block),
                    "{name}: {block} attends before its KV write"
                );
            }
        }
    }
}

#[test]
fn every_bit_identical_rule_names_a_registered_microtest() {
    let manifest: toml::Table =
        toml::from_str(&common::read("crates/model-arch/Cargo.toml")).expect("model-arch manifest");
    let examples: BTreeSet<String> = manifest["example"]
        .as_array()
        .expect("[[example]] entries")
        .iter()
        .filter_map(|e| e.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .collect();
    let rules = common::load(&common::instances()[0]).rules;
    let mut named = 0;
    for r in &rules {
        let Numerics::BitIdentical { microtest } = &r.numerics else {
            continue;
        };
        let path = common::root().join(format!("crates/model-arch/examples/{microtest}.rs"));
        assert!(
            path.is_file(),
            "rule `{}`: {} does not exist",
            r.id,
            path.display()
        );
        assert!(
            examples.contains(microtest),
            "rule `{}`: example `{microtest}` is not registered in crates/model-arch/Cargo.toml",
            r.id
        );
        named += 1;
    }
    assert!(named >= 4, "only {named} bit-identical rules");
}

// 2026-09-30: Both arms of the GDN recurrence plan for every golden instance. Under the default
// `ssm_batched_recurrent = "on"` a multi-sequence step runs the batched rules (the BA-gates twin
// from 96 rows on GB10), and the fragmented-slots route's arm is exactly the plan under "off",
// the per-row rules. One row runs the per-row rules under either setting.
#[test]
fn both_gdn_arms_plan_and_the_route_arm_is_the_off_plan() {
    let mut checked = 0;
    for inst in common::instances().iter().filter(|i| i.golden) {
        let loaded = common::load(inst);
        assert!(
            loaded
                .circuit
                .layer_kinds
                .contains(&LayerKind::LinearAttention),
            "{}: no GDN layer",
            inst.recipe
        );
        let avail = common::available(inst, &loaded.rules);
        assert_eq!(
            inst.policy.settings["ssm_batched_recurrent"], "on",
            "{}",
            inst.recipe
        );
        let mut off = inst.policy.clone();
        off.settings
            .insert("ssm_batched_recurrent".into(), "off".into());
        let fuse = |p: &metrale_circuit::Policy, mode, rows| {
            metrale_circuit::fuse(&loaded.circuit, &loaded.rules, &avail, p, mode, rows)
                .unwrap_or_else(|e| panic!("{}: {e}", inst.recipe))
        };
        let rules = |p: &metrale_circuit::FusionPlan| -> BTreeSet<String> {
            p.groups.iter().map(|g| g.rule.clone()).collect()
        };
        for rows in [16u64, 128] {
            let on = fuse(&inst.policy, Mode::MultiSeq, rows);
            let r = rules(&on);
            let ba = if rows >= 96 {
                "gdn_ba_gates_gemm_batched_twin"
            } else {
                "gdn_ba_gates_gemm_batched"
            };
            for id in [
                ba,
                "gdn_conv_l2_f32_batched",
                "gdn_recurrence_f32_batched",
                "gdn_out_norm_f32_batched",
            ] {
                assert!(r.contains(id), "{} n={rows}: no {id}", inst.recipe);
            }
            assert!(
                !r.iter()
                    .any(|x| x.starts_with("gdn_") && x.ends_with("_per_row"))
            );
            let set = (loaded.rules.as_slice(), loaded.runtime.as_slice());
            let arms = metrale_circuit::runtime::route_arms(
                &loaded.circuit,
                set,
                &avail,
                &inst.policy,
                &on,
            )
            .unwrap();
            let [(route, arm)] = arms.as_slice() else {
                panic!("{} n={rows}: {} route arms", inst.recipe, arms.len());
            };
            assert_eq!(route.id, "gdn_state_slots_fragmented");
            assert_eq!(arm.digest, fuse(&off, Mode::MultiSeq, rows).digest);
            assert!(rules(arm).contains("gdn_recurrence_f32_per_row"));
        }
        assert_eq!(
            fuse(&inst.policy, Mode::Decode, 1).digest,
            fuse(&off, Mode::Decode, 1).digest,
            "{}: one row does not read the setting",
            inst.recipe
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "golden instances checked");
}
