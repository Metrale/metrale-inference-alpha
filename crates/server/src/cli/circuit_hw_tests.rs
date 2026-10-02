// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `met circuit plan` on the checked-out tree: the gb10 plans equal the golden
//! plans, Hopper never runs an FP4-MMA kernel, H100 and H200 differ only in roofline and memory,
//! every registry device agrees with its class's build, and the matrix reports are current.
//! Regenerate the reports after an intended change with
//! `cargo test -p metrale-server circuit_hw -- --ignored regenerate_matrix`.
//!
//! Owner: server CLI tests.
//! Invariants: the tests read the repository at the workspace root; nothing is written except by
//! the ignored `regenerate_matrix`.

use std::path::PathBuf;

use metrale_circuit::hardware::avail::{check_build, guard_of};
use metrale_circuit::hardware::class::chain;
use metrale_circuit::hardware::device::MmaKind;
use metrale_circuit::hardware::exec::{Exec, exec_of};
use metrale_circuit::hardware::model::model_of;
use metrale_circuit::hardware::{self, KernelTree, ModelSpec, PrecisionChoice, Registry};
use metrale_circuit::venn::{Run, parse_families};
use metrale_circuit::{Mode, parse_instances};

use super::{CircuitSource, FsTree, matrix, source};

const MATRIX_DIR: &str = "kernels/circuits/plans/hw";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn tree() -> FsTree {
    FsTree::new(root())
}

fn registry_text() -> String {
    std::fs::read_to_string(root().join("kernels/DEVICES.toml")).expect("DEVICES.toml")
}

fn registry() -> Registry {
    hardware::parse_devices(&registry_text()).expect("DEVICES.toml parses")
}

fn model(checkpoint: &str, precision: PrecisionChoice) -> hardware::ModelUnderPlan {
    let t = tree();
    let dir = root()
        .join(super::MATRIX_CONFIGS)
        .join(checkpoint.replacen('/', "--", 1));
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let (config, hf_quant) = (read("config.json"), read("hf_quant_config.json"));
    source(&t)
        .model(&ModelSpec {
            checkpoint,
            config_json: config.as_deref(),
            hf_quant: hf_quant.as_deref(),
            precision,
        })
        .unwrap_or_else(|e| panic!("{checkpoint}: {e}"))
}

/// 2026-09-30: Every device's class exists, builds the device's arch, and defines the guard
/// macros exactly where the device lacks the guarded instruction.
fn devices_agree_with_their_builds(reg: &Registry) -> Result<(), String> {
    let t = tree();
    for d in &reg.devices {
        let c = chain(&t, &d.class).map_err(|e| format!("{}: {e}", d.id))?;
        if c[0].arch != d.arch {
            return Err(format!(
                "{}: arch {} but class builds {}",
                d.id, d.arch, c[0].arch
            ));
        }
        check_build(d, &c[0], &reg.guards).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[test]
fn every_registry_device_agrees_with_its_class_build() {
    devices_agree_with_their_builds(&registry()).unwrap();
}

// 2026-09-30: The mutation the brief names: make Hopper claim the FP4 MMA. The Hopper assertion
// below then fails (its W4A4 layers would run "native"), which is what this test proves.
#[test]
fn hopper_claiming_the_fp4_mma_fails_the_hopper_assertion() {
    let text = registry_text();
    let at = text.find("id = \"h100-sxm\"").expect("h100-sxm profile");
    let (head, tail) = text.split_at(at);
    let tail = tail
        .replacen("fp4_nvfp4 = 0.0", "fp4_nvfp4 = 1979.0", 1)
        .replacen(
            "fp4_fp4_nvfp4_block16 = false",
            "fp4_fp4_nvfp4_block16 = true",
            1,
        );
    let mutated = hardware::parse_devices(&format!("{head}{tail}")).expect("mutated parses");
    let caught = std::panic::catch_unwind(|| hopper_never_runs_fp4(&mutated, &["h100-sxm"]));
    assert!(caught.is_err(), "Hopper claiming FP4 MMA went unnoticed");
    // 2026-09-30: The same claim on the warp-level instruction contradicts the class's build.
    let warp = tail.replacen("mma_family = \"wgmma\"", "mma_family = \"mma_sync\"", 1);
    let warp = hardware::parse_devices(&format!("{head}{warp}")).expect("mutated parses");
    let e = devices_agree_with_their_builds(&warp).unwrap_err();
    assert!(e.contains("h100-sxm"), "{e}");
}

// 2026-09-30: Path A: the gb10 plan of every golden instance, at every checked-in mode and row
// count, is the golden plan byte for byte (header and digest included), runtime-route arms
// included.
#[test]
fn gb10_plans_equal_the_golden_plans() {
    let t = tree();
    let reg = registry();
    let text = std::fs::read_to_string(root().join("kernels/circuits/INSTANCES.toml")).unwrap();
    let mut checked = 0;
    for inst in parse_instances(&text).unwrap().iter().filter(|i| i.golden) {
        let m = model_of(&t, inst, "recipe".into(), PrecisionChoice::Recipe).unwrap();
        for (&mode, rows) in &inst.plans {
            for &n in rows {
                let one = hardware::plan_one(&reg, "gb10", &t, &m, Run { mode, rows: n })
                    .unwrap_or_else(|e| panic!("{} {} n={n}: {e}", inst.recipe, mode.name()));
                let got = hardware::plan_text(&m.circuit, &one);
                let file = root()
                    .join("kernels/circuits/plans")
                    .join(inst.plan_file(mode, n));
                let want = std::fs::read_to_string(&file).unwrap();
                assert!(
                    got == want,
                    "{} differs from the gb10 hardware plan",
                    file.display()
                );
                checked += 1;
            }
        }
    }
    // 2026-09-30: Golden instances (the dense recipe, it under `declared`, the MoE; 2026-10-02 the
    // NVFP4 MoE under `declared`), 16 plans each (decode, 11 multi_seq rungs, 3 verify, draft),
    // and the two dense ones' seven n-row draft widths.
    assert_eq!(checked, 78, "golden plans checked");
}

/// 2026-09-30: On `devices`, the 27B NVFP4 (recipe and declared formats) never selects a kernel
/// whose gb10 source sits in an FP4-MMA guard, the device runs no FP4 MMA, and every declared
/// NVFP4-activation layer takes the exact E4M3 path.
fn hopper_never_runs_fp4(reg: &Registry, devices: &[&str]) {
    let t = tree();
    let gb10 = t.class_sources("gb10", "qwen3.8-27b", "nvfp4").unwrap();
    for precision in PRECISIONS_27B {
        let m = model("unsloth/Qwen3.8-27B-NVFP4", precision);
        for &dev in devices {
            let d = reg.device(dev).unwrap();
            assert!(!d.runs(MmaKind::Fp4BlockScale), "{dev} claims the FP4 MMA");
            for (mode, rows) in [
                (Mode::Decode, 1),
                (Mode::MultiSeq, 16),
                (Mode::MultiSeq, 128),
                (Mode::Verify, 4),
            ] {
                let one = hardware::plan_one(reg, dev, &t, &m, Run { mode, rows }).unwrap();
                for g in &one.planned.plan.groups {
                    for k in &g.kernels {
                        let text = gb10.modules.get(&k.module).map(|m| m.text.as_str());
                        let fp4 = text
                            .and_then(|x| guard_of(x, &k.func, &reg.guards))
                            .is_some_and(|gd| gd.requires.kind == MmaKind::Fp4BlockScale);
                        assert!(!fp4, "{dev} {} n={rows} selects {k}", mode.name());
                    }
                }
            }
            for n in &m.circuit.nodes {
                let (Some(w), Some(a)) = (
                    n.weight,
                    n.inputs.first().map(|&e| m.circuit.edges[e].format),
                ) else {
                    continue;
                };
                if matches!(a, metrale_circuit::Format::Nvfp4 { .. }) {
                    assert_eq!(exec_of(d, w, a), Exec::ExactFp8Emulation, "{dev} {}", n.id);
                }
            }
        }
    }
}

/// 2026-09-30: The formats the Hopper assertion plans the 27B at.
const PRECISIONS_27B: [PrecisionChoice; 2] = [PrecisionChoice::Recipe, PrecisionChoice::Declared];

// 2026-09-30: Path A.
#[test]
fn hopper_plans_for_the_27b_never_select_an_fp4_mma_kernel() {
    hopper_never_runs_fp4(&registry(), &["h100-sxm", "h200-sxm"]);
    // 2026-09-30: The control: the guard parse does find FP4-MMA kernels in the gb10 sources,
    // so the assertion above can fail.
    let reg = registry();
    let gb10 = tree()
        .class_sources("gb10", "qwen3.8-27b", "nvfp4")
        .unwrap();
    let found: Vec<&str> = ["w4a4_gemm", "w4a4_quant_rows", "metrale_nvfp4_gemm_pipe"]
        .into_iter()
        .filter(|f| {
            gb10.modules
                .values()
                .any(|m| guard_of(&m.text, f, &reg.guards).is_some())
        })
        .collect();
    assert!(found.len() >= 2, "the FP4 guard parse found only {found:?}");
}

// 2026-09-30: Path B: H100 and H200 build the same class, so the plans are equal; only the
// roofline (estimates) and the memory fit move.
#[test]
fn h100_and_h200_differ_only_in_roofline_and_memory() {
    let t = tree();
    let reg = registry();
    let m = || model("Qwen/Qwen3.6-35B-A3B-FP8", PrecisionChoice::Recipe);
    let a = hardware::build_report(&reg, "h100-sxm", &t, m(), "t".into()).unwrap();
    let b = hardware::build_report(&reg, "h200-sxm", &t, m(), "t".into()).unwrap();
    for (x, y) in a.tables.iter().zip(&b.tables) {
        assert_eq!(x.planned.plan.digest, y.planned.plan.digest);
        let classes = |t: &hardware::gaps::GapTable| {
            let mut v: Vec<(String, String)> = t
                .rows
                .iter()
                .map(|r| (r.site.clone(), r.class.name().to_string()))
                .collect();
            v.sort();
            v
        };
        assert_eq!(classes(x), classes(y));
        assert!(y.total_us < x.total_us);
    }
    assert_eq!(a.footprint, b.footprint);
    assert_ne!(
        a.resolved.device.memory_bytes,
        b.resolved.device.memory_bytes
    );
    assert_ne!(hardware::summary_row(&a), hardware::summary_row(&b));
}

// 2026-09-30: The gb10 roofline is the measured one of its own manifest; the others are
// datasheet ceilings.
#[test]
fn only_gb10_uses_the_measured_roofline() {
    let t = tree();
    let reg = registry();
    let fams = parse_families(
        &std::fs::read_to_string(root().join("kernels/gb10/common/KERNEL_FAMILIES.toml")).unwrap(),
    )
    .unwrap();
    let m = model("unsloth/Qwen3.8-27B-NVFP4", PrecisionChoice::Recipe);
    let run = Run {
        mode: Mode::Decode,
        rows: 1,
    };
    let gb10 = hardware::plan_one(&reg, "gb10", &t, &m, run).unwrap();
    assert_eq!(gb10.resolved.roofline.roofline, fams.roofline);
    let h100 = hardware::plan_one(&reg, "h100-sxm", &t, &m, run).unwrap();
    assert_eq!(h100.resolved.roofline.roofline.dram_gbps, 3350.0);
}

// 2026-09-30: Every declared-format matrix model resolves, by the engine's own rule, to a
// kernel target directory that exists or to an explicit "no target"; and where a recipe serves
// the same checkpoint, to the recipe's target. Mutation: resolving on config.json's raw
// model_type sends the 35B-A3B to qwen3.5-35b-a3b and fails here.
#[test]
fn every_matrix_model_resolves_to_an_existing_kernel_target_or_none() {
    let root = root();
    for m in &super::MATRIX_MODELS {
        if m.precision == super::CircuitPrecision::Recipe {
            continue;
        }
        let declared = model(m.checkpoint, PrecisionChoice::Declared);
        let dir = root.join("kernels/gb10").join(&declared.kernel_model);
        assert!(
            declared.kernel_model == "(none)" || dir.join("MODEL.toml").is_file(),
            "{}: kernel target `{}` has no kernels/gb10 directory",
            m.checkpoint,
            declared.kernel_model
        );
        let recipe = super::MATRIX_MODELS.iter().find(|r| {
            r.checkpoint == m.checkpoint && r.precision == super::CircuitPrecision::Recipe
        });
        if let Some(r) = recipe {
            let pinned = model(r.checkpoint, PrecisionChoice::Recipe);
            assert_eq!(
                declared.kernel_model, pinned.kernel_model,
                "{}: declared and recipe cells resolve different kernel targets",
                m.checkpoint
            );
        }
    }
}

#[test]
fn matrix_reports_are_current() {
    matrix(&root(), MATRIX_DIR, true).unwrap_or_else(|e| {
        panic!(
            "{e:#}\nregenerate with `cargo test -p metrale-server circuit_hw -- --ignored regenerate_matrix`"
        )
    });
}

#[test]
#[ignore = "writes kernels/circuits/plans/hw/; run explicitly to regenerate"]
fn regenerate_matrix() {
    matrix(&root(), MATRIX_DIR, false).unwrap();
}
