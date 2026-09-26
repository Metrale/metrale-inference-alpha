// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The resolution rules, each proved on a fixture that can fail
//! them. The real tree is graded by
//! `crates/kernels/tests/kernels_structure.rs`.
//!
//! Owner: metrale-closure (kernel layout).
//! Invariants: none beyond the types.

use super::*;

struct Fx {
    root: PathBuf,
}

impl Fx {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("metrale-layout-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let fx = Self { root };
        fx.hw(
            "gb10",
            "[hardware]\nvendor = \"nvidia\"\narch = \"sm_121f\"\n",
        );
        fx.file("gb10/common/shared.cu", "__global__ void s() {}\n");
        fx.file("gb10/common/other.cu", "__global__ void o() {}\n");
        fx.file("gb10/common/helper.cuh", "#define H 1\n");
        fx.file("gb10/modelA/MODEL.toml", "[behavior]\n");
        fx.file("gb10/modelB/MODEL.toml", "[behavior]\n");
        fx.file("gb10/modelA/nvfp4/shared.cu", "__global__ void s2() {}\n");
        fx.file(
            "gb10/modelA/nvfp4/KERNEL.toml",
            "[shadow]\nshared = \"fork\"\n",
        );
        fx.file("gb10/modelB/nvfp4/.gitkeep", "");
        fx
    }
    fn hw(&self, hw: &str, toml: &str) {
        self.file(&format!("{hw}/HARDWARE.toml"), toml);
    }
    fn file(&self, rel: &str, text: &str) {
        let p = self.root.join("kernels").join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
    fn path(&self, rel: &str) -> PathBuf {
        self.root.join("kernels").join(rel)
    }
    fn target(&self, hw: &str, model: &str, quant: &str) -> Target {
        Target {
            hardware: hw.into(),
            model: model.into(),
            quant: quant.into(),
        }
    }
    fn discover(&self, hw: &str, model: &str, quant: &str) -> Result<Layout, LayoutError> {
        discover(&self.root, &self.target(hw, model, quant))
    }
}

impl Drop for Fx {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn names(map: &BTreeMap<String, Entry>) -> Vec<&str> {
    map.keys().map(String::as_str).collect()
}

#[test]
fn a_model_file_shadows_its_common_namesake_by_stem_when_declared() {
    let fx = Fx::new("shadow");
    let l = fx.discover("gb10", "modelA", "nvfp4").unwrap();
    let mods: Vec<(String, PathBuf)> = l
        .modules()
        .into_iter()
        .map(|(s, e)| (s, e.source.clone()))
        .collect();
    assert_eq!(
        mods,
        [
            ("other".to_string(), fx.path("gb10/common/other.cu")),
            ("shared".to_string(), fx.path("gb10/modelA/nvfp4/shared.cu")),
        ]
    );
    assert_eq!(l.shadows.len(), 1);
    assert_eq!(l.shadows[0].reason, "fork");
    assert_eq!(l.shadows[0].loser, fx.path("gb10/common/shared.cu"));
    assert_eq!(names(&l.common), ["helper.cuh", "other.cu", "shared.cu"]);
    assert_eq!(l.configs(), [fx.path("gb10/modelA/nvfp4/KERNEL.toml")]);
}

#[test]
fn an_undeclared_shadow_is_refused_and_a_dead_declaration_too() {
    let fx = Fx::new("undeclared");
    fx.file("gb10/modelA/nvfp4/KERNEL.toml", "[build]\n");
    assert!(matches!(
        fx.discover("gb10", "modelA", "nvfp4"),
        Err(LayoutError::UndeclaredShadow { ref name, .. }) if name == "shared.cu"
    ));
    fx.file(
        "gb10/modelA/nvfp4/KERNEL.toml",
        "[shadow]\nshared = \"fork\"\nother = \"nothing\"\n",
    );
    assert!(matches!(
        fx.discover("gb10", "modelA", "nvfp4"),
        Err(LayoutError::DeadShadow { ref stem, .. }) if stem == "other"
    ));
    fx.file(
        "gb10/modelA/nvfp4/KERNEL.toml",
        "[shadow]\nshared = \"  \"\n",
    );
    assert!(matches!(
        fx.discover("gb10", "modelA", "nvfp4"),
        Err(LayoutError::Manifest { .. })
    ));
}

#[test]
fn headers_stage_beside_sources_but_are_not_modules() {
    let fx = Fx::new("headers");
    let l = fx.discover("gb10", "modelB", "nvfp4").unwrap();
    assert!(l.common.contains_key("helper.cuh"));
    assert_eq!(
        l.sources(),
        [
            fx.path("gb10/common/other.cu"),
            fx.path("gb10/common/shared.cu")
        ]
    );
    assert!(l.inputs().contains(&fx.path("gb10/common/helper.cuh")));
}

#[test]
fn use_brings_a_file_into_the_layer_under_its_own_name() {
    let fx = Fx::new("use");
    fx.file(
        "gb10/modelB/nvfp4/KERNEL.toml",
        "[sources]\nuse = [\"modelA/nvfp4/shared.cu\"]\n[shadow]\nshared = \"modelA's fork\"\n",
    );
    let l = fx.discover("gb10", "modelB", "nvfp4").unwrap();
    let e = &l.leaf["shared.cu"];
    assert!(e.used);
    assert_eq!(e.source, fx.path("gb10/modelA/nvfp4/shared.cu"));
    assert_eq!(e.layer, 0);
    assert_eq!(l.shadows[0].winner, e.source);
    // 2026-09-26: Without the declaration the use still shadows common/, and
    // is refused.
    fx.file(
        "gb10/modelB/nvfp4/KERNEL.toml",
        "[sources]\nuse = [\"modelA/nvfp4/shared.cu\"]\n",
    );
    assert!(matches!(
        fx.discover("gb10", "modelB", "nvfp4"),
        Err(LayoutError::UndeclaredShadow { .. })
    ));
}

#[test]
fn a_use_that_cannot_stand_is_refused() {
    let fx = Fx::new("badu");
    let cases: [(&str, fn(&LayoutError) -> bool); 5] = [
        ("[sources]\nuse = [\"modelA/nvfp4/missing.cu\"]\n", |e| {
            matches!(e, LayoutError::Use { .. })
        }),
        ("[sources]\nuse = [\"../../Cargo.toml\"]\n", |e| {
            matches!(e, LayoutError::Use { .. })
        }),
        ("[sources]\nuse = [\"modelA/MODEL.toml\"]\n", |e| {
            matches!(e, LayoutError::Use { .. })
        }),
        ("[sources]\nuse = [\"common/shared.cu\"]\n", |e| {
            matches!(e, LayoutError::SelfShadow { .. })
        }),
        ("[sources]\nfiles = []\n", |e| {
            matches!(e, LayoutError::Manifest { .. })
        }),
    ];
    for (toml, is) in cases {
        fx.file("gb10/modelB/nvfp4/KERNEL.toml", toml);
        let err = fx.discover("gb10", "modelB", "nvfp4").unwrap_err();
        assert!(is(&err), "{toml}: {err}");
    }
    // 2026-09-26: A use naming a file the directory already holds.
    fx.file(
        "gb10/modelA/nvfp4/KERNEL.toml",
        "[sources]\nuse = [\"modelB/nvfp4/shared.cu\"]\n[shadow]\nshared = \"x\"\n",
    );
    fx.file("gb10/modelB/nvfp4/shared.cu", "// b\n");
    fx.file(
        "gb10/modelB/nvfp4/KERNEL.toml",
        "[shadow]\nshared = \"b\"\n",
    );
    assert!(matches!(
        fx.discover("gb10", "modelA", "nvfp4"),
        Err(LayoutError::UseCollides { ref name, .. }) if name == "shared.cu"
    ));
}

#[test]
fn a_symlink_in_a_consulted_directory_is_refused_and_listed() {
    let fx = Fx::new("symlink");
    std::os::unix::fs::symlink(
        "../../common/other.cu",
        fx.path("gb10/modelB/nvfp4/other.cu"),
    )
    .unwrap();
    assert!(matches!(
        fx.discover("gb10", "modelB", "nvfp4"),
        Err(LayoutError::Symlink(_))
    ));
    assert_eq!(symlinks(&fx.root), [fx.path("gb10/modelB/nvfp4/other.cu")]);
    assert!(
        fx.discover("gb10", "modelA", "nvfp4").is_ok(),
        "an unconsulted symlink is another target's problem"
    );
}

fn overlay(fx: &Fx) {
    fx.hw(
        "b300",
        "[hardware]\nvendor = \"nvidia\"\narch = \"sm_103a\"\ninherits = \"gb10\"\n",
    );
    fx.file("b300/modelA/MODEL.toml", "[behavior]\n");
    fx.file("b300/common/other.cu", "__global__ void o_b300() {}\n");
    fx.file(
        "b300/common/KERNEL.toml",
        "[shadow]\nother = \"b300-tuned\"\n",
    );
}

#[test]
fn an_overlay_reads_four_layers_own_over_parent_and_leaf_over_common() {
    let fx = Fx::new("overlay");
    overlay(&fx);
    // 2026-09-26: A parent leaf file beats an own common file of the same
    // stem.
    fx.file(
        "gb10/modelA/nvfp4/other.cu",
        "__global__ void o_model() {}\n",
    );
    fx.file(
        "gb10/modelA/nvfp4/KERNEL.toml",
        "[shadow]\nshared = \"fork\"\nother = \"fork\"\n",
    );
    let l = fx.discover("b300", "modelA", "nvfp4").unwrap();
    assert_eq!(
        l.layers
            .iter()
            .map(|l| (l.role, l.tier, l.dir.clone()))
            .collect::<Vec<_>>(),
        [
            (Role::Leaf, Tier::Own, fx.path("b300/modelA/nvfp4")),
            (Role::Leaf, Tier::Parent, fx.path("gb10/modelA/nvfp4")),
            (Role::Common, Tier::Own, fx.path("b300/common")),
            (Role::Common, Tier::Parent, fx.path("gb10/common")),
        ]
    );
    let mods: BTreeMap<String, PathBuf> = l
        .modules()
        .into_iter()
        .map(|(s, e)| (s, e.source.clone()))
        .collect();
    assert_eq!(mods["shared"], fx.path("gb10/modelA/nvfp4/shared.cu"));
    assert_eq!(mods["other"], fx.path("gb10/modelA/nvfp4/other.cu"));
    assert_eq!(l.common["other.cu"].source, fx.path("b300/common/other.cu"));
    assert_eq!(
        l.configs(),
        [
            fx.path("b300/common/KERNEL.toml"),
            fx.path("gb10/modelA/nvfp4/KERNEL.toml")
        ]
    );
    let shadows: Vec<(Role, &str)> = l
        .shadows
        .iter()
        .map(|s| (s.role, s.name.as_str()))
        .collect();
    // 2026-09-26: The leaf's other.cu is declared once, against the resolved
    // common entry (b300's); the common-role win over gb10's is b300's own
    // declaration.
    assert_eq!(
        shadows,
        [
            (Role::Leaf, "other.cu"),
            (Role::Leaf, "shared.cu"),
            (Role::Common, "other.cu")
        ]
    );
    assert_eq!(l.shadows[0].loser, fx.path("b300/common/other.cu"));
    assert_eq!(l.shadows[2].reason, "b300-tuned");
}

#[test]
fn an_overlays_own_file_over_the_parents_needs_a_declaration_too() {
    let fx = Fx::new("overlay-undeclared");
    overlay(&fx);
    fx.file("b300/common/KERNEL.toml", "[build]\n");
    assert!(matches!(
        fx.discover("b300", "modelA", "nvfp4"),
        Err(LayoutError::UndeclaredShadow { ref loser, .. }) if *loser == fx.path("gb10/common/other.cu")
    ));
    // 2026-09-26: A pure addition needs no declaration.
    std::fs::remove_file(fx.path("b300/common/other.cu")).unwrap();
    fx.file("b300/common/extra.cu", "__global__ void x() {}\n");
    let l = fx.discover("b300", "modelA", "nvfp4").unwrap();
    assert_eq!(l.common["extra.cu"].layer, 2);
    assert_eq!(l.shadows.len(), 1, "only modelA's shared.cu over common");
}

#[test]
fn walk_unions_the_quants_an_overlay_and_its_parent_own() {
    let fx = Fx::new("walk");
    overlay(&fx);
    fx.file("b300/modelA/bf16/.gitkeep", "");
    fx.file(
        "gb10/modelC/MODEL.toml",
        "[model]\nkernel_source = \"modelA\"\n",
    );
    let all: Vec<String> = walk(&fx.root)
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        all,
        [
            "b300/modelA/bf16",
            "b300/modelA/nvfp4",
            "gb10/modelA/nvfp4",
            "gb10/modelB/nvfp4",
            "gb10/modelC/nvfp4"
        ]
    );
    let l = fx.discover("b300", "modelA", "bf16").unwrap();
    assert!(l.leaf.is_empty(), "bf16 exists only as an empty own leaf");
}

#[test]
fn a_redirect_reads_the_source_owners_leaf_on_both_tiers() {
    let fx = Fx::new("redirect");
    overlay(&fx);
    fx.file(
        "gb10/modelC/MODEL.toml",
        "[model]\nkernel_source = \"modelA\"\n",
    );
    fx.file("b300/modelC/MODEL.toml", "[behavior]\n");
    let l = fx.discover("b300", "modelC", "nvfp4").unwrap();
    // 2026-09-26: The own tier does not redirect, so the own leaf and
    // `source_model` stay modelC's; the parent tier follows its own redirect.
    assert_eq!(l.source_model, "modelC");
    assert_eq!(l.layers[0].dir, fx.path("b300/modelC/nvfp4"));
    assert_eq!(l.layers[1].dir, fx.path("gb10/modelA/nvfp4"));
    assert_eq!(l.model_dir, fx.path("b300/modelC"));
    fx.file(
        "b300/modelC/MODEL.toml",
        "[model]\nkernel_source = \"modelA\"\n",
    );
    let l = fx.discover("b300", "modelC", "nvfp4").unwrap();
    assert_eq!(l.layers[0].dir, fx.path("b300/modelA/nvfp4"));
    assert_eq!(
        l.leaf["shared.cu"].source,
        fx.path("gb10/modelA/nvfp4/shared.cu")
    );
    // 2026-09-26: Chains and dangling redirects are refused.
    fx.file(
        "gb10/modelD/MODEL.toml",
        "[model]\nkernel_source = \"modelC\"\n",
    );
    assert!(matches!(
        fx.discover("gb10", "modelD", "nvfp4"),
        Err(LayoutError::Redirect { .. })
    ));
    fx.file(
        "gb10/modelD/MODEL.toml",
        "[model]\nkernel_source = \"nowhere\"\n",
    );
    assert!(matches!(walk(&fx.root), Err(LayoutError::Redirect { .. })));
}

#[test]
fn inheritance_is_one_level_and_the_retired_inventory_key_is_refused() {
    let fx = Fx::new("inherits");
    overlay(&fx);
    fx.hw(
        "b200",
        "[hardware]\nvendor = \"nvidia\"\ninherits = \"b300\"\n",
    );
    fx.file("b200/modelA/MODEL.toml", "[behavior]\n");
    assert!(matches!(
        fx.discover("b200", "modelA", "nvfp4"),
        Err(LayoutError::Inherits { .. })
    ));
    fx.hw(
        "b200",
        "[hardware]\nvendor = \"nvidia\"\ninherits = \"b200\"\n",
    );
    assert!(matches!(
        fx.discover("b200", "modelA", "nvfp4"),
        Err(LayoutError::Inherits { .. })
    ));
    fx.hw(
        "b200",
        "[hardware]\nvendor = \"nvidia\"\ninherits = \"metal\"\n",
    );
    assert!(matches!(
        fx.discover("b200", "modelA", "nvfp4"),
        Err(LayoutError::Inherits { .. })
    ));
    fx.hw(
        "b200",
        "[hardware]\nvendor = \"nvidia\"\n[kernels]\noverrides = [\"x.cu\"]\n",
    );
    assert!(matches!(
        fx.discover("b200", "modelA", "nvfp4"),
        Err(LayoutError::RetiredKey { .. })
    ));
    fx.hw("b200", "[hardware]\nvendor = \"quantum-abacus\"\n");
    assert!(matches!(
        fx.discover("b200", "modelA", "nvfp4"),
        Err(LayoutError::UnknownVendor { .. })
    ));
}

#[test]
fn a_target_with_no_directory_anywhere_is_an_error_not_an_empty_set() {
    let fx = Fx::new("empty");
    std::fs::remove_dir_all(fx.path("gb10/common")).unwrap();
    assert!(matches!(
        fx.discover("gb10", "modelA", "absent"),
        Err(LayoutError::NoKernelDirectory(_))
    ));
    assert!(matches!(
        fx.discover("gb10", "nobody", "nvfp4"),
        Err(LayoutError::UnknownModel(_))
    ));
    assert!(matches!(
        fx.discover("nohw", "modelA", "nvfp4"),
        Err(LayoutError::UnknownHardware(_))
    ));
    // 2026-09-26: A quant the model does not own still resolves, to common/
    // alone.
    let fx = Fx::new("phantom");
    let l = fx.discover("gb10", "modelB", "phantom").unwrap();
    assert_eq!(l.modules().len(), 2);
}

#[test]
fn vendored_subdirectories_ride_with_their_role() {
    let fx = Fx::new("subdir");
    fx.file("gb10/modelA/nvfp4/vendor/v.h", "// v\n");
    let l = fx.discover("gb10", "modelA", "nvfp4").unwrap();
    assert_eq!(
        l.leaf_subdirs["vendor"],
        fx.path("gb10/modelA/nvfp4/vendor")
    );
    assert!(
        l.inputs()
            .contains(&fx.path("gb10/modelA/nvfp4/vendor/v.h"))
    );
    assert!(l.common_subdirs.is_empty());
}
