// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The serve's MoE expert-table decision, before the model loads: `met circuit
//! memory`'s model of this checkpoint under these serve flags on this device
//! (`circuit_memory_point::prepare_with`, which decides through `circuit_memory_tables::decide`),
//! read from the kernel tree embedded at build time (`metrale-kernel-tree`). It replaces the
//! loader's free-memory heuristic: the decision depends on the checkpoint, the flags and the
//! device, never on what else is resident.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - Planned only for a loader that reads the decision (`reads_expert_table_plan`); published
//!   once, before `build_model`, and logged.
//! - The device is the DEVICES.toml entry with the live GPU's architecture and SM count whose
//!   memory is closest to the live total; none is an error, as is a plan that cannot be made.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_model_layers::layers::{MoeExpertTables, set_moe_expert_tables_from_plan};

use crate::cli::circuit_hw::CheckpointTexts;
use crate::cli::circuit_hw_tree::FsTree;
use crate::cli::circuit_memory_point::prepare_with;
use crate::cli::circuit_memory_tables::TablesDecision;
use crate::cli::{CircuitMemoryArgs, ServeArgs};

/// 2026-10-02: Where the embedded kernel tree is unpacked: `$HOME/.cache/metrale/kernel-tree`,
/// beside the serve's logs (`tui/init.rs`).
fn tree_cache() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context(
        "HOME is unset: the serve unpacks its kernel tree (the memory plan's inputs) under \
         $HOME/.cache/metrale/kernel-tree",
    )?;
    Ok(PathBuf::from(home).join(".cache/metrale/kernel-tree"))
}

/// 2026-10-02: What the live GPU is, for the device match.
pub(crate) struct LiveDevice<'a> {
    /// 2026-10-02: The kernel target's base SM architecture (`sm_121`).
    pub arch: &'a str,
    pub sms: u32,
    pub total_memory: u64,
}

/// 2026-10-02: The DEVICES.toml entry `live` is.
fn device_id(reg: &metrale_circuit::hardware::Registry, live: &LiveDevice<'_>) -> Result<String> {
    let base = |a: &str| a.trim_end_matches(['a', 'f']).to_string();
    reg.devices
        .iter()
        .filter(|d| base(&d.arch) == live.arch && d.sms == live.sms)
        .min_by_key(|d| (d.memory_bytes as i128 - live.total_memory as i128).unsigned_abs())
        .map(|d| d.id.clone())
        .with_context(|| {
            format!(
                "no kernels/DEVICES.toml device is {} with {} SMs; the memory plan needs one",
                live.arch, live.sms
            )
        })
}

/// 2026-10-02: Plan the decision for `args` serving the checkpoint at `model_dir` (`model_id`)
/// on `live`; `None` when the plan has no expert tables.
pub(crate) fn plan(
    args: &ServeArgs,
    model_id: &str,
    model_dir: &Path,
    live: &LiveDevice<'_>,
) -> Result<Option<TablesDecision>> {
    let root = metrale_kernel_tree::materialize(&tree_cache()?)?;
    let tree = FsTree::new(root.clone());
    let devices = std::fs::read_to_string(root.join("kernels/DEVICES.toml"))?;
    let reg = metrale_circuit::hardware::parse_devices(&devices)?;
    let read = |f: &str| -> Result<Option<String>> {
        match std::fs::read_to_string(model_dir.join(f)) {
            Ok(t) => Ok(Some(t)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("{}/{f}", model_dir.display())),
        }
    };
    let texts = CheckpointTexts {
        id: model_id.to_string(),
        config: read("config.json")?,
        hf_quant: read("hf_quant_config.json")?,
        dir: Some(model_dir.to_path_buf()),
    };
    let a = CircuitMemoryArgs {
        checkpoint: model_id.to_string(),
        hardware: device_id(&reg, live)?,
        recipe: None,
        isl: 1,
        osl: 1,
        concurrency: 1,
        slots: None,
        capture_rows: 0,
        prompt_lookup: false,
        tree_nodes: None,
        per_node: false,
        json: false,
        allow_network: false,
        root: Some(root),
        serve: Vec::new(),
    };
    let p = prepare_with(&a, args.clone(), &texts, &tree, &reg)
        .context("the MoE expert-table memory plan")?;
    Ok(p.tables)
}

/// 2026-10-02: Publish `d` to the loader and log it; `None` publishes nothing (the plan has no
/// tables, so the loader reads no decision).
pub(crate) fn publish(d: Option<&TablesDecision>) -> Result<()> {
    let Some(d) = d else {
        tracing::info!("MoE expert tables: none in the memory plan");
        return Ok(());
    };
    let in_force = set_moe_expert_tables_from_plan(d.tables);
    if in_force != d.tables {
        bail!(
            "MoE expert tables were already resolved ({}); the plan's ({}) cannot take effect",
            in_force.name(),
            d.tables.name()
        );
    }
    match d.tables {
        MoeExpertTables::Build => tracing::info!("MoE expert tables: {}", d.describe()),
        MoeExpertTables::Skip => tracing::warn!(
            "MoE expert tables: {}; MoE prefill runs the grouped GEMM on the row-major experts",
            d.describe()
        ),
    }
    Ok(())
}
