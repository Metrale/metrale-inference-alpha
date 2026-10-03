// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `met circuit memory`: the I/O side of `metrale_circuit::memory`. It reads the
//! device registry, the class's HARDWARE.toml `[memory]`, the copy rules, the checkpoint's
//! config (and, when the weights are local, their safetensors headers), resolves the serve's
//! settings and the engine's counts ([`super::circuit_memory_serve`]), fuses the activation runs
//! on the device's class ([`super::circuit_memory_point`]), evaluates the model and prints it.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here sizes a term: the crate does, from the counts gathered.
//! - One evaluator answers the forward report and both inverse queries.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use metrale_circuit::hardware::{self, Registry};
use metrale_circuit::memory::{budget, render};
use metrale_circuit::venn::Repo;

use super::CircuitMemoryArgs;
use super::circuit_hw::FsTree;
use super::circuit_memory_point::{Point, prepare};
use super::circuit_memory_serve::EngineFacts;
use super::circuit_venn::find_root;

/// 2026-10-02: The repository's kernel tree and device registry, with `a`'s device checked.
pub(crate) fn load(a: &CircuitMemoryArgs, root: &std::path::Path) -> Result<(FsTree, Registry)> {
    let tree = FsTree::new(root.to_path_buf());
    let text = tree
        .read("kernels/DEVICES.toml")
        .map_err(anyhow::Error::msg)?;
    let reg = hardware::parse_devices(&text)?;
    if !reg.devices.iter().any(|d| d.id == a.hardware) {
        bail!("unknown device `{}` (kernels/DEVICES.toml)", a.hardware);
    }
    Ok((tree, reg))
}

/// 2026-10-02: The header lines: what was planned, at which settings, from which sources.
fn header(p: &Point<'_>, f: &EngineFacts) -> Vec<(String, String)> {
    let (device, util) = (&p.device, p.args.gpu_memory_utilization);
    let gib = |b: f64| format!("{:.2} GiB", b / (1u64 << 30) as f64);
    let a = p.a;
    let mut h = vec![
        ("checkpoint".to_string(), a.checkpoint.clone()),
        ("recipe".to_string(), a.recipe.clone().unwrap_or_else(|| "none (met serve defaults)".into())),
        ("serve flags".to_string(), if a.serve.is_empty() { "none".into() } else { a.serve.join(" ") }),
        ("device".to_string(), format!(
            "{} ({}): {} x util {util} = budget {}{}",
            device.id,
            device.class,
            gib(device.memory_bytes),
            gib(p.budget_bytes as f64),
            if util > p.driver.util_ceiling {
                format!(
                    " (above the {} ceiling {}, kernels/{}/HARDWARE.toml [memory] util_ceiling)",
                    device.class, p.driver.util_ceiling, device.class
                )
            } else {
                String::new()
            }
        )),
        ("workload".to_string(), format!("ISL {} + OSL {} tokens, concurrency {}", a.isl, a.osl, a.concurrency)),
        ("formats".to_string(), format!(
            "--weight-quantization {}, lm_head {}, KV {}, SSM h {}",
            p.settings["weight_quantization"],
            p.settings["lm_head_dtype"],
            f.kv_dtype,
            if f.h_f16_pool { "f16 pool" } else { "f32" }
        )),
        ("slots".to_string(), format!(
            "{} (+1 dummy); speculation {}; MTP verify slots {}",
            f.slots,
            if f.spec { format!("on, K = {}", f.num_drafts + 1) } else { "off".into() },
            f.pool.verify.as_ref().map_or(0, |v| v.slots()),
        )),
        ("caches".to_string(), format!(
            "Marconi slots {}, decode ring {} x {} slots (requested depth; the preflight auto-fit may lower it), carry slots {}, verify-table rows {}, capture rows {}, prompt lookup {}, tree nodes {}",
            f.marconi_slots,
            f.ring_slots,
            f.slots,
            f.carry.0,
            f.verify_rows,
            a.capture_rows,
            if a.prompt_lookup { "on" } else { "off" },
            a.tree_nodes.map_or("none".into(), |n| n.to_string()),
        )),
        ("MoE expert tables".to_string(), match &p.tables {
            Some(d) => format!("{} (decided once, as `met serve` does)", d.describe()),
            None => "none in this plan".into(),
        }),
        ("legacy arena".to_string(), format!("BufferSizes at max_batch_tokens {} + GDN two-phase", f.max_batch_tokens)),
        ("MODEL.toml".to_string(), p.behavior.source.clone()),
        ("outside the circuit".to_string(), match p.outside {
            Some(b) => format!("{} of checkpoint tensors no weight node binds (norms, conv, gates, vision), from the safetensors headers", gib(b as f64)),
            None => "not counted: the checkpoint's weights are not local (norms, conv, gates, a vision tower)".into(),
        }),
    ];
    if p.driver.unified {
        h.push((
            "unified memory".into(),
            "host caches compete for the same LPDDR5X pool, outside the util budget".into(),
        ));
    }
    h
}

/// 2026-10-02: Run `met circuit memory`.
pub(crate) fn run(a: CircuitMemoryArgs) -> Result<()> {
    let root: PathBuf = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let (tree, reg) = load(&a, &root)?;
    let texts = super::circuit_hw::checkpoint_texts(&a.checkpoint, a.allow_network)?;
    let point = prepare(&a, &texts, &root, &tree, &reg).context("preparing the memory model")?;
    let (report, f) = point.eval(a.concurrency, a.isl, a.osl)?;
    let osl = a.osl;
    let cap = point.args.max_seq_len as u64;
    // 2026-10-02: The widest batch a serve takes is one batched verify's width
    // (`VERIFY_WY_TABLE_SEQS`, `MAX_MTP_MAX_SEQS`); `auto` slots search up to it.
    let widest = match point.query.slots.as_deref() {
        Some("auto") => metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS as u64,
        _ => point.slots(a.concurrency)? as u64,
    };
    let inverse = render::Inverse {
        max_concurrency: Some((
            a.isl,
            osl,
            budget::largest_fitting(1, widest.max(1), &mut |c| point.fits(c, a.isl, osl))?,
        )),
        max_isl: Some((
            a.concurrency,
            osl,
            budget::largest_fitting(1, cap.saturating_sub(osl).max(1), &mut |i| {
                point.fits(a.concurrency, i, osl)
            })?,
        )),
        kv_pool: Some(point.kv_pool(&report)),
    };
    let head = header(&point, &f);
    let out = if a.json {
        serde_json::to_string_pretty(&render::render_json(
            &point.served,
            &report,
            &head,
            &inverse,
        ))? + "\n"
    } else {
        render::render_text(&point.served, &report, &head, &inverse, a.per_node)
    };
    use std::io::Write as _;
    match std::io::stdout().lock().write_all(out.as_bytes()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

#[cfg(test)]
#[path = "circuit_memory_ledger_tests.rs"]
pub(crate) mod circuit_memory_ledger_tests;
