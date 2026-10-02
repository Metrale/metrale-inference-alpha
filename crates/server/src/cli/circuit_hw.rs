// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `met circuit plan --checkpoint <id|dir> --hardware <device>`: the I/O side of
//! `metrale_circuit::hardware`. It reads the device registry and the kernel tree from the
//! working tree ([`FsTree`]), the checkpoint's `config.json` / `hf_quant_config.json` from a
//! directory, the local Hugging Face cache or (with `--allow-network`) huggingface.co, and
//! writes, prints or checks the report. `--matrix` does the same for every cell of the roadmap
//! matrix and writes the summary table.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here plans, classifies or estimates; the crate does, over [`FsTree`].
//! - The model comes through one [`CircuitSource`] ([`source`]): the single place the
//!   instantiation behind a checkpoint is chosen.
//! - `--check` compares and never writes; a stale or missing report is an error.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_circuit::hardware::{
    self, CheckpointSource, CircuitSource, ModelSpec, PrecisionChoice, Registry,
};
use metrale_circuit::venn::Run;

pub(crate) use super::circuit_hw_tree::FsTree;
use super::circuit_venn::find_root;
use super::{CircuitHwArgs, CircuitHwFormat, CircuitMode, CircuitPrecision};

/// 2026-09-30: The devices of the roadmap matrix.
pub(crate) const MATRIX_DEVICES: [&str; 5] = ["h100-sxm", "h200-sxm", "b200", "gb300", "gb10"];

/// 2026-09-30: One model of the roadmap matrix.
pub(crate) struct MatrixModel {
    /// 2026-09-30: Report file slug.
    pub(crate) slug: &'static str,
    /// 2026-09-30: Checkpoint id.
    pub(crate) checkpoint: &'static str,
    /// 2026-09-30: Formats.
    pub(crate) precision: CircuitPrecision,
}

/// 2026-09-30: The checkpoint configs the matrix reads, checked in so the `--check` test is
/// hermetic: the model axis's fixtures, `<dir>/<org>--<name>/config.json` (+ sidecar).
pub(crate) const MATRIX_CONFIGS: &str = "crates/circuit/tests/fixtures/checkpoints";

/// 2026-09-30: The models of the roadmap matrix. `recipe` where a golden recipe pins the formats
/// (its gb10 cell is the golden plan); `declared` plans the checkpoint's own formats.
pub(crate) const MATRIX_MODELS: [MatrixModel; 10] = [
    MatrixModel {
        slug: "qwen3.8-27b-nvfp4",
        checkpoint: "unsloth/Qwen3.8-27B-NVFP4",
        precision: CircuitPrecision::Recipe,
    },
    MatrixModel {
        slug: "qwen3.8-27b-nvfp4-declared",
        checkpoint: "unsloth/Qwen3.8-27B-NVFP4",
        precision: CircuitPrecision::Declared,
    },
    MatrixModel {
        slug: "qwen3.6-35b-a3b-fp8",
        checkpoint: "Qwen/Qwen3.6-35B-A3B-FP8",
        precision: CircuitPrecision::Recipe,
    },
    MatrixModel {
        slug: "qwen3.6-35b-a3b-fp8-declared",
        checkpoint: "Qwen/Qwen3.6-35B-A3B-FP8",
        precision: CircuitPrecision::Declared,
    },
    // 2026-10-02: NVIDIA's NVFP4 35B-A3B at its declared formats (NVFP4 W4A16 experts and head,
    // FP8 attention and GDN).
    MatrixModel {
        slug: "qwen3.6-35b-a3b-nvfp4-declared",
        checkpoint: "nvidia/Qwen3.6-35B-A3B-NVFP4",
        precision: CircuitPrecision::Declared,
    },
    MatrixModel {
        slug: "nemotron-3-nano-30b-a3b-nvfp4",
        checkpoint: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        precision: CircuitPrecision::Declared,
    },
    MatrixModel {
        slug: "nemotron-3.5-lightning-30b-a3b-nvfp4",
        checkpoint: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        precision: CircuitPrecision::Declared,
    },
    MatrixModel {
        slug: "nemotron-3-super-120b-a12b-nvfp4",
        checkpoint: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        precision: CircuitPrecision::Declared,
    },
    MatrixModel {
        slug: "qwen3.6-27b-fp8",
        checkpoint: "Qwen/Qwen3.6-27B-FP8",
        precision: CircuitPrecision::Declared,
    },
    MatrixModel {
        slug: "llama-3.1-8b-instruct",
        checkpoint: "NousResearch/Meta-Llama-3.1-8B-Instruct",
        precision: CircuitPrecision::Declared,
    },
];

/// 2026-09-30: The model source: the one line that picks how a checkpoint becomes a circuit
/// (the checkpoint's own config for declared formats, the recipe's instance for recipe formats).
pub(crate) fn source(tree: &FsTree) -> impl CircuitSource + '_ {
    CheckpointSource { tree }
}

pub(crate) fn precision_of(p: CircuitPrecision) -> PrecisionChoice {
    match p {
        CircuitPrecision::Recipe => PrecisionChoice::Recipe,
        CircuitPrecision::Declared => PrecisionChoice::Declared,
    }
}

pub(crate) fn mode_of(m: CircuitMode) -> metrale_circuit::Mode {
    match m {
        CircuitMode::Decode => metrale_circuit::Mode::Decode,
        CircuitMode::MultiSeq => metrale_circuit::Mode::MultiSeq,
        CircuitMode::Verify => metrale_circuit::Mode::Verify,
        CircuitMode::Draft => metrale_circuit::Mode::Draft,
    }
}

/// 2026-09-30: A checkpoint's config texts: (id, config.json, hf_quant_config.json).
pub(crate) struct CheckpointTexts {
    pub(crate) id: String,
    pub(crate) config: Option<String>,
    pub(crate) hf_quant: Option<String>,
}

fn read_optional(p: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(p) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", p.display())),
    }
}

fn fetch(repo: &str, file: &str) -> Result<Option<String>> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let out = std::process::Command::new("curl")
        .args(["-sfL", &url])
        .output()
        .with_context(|| format!("running curl for {url}"))?;
    // 2026-09-30: curl -f exits 22 on an HTTP error (a checkpoint without the sidecar).
    match out.status.code() {
        Some(0) => Ok(Some(
            String::from_utf8(out.stdout).context("non-UTF-8 JSON")?,
        )),
        Some(22) => Ok(None),
        other => bail!("curl {url} failed ({other:?})"),
    }
}

/// 2026-09-30: The cached snapshot of Hub id `id` that holds `config.json`: `refs/main`'s, else
/// none. Unlike serving (`model_resolver::resolve_model_dir`), a plan needs no weights, so a
/// metadata-only snapshot counts.
fn cached_snapshot(id: &str) -> Result<Option<PathBuf>> {
    let root = crate::model_resolver::resolve_cache_root(None)?;
    let model = root.join(format!("models--{}", id.replace('/', "--")));
    let Ok(rev) = std::fs::read_to_string(model.join("refs/main")) else {
        return Ok(None);
    };
    let snap = model.join("snapshots").join(rev.trim());
    Ok(snap.join("config.json").is_file().then_some(snap))
}

/// 2026-09-30: The texts `spec` names: a directory's files, the local cache's, or the Hub's.
pub(crate) fn checkpoint_texts(spec: &str, allow_network: bool) -> Result<CheckpointTexts> {
    let dir = Path::new(spec);
    let from_dir = |d: &Path, id: String| -> Result<CheckpointTexts> {
        Ok(CheckpointTexts {
            id,
            config: read_optional(&d.join("config.json"))?,
            hf_quant: read_optional(&d.join("hf_quant_config.json"))?,
        })
    };
    if dir.join("config.json").is_file() {
        let id = metrale_circuit::venn::checkpoint_id_of(spec)
            .with_context(|| format!("cannot tell which checkpoint {spec} holds"))?;
        return from_dir(dir, id);
    }
    if let Some(d) = cached_snapshot(spec)? {
        return from_dir(&d, spec.to_string());
    }
    if allow_network && spec.contains('/') {
        return Ok(CheckpointTexts {
            id: spec.to_string(),
            config: fetch(spec, "config.json")?,
            hf_quant: fetch(spec, "hf_quant_config.json")?,
        });
    }
    Ok(CheckpointTexts {
        id: spec.to_string(),
        config: None,
        hf_quant: None,
    })
}

pub(crate) fn registry(tree: &FsTree) -> Result<Registry> {
    let text = metrale_circuit::venn::Repo::read(tree, "kernels/DEVICES.toml")
        .map_err(anyhow::Error::msg)?;
    Ok(hardware::parse_devices(&text)?)
}

/// 2026-09-30: One matrix cell: its report, its summary row, and the kernels it lacks (kernel,
/// why).
pub(crate) struct CellReport {
    pub(crate) text: String,
    pub(crate) row: String,
    pub(crate) absent: Vec<(String, String)>,
}

/// 2026-09-30: The report of one cell, rendered.
pub(crate) fn report_text(
    tree: &FsTree,
    reg: &Registry,
    texts: &CheckpointTexts,
    device: &str,
    precision: CircuitPrecision,
    command: String,
) -> Result<CellReport> {
    let model = source(tree).model(&ModelSpec {
        checkpoint: &texts.id,
        config_json: texts.config.as_deref(),
        hf_quant: texts.hf_quant.as_deref(),
        precision: precision_of(precision),
    })?;
    let r = hardware::build_report(reg, device, tree, model, command)?;
    let absent = r
        .absent
        .iter()
        .map(|(k, a)| (k.to_string(), a.describe()))
        .collect();
    Ok(CellReport {
        text: hardware::render_report(&r),
        row: hardware::summary_row(&r),
        absent,
    })
}

fn write_or_check(root: &Path, rel: &str, text: &str, check: bool) -> Result<()> {
    let path = root.join(rel);
    if check {
        let on_disk = std::fs::read_to_string(&path)
            .with_context(|| format!("--check: reading {}", path.display()))?;
        if on_disk != text {
            let line = on_disk
                .lines()
                .zip(text.lines())
                .position(|(x, y)| x != y)
                .map_or_else(|| "the end".to_string(), |i| format!("line {}", i + 1));
            bail!("{rel} is stale (first difference at {line})");
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

/// 2026-09-30: The summary table header of the matrix.
pub(crate) const SUMMARY_HEAD: &str = "| model | device | shared, measured here (C1 / C16) | shared incl. unmeasured (C16) | tok/s C1 / C16 / C128 (roofline) | fits C1 @4k | max C @4k | top gaps at C16 (share of step) |\n|---|---|---:|---:|---|---|---:|---|";

/// 2026-09-30: Every matrix report under `dir`, and the summary text.
pub(crate) fn matrix(root: &Path, dir: &str, check: bool) -> Result<String> {
    let tree = FsTree::new(root.to_path_buf());
    let reg = registry(&tree)?;
    let mut rows = Vec::new();
    let mut ports: Vec<(String, hardware::PortList)> = MATRIX_DEVICES
        .iter()
        .map(|d| (d.to_string(), hardware::PortList::new()))
        .collect();
    for MatrixModel {
        slug,
        checkpoint,
        precision,
    } in MATRIX_MODELS
    {
        let configs = root
            .join(MATRIX_CONFIGS)
            .join(checkpoint.replacen('/', "--", 1));
        let texts = CheckpointTexts {
            id: checkpoint.to_string(),
            config: read_optional(&configs.join("config.json"))?,
            hf_quant: read_optional(&configs.join("hf_quant_config.json"))?,
        };
        for device in MATRIX_DEVICES {
            let rel = format!("{dir}/{slug}--{device}.md");
            let command = format!("met circuit plan --matrix {dir}");
            let CellReport { text, row, absent } =
                report_text(&tree, &reg, &texts, device, precision, command)
                    .with_context(|| format!("{slug} on {device}"))?;
            write_or_check(root, &rel, &text, check)?;
            rows.push(row);
            let list = ports
                .iter_mut()
                .find(|(d, _)| d == device)
                .map(|(_, l)| l)
                .context("matrix device")?;
            for (k, why) in absent {
                list.entry(k).or_insert((why, 0)).1 += 1;
            }
        }
    }
    let summary = format!(
        "# Circuit hardware matrix\n\nGenerated by `met circuit plan --matrix {dir}` (one report per cell in `{dir}/`). Roofline ceilings on each device's datasheet unless its class measured its own; a row whose tok/s is marked \"roofline projection, unmeasured\" is estimated from datasheet ceilings alone, its shares included. \"measured here\" counts only kernels with microbench evidence on the device's own class.\n\n{SUMMARY_HEAD}\n{}\n\n{}",
        rows.join("\n"),
        hardware::port_lists(&ports)
    );
    write_or_check(root, &format!("{dir}/MATRIX.md"), &summary, check)?;
    Ok(summary)
}

fn print(text: &str) -> Result<()> {
    use std::io::Write;
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

/// 2026-09-30: Run `met circuit plan`.
pub(crate) fn run(a: CircuitHwArgs) -> Result<()> {
    let root = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    if let Some(dir) = &a.matrix {
        let summary = matrix(&root, dir, a.check)?;
        if let Some(p) = &a.summary {
            std::fs::write(p, &summary).with_context(|| format!("writing {}", p.display()))?;
        }
        eprintln!(
            "{} matrix reports {}",
            MATRIX_MODELS.len() * MATRIX_DEVICES.len(),
            if a.check { "current" } else { "written" }
        );
        return Ok(());
    }
    let (Some(checkpoint), Some(device), Some(precision)) =
        (&a.checkpoint, &a.hardware, a.precision)
    else {
        bail!("--checkpoint, --hardware and --precision are required without --matrix");
    };
    let tree = FsTree::new(root.clone());
    let reg = registry(&tree)?;
    let texts = checkpoint_texts(checkpoint, a.allow_network)?;
    let text = match a.format {
        CircuitHwFormat::Report => {
            let mut command = format!(
                "met circuit plan --checkpoint {checkpoint} --hardware {device} --precision {}",
                precision_of(precision).name()
            );
            if let Some(o) = &a.out {
                command.push_str(&format!(" --out {o}"));
            }
            report_text(&tree, &reg, &texts, device, precision, command)?.text
        }
        CircuitHwFormat::Plan => {
            let model = source(&tree).model(&ModelSpec {
                checkpoint: &texts.id,
                config_json: texts.config.as_deref(),
                hf_quant: texts.hf_quant.as_deref(),
                precision: precision_of(precision),
            })?;
            let mode = mode_of(a.mode);
            let rows = match (a.rows, mode) {
                (Some(0), _) => bail!("--rows must be at least 1"),
                (Some(r), _) => r,
                (None, metrale_circuit::Mode::Decode | metrale_circuit::Mode::Draft) => 1,
                (None, _) => bail!("--mode {} needs --rows", mode.name()),
            };
            let one = hardware::plan_one(&reg, device, &tree, &model, Run { mode, rows })?;
            hardware::plan_text(&model.circuit, &one)
        }
    };
    match &a.out {
        Some(rel) => {
            write_or_check(&root, rel, &text, a.check)?;
            eprintln!("{rel}: {}", if a.check { "current" } else { "written" });
            Ok(())
        }
        None => print(&text),
    }
}

/// 2026-09-30: The repository root for `met circuit display --hardware` (the working tree).
pub(crate) fn tree_here() -> Result<(FsTree, Registry, PathBuf)> {
    let root = find_root(&std::env::current_dir()?)?;
    let tree = FsTree::new(root.clone());
    let reg = registry(&tree)?;
    Ok((tree, reg, root))
}

#[cfg(test)]
#[path = "circuit_hw_tests.rs"]
mod circuit_hw_tests;
