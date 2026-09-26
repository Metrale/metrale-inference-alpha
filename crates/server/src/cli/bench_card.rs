// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Result cards: render a gate record into the SVG card template.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};

use super::bench_run::repo_root;

/// 2026-09-26: Where `--output-image` writes.
///
/// A target containing a path separator or `/`, or ending in `.svg` (any case), is
/// taken literally. Anything else is a name and gets `.svg` appended, never
/// substituted, so `qwen3.8-27b` becomes `qwen3.8-27b.svg`.
pub(crate) fn card_output_path(target: &str) -> std::path::PathBuf {
    let looks_like_a_path = target.contains(std::path::MAIN_SEPARATOR)
        || target.contains('/')
        || std::path::Path::new(target)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("svg"));
    if looks_like_a_path {
        std::path::PathBuf::from(target)
    } else {
        std::path::PathBuf::from(format!("{target}.svg"))
    }
}

/// 2026-09-26: Render `record` with `<root>/assets/cards/result-card.svg` and write it
/// to `card_output_path(target)`, creating the parent directory. Returns the path.
pub(crate) fn write_card(
    root: &std::path::Path,
    record: &metrale_bench::gate::GateRecord,
    target: &str,
    card_args: &std::collections::BTreeMap<String, String>,
) -> Result<std::path::PathBuf> {
    let template_path = root.join("assets/cards/result-card.svg");
    let template = std::fs::read_to_string(&template_path)
        .with_context(|| format!("reading the card template at {}", template_path.display()))?;
    let svg = metrale_bench::gate::card::render(&template, record, card_args);
    let out = card_output_path(target);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&out, svg).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

/// 2026-09-26: `met benchmark card <record>`: a card from an already-written record.
pub(crate) fn card_cmd(args: crate::cli::bench_args::CardArgs) -> Result<()> {
    let record_path = resolve_record(&args.record)?;
    let record = metrale_bench::gate::read_record(&record_path)
        .with_context(|| format!("reading the record at {}", record_path.display()))?;
    let card_args = args
        .output_image_args
        .as_deref()
        .map(metrale_bench::gate::card::parse_args)
        .transpose()
        .map_err(|e| anyhow::anyhow!("--output-image-args: {e}"))?
        .unwrap_or_default();
    // 2026-09-26: Without `--output-image` the card is `<benchmark_id>-<git_sha>.svg`.
    let target = args
        .output_image
        .unwrap_or_else(|| format!("{}-{}", record.benchmark_id, record.git_sha));
    // 2026-09-26: The template is looked up by walking up from the record, so a
    // record outside the current checkout uses the template of the repo that
    // holds it; `repo_root()` is the fallback.
    let root = template_root_for(&record_path).or_else(|_| repo_root())?;
    let out = write_card(&root, &record, &target, &card_args)?;
    println!("{}", out.display());
    Ok(())
}

/// 2026-09-26: The nearest ancestor of `record` (canonicalized when possible) that
/// holds `assets/cards/result-card.svg`.
fn template_root_for(record: &std::path::Path) -> Result<std::path::PathBuf> {
    let start = record
        .canonicalize()
        .unwrap_or_else(|_| record.to_path_buf());
    for dir in start.ancestors().skip(1) {
        if dir.join("assets/cards/result-card.svg").is_file() {
            return Ok(dir.to_path_buf());
        }
    }
    bail!(
        "no assets/cards/result-card.svg above {} — pass a record inside a checkout",
        record.display()
    )
}

/// 2026-09-26: A benchmark id or a record path -> a record path.
///
/// An existing file wins. Otherwise the argument is a benchmark id, and the
/// result is the lexically last `.json` in `.benchmarks/<id>/` other than
/// `BASELINE.json`.
fn resolve_record(arg: &str) -> Result<std::path::PathBuf> {
    let direct = std::path::Path::new(arg);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }
    let root = repo_root().context(
        "not inside a checkout, so a benchmark id cannot be resolved — pass a record path",
    )?;
    let dir = root.join(".benchmarks").join(arg);
    if !dir.is_dir() {
        bail!(
            "no benchmark or record called `{arg}` ({} does not exist). \
             `met benchmark list` prints the ids.",
            dir.display()
        );
    }
    // 2026-09-26: Record names start with their UTC day, so a lexical sort orders
    // days. A same-day re-run is named `<day>T<HHMMSS>Z-…` and sorts after the
    // `<day>-…` record it follows (`-` < `T`). Two records of one day at
    // different commits are ordered by sha, not by time.
    let mut records: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "json")
                && p.file_name().is_some_and(|n| n != "BASELINE.json")
        })
        .collect();
    records.sort();
    records.pop().with_context(|| {
        format!("`{arg}` has no committed records yet — run it first, or pass a record path")
    })
}

/// 2026-09-26: Render a card for a benchmark id (or record path) through the same
/// `resolve_record` and `write_card` as `benchmark card`. The TUI History
/// pane's card export (`BenchState::export_card`) calls it.
pub fn render_card_for_benchmark(id: &str, output: Option<&str>) -> Result<std::path::PathBuf> {
    let record_path = resolve_record(id)?;
    let record = metrale_bench::gate::read_record(&record_path)
        .with_context(|| format!("reading {}", record_path.display()))?;
    let target = output
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}-{}", record.benchmark_id, record.git_sha));
    let root = template_root_for(&record_path).or_else(|_| repo_root())?;
    write_card(&root, &record, &target, &Default::default())
}
