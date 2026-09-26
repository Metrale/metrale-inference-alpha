// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Diagnostics for kernel-target selection: panic texts that list what exists, and a warning when the target triple was left implicit.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - Nothing here panics: a missing or malformed `kernels/` tree or
//!   HARDWARE.toml degrades to a less specific message.
//!
//! `METRALE_TARGET_HW` selects which `kernels/<hw>/` tree gets compiled and
//! defaults to [`DEFAULT_HW`]; these messages name the env vars so a wrong
//! default is visible in the build output.

use std::path::Path;
use std::sync::Once;

/// 2026-09-25: Hardware target compiled when `METRALE_TARGET_HW` is unset.
pub const DEFAULT_HW: &str = "gb10";

/// 2026-09-25: Read `vendor = "..."` out of a HARDWARE.toml by hand.
///
/// Deliberately not a TOML parse: this is used to build panic messages, so it
/// must not itself fail on a file that is malformed — which is exactly the
/// situation some of these messages are reporting.
fn vendor_of(hw_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(hw_dir.join("HARDWARE.toml")).ok()?;
    text.lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("vendor"))
        .and_then(|rest| rest.trim_start().strip_prefix('='))
        .map(|v| v.trim().trim_matches('"').to_string())
}

/// 2026-09-25: Immediate subdirectories of `dir`, sorted, ignoring anything unreadable.
fn subdirs(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    out.sort();
    out
}

/// 2026-09-25: Hardware targets (directories with a HARDWARE.toml), as `name (vendor)`.
pub fn available_hw(kernels_root: &Path) -> Vec<String> {
    subdirs(kernels_root)
        .into_iter()
        .filter(|name| kernels_root.join(name).join("HARDWARE.toml").is_file())
        .map(|name| match vendor_of(&kernels_root.join(&name)) {
            Some(v) => format!("{name} ({v})"),
            None => name,
        })
        .collect()
}

/// 2026-09-25: Models under a hardware target: subdirs carrying a MODEL.toml.
pub fn available_models(hw_dir: &Path) -> Vec<String> {
    subdirs(hw_dir)
        .into_iter()
        .filter(|d| hw_dir.join(d).join("MODEL.toml").is_file())
        .collect()
}

fn or_none(items: Vec<String>) -> String {
    if items.is_empty() {
        "(none found)".to_string()
    } else {
        items.join(", ")
    }
}

/// 2026-09-25: One ready-to-paste invocation per hardware target that exists,
/// using its first model and that model's first subdirectory as the quant.
/// Listing every hardware target rather than one avoids steering a user to
/// whichever target sorts first.
fn examples(kernels_root: &Path) -> String {
    let lines: Vec<String> = subdirs(kernels_root)
        .into_iter()
        .filter(|h| kernels_root.join(h).join("HARDWARE.toml").is_file())
        .map(|hw| {
            let hw_dir = kernels_root.join(&hw);
            let model = available_models(&hw_dir)
                .into_iter()
                .next()
                .unwrap_or_else(|| "<model>".to_string());
            let quant = subdirs(&hw_dir.join(&model))
                .into_iter()
                .next()
                .unwrap_or_else(|| "<quant>".to_string());
            format!("   METRALE_TARGET_HW={hw} METRALE_TARGET_MODEL={model} METRALE_TARGET_QUANT={quant} cargo build --release -p metrale-server")
        })
        .collect();
    if lines.is_empty() {
        "   (no kernels/<hw>/HARDWARE.toml found - is this a full checkout?)".to_string()
    } else {
        lines.join("\n")
    }
}

/// 2026-09-25: Panic text for an `METRALE_TARGET_HW` naming no `kernels/<hw>/` directory.
pub fn unknown_hw(kernels_root: &Path, hw: &str) -> String {
    format!(
        "\n\
         METRALE_TARGET_HW={hw} selects kernels/{hw}/, which does not exist.\n\
         \n\
         Available hardware targets: {}\n\
         \n\
         Build a target that exists:\n{}\n",
        or_none(available_hw(kernels_root)),
        examples(kernels_root),
    )
}

/// 2026-09-25: Panic text for an `METRALE_TARGET_MODEL` with no directory under this target.
pub fn unknown_model(hw_dir: &Path, hw: &str, model: &str) -> String {
    format!(
        "\n\
         METRALE_TARGET_MODEL={model} selects kernels/{hw}/{model}/, which does not exist.\n\
         \n\
         Models available for {hw}: {}\n",
        or_none(available_models(hw_dir)),
    )
}

/// 2026-09-25: Panic text for a target selection that resolved to nothing at all.
pub fn no_targets(kernels_root: &Path, hw: &str, model: &str, quant: &str) -> String {
    format!(
        "\n\
         No kernel targets resolved for METRALE_TARGET_HW={hw} \
         METRALE_TARGET_MODEL={model} METRALE_TARGET_QUANT={quant}.\n\
         \n\
         Models available for {hw}: {}\n\
         \n\
         Build a target that exists:\n{}\n",
        or_none(available_models(&kernels_root.join(hw))),
        examples(kernels_root),
    )
}

/// 2026-09-25: Say what is being built whenever any of the
/// `METRALE_TARGET_{HW,MODEL,QUANT}` triple is unset.
///
/// A `cargo:warning=` rather than an error: requiring the triple would break
/// every build that relies on the defaults. Emitted at most once per
/// build-script run.
pub fn warn_if_defaulted(kernels_root: &Path, hw: &str, model: &str, quant: &str) {
    static ONCE: Once = Once::new();

    let implicit: Vec<&str> = [
        ("METRALE_TARGET_HW", hw),
        ("METRALE_TARGET_MODEL", model),
        ("METRALE_TARGET_QUANT", quant),
    ]
    .iter()
    .filter(|(var, _)| std::env::var_os(var).is_none())
    .map(|(var, _)| *var)
    .collect();

    if implicit.is_empty() {
        return;
    }

    // 2026-09-25: "a, b and c" rather than join(" and "), which produces "a and b and c".
    let unset = match implicit.split_last() {
        Some((last, [])) => (*last).to_string(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => unreachable!("empty case returned above"),
    };

    ONCE.call_once(|| {
        let vendor = vendor_of(&kernels_root.join(hw)).unwrap_or_else(|| "unknown vendor".into());
        println!(
            "cargo:warning=Building kernels for {hw} ({vendor}), model={model}, quant={quant} \
             - {unset} not set. Hardware targets available: {}. Set all three explicitly \
             to choose.",
            or_none(available_hw(kernels_root)),
        );
    });
}
