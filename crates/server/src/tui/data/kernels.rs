// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Rows for Main ▸ Kernels: the embedded modules of the kernel
//! target serve resolved (`publish_loaded_target`, looked up with
//! `ptx_for_exact_target`), joined with the runtime lookup audit. The inputs
//! are the ones `kernel_audit::render_kernel_table` logs at load.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::sync::Mutex;

use metrale_telemetry::kernel_audit::AuditRow;

/// 2026-09-26: One row of the kernel table.
#[derive(Clone, Debug)]
pub struct KernelRow {
    pub module: String,
    pub ptx_hash: String,
    /// 2026-09-26: `None`: no audited lookup of this module. `Some(true)`: at
    /// least one lookup loaded. `Some(false)`: every audited lookup failed.
    pub resolution: Option<bool>,
}

/// 2026-09-26: A `(module, func)` lookup that failed.
#[derive(Clone, Debug)]
pub struct MissingKernel {
    pub module: String,
    pub func: String,
    /// 2026-09-26: `file:line` of the first lookup's dispatch site
    /// (`AuditRow::site`).
    pub site: String,
}

impl MissingKernel {
    fn from_row(r: &AuditRow) -> Self {
        Self {
            module: r.module.clone(),
            func: r.func.clone(),
            site: format!("{}:{}", r.site.file(), r.site.line()),
        }
    }
}

#[derive(Default)]
pub struct KernelTableModel {
    pub rows: Vec<KernelRow>,
    /// 2026-09-26: Failed lookups the target does not declare absent
    /// (`split_failures(..).required`). Only this list raises the toast and
    /// the warning panel.
    pub missing_required: Vec<MissingKernel>,
    /// 2026-09-26: Failed lookups the target's MODEL.toml `[expected_absent]`
    /// declares; the build requires a reason for each. Counted in the warning
    /// panel's title only.
    pub missing_expected: Vec<MissingKernel>,
}

/// 2026-09-26: `(target model, quant)` of the kernel target serve resolved for
/// the loaded model, set by `publish_loaded_target`. The table looks the
/// target up by this pair, so it shows the target serve selected.
static LOADED_TARGET: Mutex<Option<(String, String)>> = Mutex::new(None);

/// 2026-09-26: Record which kernel target the serve path resolved.
pub fn publish_loaded_target(model: &str, quant: &str) {
    if let Ok(mut g) = LOADED_TARGET.lock() {
        *g = Some((model.to_string(), quant.to_string()));
    }
}

fn loaded_target() -> Option<(String, String)> {
    let guard = LOADED_TARGET.lock().ok()?;
    guard.clone()
}

/// 2026-09-26: FNV-1a 64-bit hash, low 48 bits as 12 hex digits; the same
/// function as `ptx_hash` in `metrale_telemetry::kernel_audit::report`.
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// 2026-09-26: Build the table from the loaded target's modules and the audit
/// rows, sorted by module. `App` rebuilds it once each time a different model
/// finishes loading.
pub fn build() -> KernelTableModel {
    let audit = metrale_telemetry::kernel_audit::audit_rows();
    // 2026-09-26: No target published yet, or none compiled for it: an empty
    // table.
    let Some(ptx) = loaded_target()
        .and_then(|(model, quant)| metrale_kernels::ptx_for_exact_target(&model, &quant))
    else {
        return KernelTableModel::default();
    };
    let mut rows: Vec<KernelRow> = ptx
        .modules
        .iter()
        .map(|(module, blob)| {
            let mut resolution = None;
            for r in &audit {
                if r.module == *module {
                    resolution = Some(resolution.unwrap_or(false) || r.loaded);
                }
            }
            KernelRow {
                module: (*module).to_string(),
                ptx_hash: ptx_hash(blob),
                resolution,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.module.cmp(&b.module));
    // 2026-09-26: The same `split_failures` the load-time kernel gate uses.
    let split = metrale_telemetry::kernel_audit::split_failures(&audit, ptx.expected_absent);
    KernelTableModel {
        rows,
        missing_required: split.required.iter().map(MissingKernel::from_row).collect(),
        missing_expected: split.expected.iter().map(MissingKernel::from_row).collect(),
    }
}
