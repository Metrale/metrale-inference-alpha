// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The model-variant step of the Benchmarks section: pick which checkpoint a benchmark runs on.
//!
//! The rows come from `gate::read_baseline`, the baseline the gate itself
//! reads (every `kernels/<hw>/<model>/BENCH.toml`), so the TUI offers exactly
//! the variants the gate knows. A benchmark with one variant still shows the
//! step, where the variant's `note` is readable. With no repository checkout
//! there is no baseline and the flow goes straight to the parameters.
//!
//! Owner: server tui.
//! Invariants:
//! - `choose_variant` pins the target model (`target_model_pinned` and
//!   `variant_pinned`) whenever the index names a variant.

use metrale_bench::gate;

use super::bench_state::{BenchState, View};

/// 2026-09-26: One selectable variant: a (hardware, checkpoint) baseline entry.
#[derive(Clone, Debug)]
pub struct VariantRow {
    pub hardware: String,
    pub checkpoint: String,
    /// 2026-09-26: The entry's `label`, or the checkpoint id when it carries none.
    pub title: String,
    pub recipe: Option<String>,
    /// 2026-09-26: Whether this is the hardware's declared `default` checkpoint.
    pub is_default: bool,
    /// 2026-09-26: The entry's `note`.
    pub note: String,
    /// 2026-09-26: The committed bounds, for the detail pane.
    pub metrics: Vec<(String, gate::Bound)>,
}

/// 2026-09-26: The variants a benchmark is defined on, the default first within each box class.
/// Empty, not an error, when there is no checkout or no baseline.
pub fn variants_for(benchmark_id: &str) -> Vec<VariantRow> {
    let Ok(root) = crate::cli::bench_run::repo_root() else {
        return Vec::new();
    };
    let baseline = match gate::read_baseline(&root, benchmark_id) {
        Ok(b) => b,
        Err(e) => {
            tracing::debug!("no variant baseline for {benchmark_id}: {e:#}");
            return Vec::new();
        }
    };
    let mut rows = Vec::new();
    for (hardware, hw) in &baseline.hardware {
        let mut here: Vec<VariantRow> = hw
            .models
            .iter()
            .map(|(checkpoint, entry)| VariantRow {
                hardware: hardware.clone(),
                checkpoint: checkpoint.clone(),
                title: if entry.label.is_empty() {
                    checkpoint.clone()
                } else {
                    entry.label.clone()
                },
                recipe: entry.recipe.clone(),
                is_default: *checkpoint == hw.default,
                note: entry.note.clone(),
                metrics: entry
                    .metrics
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            })
            .collect();
        // 2026-09-26: The default leads; the rest keep the `BTreeMap`'s sorted order (the sort is stable).
        here.sort_by_key(|r| !r.is_default);
        rows.extend(here);
    }
    rows
}

impl BenchState {
    /// 2026-09-26: Enter the selected benchmark from the Suite list: through the variant
    /// step when it has variants, straight to the form otherwise.
    pub fn enter_selected(&mut self) {
        let Some(descriptor) = self.descriptor() else {
            return;
        };
        self.variants = variants_for(descriptor.id);
        self.variant_row = self.variant_row.min(self.variants.len().saturating_sub(1));
        self.view = if self.variants.is_empty() {
            View::Params
        } else {
            View::Variants
        };
    }

    /// 2026-09-26: Adopt the selected variant and open the form.
    ///
    /// The target model is pinned, so `follow_live_model` leaves it alone, and
    /// each parameter in the descriptor's `threshold_params` takes the variant's
    /// committed bound for its metric.
    pub fn choose_variant(&mut self, index: usize) {
        let Some(row) = self.variants.get(index).cloned() else {
            return;
        };
        self.variant_row = index;
        self.target =
            metrale_bench::TargetEndpoint::new(self.target.base_url.clone(), &row.checkpoint);
        self.target_model_pinned = true;
        self.variant_pinned = true;
        if let Some(descriptor) = self.descriptor() {
            for (param, metric) in descriptor.threshold_params {
                let Some(bound) = row
                    .metrics
                    .iter()
                    .find(|(k, _)| k == metric)
                    .map(|(_, b)| b)
                else {
                    continue;
                };
                // 2026-09-26: `max` if declared (a ceiling), else `min` (a floor); both at once adopts nothing.
                let derived = match (bound.min, bound.max) {
                    (Some(min), Some(max)) => {
                        tracing::warn!(
                            "variant {}: metric {metric} declares BOTH min ({min}) and max \
                             ({max}) — ambiguous for {param}, adopting neither",
                            row.title
                        );
                        continue;
                    }
                    (None, Some(max)) => max,
                    (Some(min), None) => min,
                    (None, None) => continue,
                };
                let Some(pos) = self.specs.iter().position(|s| s.key == *param) else {
                    continue;
                };
                // 2026-09-26: Parsed by the spec's own kind, like a typed value, so its bounds apply.
                match self.specs[pos].kind.parse(&format!("{derived}")) {
                    Ok(value) => {
                        self.values.set(param.to_string(), value);
                        if let Some(buf) = self.edit.get_mut(pos) {
                            *buf = format!("{derived}");
                        }
                        self.errors.remove(*param);
                    }
                    Err(e) => {
                        tracing::warn!("variant {} bound for {param} rejected: {e:#}", row.title);
                    }
                }
            }
        }
        // 2026-09-26: The model row of the form shows the adopted checkpoint.
        if let Some(buf) = self.edit.get_mut(self.specs.len() + 1) {
            *buf = row.checkpoint.clone();
        }
        self.view = View::Params;
    }

    /// 2026-09-26: Keys for the variant list: j/k move, Enter/→/l choose, Esc/←/h return to the list.
    pub(super) fn variants_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        let n = self.variants.len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.variant_row = (self.variant_row + 1).min(n - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.variant_row = self.variant_row.saturating_sub(1);
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                self.choose_variant(self.variant_row);
            }
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            _ => {}
        }
    }
}

#[cfg(test)]
#[path = "bench_variants_tests.rs"]
mod tests;
