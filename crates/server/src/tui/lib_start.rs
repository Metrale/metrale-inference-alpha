// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Starting points: the cards Enter offers on a model no recipe covers.
//!
//! Each donor recipe is copied and re-aimed at the model, and a blank card
//! with no pinned settings comes last. None of it is a measurement, so every
//! card sets `Recipe::starting_point`, says in its description where its
//! settings came from, and drops the donor's date and checkpoint metadata.
//!
//! Owner: server tui.
//! Invariants:
//! - Every card `starting_points` returns has `starting_point` set, and the
//!   list ends with the blank card, so it is never empty.

use crate::recipe::Recipe;
use crate::tui::data::catalogue::Entry;

use super::lib_state::LibState;

/// 2026-09-26: Alphanumerics only, lowercased, so `qwen3.6`, `qwen3_6_moe` and `Qwen3.6-27B` all contain `qwen36`.
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// 2026-09-26: Whether the donor's family, the first segment of its id (`qwen3.6/...`), appears in the
/// target's HF id or `model_type`, both compared through [`norm`].
fn same_family(donor: &Recipe, model: &str, model_type: &str) -> bool {
    let family = norm(donor.id.split('/').next().unwrap_or_default());
    if family.is_empty() {
        return false;
    }
    norm(model).contains(&family) || norm(model_type).contains(&family)
}

/// 2026-09-26: One donor recipe, re-aimed at `model` and marked as a starting point.
fn template_from(donor: &Recipe, model: &str) -> Recipe {
    let mut t = donor.clone();
    t.model = model.to_string();
    t.starting_point = Some(donor.id.clone());
    t.description = format!(
        "Starting point, not a measurement: settings copied from {}, which was \
         measured on {} — none of it has been verified on {}. Review each value \
         before launching.",
        donor.id, donor.model, model
    );
    // 2026-09-26: The date and checkpoint metadata describe the donor's model; only `defaults:` is offered.
    t.updated.clear();
    t.maintainer.clear();
    t.category.clear();
    t.model_params.clear();
    t.quantization.clear();
    t.kv_dtype.clear();
    t
}

/// 2026-09-26: The no-donor card: no pinned settings, so every flag is at the server's own default.
fn blank(model: &str) -> Recipe {
    Recipe {
        id: "starting-point/metrale-defaults".into(),
        version: "0".into(),
        model: model.to_string(),
        runtime: Some("metrale".into()),
        container: String::new(),
        min_nodes: 1,
        description: "Starting point, not a measurement: no settings pinned, so every \
                      flag is the server's own default. There is nothing to edit on \
                      this card — launch as-is, or pick a donor card to start from \
                      its settings."
            .into(),
        maintainer: String::new(),
        category: String::new(),
        model_params: String::new(),
        quantization: String::new(),
        kv_dtype: String::new(),
        updated: String::new(),
        defaults: std::collections::BTreeMap::new(),
        env: std::collections::BTreeMap::new(),
        starting_point: Some("no donor — the server's own defaults".into()),
    }
}

/// 2026-09-26: The recipes whose parameters may be offered for `model`: family matches first, then by id.
///
/// Only `runtime: metrale` recipes with `min_nodes <= 1`. Both the
/// starting-point cards and the Config form's borrow picker use this list.
pub(super) fn ranked_donors<'a>(
    recipes: &'a [Recipe],
    model: &str,
    model_type: &str,
) -> Vec<&'a Recipe> {
    let mut donors: Vec<&Recipe> = recipes
        .iter()
        .filter(|r| r.is_metrale() && r.min_nodes <= 1)
        .collect();
    donors.sort_by_key(|r| (!same_family(r, model, model_type), r.id.clone()));
    donors
}

/// 2026-09-26: Every starting point for `entry`, in [`ranked_donors`] order, with the blank card last.
pub(super) fn starting_points(recipes: &[Recipe], entry: &Entry) -> Vec<Recipe> {
    let model_type = entry
        .local
        .as_ref()
        .map(|l| l.model_type.as_str())
        .unwrap_or_default();
    let mut out: Vec<Recipe> = ranked_donors(recipes, &entry.model, model_type)
        .into_iter()
        .map(|d| template_from(d, &entry.model))
        .collect();
    out.push(blank(&entry.model));
    out
}

impl LibState {
    /// 2026-09-26: Build and hold the starting points for the selected model; `open_cards` calls it for a
    /// model with no recipe.
    ///
    /// Rebuilt each time, since a refresh can replace the index they are
    /// copied from.
    pub(super) fn open_starting_points(&mut self) {
        let Some(entry) = self.current() else {
            return;
        };
        let cards = starting_points(&self.index.recipes, entry);
        let model = entry.model.clone();
        self.starting = Some((model, cards));
    }
}

#[cfg(test)]
#[path = "lib_start_tests.rs"]
mod tests;
