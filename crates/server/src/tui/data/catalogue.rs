// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Library's row model: recipes outer-joined with the locally
//! cached checkpoints on the HuggingFace id, one row per model. Rows sort by
//! rank, then model id: runnable now (weights and a Metrale Engine recipe),
//! then any other row with a recipe, then local-only rows. Choosing between a
//! model's recipes is left to the recipe cards.
//!
//! Owner: server tui.
//! Invariants:
//! - `join` yields exactly one row per distinct recipe model, plus one per
//!   local entry that no recipe names.

use crate::recipe::Recipe;
use crate::tui::data::library::{LibraryEntry, human_size};

/// 2026-09-26: One row: every recipe for a model, its local checkpoint, or
/// both.
#[derive(Clone, Debug)]
pub struct Entry {
    /// 2026-09-26: The HuggingFace id, the join key.
    pub model: String,
    /// 2026-09-26: Every recipe naming this model, in id order. Empty for a
    /// local-only row.
    pub recipes: Vec<Recipe>,
    pub local: Option<LibraryEntry>,
}

impl Entry {
    pub fn has_recipe(&self) -> bool {
        !self.recipes.is_empty()
    }

    /// 2026-09-26: The recipe to describe the row by: the first Metrale Engine
    /// recipe (`Recipe::is_metrale`), else the first recipe of any runtime.
    pub fn primary(&self) -> Option<&Recipe> {
        self.recipes
            .iter()
            .find(|r| r.is_metrale())
            .or_else(|| self.recipes.first())
    }

    pub fn has_weights(&self) -> bool {
        self.local.as_ref().is_some_and(|l| l.has_weights)
    }

    /// 2026-09-26: `LibraryEntry::optimized` of the local checkpoint (a compiled
    /// kernel target resolves for it); false for a row with no local entry.
    /// Independent of `has_recipe`.
    pub fn optimized(&self) -> bool {
        self.local.as_ref().is_some_and(|l| l.optimized)
    }

    /// 2026-09-26: The weights are cached and at least one recipe is a Metrale
    /// Engine recipe.
    pub fn runnable_now(&self) -> bool {
        self.has_weights() && self.recipes.iter().any(Recipe::is_metrale)
    }

    /// 2026-09-26: Sort key: 0 runnable now, 1 any other row with a recipe, 2
    /// the rest.
    fn rank(&self) -> u8 {
        match (self.runnable_now(), self.has_recipe(), self.has_weights()) {
            (true, _, _) => 0,
            (_, true, _) => 1,
            _ => 2,
        }
    }

    /// 2026-09-26: The size on disk when the weights are complete, `partial`
    /// for a local entry without them, and a dash with no local entry.
    pub fn size_text(&self) -> String {
        match &self.local {
            Some(l) if l.has_weights => human_size(l.size_bytes),
            Some(_) => "partial".into(),
            None => "—".into(),
        }
    }

    /// 2026-09-26: The one-line subtitle under the model id: the primary
    /// recipe's params, quantization and category, then the local model type
    /// and layer count, with adjacent duplicates removed.
    pub fn subtitle(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(r) = self.primary() {
            if !r.model_params.is_empty() {
                parts.push(r.model_params.clone());
            }
            if !r.quantization.is_empty() {
                parts.push(r.quantization.clone());
            }
            if !r.category.is_empty() {
                parts.push(r.category.clone());
            }
        }
        if let Some(l) = &self.local {
            if !l.model_type.is_empty() {
                parts.push(l.model_type.clone());
            }
            if l.layers > 0 {
                parts.push(format!("{}L", l.layers));
            }
        }
        parts.dedup();
        parts.join(" · ")
    }

    /// 2026-09-26: Does this row match a filter? A case-insensitive substring
    /// test against the model id, every recipe id, and the local model type.
    pub fn matches(&self, needle: &str) -> bool {
        if needle.is_empty() {
            return true;
        }
        let needle = needle.to_lowercase();
        let hay = [
            self.model.to_lowercase(),
            self.recipes
                .iter()
                .map(|r| r.id.to_lowercase())
                .collect::<Vec<_>>()
                .join(" "),
            self.local
                .as_ref()
                .map(|l| l.model_type.to_lowercase())
                .unwrap_or_default(),
        ];
        hay.iter().any(|h| h.contains(&needle))
    }
}

/// 2026-09-26: Join recipes and local checkpoints into one sorted list, one row
/// per model; a model with several recipes yields one row carrying all of
/// them.
pub fn join(recipes: &[Recipe], local: &[LibraryEntry]) -> Vec<Entry> {
    let mut rows: Vec<Entry> = Vec::new();

    // 2026-09-26: Grouped by model; each group is then sorted by recipe id,
    // whatever order the fetch returned.
    let mut by_model: std::collections::BTreeMap<&str, Vec<Recipe>> =
        std::collections::BTreeMap::new();
    for recipe in recipes {
        by_model
            .entry(recipe.model.as_str())
            .or_default()
            .push(recipe.clone());
    }
    for (model, mut group) in by_model {
        group.sort_by(|a, b| a.id.cmp(&b.id));
        rows.push(Entry {
            model: model.to_string(),
            recipes: group,
            local: local.iter().find(|l| l.id == model).cloned(),
        });
    }
    // 2026-09-26: Local checkpoints that no recipe names get their own rows.
    for entry in local {
        if recipes.iter().any(|r| r.model == entry.id) {
            continue;
        }
        rows.push(Entry {
            model: entry.id.clone(),
            recipes: Vec::new(),
            local: Some(entry.clone()),
        });
    }

    rows.sort_by(|a, b| a.rank().cmp(&b.rank()).then_with(|| a.model.cmp(&b.model)));
    rows
}

#[cfg(test)]
#[path = "catalogue_tests.rs"]
mod tests;
