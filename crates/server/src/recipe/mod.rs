// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Serving recipes from `Metrale/metralectl`: parse one and render it as `met serve` argv.
//!
//! Owner: server (recipe). `schema` owns the key→flag mapping.
//! Invariants:
//! - `Recipe::parse` always sets `starting_point` to `None`.
//! - Every argv and `ServeArgs` built here comes from `argv_edited`, which refuses a
//!   recipe whose `runtime` is not `metrale`.

pub mod fetch;
mod fetch_github;
pub mod schema;
pub mod yaml;

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;

use yaml::Yaml;

/// 2026-09-26: One recipe, flattened to what the dashboard and the launchers read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// 2026-09-26: `family/stem`, derived from the recipe's path.
    pub id: String,
    pub version: String,
    pub model: String,
    /// 2026-09-26: The `runtime:` key, `None` when absent. Only `metrale` can be served
    /// from here (`is_metrale`).
    pub runtime: Option<String>,
    pub container: String,
    /// 2026-09-26: Ranks this recipe requires, from the top-level `min_nodes` key, not a
    /// `defaults:` key. The EP=2 recipes carry `ep_size: 2` in `defaults` and
    /// `min_nodes: 2` here, so a launcher reading only `defaults` builds a config
    /// the validator rejects with "--ep-size 2 exceeds --world-size 1".
    pub min_nodes: usize,
    pub description: String,
    pub maintainer: String,
    pub category: String,
    pub model_params: String,
    pub quantization: String,
    pub kv_dtype: String,
    /// 2026-09-26: `metadata.updated`, empty when the recipe does not carry it. A plain
    /// string, not a parsed date: it is only displayed, so a malformed date still
    /// loads.
    pub updated: String,
    /// 2026-09-26: The `defaults:` block; every value must be a scalar.
    pub defaults: BTreeMap<String, String>,
    /// 2026-09-26: The `env:` block verbatim: the `METRALE_*` serve levers this recipe is
    /// measured under. It is not validated here; the bench gate validates it with
    /// `serve_env::declared`, which names the recipe in its refusal
    /// (`cli::bench_serve_plan`). An absent block and `{}` both read as empty.
    pub env: BTreeMap<String, String>,
    /// 2026-09-26: `Some(provenance)` when this is a dashboard-synthesized starting point
    /// rather than a recipe read from the index; the provenance names the donor
    /// recipe the settings were copied from, or says there was none. `parse`
    /// always sets `None`.
    pub starting_point: Option<String>,
}

impl Recipe {
    /// 2026-09-26: Parse one recipe. `id` is `family/stem`, derived from its path.
    ///
    /// # Errors
    /// YAML the reader refuses; a missing `recipe_version`, `model` or `container`;
    /// a `defaults:` or `env:` that is not a mapping of scalars (`defaults:` is
    /// required); a `min_nodes` that is not a number.
    pub fn parse(id: impl Into<String>, text: &str) -> Result<Self> {
        let id = id.into();
        let doc = yaml::parse(text).with_context(|| format!("reading recipe {id}"))?;
        let map = doc.as_map().expect("parse guarantees a mapping");

        let scalar = |key: &str| -> Option<String> {
            map.get(key).and_then(Yaml::as_str).map(str::to_string)
        };
        let required = |key: &str| -> Result<String> {
            scalar(key).with_context(|| format!("{id}: missing required key {key:?}"))
        };

        let meta = map.get("metadata").and_then(Yaml::as_map);
        let meta_str = |key: &str| -> String {
            meta.and_then(|m| m.get(key))
                .and_then(Yaml::as_str)
                .unwrap_or_default()
                .to_string()
        };

        let defaults_block = map
            .get("defaults")
            .and_then(Yaml::as_map)
            .with_context(|| format!("{id}: `defaults:` must be a mapping"))?;
        let mut defaults = BTreeMap::new();
        for (key, value) in defaults_block {
            let Some(text) = value.as_str() else {
                bail!("{id}: defaults.{key} is not a scalar");
            };
            defaults.insert(key.clone(), text.to_string());
        }

        // 2026-09-26: An absent `env:` means no levers, the same as `{}`.
        let mut env = BTreeMap::new();
        if let Some(block) = map.get("env") {
            let Some(entries) = block.as_map() else {
                bail!("{id}: `env:` must be a mapping");
            };
            for (key, value) in entries {
                let Some(text) = value.as_str() else {
                    bail!("{id}: env.{key} is not a scalar");
                };
                env.insert(key.clone(), text.to_string());
            }
        }

        // 2026-09-26: An absent `min_nodes` means one node.
        let min_nodes = match scalar("min_nodes") {
            Some(n) => n
                .parse()
                .with_context(|| format!("{id}: min_nodes {n:?} is not a number"))?,
            None => 1,
        };

        Ok(Self {
            version: required("recipe_version")?,
            model: required("model")?,
            runtime: scalar("runtime"),
            container: required("container")?,
            min_nodes,
            // 2026-09-26: Version 1 recipes carry a top-level `description`; version 2
            // puts it under `metadata`.
            description: match meta_str("description") {
                d if d.is_empty() => scalar("description").unwrap_or_default(),
                d => d,
            },
            maintainer: meta_str("maintainer"),
            category: meta_str("category"),
            model_params: meta_str("model_params"),
            quantization: meta_str("quantization"),
            kv_dtype: meta_str("kv_dtype"),
            updated: meta_str("updated"),
            defaults,
            env,
            id,
            starting_point: None,
        })
    }

    /// 2026-09-26: Whether `runtime:` is `metrale`. Any other recipe is listed but not
    /// launchable, and `argv_edited` refuses it.
    pub fn is_metrale(&self) -> bool {
        self.runtime.as_deref() == Some("metrale")
    }

    /// 2026-09-26: The full `met serve` argv, with `overrides` replacing recipe values.
    ///
    /// Overrides are merged into `defaults` before rendering, so each key is
    /// rendered once.
    pub fn argv(&self, overrides: &BTreeMap<String, String>) -> Result<Vec<String>> {
        self.argv_edited(overrides, &std::collections::BTreeSet::new())
    }

    /// 2026-09-26: [`Recipe::argv`], minus the keys in `removed`.
    ///
    /// A removed key's flag is not passed, so the server's own default applies.
    /// Removal is applied after the override merge, so an override for the same
    /// key cannot bring it back.
    pub fn argv_edited(
        &self,
        overrides: &BTreeMap<String, String>,
        removed: &std::collections::BTreeSet<String>,
    ) -> Result<Vec<String>> {
        if !self.is_metrale() {
            bail!(
                "{} is a {} recipe — only `runtime: metrale` recipes can be served from here",
                self.id,
                self.runtime.as_deref().unwrap_or("non-metrale")
            );
        }
        for (key, value) in &self.defaults {
            schema::check_recipe_default(key, value)
                .map_err(|why| anyhow::anyhow!("{}: {why}", self.id))?;
        }
        let mut merged = self.defaults.clone();
        for (key, value) in overrides {
            // 2026-09-26: A key the recipe does not list is an addition, not an
            // error: a setting absent from `defaults:` is one a user may need to
            // add. An unknown flag is still refused by name when clap parses the
            // rendered argv. What clap cannot see is a key `flag_for` maps to no
            // flag (a key in `NOT_FLAGS`): it would render to nothing, so it is
            // refused here. `NOT_FLAGS` is empty, so no key reaches this today.
            if !merged.contains_key(key) && schema::flag_for(key).is_none() {
                let known: Vec<&str> = merged.keys().map(String::as_str).collect();
                bail!(
                    "{}: {key:?} is not a serve flag, so setting it would change nothing. \
                     This recipe's settings: {}",
                    self.id,
                    known.join(", ")
                );
            }
            merged.insert(key.clone(), value.clone());
        }
        for key in removed {
            merged.remove(key);
        }

        let mut argv = vec!["met".to_string(), "serve".to_string(), self.model.clone()];
        for (key, value) in &merged {
            if let Some(mut rendered) = schema::argv_for(key, value) {
                argv.append(&mut rendered);
            }
        }
        // 2026-09-26: The world size comes from `min_nodes`, outside `defaults:`.
        if self.min_nodes > 1 {
            argv.push("--world-size".into());
            argv.push(self.min_nodes.to_string());
        }
        Ok(argv)
    }

    /// 2026-09-26: Render to argv, parse it with clap, and run `validate_serve_args`.
    pub fn serve_args(
        &self,
        overrides: &BTreeMap<String, String>,
    ) -> Result<crate::cli::ServeArgs> {
        self.serve_args_edited(overrides, &std::collections::BTreeSet::new())
    }

    /// 2026-09-26: [`Recipe::serve_args`] with removals: [`Recipe::argv_edited`], then
    /// the same parse and whole-config validation as an edit.
    pub fn serve_args_edited(
        &self,
        overrides: &BTreeMap<String, String>,
        removed: &std::collections::BTreeSet<String>,
    ) -> Result<crate::cli::ServeArgs> {
        use clap::Parser as _;
        let argv = self.argv_edited(overrides, removed)?;
        let cli = crate::cli::Cli::try_parse_from(&argv)
            .with_context(|| format!("{}: recipe produced an invalid command line", self.id))?;
        let crate::cli::Command::Serve(args) = cli.command else {
            bail!("{}: recipe did not produce a serve command", self.id);
        };
        crate::cli::validate_serve_args(&args).map_err(|e| anyhow::anyhow!("{}: {e}", self.id))?;
        Ok(args)
    }
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
