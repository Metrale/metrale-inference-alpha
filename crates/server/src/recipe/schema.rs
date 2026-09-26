// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Recipe `defaults:` keys → `met serve` flags.
//!
//! Owner: server (recipe).
//! Invariants:
//! - `flag_for` returns `None` only for a key in `NOT_FLAGS`.
//! - The presence-only set is read from `ServeArgs::command()`, never written down.
//!
//! clap is the single source of truth for the flag surface. `ServeArgs` has no
//! `Serialize` derive: a recipe is rendered to argv and parsed back by clap
//! exactly as typed input, so a key that maps to no existing flag fails at
//! parse. Most keys are the field name with underscores swapped for dashes;
//! `RENAMES` lists the exceptions.

/// 2026-09-26: Keys whose recipe spelling differs from the flag. `ServeArgs` has no
/// `max_model_len` or `tensor_parallel` field, so clap would reject those two
/// unrenamed; the listen address is `--bind`, which also accepts `--host` as an
/// alias.
pub(crate) const RENAMES: &[(&str, &str)] = &[
    ("max_model_len", "max-seq-len"),
    ("tensor_parallel", "tp-size"),
    ("host", "bind"),
];

/// 2026-09-26: `defaults:` keys that are not `met serve` flags. Empty, so every key
/// maps to a flag.
const NOT_FLAGS: &[&str] = &[];

/// 2026-09-26: The flag for a recipe key, or `None` if the key is in `NOT_FLAGS`.
pub fn flag_for(key: &str) -> Option<String> {
    if NOT_FLAGS.contains(&key) {
        return None;
    }
    if let Some((_, flag)) = RENAMES.iter().find(|(k, _)| *k == key) {
        return Some((*flag).to_string());
    }
    Some(key.replace('_', "-"))
}

/// 2026-09-26: Render one `key: value` pair as argv.
///
/// A presence-only flag (clap `SetTrue`, e.g. `--speculative`) takes no value:
/// `true` becomes the bare `--flag` and `false` is omitted, because the flag's
/// absence is its default. Every boolean on the command line is such a flag
/// (`cli::bool_surface_tests`). Any other value, including an `auto`/`on`/`off`
/// enum such as `--tool-grammar`, passes through after the flag.
///
/// Whether a recipe may write `false` at all is [`check_recipe_default`]'s
/// question.
pub fn argv_for(key: &str, value: &str) -> Option<Vec<String>> {
    let flag = flag_for(key)?;
    let dashed = format!("--{flag}");
    match value {
        "true" if presence_only(&flag) => Some(vec![dashed]),
        "false" if presence_only(&flag) => None,
        other => Some(vec![dashed, other.to_string()]),
    }
}

/// 2026-09-26: Refuse a recipe `defaults:` entry that gives a presence-only flag
/// anything but `true`.
///
/// `speculative: false` renders to nothing, exactly like leaving the key out, so
/// a recipe line that writes it states a decision it does not make. Any other
/// value is refused because the flag takes none. Keys that are not presence-only
/// flags pass.
///
/// `Recipe::argv_edited` checks only the recipe's own `defaults:`, not overrides:
/// an override's `false` removes a `true` the recipe set, as `--hermetic` does
/// with `enable_prefix_caching=false`.
pub fn check_recipe_default(key: &str, value: &str) -> Result<(), String> {
    let Some(flag) = flag_for(key) else {
        return Ok(());
    };
    if !presence_only(&flag) {
        return Ok(());
    }
    match value {
        "true" => Ok(()),
        "false" => Err(format!(
            "defaults.{key}: false restates the default of --{flag}; a boolean flag is \
             off unless given, so delete the line (write `{key}: true` to turn it on)"
        )),
        other => Err(format!(
            "defaults.{key}: --{flag} takes no value, so {other:?} means nothing; write \
             `{key}: true` to set it or delete the line"
        )),
    }
}

/// 2026-09-26: Whether `flag` is a presence-only boolean (clap action `SetTrue`).
///
/// The set is read from `ServeArgs::command()` once per process, so it cannot
/// drift from the CLI. A flag not in `ServeArgs` is not presence-only: its value
/// passes through and clap rejects the unknown flag by name at parse.
fn presence_only(flag: &str) -> bool {
    use std::sync::OnceLock;
    static SET_TRUE: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    SET_TRUE
        .get_or_init(|| {
            use clap::CommandFactory as _;
            crate::cli::ServeArgs::command()
                .get_arguments()
                .filter(|a| matches!(a.get_action(), clap::ArgAction::SetTrue))
                .filter_map(|a| a.get_long().map(str::to_string))
                .collect()
        })
        .contains(flag)
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
