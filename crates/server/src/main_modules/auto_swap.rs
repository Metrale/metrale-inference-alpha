// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Auto-swap: with `--auto-swap`, a chat request whose `model`
//! names a different model known to the recipe catalogue loads it first.
//!
//! | request `model`                     | action                 |
//! |-------------------------------------|------------------------|
//! | empty or blank                      | serve the current model |
//! | not a `metrale` recipe's model      | serve the current model |
//! | the model already live              | serve the current model |
//! | a different recipe's model          | swap, then serve       |
//!
//! Owner: server (model hosting).
//! Invariants: [`decide`] returns `SwapTo` only for the id of a `metrale`
//! recipe whose `model` equals the trimmed request exactly.

use crate::recipe::Recipe;

/// 2026-09-26: What a request's `model` field asks of the server.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    ServeCurrent,
    /// 2026-09-26: Load this recipe first. Carries the recipe id, not the
    /// request string.
    SwapTo(String),
}

/// 2026-09-26: Whether request-triggered swapping is on (`--auto-swap`).
pub(crate) fn enabled(args: &crate::cli::ServeArgs) -> bool {
    args.auto_swap
}

/// 2026-09-26: Decide what to do about `requested`, given the live model and
/// the catalogue. Matching is exact against a recipe's `model`; a fuzzy match
/// could swap to a model the caller did not ask for.
pub(crate) fn decide(requested: &str, live_model: &str, catalogue: &[Recipe]) -> Decision {
    let requested = requested.trim();
    if requested.is_empty() || requested == live_model {
        return Decision::ServeCurrent;
    }
    match catalogue
        .iter()
        .filter(|r| r.is_metrale())
        .find(|r| r.model == requested)
    {
        Some(recipe) => Decision::SwapTo(recipe.id.clone()),
        None => Decision::ServeCurrent,
    }
}

/// 2026-09-26: Load `recipe_id` unless `requested_model` is already live.
///
/// Blocking, and long: call it from `spawn_blocking`. `model_swap::swap` takes
/// the swap guard and skips the load when its argv is already serving.
pub(crate) fn ensure_loaded(
    host: &std::sync::Arc<super::model_host::ModelHost>,
    recipe_id: &str,
    requested_model: &str,
    catalogue: &[Recipe],
) -> anyhow::Result<()> {
    // 2026-09-26: An early exit only; the guard is taken in `swap`, and taking
    // it here as well would deadlock on the non-reentrant mutex.
    if host.live_model().as_deref() == Some(requested_model) {
        return Ok(());
    }
    let recipe = catalogue
        .iter()
        .find(|r| r.id == recipe_id)
        .ok_or_else(|| anyhow::anyhow!("recipe {recipe_id} vanished between decide and load"))?;
    let args = recipe.serve_args(&std::collections::BTreeMap::new())?;
    super::model_swap::swap(host, args)?;
    Ok(())
}

#[cfg(test)]
#[path = "auto_swap_tests.rs"]
mod tests;
