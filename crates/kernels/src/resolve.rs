// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel-target resolution: which compiled target serves a
//! checkpoint. The rules are pure functions over [`ResolveCandidate`]s, so
//! they are tested without compiled kernels.
//!
//! `(model_type, hidden_size)` alone can collide: `kernels/gb10/qwen3.6-27b`
//! and `kernels/gb10/qwen3.8-27b` both declare `(qwen3_5, 5120)`.
//!
//! The rules:
//! 1. Exact `(model_type, Some(hidden_size))` declarations are tried before
//!    wildcard `(model_type, None)` ones.
//! 2. Within the first tier that matches, one target name wins outright.
//! 3. Several names are a collision, broken by `match_names`: a candidate
//!    survives when one of its needles is a case-insensitive substring of a
//!    checkpoint reference (HF id, `--model-name`, resolved model dir).
//!    Exactly one surviving name wins.
//! 4. Zero or several survivors are [`TargetResolveError::Ambiguous`]; the
//!    wildcard tier is not tried.
//! 5. `--kernel-target <name>` pins a target and skips the tie-break, but the
//!    pinned target must declare the checkpoint's `model_type` for this
//!    `hidden_size` or a wildcard ([`TargetResolveError::PinIncompatible`]).
//!
//! Owner: kernels crate.
//! Invariants:
//! - Resolution never chooses between different target names by iteration
//!   order: a collision it cannot break is an error.
//! - `build.rs` (`validate_collision_match_names`) panics when a target in a
//!   collision of the build's targets has no `match_names`.

use crate::{ModelTypeMatch, TargetPtxSet};

/// 2026-09-25: The parts of one compiled target that resolution reads,
/// borrowed so production can wrap `TargetPtxSet`s without copying blobs.
pub struct ResolveCandidate<'a> {
    /// 2026-09-25: Kernel-target directory name (`KernelTarget::model`), e.g.
    /// `"qwen3.8-27b"`. A multi-quant build repeats a name once per quant;
    /// candidates with the same name never collide, and the first one wins.
    pub name: &'a str,
    /// 2026-09-25: MODEL.toml `[[model_types]]`.
    pub type_matches: &'a [ModelTypeMatch],
    /// 2026-09-25: MODEL.toml `[model] match_names` needles.
    pub match_names: &'a [&'a str],
}

/// 2026-09-25: Why resolution could not choose a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetResolveError {
    /// 2026-09-25: Several target names claim the checkpoint's
    /// `(model_type, hidden_size)` in one tier, and the reference tie-break
    /// did not leave exactly one.
    Ambiguous {
        model_type: String,
        hidden_size: usize,
        /// 2026-09-25: `"exact"` or `"wildcard"`: the tier that collided.
        tier: &'static str,
        /// 2026-09-25: Distinct target names in that tier, with their needles.
        candidates: Vec<(String, Vec<String>)>,
        /// 2026-09-25: The names whose needles matched a reference.
        matched: Vec<String>,
        /// 2026-09-25: The checkpoint references searched.
        model_refs: Vec<String>,
    },
    /// 2026-09-25: `--kernel-target` named a target this binary did not
    /// compile.
    PinNotFound { pin: String, available: Vec<String> },
    /// 2026-09-25: `--kernel-target` named a compiled target that declares
    /// neither the checkpoint's `(model_type, hidden_size)` nor a wildcard for
    /// its `model_type`.
    PinIncompatible {
        pin: String,
        model_type: String,
        hidden_size: usize,
        declared: Vec<String>,
    },
}

impl std::fmt::Display for TargetResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ambiguous {
                model_type,
                hidden_size,
                tier,
                candidates,
                matched,
                model_refs,
            } => {
                let cands = candidates
                    .iter()
                    .map(|(n, needles)| format!("{n} (match_names: {needles:?})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let outcome = if matched.is_empty() {
                    "no checkpoint reference names any of them".to_string()
                } else {
                    format!("the references match several of them: {matched:?}")
                };
                write!(
                    f,
                    "AMBIGUOUS kernel target: {} compiled targets declare {tier} support for \
                     (model_type '{model_type}', hidden_size {hidden_size}) — [{cands}] — and \
                     {outcome} (references searched: {model_refs:?}). Refusing to pick by build \
                     order. Fix: serve with a model id/path that contains exactly one target's \
                     match_names needle, pin explicitly with --kernel-target <name>, or build \
                     single-target with METRALE_TARGET_MODEL=<name>.",
                    candidates.len(),
                )
            }
            Self::PinNotFound { pin, available } => write!(
                f,
                "--kernel-target '{pin}' does not name a compiled kernel target \
                 (available: {available:?})"
            ),
            Self::PinIncompatible {
                pin,
                model_type,
                hidden_size,
                declared,
            } => write!(
                f,
                "--kernel-target '{pin}' is compiled but declares no support for this \
                 checkpoint's (model_type '{model_type}', hidden_size {hidden_size}) — it \
                 declares {declared:?}. Serving another architecture's kernels would be \
                 garbage; refusing."
            ),
        }
    }
}

impl std::error::Error for TargetResolveError {}

/// 2026-09-25: Whether any non-empty needle, lower-cased, is a substring of
/// any lower-cased reference.
fn needles_hit(match_names: &[&str], refs_lower: &[String]) -> bool {
    match_names.iter().any(|needle| {
        let n = needle.to_lowercase();
        !n.is_empty() && refs_lower.iter().any(|r| r.contains(&n))
    })
}

/// 2026-09-25: Distinct candidate names among `idxs`, in first-seen order.
fn distinct_names<'a>(candidates: &[ResolveCandidate<'a>], idxs: &[usize]) -> Vec<&'a str> {
    let mut names: Vec<&str> = Vec::new();
    for &i in idxs {
        if !names.contains(&candidates[i].name) {
            names.push(candidates[i].name);
        }
    }
    names
}

/// 2026-09-25: Which candidate serves `(model_type, hidden_size)` for a
/// checkpoint identified by `model_refs`: its index, `Ok(None)` when no tier
/// declares the pair, or [`TargetResolveError::Ambiguous`] when a collision
/// does not break to exactly one name.
pub fn resolve_target(
    candidates: &[ResolveCandidate<'_>],
    model_type: &str,
    hidden_size: usize,
    model_refs: &[&str],
) -> Result<Option<usize>, TargetResolveError> {
    let refs_lower: Vec<String> = model_refs.iter().map(|r| r.to_lowercase()).collect();

    let tiers: [(&'static str, Option<usize>); 2] =
        [("exact", Some(hidden_size)), ("wildcard", None)];
    for (tier, want_hidden) in tiers {
        let idxs: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.type_matches
                    .iter()
                    .any(|m| m.model_type == model_type && m.hidden_size == want_hidden)
            })
            .map(|(i, _)| i)
            .collect();
        if idxs.is_empty() {
            continue;
        }
        let names = distinct_names(candidates, &idxs);
        if names.len() == 1 {
            // 2026-09-25: One name, possibly several quant variants: the first
            // wins.
            return Ok(Some(idxs[0]));
        }
        let matched: Vec<&str> = names
            .iter()
            .copied()
            .filter(|n| {
                idxs.iter().any(|&i| {
                    candidates[i].name == *n && needles_hit(candidates[i].match_names, &refs_lower)
                })
            })
            .collect();
        if let [winner] = matched.as_slice() {
            let idx = idxs
                .iter()
                .copied()
                .find(|&i| candidates[i].name == *winner)
                .expect("winner name came from idxs");
            return Ok(Some(idx));
        }
        // 2026-09-25: Zero or several survivors: error, without trying the
        // wildcard tier.
        return Err(TargetResolveError::Ambiguous {
            model_type: model_type.to_string(),
            hidden_size,
            tier,
            candidates: names
                .iter()
                .map(|n| {
                    let needles = idxs
                        .iter()
                        .filter(|&&i| candidates[i].name == *n)
                        .flat_map(|&i| candidates[i].match_names.iter().map(|s| s.to_string()))
                        .collect();
                    (n.to_string(), needles)
                })
                .collect(),
            matched: matched.iter().map(|n| n.to_string()).collect(),
            model_refs: model_refs.iter().map(|r| r.to_string()).collect(),
        });
    }
    Ok(None)
}

/// 2026-09-25: Resolve a `--kernel-target` pin, matched case-insensitively
/// against target names. The first same-named candidate that declares the
/// checkpoint's `model_type` with this `hidden_size` or a wildcard wins; the
/// tie-break is not consulted.
pub fn resolve_pinned(
    candidates: &[ResolveCandidate<'_>],
    pin: &str,
    model_type: &str,
    hidden_size: usize,
) -> Result<usize, TargetResolveError> {
    let pinned: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name.eq_ignore_ascii_case(pin))
        .map(|(i, _)| i)
        .collect();
    if pinned.is_empty() {
        let all: Vec<usize> = (0..candidates.len()).collect();
        return Err(TargetResolveError::PinNotFound {
            pin: pin.to_string(),
            available: distinct_names(candidates, &all)
                .into_iter()
                .map(String::from)
                .collect(),
        });
    }
    let compatible = pinned.iter().copied().find(|&i| {
        candidates[i].type_matches.iter().any(|m| {
            m.model_type == model_type
                && (m.hidden_size.is_none() || m.hidden_size == Some(hidden_size))
        })
    });
    compatible.ok_or_else(|| TargetResolveError::PinIncompatible {
        pin: pin.to_string(),
        model_type: model_type.to_string(),
        hidden_size,
        declared: pinned
            .iter()
            .flat_map(|&i| candidates[i].type_matches.iter())
            .map(|m| format!("({}, {:?})", m.model_type, m.hidden_size))
            .collect(),
    })
}

/// 2026-09-25: The compiled target that serves a checkpoint, by the module's
/// rules: [`resolve_pinned`] when `pinned_target` (`--kernel-target`) is set,
/// else [`resolve_target`] over `all_ptx_sets()`. `Ok(None)` when no compiled
/// target declares the pair.
pub fn ptx_for_config(
    model_type: &str,
    hidden_size: usize,
    model_refs: &[&str],
    pinned_target: Option<&str>,
) -> Result<Option<TargetPtxSet>, TargetResolveError> {
    let targets = crate::all_ptx_sets();
    let candidates: Vec<ResolveCandidate<'_>> = targets
        .iter()
        .map(|t| ResolveCandidate {
            name: t.target.model,
            type_matches: &t.model_type_matches,
            match_names: t.match_names,
        })
        .collect();
    let idx = match pinned_target {
        Some(pin) => Some(resolve_pinned(&candidates, pin, model_type, hidden_size)?),
        None => resolve_target(&candidates, model_type, hidden_size, model_refs)?,
    };
    drop(candidates);
    Ok(idx.and_then(|i| targets.into_iter().nth(i)))
}

/// 2026-09-25: The compiled target with exactly this `(model, quant)`
/// identity, or `None`. For callers that already know the resolved target,
/// such as the dashboard's kernel table, which looks up the target `serve`
/// published.
pub fn ptx_for_exact_target(model: &str, quant: &str) -> Option<TargetPtxSet> {
    crate::all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model == model && t.target.quant == quant)
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
