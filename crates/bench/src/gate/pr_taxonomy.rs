// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The PR intent taxonomy (`.github/pr-taxonomy.json`): a tree of
//! what a change is for (`performance/decode`, `correctness/kv-cache`), and
//! the benchmarks each path implies. [`crate::gate::taxon`] is the separate
//! hardware taxonomy derived from `kernels/` paths.
//!
//! Owner: bench gate (intent).
//! Invariants:
//! - `benches_for` is the union of `_benches` along the matched prefix, so
//!   descending further never removes a benchmark.
//! - `load` returns only a tree that passed `validate`: at least two roots,
//!   kebab-case keys, no single-child node, and every `_benches` id in
//!   `coverage::REQUIRED`.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// 2026-09-26: Keys in the JSON that are metadata rather than child nodes.
const RESERVED: [&str; 2] = ["_doc", "_benches"];

/// 2026-09-26: One node of the parsed tree. Children keep the JSON file's key
/// order (the workspace enables serde_json's `preserve_order`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub benches: Vec<String>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// 2026-09-26: Load and validate `.github/pr-taxonomy.json` from a repo root.
pub fn load(root: &Path) -> Result<Vec<Node>> {
    let path = root.join(".github/pr-taxonomy.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let roots = parse_children(&json)?;
    validate(&roots)?;
    Ok(roots)
}

fn parse_children(value: &serde_json::Value) -> Result<Vec<Node>> {
    let Some(obj) = value.as_object() else {
        bail!("every taxonomy node must be a JSON object");
    };
    let mut out = Vec::new();
    for (key, child) in obj {
        if RESERVED.contains(&key.as_str()) {
            continue;
        }
        // 2026-09-26: `_benches` must be an array of strings. Any other shape
        // is an error rather than an empty list, which would drop benchmarks.
        let benches = match child.get("_benches") {
            None => Vec::new(),
            Some(serde_json::Value::Array(items)) => {
                let mut v = Vec::with_capacity(items.len());
                for item in items {
                    match item.as_str() {
                        Some(s) => v.push(s.to_string()),
                        None => bail!(
                            "{key}: _benches contains a non-string entry ({item}). \
                             A silently-dropped entry removes a benchmark."
                        ),
                    }
                }
                v
            }
            Some(other) => bail!(
                "{key}: _benches must be an ARRAY of benchmark ids, got {other}. \
                 A bare string parses as empty here while jq reads it, so the two \
                 halves would disagree — in the removing direction."
            ),
        };
        out.push(Node {
            name: key.clone(),
            benches,
            children: parse_children(child)?,
        });
    }
    Ok(out)
}

/// 2026-09-26: The shape rules the JSON's `_doc` states: at least two roots,
/// lowercase kebab-case keys, no node with exactly one child, and every
/// `_benches` id in `coverage::REQUIRED`.
fn validate(roots: &[Node]) -> Result<()> {
    if roots.len() < 2 {
        bail!("the taxonomy needs at least two roots; one root is not a choice");
    }
    let known: BTreeSet<&str> = super::coverage::REQUIRED.iter().map(|g| g.id).collect();
    fn walk(nodes: &[Node], trail: &str, known: &BTreeSet<&str>) -> Result<()> {
        for n in nodes {
            let here = if trail.is_empty() {
                n.name.clone()
            } else {
                format!("{trail}/{}", n.name)
            };
            if !n
                .name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                || n.name.is_empty()
            {
                bail!("{here}: keys must be lowercase kebab-case so a path is a safe label");
            }
            for b in &n.benches {
                if !known.contains(b.as_str()) {
                    bail!(
                        "{here}: _benches names {b:?}, which is not a required benchmark. \
                         A path that selects a benchmark nobody runs is a silent no-op."
                    );
                }
            }
            // 2026-09-26: A single child is not a choice for the classifier.
            if n.children.len() == 1 {
                bail!(
                    "{here} has exactly one child ({}). Either give it a sibling or \
                     make {here} a leaf.",
                    n.children[0].name
                );
            }
            walk(&n.children, &here, known)?;
        }
        Ok(())
    }
    walk(roots, "", &known)
}

/// 2026-09-26: The children of the node at `path`, when there are at least
/// two. `None` for a leaf, a lone child (which [`resolve`] follows) or a path
/// that does not match.
pub fn options_at(roots: &[Node], path: &[String]) -> Option<Vec<String>> {
    let node = walk_to(roots, path)?;
    let kids: Vec<String> = node.iter().map(|n| n.name.clone()).collect();
    (kids.len() > 1).then_some(kids)
}

fn walk_to<'a>(roots: &'a [Node], path: &[String]) -> Option<&'a [Node]> {
    let mut level = roots;
    for step in path {
        level = &level.iter().find(|n| &n.name == step)?.children;
    }
    Some(level)
}

/// 2026-09-26: Every benchmark a path implies: the union of `_benches` along
/// it, so an ancestor's benchmarks apply to every descendant. The walk stops
/// at the first unknown segment and returns what matched so far.
pub fn benches_for(roots: &[Node], path: &[String]) -> BTreeSet<String> {
    benches_for_matched(roots, path).0
}

/// 2026-09-26: [`benches_for`], plus how many leading segments matched, so a
/// caller can report a stale segment (`performance/decodes` implies the same
/// set as `performance`).
pub fn benches_for_matched(roots: &[Node], path: &[String]) -> (BTreeSet<String>, usize) {
    let mut out = BTreeSet::new();
    let mut level = roots;
    let mut matched = 0usize;
    for step in path {
        let Some(node) = level.iter().find(|n| &n.name == step) else {
            break;
        };
        out.extend(node.benches.iter().cloned());
        level = &node.children;
        matched += 1;
    }
    (out, matched)
}

/// 2026-09-26: `path` extended through every single-child step below it.
/// `validate` refuses single-child nodes, so on a loaded tree this returns
/// `path` unchanged.
pub fn resolve(roots: &[Node], path: &[String]) -> Vec<String> {
    let mut out = path.to_vec();
    loop {
        let Some(level) = walk_to(roots, &out) else {
            return out;
        };
        if level.len() == 1 {
            out.push(level[0].name.clone());
        } else {
            return out;
        }
    }
}

/// 2026-09-26: Whether `path` matches and ends at a leaf.
pub fn is_complete(roots: &[Node], path: &[String]) -> bool {
    walk_to(roots, path).is_some_and(<[Node]>::is_empty)
}

#[cfg(test)]
#[path = "pr_taxonomy_tests.rs"]
mod pr_taxonomy_tests;
