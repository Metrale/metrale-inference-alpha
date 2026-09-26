// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Reading the materialised BFCL table and applying the draw to it.
//! The draw takes the first `n` rows of each subset in file order, so the row
//! order within a subset is the sample selection.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use super::draw::{self, DrawSpec};

/// 2026-09-26: One materialised BFCL sample.
#[derive(Clone, Debug, Deserialize)]
pub struct Sample {
    pub subset: String,
    pub sample_id: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
}

/// 2026-09-26: One shard of a draw: `index` (0-based) of `count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    /// 2026-09-26: Which shard, in `0..count`.
    pub index: usize,
    /// 2026-09-26: How many shards the draw is split into.
    pub count: usize,
}

impl Shard {
    /// 2026-09-26: Parses `"i/n"`, the `--param shard=2/7` surface. `i` is
    /// 0-based, as in `draw::shard_owns`, so the whole draw is `0/1` and `1/1`
    /// is refused. The error is the message the operator sees, naming the
    /// value and what is wrong with it.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let (i, n) = s
            .split_once('/')
            .ok_or_else(|| format!("shard {s:?} is not `index/count`, e.g. `2/4`"))?;
        let index: usize = i
            .trim()
            .parse()
            .map_err(|_| format!("shard index {i:?} is not a number"))?;
        let count: usize = n
            .trim()
            .parse()
            .map_err(|_| format!("shard count {n:?} is not a number"))?;
        if count == 0 {
            return Err("shard count must be at least 1".to_string());
        }
        if index >= count {
            return Err(format!(
                "shard index {index} is out of range for {count} shards — \
                 indices are 0-based, so the last is {}",
                count - 1
            ));
        }
        Ok(Self { index, count })
    }
}

pub fn load(path: &Path, spec: &DrawSpec) -> Result<Vec<Sample>> {
    load_shard(path, spec, None)
}

/// 2026-09-26: Loads one shard of the draw, or all of it when `shard` is
/// `None`. The plan decides how many rows of each subset the draw takes, and
/// the shard then keeps every `count`-th of those; filtering before the plan
/// would change which rows the draw contains. An empty selection is an error.
pub fn load_shard(path: &Path, spec: &DrawSpec, shard: Option<Shard>) -> Result<Vec<Sample>> {
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "reading {} — delete ~/.metrale/artifacts/bfcl to re-provision",
            path.display()
        )
    })?;
    let totals = totals_of(&text)?;
    let plan: BTreeMap<String, usize> = draw::plan(spec, &totals).into_iter().collect();

    let mut taken: BTreeMap<String, usize> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let sample: Sample = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: malformed sample", path.display(), i + 1))?;
        let Some(limit) = plan.get(&sample.subset) else {
            continue;
        };
        let count = taken.entry(sample.subset.clone()).or_insert(0);
        if *count >= *limit {
            continue;
        }
        // 2026-09-26: `*count` is this row's 0-based position within the drawn
        // rows of its subset, which `shard_owns` strides over. It advances
        // whether or not this shard keeps the row, so every shard agrees on
        // each row's position.
        let position = *count;
        *count += 1;
        if let Some(sh) = shard
            && !draw::shard_owns(position, sh.index, sh.count)
        {
            continue;
        }
        out.push(sample);
    }
    if out.is_empty() {
        match shard {
            None => bail!("the draw selected no samples — check the categories and percentages"),
            Some(sh) => bail!(
                "shard {} of {} selected no samples — the draw is smaller than the \
                 shard count, or the plan is empty",
                sh.index,
                sh.count
            ),
        }
    }
    // 2026-09-26: A stable sort: by subset name, then file order.
    out.sort_by(|a, b| a.subset.cmp(&b.subset));
    Ok(out)
}

/// 2026-09-26: Per-subset row counts, without parsing whole samples.
pub fn totals(path: &Path) -> Result<BTreeMap<String, usize>> {
    totals_of(&std::fs::read_to_string(path)?)
}

fn totals_of(text: &str) -> Result<BTreeMap<String, usize>> {
    #[derive(Deserialize)]
    struct SubsetOnly {
        subset: String,
    }
    let mut totals = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // 2026-09-26: `load_shard` counts before it parses, so this pass meets a
        // corrupt line first and must name the line.
        let row: SubsetOnly = serde_json::from_str(line)
            .with_context(|| format!("line {}: malformed sample", i + 1))?;
        *totals.entry(row.subset).or_insert(0) += 1;
    }
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, rows: &[(&str, usize)]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "metrale-bfcl-ds-{name}-{}.jsonl",
            std::process::id()
        ));
        let mut text = String::new();
        for (subset, n) in rows {
            for i in 0..*n {
                text.push_str(&format!(
                    r#"{{"subset":"{subset}","sample_id":"{subset}_{i}","messages":[],"tools":[],"tool_choice":"auto"}}"#
                ));
                text.push('\n');
            }
        }
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn the_draw_takes_the_first_n_of_each_subset() {
        let p = write(
            "head",
            &[
                ("simple_python", 10),
                ("simple_java", 10),
                ("irrelevance", 10),
            ],
        );
        let spec = DrawSpec {
            categories: vec!["non_live".into()],
            category_pct: [("non_live".to_string(), 50.0)].into_iter().collect(),
            subset_floor: None,
        };
        let s = load(&p, &spec).unwrap();
        assert_eq!(
            s.len(),
            10,
            "50% of both selected subsets, and irrelevance is excluded"
        );
        let ids: Vec<&str> = s.iter().map(|x| x.sample_id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "simple_java_0",
                "simple_java_1",
                "simple_java_2",
                "simple_java_3",
                "simple_java_4",
                "simple_python_0",
                "simple_python_1",
                "simple_python_2",
                "simple_python_3",
                "simple_python_4"
            ],
            "head(n), not a random sample"
        );
    }

    #[test]
    fn totals_are_counted_without_parsing_whole_samples() {
        let p = write("totals", &[]);
        std::fs::write(
            &p,
            concat!(
                "{\"subset\":\"multiple\"}\n",
                "{\"subset\":\"multiple\"}\n",
                "{\"subset\":\"multiple\"}\n",
                "{\"subset\":\"live_simple\"}\n",
                "{\"subset\":\"live_simple\"}\n",
            ),
        )
        .unwrap();
        let t = totals(&p).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t["multiple"], 3);
        assert_eq!(t["live_simple"], 2);
    }

    #[test]
    fn an_empty_draw_is_an_error_rather_than_a_zero_sample_run() {
        let p = write("empty", &[("live_relevance", 4)]);
        let err = load(&p, &DrawSpec::golden()).unwrap_err().to_string();
        assert!(err.contains("selected no samples"), "{err}");
    }

    #[test]
    fn a_malformed_line_names_its_line_number() {
        let p = write("bad", &[("multiple", 1)]);
        let mut text = std::fs::read_to_string(&p).unwrap();
        text.push_str("{not json}\n");
        std::fs::write(&p, text).unwrap();
        let err = load(&p, &DrawSpec::full()).unwrap_err().to_string();
        assert!(err.contains("line 2: malformed sample"), "{err}");
    }
}
