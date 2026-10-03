// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The MoE expert-table decision: whether a serve builds its MoE's transposed NVFP4
//! prefill tables. Made from the memory model at the serve's declared length and batch, by
//! `met circuit memory` (whose report then carries it) and by `met serve` before the model loads
//! (which publishes it to the loader, `MoeExpertTables`), through this one function.
//!
//! Owner: server CLI.
//! Invariants:
//! - The tables are the copies whose COPIES.toml rule reads [`SETTING`]; the rules, not this
//!   file, say which they are.
//! - Build when the plan, with the tables, fits the util budget at `--max-seq-len` tokens and
//!   `--max-batch-size` sequences (one under `auto`, which then gives every slot the KV left);
//!   skip otherwise. Only the serve's own flags count ([`Query::SERVE`]), never a CLI what-if.
//! - A plan whose rules match no table decides nothing (`None`).

use anyhow::Result;
use metrale_circuit::memory::MemoryReport;
use metrale_model_engine::factory::SlotRequest;
use metrale_model_layers::layers::MoeExpertTables;

use super::circuit_memory_point::{Point, Query};
use crate::cli::ServeArgs;

/// 2026-10-02: The copy-rule setting that carries the decision (`build` or `skip`).
pub(crate) const SETTING: &str = "moe_expert_tables";

/// 2026-10-02: The decision and the plan it was read off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TablesDecision {
    pub tables: MoeExpertTables,
    /// 2026-10-02: The tables' bytes in the plan.
    pub bytes: u64,
    /// 2026-10-02: Headroom with the tables.
    pub headroom_with: i128,
    /// 2026-10-02: Headroom without them; evaluated only when they do not fit.
    pub headroom_without: Option<i128>,
    /// 2026-10-02: The workload it was planned at: sequences, ISL, OSL.
    pub workload: (u64, u64, u64),
}

/// 2026-10-02: The serve's declared workload: `--max-batch-size` sequences (one under `auto`)
/// of `--max-seq-len` tokens, as ISL `max_seq_len - 1` plus one output token.
pub(crate) fn workload(args: &ServeArgs) -> (u64, u64, u64) {
    let c = match args.max_batch_size {
        SlotRequest::Count(n) => n as u64,
        SlotRequest::Auto => 1,
    };
    (
        (c.max(1)),
        (args.max_seq_len as u64).saturating_sub(1).max(1),
        1,
    )
}

fn table_bytes(p: &Point<'_>, r: &MemoryReport) -> u64 {
    let ids: Vec<&str> = p
        .copies
        .iter()
        .filter(|c| c.when.contains_key(SETTING))
        .map(|c| c.id.as_str())
        .collect();
    r.weights
        .iter()
        .flat_map(|w| &w.derived)
        .filter(|d| ids.contains(&d.rule.as_str()))
        .map(|d| d.bytes)
        .sum()
}

/// 2026-10-02: Decide, and leave `p.settings` holding the decision.
pub(crate) fn decide(p: &mut Point<'_>) -> Result<Option<TablesDecision>> {
    let workload = workload(&p.args);
    let (c, isl, osl) = workload;
    let cli = std::mem::replace(&mut p.query, Query::SERVE);
    let eval = |p: &mut Point<'_>, t: MoeExpertTables| -> Result<MemoryReport> {
        p.settings.insert(SETTING.to_string(), t.name().to_string());
        Ok(p.eval(c, isl, osl)?.0)
    };
    let decided = (|| -> Result<Option<TablesDecision>> {
        let with = eval(p, MoeExpertTables::Build)?;
        let bytes = table_bytes(p, &with);
        if bytes == 0 {
            return Ok(None);
        }
        let mut d = TablesDecision {
            tables: MoeExpertTables::Build,
            bytes,
            headroom_with: with.headroom(),
            headroom_without: None,
            workload,
        };
        if d.headroom_with < 0 {
            d.tables = MoeExpertTables::Skip;
            d.headroom_without = Some(eval(p, MoeExpertTables::Skip)?.headroom());
        }
        Ok(Some(d))
    })();
    p.query = cli;
    decided
}

impl TablesDecision {
    /// 2026-10-02: One line for the serve log and the CLI header.
    pub(crate) fn describe(&self) -> String {
        let mib = |b: i128| format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64);
        let (c, isl, osl) = self.workload;
        let without = self
            .headroom_without
            .map_or(String::new(), |h| format!(", {} without them", mib(h)));
        format!(
            "{}: {} of tables; headroom {} with them{without} (memory plan at {c} x ISL {isl} + \
             OSL {osl})",
            self.tables.name(),
            mib(self.bytes as i128),
            mib(self.headroom_with),
        )
    }
}

#[cfg(test)]
#[path = "circuit_memory_tables_tests.rs"]
mod tests;
