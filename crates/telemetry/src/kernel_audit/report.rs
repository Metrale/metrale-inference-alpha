// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The rendering half of the kernel audit: the embedded-kernel-set
//! table and the unresolved-lookup report the boot gate prints.
//!
//! [`unresolved_report`]'s own text is plain ASCII, so it survives a non-TTY
//! log pipe (`the_report_is_plain_ascii`); the table uses box-drawing
//! characters.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::{FailureSplit, split_failures};

/// 2026-09-26: FNV-1a 64-bit content fingerprint, low 48 bits as 12 hex chars:
/// the same function as `content_hash` in the kernels crate's `build.rs`.
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// 2026-09-26: Render the embedded kernel set (`embedded`, the loaded target's
/// `TargetPtxSet::modules`) with each module's resolution. `set_hash` is
/// `metrale_kernels::KERNEL_SET_HASH`, `shadowed_dropped` fills the SHADOWED
/// column, and `expected_absent` selects the failures listed as expected (via
/// [`super::split_failures`]).
pub fn render_kernel_table(
    embedded: &[(&str, &[u8])],
    set_hash: &str,
    shadowed_dropped: &[(&str, &str)],
    expected_absent: &[(&str, &str)],
) -> String {
    let rows = super::audit_rows();
    // 2026-09-26: Per module: absent if never looked up, else whether any
    // lookup resolved.
    let mut mod_resolved: BTreeMap<&str, bool> = BTreeMap::new();
    for r in &rows {
        let e = mod_resolved.entry(r.module.as_str()).or_insert(false);
        *e = *e || r.loaded;
    }

    let mut out = String::new();
    out.push_str(&format!(
        "\n\u{250c}\u{2500} Kernel load audit \u{2500} {} kernels embedded \u{b7} \
         set-hash {} \u{2500}\n",
        embedded.len(),
        set_hash
    ));
    out.push_str(&format!(
        "\u{2502} {:<34} {:<14} {:<20} {}\n",
        "MODULE (operation)", "PTX-HASH", "RESOLUTION", "SHADOWED"
    ));
    out.push_str(&format!("\u{2502} {}\n", "\u{2500}".repeat(84)));
    let mut sorted: Vec<&(&str, &[u8])> = embedded.iter().collect();
    sorted.sort_by_key(|(m, _)| *m);
    for (m, blob) in sorted {
        let h = ptx_hash(blob);
        let res = match mod_resolved.get(m) {
            Some(true) => "used",
            Some(false) => "** lookup FAILED **",
            None => "-", // 2026-09-26: embedded, never looked up
        };
        // 2026-09-26: Y when this target's file of that name lacks kernels its
        // `common/` namesake defines (`shadowed_dropped`).
        let n_dropped = shadowed_dropped.iter().filter(|(sm, _)| sm == m).count();
        let shadow = if n_dropped > 0 {
            format!("Y ({n_dropped} dropped)")
        } else {
            "N".to_string()
        };
        out.push_str(&format!("\u{2502} {m:<34} {h:<14} {res:<20} {shadow}\n"));
    }
    out.push_str("\u{2514}\u{2500}\n");

    let split = split_failures(&rows, expected_absent);
    if !split.expected.is_empty() {
        out.push_str(&format!(
            "\n{} kernel(s) declared expected-absent in this target's MODEL.toml \
             [expected_absent] (no action):\n    {}\n",
            split.expected.len(),
            split
                .expected
                .iter()
                .map(|r| r.name())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out
}

/// 2026-09-26: The unresolved-lookup report: the numbered list of
/// `split.required`, then the remediation paragraph once. `allowed` selects
/// the closing paragraph: the offer to pass the allow flag, or, when it is
/// already set, only the consequence.
pub fn unresolved_report(
    split: &FailureSplit,
    shadowed_dropped: &[(&str, &str)],
    model: &str,
    arch: &str,
    quant: &str,
    allowed: bool,
) -> String {
    let mut out = format!(
        "{} unresolved kernel lookup(s) for ({model}, {arch}, {quant}). Each one resolved to \
         handle 0, so its dispatch site is on a silent fallback path:\n",
        split.required.len()
    );
    for (i, r) in split.required.iter().enumerate() {
        let n = i + 1;
        let name = r.name();
        let dropped = shadowed_dropped
            .iter()
            .any(|(sm, sf)| *sm == r.module.as_str() && *sf == r.func.as_str());
        // 2026-09-26: A shadow-dropped kernel's fix is appended to its line.
        let note = if dropped {
            "  [SHADOW-DROPPED: common/ defines it, this target's kernel file shadows common/ \
             without it - port it in as an exact piecewise copy]"
        } else {
            ""
        };
        out.push_str(&format!(
            "  {n}. {name}  at {}:{}  ({model}, {arch}, {quant}){note}\n",
            r.site.file(),
            r.site.line(),
        ));
    }
    out.push_str(
        "\nEither gate the lookup on this model's config so it is never issued (see \
         `qwen3_attention::init`), fix the build so the kernel is compiled, or declare it in \
         this target's MODEL.toml [expected_absent] with a stated reason.\n",
    );
    if allowed {
        out.push_str(
            "\nPerformance may be seriously degraded. We recommend you open a GitHub issue \
             and/or open a PR to solve this issue.\n",
        );
    } else {
        out.push_str(
            "\nIf you wish to allow this model to be served, you can pass\n\
             --dangerously-allow-unresolved-kernel-lookups. But note that performance may be\n\
             seriously degraded. We recommend you open a GitHub issue and/or open a PR to\n\
             solve this issue.\n",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::AuditRow;
    use super::*;

    fn row(module: &str, func: &str) -> AuditRow {
        AuditRow {
            module: module.to_string(),
            func: func.to_string(),
            loaded: false,
            site: std::panic::Location::caller(),
        }
    }

    /// 2026-09-26: Both modes print the numbered list and one closing
    /// paragraph, and only the deny mode offers the flag.
    #[test]
    fn the_report_enumerates_then_remediates_once() {
        let split = FailureSplit {
            required: vec![row("gdn", "gdn_decode_multi_seq")],
            expected: vec![],
        };
        let deny = unresolved_report(&split, &[], "qwen3.6-27b", "sm_121", "nvfp4", false);
        assert!(deny.starts_with("1 unresolved kernel lookup(s)"));
        assert!(deny.contains("  1. gdn::gdn_decode_multi_seq"));
        assert!(deny.contains("--dangerously-allow-unresolved-kernel-lookups"));
        assert_eq!(deny.matches("open a GitHub issue").count(), 1);

        let allow = unresolved_report(&split, &[], "qwen3.6-27b", "sm_121", "nvfp4", true);
        assert!(allow.contains("  1. gdn::gdn_decode_multi_seq"));
        assert!(
            !allow.contains("If you wish to allow"),
            "the flag is already set; do not offer it again"
        );
        assert_eq!(allow.matches("open a GitHub issue").count(), 1);
    }

    /// 2026-09-26: The report is plain ASCII, shadow-drop note included.
    #[test]
    fn the_report_is_plain_ascii() {
        let split = FailureSplit {
            required: vec![row("gdn", "gdn_decode_multi_seq")],
            expected: vec![],
        };
        let dropped = [("gdn", "gdn_decode_multi_seq")];
        let text = unresolved_report(&split, &dropped, "m", "a", "q", false);
        assert!(text.is_ascii(), "report must survive a non-TTY pipe");
        assert!(text.contains("SHADOW-DROPPED"));
    }

    #[test]
    fn expected_absent_never_lands_in_the_required_list() {
        let rows = vec![row("mla_absorbed", "mla_batched_gemv"), row("gdn", "x")];
        let split = split_failures(&rows, &[("mla_absorbed", "mla_batched_gemv")]);
        assert_eq!(split.required.len(), 1);
        assert_eq!(split.required[0].module, "gdn");
        assert_eq!(split.expected.len(), 1);
    }
    /// 2026-09-26: Pins the deny-mode closing sentence word for word.
    /// Whitespace is collapsed, because the source wraps the sentence across
    /// lines.
    #[test]
    fn the_remediation_wording_is_the_one_that_was_asked_for() {
        let split = FailureSplit {
            required: vec![row("gdn", "gdn_decode_multi_seq")],
            expected: vec![],
        };
        let deny = unresolved_report(&split, &[], "qwen3.6-27b", "sm_121", "nvfp4", false);
        let flat = deny.split_whitespace().collect::<Vec<_>>().join(" ");
        let required = "If you wish to allow this model to be served, you can pass \
                        --dangerously-allow-unresolved-kernel-lookups. But note that \
                        performance may be seriously degraded. We recommend you open a \
                        GitHub issue and/or open a PR to solve this issue.";
        let required = required.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            flat.contains(&required),
            "the remediation sentence has drifted from the specified wording.\n\
             wanted: {required}\n\
             got:    {flat}"
        );
    }
}
