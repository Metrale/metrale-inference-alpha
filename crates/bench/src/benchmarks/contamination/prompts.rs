// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The probe corpus: prompts built so that contamination is
//! detectable.
//!
//! - Each prompt carries a canary and asks the model to echo it, so the canary
//!   is in both the prompt tokens and the generated tokens. A foreign canary
//!   in a reply is leakage on its own evidence
//!   ([`super::transcript::Transcript::carries_foreign_canary`]).
//! - The canaries share no `-`-separated component and neither contains the
//!   other (tested below).
//! - The prompts are identical up to the canary: the shared preamble is a
//!   prefix-cache collision surface, and the text right after it is the
//!   canary sentence, so resuming the wrong continuation at that boundary
//!   emits the foreign canary.
//! - The topics differ (lighthouse, water mill), so a leak that mangles the
//!   canary still diverges from the solo reference (`Class::Diverged`).
//! - The driver sends temperature 0, and the prompts ask for a fixed shape
//!   (code, three sentences, code).
//!
//! Owner: bench (contamination).
//! Invariants: every canary occurs in its own prompt's tail and in no other
//! prompt (tested below).

/// 2026-09-26: A prompt and the canary that must not appear in any other
/// probe's reply.
pub struct Probe {
    pub name: &'static str,
    pub canary: &'static str,
    /// 2026-09-26: The probe-specific instruction; [`Probe::prompt`] prefixes
    /// the shared preamble.
    tail: &'static str,
}

impl Probe {
    /// 2026-09-26: Shared preamble followed by this probe's tail. The preamble
    /// is one const, so the prompts are identical up to the tail.
    pub fn prompt(&self) -> String {
        format!("{PREAMBLE}{}", self.tail)
    }
}

const PREAMBLE: &str = "You are a precise assistant taking part in a determinism audit. \
                        Follow the instructions exactly and do not add anything else. ";

pub const PROBES: [Probe; 2] = [
    Probe {
        name: "A",
        canary: "XK-AZURE-HERON-41",
        tail: "Your reference code is XK-AZURE-HERON-41. Begin your reply with the \
               reference code on its own line, then explain in exactly three short \
               sentences how a lighthouse warns ships at night, then end with the \
               reference code again on its own line.",
    },
    Probe {
        name: "B",
        canary: "QJ-CRIMSON-OTTER-77",
        tail: "Your reference code is QJ-CRIMSON-OTTER-77. Begin your reply with the \
               reference code on its own line, then explain in exactly three short \
               sentences how a water mill grinds grain into flour, then end with the \
               reference code again on its own line.",
    },
];

/// 2026-09-26: The canaries in probe order: [`super::score::score`] reads the
/// canary at a probe's index as that probe's own.
pub fn canaries() -> Vec<String> {
    PROBES.iter().map(|p| p.canary.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::contamination::transcript::Transcript;

    #[test]
    fn every_canary_is_unique_to_its_probe() {
        for (i, a) in PROBES.iter().enumerate() {
            assert!(
                a.prompt().contains(a.canary),
                "probe {} does not carry its own canary",
                a.name
            );
            for (j, b) in PROBES.iter().enumerate() {
                if i == j {
                    continue;
                }
                assert!(
                    !b.prompt().contains(a.canary),
                    "probe {}'s prompt contains probe {}'s canary — a clean run \
                     would read as contaminated",
                    b.name,
                    a.name
                );
                assert!(
                    !a.canary.contains(b.canary) && !b.canary.contains(a.canary),
                    "canaries {} / {} overlap as substrings",
                    a.canary,
                    b.canary
                );
                let a_parts: std::collections::BTreeSet<_> = a.canary.split('-').collect();
                let b_parts: std::collections::BTreeSet<_> = b.canary.split('-').collect();
                assert!(
                    a_parts.is_disjoint(&b_parts),
                    "canaries {} / {} share lexical components: {:?}",
                    a.canary,
                    b.canary,
                    a_parts.intersection(&b_parts).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn prompts_share_the_preamble_and_diverge_at_the_canary() {
        let full: Vec<String> = PROBES.iter().map(Probe::prompt).collect();
        let canary_offsets: Vec<usize> = PROBES
            .iter()
            .zip(&full)
            .map(|(probe, prompt)| prompt.find(probe.canary).expect("canary in own prompt"))
            .collect();
        assert_eq!(canary_offsets[0], canary_offsets[1]);
        let boundary = canary_offsets[0];
        assert_eq!(&full[0][..boundary], &full[1][..boundary]);
        assert_eq!(
            &full[0][..boundary],
            format!("{PREAMBLE}Your reference code is ")
        );
        for p in &PROBES {
            assert!(
                p.prompt().starts_with(PREAMBLE),
                "probe {} lost the shared preamble",
                p.name
            );
            assert!(
                !PREAMBLE.contains(p.canary) && p.tail.contains(p.canary),
                "probe {}'s canary must live in the divergent tail, not the \
                 shared span",
                p.name
            );
        }
    }

    #[test]
    fn names_are_distinct_and_canaries_ride_in_probe_order() {
        let mut names = std::collections::BTreeSet::new();
        for p in &PROBES {
            assert!(names.insert(p.name), "duplicate probe name {}", p.name);
        }
        assert_eq!(
            canaries(),
            PROBES
                .iter()
                .map(|p| p.canary.to_string())
                .collect::<Vec<_>>(),
            "Legs indexes canaries by prompt position; order is load-bearing"
        );
    }

    #[test]
    fn the_core_detector_fires_on_these_canaries() {
        let cans = canaries();
        let all: Vec<&str> = cans.iter().map(String::as_str).collect();
        let clean = Transcript {
            text: format!(
                "{}\nA lighthouse shines a rotating beam. Ships see it from afar. \
                 They steer clear of the rocks.\n{}",
                PROBES[0].canary, PROBES[0].canary
            ),
            ..Default::default()
        };
        assert_eq!(
            clean.carries_foreign_canary(PROBES[0].canary, &all),
            None,
            "a probe's own canary must not read as leakage"
        );
        let leaked = Transcript {
            text: format!("{}\nA lighthouse shines...", PROBES[1].canary),
            ..Default::default()
        };
        assert_eq!(
            leaked.carries_foreign_canary(PROBES[0].canary, &all),
            Some(PROBES[1].canary),
            "B's canary in A's reply must be attributed to B"
        );
    }
}
