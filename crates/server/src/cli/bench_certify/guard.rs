// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The drift guard: has a `PERF_PATHS` file changed on the guarded branch since the anchor?
//!
//! Owner: server CLI (`met benchmark certify`).
//! A branch that moved without touching `PERF_PATHS` is not drift. A guard
//! that cannot answer is an error, never `Unmoved`; a `Poll` tolerates
//! `BLIND_TICKS_ALLOWED` such errors in a row and stops on the next one.
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_bench::gate::PERF_PATHS;

/// 2026-09-26: The two git questions the guard asks, behind a trait so the decision is
/// testable without a repository.
pub trait Git {
    /// 2026-09-26: Fetch `remote_ref` (`REMOTE/BRANCH`) and return its head sha.
    fn fetch_head(&self, remote_ref: &str) -> Result<String>;
    /// 2026-09-26: The `PERF_PATHS` files that differ between two commits.
    fn changed_perf_paths(&self, from: &str, to: &str) -> Result<Vec<String>>;
}

/// 2026-09-26: What the guard found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Drift {
    /// 2026-09-26: The branch head and the anchor are the same commit (one is a
    /// prefix of the other).
    Unmoved,
    /// 2026-09-26: The branch moved, but no `PERF_PATHS` file changed.
    MovedHarmlessly { head: String },
    /// 2026-09-26: These `PERF_PATHS` files changed between the anchor and `head`.
    PerfPathMoved { head: String, paths: Vec<String> },
}

/// 2026-09-26: How many consecutive checks may fail to answer before a `Poll` stops:
/// the next failure after this many in a row is `Judgement::Stop`. The guard
/// runs every [`super::GUARD_EVERY`] (60 s) while a unit is in flight.
pub const BLIND_TICKS_ALLOWED: u32 = 5;

/// 2026-09-26: What a poller does with one guard result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Judgement {
    /// 2026-09-26: `Unmoved` or `MovedHarmlessly`; carry on.
    Fine,
    /// 2026-09-26: The guard could not answer, `n` times in a row so far; carry on and
    /// say so.
    Blind(u32),
    /// 2026-09-26: Hand this to the campaign (`Campaign::guard`), which aborts.
    Stop(Result<Drift, String>),
}

/// 2026-09-26: The count of consecutive unanswered checks, owned by whichever loop
/// polls the guard. Any answer resets it.
#[derive(Debug, Default)]
pub struct Poll {
    blind: u32,
}

impl Poll {
    pub fn judge(&mut self, result: Result<Drift, String>) -> Judgement {
        match result {
            Ok(Drift::Unmoved) | Ok(Drift::MovedHarmlessly { .. }) => {
                self.blind = 0;
                Judgement::Fine
            }
            Ok(moved @ Drift::PerfPathMoved { .. }) => {
                self.blind = 0;
                Judgement::Stop(Ok(moved))
            }
            Err(e) => {
                self.blind += 1;
                if self.blind <= BLIND_TICKS_ALLOWED {
                    Judgement::Blind(self.blind)
                } else {
                    Judgement::Stop(Err(format!("{e} — {} checks in a row", self.blind)))
                }
            }
        }
    }
}

/// 2026-09-26: Ask the two questions. `Err` means the guard could not answer.
pub fn drift(git: &dyn Git, anchor: &str, remote_ref: &str) -> Result<Drift> {
    let head = git.fetch_head(remote_ref)?;
    if head.starts_with(anchor) || anchor.starts_with(&head) {
        return Ok(Drift::Unmoved);
    }
    let paths = git.changed_perf_paths(anchor, &head)?;
    if paths.is_empty() {
        Ok(Drift::MovedHarmlessly { head })
    } else {
        Ok(Drift::PerfPathMoved { head, paths })
    }
}

/// 2026-09-26: The real thing: `git` in a checkout.
pub struct GitCli {
    pub root: PathBuf,
}

impl GitCli {
    fn run(&self, args: &[&str]) -> Result<String> {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .with_context(|| format!("running git {args:?}"))?;
        if !out.status.success() {
            bail!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

impl Git for GitCli {
    fn fetch_head(&self, remote_ref: &str) -> Result<String> {
        let (remote, branch) = remote_ref
            .split_once('/')
            .with_context(|| format!("guard ref {remote_ref:?} is not REMOTE/BRANCH"))?;
        self.run(&["fetch", "-q", remote, branch])?;
        self.run(&["rev-parse", "FETCH_HEAD"])
    }

    fn changed_perf_paths(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let mut args = vec!["diff", "--name-only", from, to, "--"];
        args.extend(PERF_PATHS);
        Ok(self
            .run(&args)?
            .lines()
            .map(str::to_owned)
            .filter(|l| !l.is_empty())
            .collect())
    }
}

/// 2026-09-26: `REMOTE/BRANCH` for HEAD's upstream, if it has one.
pub fn upstream_of_head(root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (s.contains('/') && !s.is_empty()).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        head: Result<String, String>,
        changed: Result<Vec<String>, String>,
    }

    impl Git for Fake {
        fn fetch_head(&self, _: &str) -> Result<String> {
            self.head.clone().map_err(|e| anyhow::anyhow!(e))
        }
        fn changed_perf_paths(&self, _: &str, _: &str) -> Result<Vec<String>> {
            self.changed.clone().map_err(|e| anyhow::anyhow!(e))
        }
    }

    #[test]
    fn an_unmoved_branch_is_unmoved_even_with_a_short_anchor() {
        let g = Fake {
            head: Ok("1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18".into()),
            changed: Ok(vec!["crates/x.rs".into()]),
        };
        assert_eq!(
            drift(&g, "1a0dc88a8c", "metrale/main").unwrap(),
            Drift::Unmoved
        );
    }

    #[test]
    fn a_docs_only_move_is_harmless() {
        let g = Fake {
            head: Ok("bbbb".into()),
            changed: Ok(vec![]),
        };
        assert_eq!(
            drift(&g, "aaaa", "metrale/main").unwrap(),
            Drift::MovedHarmlessly {
                head: "bbbb".into()
            }
        );
    }

    #[test]
    fn a_perf_path_move_names_the_paths() {
        let g = Fake {
            head: Ok("bbbb".into()),
            changed: Ok(vec!["crates/model-layers/src/lib.rs".into()]),
        };
        assert_eq!(
            drift(&g, "aaaa", "metrale/main").unwrap(),
            Drift::PerfPathMoved {
                head: "bbbb".into(),
                paths: vec!["crates/model-layers/src/lib.rs".into()]
            }
        );
    }

    /// 2026-09-26: A poller retries a mute guard for the blind budget and stops after
    /// it; any answer resets the count. Negative control: a perf-path move stops
    /// at once, and the (budget + 1)-th miss stops too.
    #[test]
    fn a_poller_tolerates_a_mute_guard_for_the_budget_and_no_longer() {
        let mut p = Poll::default();
        for i in 1..=BLIND_TICKS_ALLOWED {
            assert_eq!(p.judge(Err("dns".into())), Judgement::Blind(i));
        }
        assert_eq!(p.judge(Ok(Drift::Unmoved)), Judgement::Fine);
        assert_eq!(p.judge(Err("dns".into())), Judgement::Blind(1));
        for _ in 1..BLIND_TICKS_ALLOWED {
            assert!(matches!(p.judge(Err("dns".into())), Judgement::Blind(_)));
        }
        match p.judge(Err("dns".into())) {
            Judgement::Stop(Err(why)) => assert!(why.contains("6 checks in a row"), "{why}"),
            other => panic!("{other:?}"),
        }
        let moved = Drift::PerfPathMoved {
            head: "b".into(),
            paths: vec!["crates/x.rs".into()],
        };
        let mut fresh = Poll::default();
        assert_eq!(fresh.judge(Ok(moved.clone())), Judgement::Stop(Ok(moved)));
    }

    /// 2026-09-26: Negative control: "could not answer" is an error, never `Unmoved`.
    #[test]
    fn a_fetch_failure_is_an_error_not_a_pass() {
        let g = Fake {
            head: Err("could not fetch".into()),
            changed: Ok(vec![]),
        };
        assert!(drift(&g, "aaaa", "metrale/main").is_err());
        let g = Fake {
            head: Ok("bbbb".into()),
            changed: Err("diff failed".into()),
        };
        assert!(drift(&g, "aaaa", "metrale/main").is_err());
    }
}
