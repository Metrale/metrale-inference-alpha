// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Section navigation: the sidebar order, `1`-`7` jumps, `⇥` cycling, and the two click entry points.
//!
//! Owner: server tui.
//! Invariants:
//! - `nav_rows` is `Section::ALL` in order, each section expanded into its
//!   `Section::subs` rows: the order `render::draw_sidebar` draws (it expands
//!   only the active section) and `events::sidebar_row` hit-tests.

use super::app::{App, BenchSub, Focus, MainSub, TermSub};
use super::help_state::HelpSub;
use super::section::Section;

impl App {
    /// 2026-09-26: Which subsection of `s` is active, as an index into [`Section::subs`].
    pub fn sub_index(&self, s: Section) -> usize {
        match s {
            Section::Main => (self.main_sub == MainSub::Kernels) as usize,
            Section::Benchmarks => (self.bench_sub == BenchSub::History) as usize,
            Section::Terminal => (self.term_sub == TermSub::Chat) as usize,
            Section::Help => (self.help.sub == HelpSub::Report) as usize,
            _ => 0,
        }
    }

    fn set_sub(&mut self, s: Section, i: usize) {
        match s {
            Section::Main => {
                self.main_sub = if i == 0 {
                    MainSub::Overview
                } else {
                    MainSub::Kernels
                }
            }
            Section::Benchmarks => {
                self.bench_sub = if i == 0 {
                    BenchSub::Suite
                } else {
                    BenchSub::History
                }
            }
            Section::Terminal => self.term_sub = if i == 0 { TermSub::Ops } else { TermSub::Chat },
            Section::Help => {
                self.help.sub = if i == 0 {
                    HelpSub::Guide
                } else {
                    HelpSub::Report
                }
            }
            _ => {}
        }
    }

    /// 2026-09-26: Every navigable sidebar row, flattened in the order the sidebar draws them:
    /// one entry per subsection, or a single entry for a section that has none.
    pub(super) fn nav_rows() -> Vec<(Section, usize)> {
        Section::ALL
            .iter()
            .flat_map(|s| (0..s.subs().len().max(1)).map(move |i| (*s, i)))
            .collect()
    }

    pub(super) fn jump(&mut self, s: Section) {
        if self.section != s {
            self.repaint = true;
        }
        if self.section == s {
            // 2026-09-26: Repeat-press cycles this section's subsections.
            let n = s.subs().len();
            if n > 1 {
                self.set_sub(s, (self.sub_index(s) + 1) % n);
            }
        }
        self.section = s;
        self.focus = Focus::Content;
    }

    /// 2026-09-26: `⇥` / `⇧⇥` walk every row of [`Self::nav_rows`], subsection rows included.
    pub(super) fn cycle_section(&mut self, dir: i32) {
        let rows = Self::nav_rows();
        let cur = rows
            .iter()
            .position(|(s, i)| *s == self.section && *i == self.sub_index(*s))
            .unwrap_or(0) as i32;
        let (s, i) = rows[((cur + dir).rem_euclid(rows.len() as i32)) as usize];
        self.section = s;
        self.set_sub(s, i);
        self.focus = Focus::Content;
    }

    pub fn sidebar_click(&mut self, row_in_sidebar: usize) {
        // 2026-09-26: `row_in_sidebar` is an index into `Section::ALL`; `events::sidebar_row` has already removed the subsection rows.
        if let Some(s) = Section::ALL.get(row_in_sidebar) {
            self.jump(*s);
        }
    }

    /// 2026-09-26: Click on one of the active section's subsection rows, the only ones the sidebar draws.
    /// Selects that row outright; a repeat press of the section's key cycles instead.
    pub fn sidebar_sub_click(&mut self, sub: usize) {
        if sub < self.section.subs().len() {
            self.set_sub(self.section, sub);
            self.focus = Focus::Content;
        }
    }
}
