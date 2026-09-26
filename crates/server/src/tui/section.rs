// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`Section`], the sidebar's navigation model. The sidebar draws
//! its rows from `Section::ALL` and `subs`, `⇥` steps through the same rows
//! (`App::nav_rows`), and a click maps a sidebar row back to a section
//! (`App::sidebar_click`).
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Section {
    Main,
    Stats,
    Network,
    Library,
    Benchmarks,
    Terminal,
    Help,
}

impl Section {
    pub const ALL: [Section; 7] = [
        Section::Main,
        Section::Stats,
        Section::Network,
        Section::Library,
        Section::Benchmarks,
        Section::Terminal,
        Section::Help,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Section::Main => "Main",
            Section::Stats => "Stats",
            Section::Network => "Network",
            Section::Library => "Library",
            Section::Benchmarks => "Benchmarks",
            Section::Terminal => "Terminal",
            Section::Help => "Help",
        }
    }
    pub fn icon(self) -> &'static str {
        match self {
            Section::Main => "◆",
            Section::Stats => "∿",
            Section::Network => "⬡",
            Section::Library => "▤",
            Section::Benchmarks => "▰",
            Section::Terminal => "❯",
            Section::Help => "✚",
        }
    }
    /// 2026-09-26: Subsection labels, in sidebar order. The sidebar draws
    /// them, a repeated section key cycles through them (`App::jump`), and `⇥`
    /// stops on each (`App::nav_rows`).
    pub fn subs(self) -> &'static [&'static str] {
        match self {
            Section::Main => &["Overview", "Kernels"],
            Section::Benchmarks => &["Suite", "History"],
            Section::Terminal => &["Ops", "Chat"],
            Section::Help => &["Guide", "Report Issue"],
            Section::Stats | Section::Network | Section::Library => &[],
        }
    }
}

#[cfg(test)]
#[path = "section_tests.rs"]
mod tests;
