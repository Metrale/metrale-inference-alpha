// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Plugin provenance: who wrote a plugin, where it came from,
//! where to report it. Returned by [`crate::Plugin::metadata`].
//!
//! Every field is `&'static str` or `bool`, so a plugin's identity is fixed
//! at compile time.
//!
//! Owner: bench (plugin API).
//! Invariants:
//! - [`PluginMetadata::third_party`] always yields `official == false`.

/// 2026-09-26: Authorship and support links for one plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluginMetadata {
    /// 2026-09-26: One line: what the plugin is.
    pub description: &'static str,
    /// 2026-09-26: The plugin's version. First-party plugins carry the crate
    /// version.
    pub version: &'static str,
    pub author: &'static str,
    /// 2026-09-26: The author's home on the web. Empty when there is none.
    pub author_url: &'static str,
    pub email: &'static str,
    pub repository: &'static str,
    /// 2026-09-26: Documentation for this plugin specifically.
    pub help_url: &'static str,
    pub bug_report_url: &'static str,
    pub license: &'static str,
    /// 2026-09-26: True for plugins shipped inside Metrale Engine itself.
    ///
    /// It drives the TUI's origin badge, a trust signal. First-party plugins
    /// get it from [`PluginMetadata::metrale`]; anything built with
    /// [`PluginMetadata::third_party`] cannot set it.
    pub official: bool,
}

impl PluginMetadata {
    /// 2026-09-26: A first-party Metrale Engine plugin. Every field except the
    /// description is the same for all of them, so this is the one place they
    /// are written.
    pub const fn metrale(description: &'static str) -> Self {
        Self {
            description,
            version: env!("CARGO_PKG_VERSION"),
            author: "Metrale Engine Cybersecurity",
            author_url: "https://metrale.ai/engine",
            email: "support@metrale.ai",
            repository: "https://github.com/Metrale/metrale-inference-alpha",
            help_url: "https://book.dev.metrale.ai/benchmarks",
            bug_report_url: "https://github.com/Metrale/metrale-inference-alpha/issues/new",
            license: "MIT OR Apache-2.0",
            official: true,
        }
    }

    /// 2026-09-26: A plugin from outside the Metrale Engine tree. `official`
    /// is forced false.
    #[allow(clippy::too_many_arguments)]
    pub const fn third_party(
        description: &'static str,
        version: &'static str,
        author: &'static str,
        author_url: &'static str,
        email: &'static str,
        repository: &'static str,
        help_url: &'static str,
        bug_report_url: &'static str,
        license: &'static str,
    ) -> Self {
        Self {
            description,
            version,
            author,
            author_url,
            email,
            repository,
            help_url,
            bug_report_url,
            license,
            official: false,
        }
    }

    /// 2026-09-26: Label/value pairs for the detail pane, skipping empty
    /// values so a missing field does not render a blank row.
    pub fn rows(&self) -> Vec<(&'static str, &'static str)> {
        [
            ("Author", self.author),
            ("Website", self.author_url),
            ("Contact", self.email),
            ("Repository", self.repository),
            ("Docs", self.help_url),
            ("Report a bug", self.bug_report_url),
            ("License", self.license),
        ]
        .into_iter()
        .filter(|(_, v)| !v.is_empty())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_party_metadata_tracks_the_crate_version() {
        let m = PluginMetadata::metrale("a benchmark");
        assert_eq!(
            m,
            PluginMetadata {
                description: "a benchmark",
                version: env!("CARGO_PKG_VERSION"),
                author: "Metrale Engine Cybersecurity",
                author_url: "https://metrale.ai/engine",
                email: "support@metrale.ai",
                repository: "https://github.com/Metrale/metrale-inference-alpha",
                help_url: "https://book.dev.metrale.ai/benchmarks",
                bug_report_url: "https://github.com/Metrale/metrale-inference-alpha/issues/new",
                license: "MIT OR Apache-2.0",
                official: true,
            }
        );
    }

    #[test]
    fn third_party_cannot_claim_to_be_official() {
        let m = PluginMetadata::third_party(
            "community sweep",
            "0.2.1",
            "Someone",
            "",
            "",
            "https://example.invalid/repo",
            "",
            "",
            "MIT",
        );
        assert!(!m.official);
    }

    #[test]
    fn empty_fields_are_not_rendered_as_blank_rows() {
        let m = PluginMetadata::third_party("d", "1", "A", "", "", "r", "", "", "MIT");
        assert_eq!(
            m.rows(),
            [("Author", "A"), ("Repository", "r"), ("License", "MIT")]
        );
    }
}
