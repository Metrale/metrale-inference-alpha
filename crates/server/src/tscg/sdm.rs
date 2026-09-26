// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: TSCG operator SDM: shorten tool and parameter descriptions.
//!
//! Owner: server (tool prompt).
//! Invariants: `densify` returns one line with no leading, trailing or repeated
//! whitespace.
//!
//! `densify` deletes the `FILLER` phrases, applies the `REWRITES`, and collapses
//! whitespace. Matching is case-insensitive on substrings, not words, and the
//! surviving text keeps its case. Some rewrites drop words that carry meaning
//! (`must be `, `should be `, `optional `).

/// 2026-09-26: Filler phrases deleted outright (case-insensitive substrings). Each is
/// applied in list order until it no longer matches, so an earlier phrase inside
/// a later one (`used to ` in `is used to `) wins.
const FILLER: &[&str] = &[
    "please note that ",
    "please be aware that ",
    "it should be noted that ",
    "it is important to note that ",
    "as a general rule, ",
    "in order to ",
    "for the purpose of ",
    "with the goal of ",
    "this function ",
    "this tool ",
    "this method ",
    "use this to ",
    "used to ",
    "you can use this ",
    "can be used to ",
    "is used to ",
    "allows you to ",
    "lets you ",
    "helps you ",
    "the following ",
    "a list of ",
    "one or more ",
    "if applicable, ",
    "if necessary, ",
    "if needed, ",
    "as needed, ",
    "where appropriate, ",
    "kindly ",
    "simply ",
    "just ",
    "basically ",
    "essentially ",
    "note: ",
];

/// 2026-09-26: Verbose → compact phrase rewrites, applied after filler removal.
/// Case-insensitive on the key.
const REWRITES: &[(&str, &str)] = &[
    ("corresponds to", "→"),
    ("for example", "e.g."),
    ("for instance", "e.g."),
    ("such as", "e.g."),
    ("and so on", "etc."),
    ("optional ", ""),
    ("a string representing ", ""),
    ("a string containing ", ""),
    ("an integer representing ", ""),
    ("the name of the ", ""),
    ("the path to the ", "path: "),
    ("must be ", ""),
    ("should be ", ""),
];

/// 2026-09-26: Compress a description string to one whitespace-collapsed line.
/// Whitespace-only input gives an empty string.
pub fn densify(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    // 2026-09-26: Newlines and tabs become spaces first, so a multi-line description
    // becomes one line (the TSCG block is line-structured).
    let mut s: String = text
        .chars()
        .map(|c| {
            if c == '\n' || c == '\t' || c == '\r' {
                ' '
            } else {
                c
            }
        })
        .collect();

    // 2026-09-26: Filler removal: find in a lowercased copy, splice the original, so
    // surviving text keeps its case.
    for pat in FILLER {
        loop {
            let lower = s.to_lowercase();
            match lower.find(pat) {
                Some(idx) => {
                    s.replace_range(idx..idx + pat.len(), "");
                }
                None => break,
            }
        }
    }

    for (from, to) in REWRITES {
        loop {
            let lower = s.to_lowercase();
            match lower.find(from) {
                Some(idx) => {
                    s.replace_range(idx..idx + from.len(), to);
                }
                None => break,
            }
        }
    }

    // 2026-09-26: Collapse runs of whitespace.
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_filler() {
        assert_eq!(
            densify("This tool allows you to search the project files"),
            "search the project files"
        );
    }

    #[test]
    fn collapses_multiline() {
        assert_eq!(
            densify("line one\n  line two\n\tline three"),
            "line one line two line three"
        );
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(densify(""), "");
        assert_eq!(densify("   \n  "), "");
    }

    #[test]
    fn preserves_meaningful_casing() {
        // 2026-09-26: Only the filler prefix is removed; "Bash" keeps its case.
        assert_eq!(
            densify("Use this to run Bash commands"),
            "run Bash commands"
        );
    }
}
