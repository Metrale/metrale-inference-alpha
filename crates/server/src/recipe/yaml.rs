// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A small YAML reader for recipe files; the workspace has no YAML dependency.
//!
//! Owner: server (recipe).
//! Invariants:
//! - `parse` returns `Ok` only for a document whose top level is a mapping.
//! - A map line with no `key:` or at an unexpected indent, a list line without
//!   `- `, an empty key, or a key with neither a value nor a deeper block is an
//!   error, never skipped: a skipped key would change what gets served.
//!
//! Supported, as the vendored recipes under `tests/fixtures/recipes` use them:
//! `key: scalar`, nested maps by deeper indent, `key: |` and `key: |-` literal
//! blocks, `- item` sequences, `{}` empty maps, `#` comments and blank lines.
//! No other flow collection is parsed: `key: [a, b]` reads as the scalar
//! `[a, b]`.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Yaml {
    Scalar(String),
    List(Vec<Yaml>),
    Map(BTreeMap<String, Yaml>),
}

impl Yaml {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Yaml::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&BTreeMap<String, Yaml>> {
        match self {
            Yaml::Map(m) => Some(m),
            _ => None,
        }
    }
}

/// 2026-09-26: One significant line: its indent and its content, comments already gone.
struct Line {
    no: usize,
    indent: usize,
    text: String,
}

/// 2026-09-26: Parse a whole document. The top level must be a mapping.
pub fn parse(input: &str) -> Result<Yaml> {
    let lines = significant(input)?;
    let (value, end) = parse_block(&lines, 0, 0)?;
    if end != lines.len() {
        let l = &lines[end];
        bail!("line {}: unexpected indentation at {:?}", l.no, l.text);
    }
    match value {
        Yaml::Map(_) => Ok(value),
        _ => bail!("the document must be a mapping"),
    }
}

/// 2026-09-26: Strip comments and blanks, but never inside a literal block: a `#`
/// there is content.
fn significant(input: &str) -> Result<Vec<Line>> {
    let mut out = Vec::new();
    let mut raw = input.lines().enumerate().peekable();
    while let Some((i, line)) = raw.next() {
        let no = i + 1;
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let text = strip_comment(trimmed);
        if text.is_empty() {
            continue;
        }
        let is_literal = text.ends_with(": |") || text.ends_with(": |-");
        if !is_literal {
            out.push(Line { no, indent, text });
            continue;
        }
        // 2026-09-26: Consume the block verbatim: every following line indented deeper
        // than the key, comments and blanks included.
        let mut body: Vec<String> = Vec::new();
        let mut block_indent = None;
        while let Some((_, peek)) = raw.peek() {
            let peek_indent = peek.len() - peek.trim_start().len();
            if !peek.trim().is_empty() && peek_indent <= indent {
                break;
            }
            let (_, l) = raw.next().expect("peeked");
            if l.trim().is_empty() {
                body.push(String::new());
                continue;
            }
            let base = *block_indent.get_or_insert(peek_indent);
            body.push(l.chars().skip(base).collect());
        }
        while body.last().is_some_and(|l| l.is_empty()) {
            body.pop();
        }
        // 2026-09-26: Re-emit the key with the block's lines joined by `\n` after a
        // NUL marker, so parse_block knows.
        let key = text.trim_end_matches(['|', '-', ' ']).trim_end_matches(':');
        out.push(Line {
            no,
            indent,
            text: format!("{key}:\u{0}{}", body.join("\n")),
        });
    }
    Ok(out)
}

/// 2026-09-26: Remove a trailing `# comment` (a `#` at the start or after a space),
/// respecting double quotes.
fn strip_comment(s: &str) -> String {
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes && (i == 0 || s.as_bytes()[i - 1] == b' ') => {
                return s[..i].trim_end().to_string();
            }
            _ => {}
        }
    }
    s.trim_end().to_string()
}

/// 2026-09-26: Parse the run of lines at `indent`, returning the value and the next index.
fn parse_block(lines: &[Line], start: usize, indent: usize) -> Result<(Yaml, usize)> {
    if lines.get(start).is_some_and(|l| l.text.starts_with("- ")) {
        return parse_list(lines, start, indent);
    }
    let mut map = BTreeMap::new();
    let mut i = start;
    while i < lines.len() {
        let line = &lines[i];
        if line.indent < indent {
            break;
        }
        if line.indent > indent {
            bail!(
                "line {}: unexpected indentation at {:?}",
                line.no,
                line.text
            );
        }
        let Some((key, rest)) = line.text.split_once(':') else {
            bail!(
                "line {}: expected `key: value`, found {:?}",
                line.no,
                line.text
            );
        };
        let key = key.trim().to_string();
        if key.is_empty() {
            bail!("line {}: empty key", line.no);
        }
        // 2026-09-26: A `|` literal block, already joined and NUL-marked by `significant`.
        if let Some(body) = rest.strip_prefix('\u{0}') {
            map.insert(key, Yaml::Scalar(body.to_string()));
            i += 1;
            continue;
        }
        let rest = rest.trim();
        if rest == "{}" {
            map.insert(key, Yaml::Map(BTreeMap::new()));
            i += 1;
            continue;
        }
        if !rest.is_empty() {
            map.insert(key, Yaml::Scalar(unquote(rest)));
            i += 1;
            continue;
        }
        // 2026-09-26: Nested block: whatever is indented deeper.
        let next_indent = lines.get(i + 1).map(|l| l.indent).unwrap_or(0);
        if next_indent <= indent {
            bail!(
                "line {}: {key:?} has no value and no indented block",
                line.no
            );
        }
        let (value, next) = parse_block(lines, i + 1, next_indent)?;
        map.insert(key, value);
        i = next;
    }
    Ok((Yaml::Map(map), i))
}

fn parse_list(lines: &[Line], start: usize, indent: usize) -> Result<(Yaml, usize)> {
    let mut items = Vec::new();
    let mut i = start;
    while i < lines.len() {
        let line = &lines[i];
        if line.indent < indent {
            break;
        }
        let Some(item) = line.text.strip_prefix("- ") else {
            bail!(
                "line {}: expected a `- ` item, found {:?}",
                line.no,
                line.text
            );
        };
        items.push(Yaml::Scalar(unquote(item.trim())));
        i += 1;
    }
    Ok((Yaml::List(items), i))
}

/// 2026-09-26: `"2"` and `'x'` become `2` and `x`. Every scalar is text here, so a
/// quote only delimits.
fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() >= 2
        && (b[0] == b'"' && b[b.len() - 1] == b'"' || b[0] == b'\'' && b[b.len() - 1] == b'\'')
    {
        return s[1..s.len() - 1].to_string();
    }
    s.to_string()
}

#[cfg(test)]
#[path = "yaml_tests.rs"]
mod tests;
