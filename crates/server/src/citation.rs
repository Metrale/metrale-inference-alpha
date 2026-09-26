// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: URL citations found in assistant text, as the neutral
//! [`Citation`]; `openai::Annotation` is built from it.
//!
//! Owner: server API.
//! Invariants: `extract`, `extract_url_citations`, `merge_dedupe` and
//! `merged_citations` return citations sorted by `start_index`.
//!
//! * [`extract_url_citations`] finds markdown-link and bare URLs, skipping
//!   fenced and inline code so `curl https://…` is not a citation.
//! * [`extract`] finds three structured patterns, parsed in
//!   [`crate::citation_structured`]:
//!
//!   1. Markdown footnotes
//!      ```text
//!      See the source[^1] for details.
//!      ...
//!      [^1]: https://example.com/source The title text
//!      ```
//!      A citation at the `[^1]` reference site, with the URL of the
//!      definition and its title text (else the label) as `title`.
//!
//!   2. Numeric bracket refs
//!      ```text
//!      See [1] for details.
//!      ...
//!      [1] https://example.com/source
//!      ```
//!      Same shape, without the `^`; the definition may also be `[1]: url`.
//!
//!   3. `Sources:`, `References:` or `Citations:` sections
//!      ```text
//!      Sources:
//!      - https://a.example.com
//!      - https://b.example.com
//!      ```
//!      One citation per line at its URL span.
//!
//! A definition or section line counts only when it starts with an http(s)
//! URL. [`merged_citations`] runs both extractors; [`merge_dedupe`] decides
//! which repeated URLs stay.

use crate::citation_structured::{
    footnote_citations, numeric_ref_citations, sources_block_citations,
};

/// 2026-09-26: A URL citation with its byte range `[start_index, end_index)`
/// in the text it was found in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    pub start_index: usize,
    pub end_index: usize,
    pub url: String,
    pub title: String,
}

/// 2026-09-26: [`extract_url_citations`] and [`extract`] combined by
/// [`merge_dedupe`], with the URL-scan results as `primary`. `None` when
/// nothing matched, so wire fields can be skipped.
pub fn merged_citations(content: &str) -> Option<Vec<Citation>> {
    let bare = extract_url_citations(content);
    let structured = extract(content);
    let merged = merge_dedupe(bare, structured);
    if merged.is_empty() {
        None
    } else {
        Some(merged)
    }
}

/// 2026-09-26: The structured citations in `content`: one per footnote or
/// numeric reference site and one per sources-section line, in document
/// order. Empty when nothing matched.
pub fn extract(content: &str) -> Vec<Citation> {
    let mut out = Vec::new();
    out.extend(footnote_citations(content));
    out.extend(numeric_ref_citations(content));
    out.extend(sources_block_citations(content));
    out.sort_by_key(|c| c.start_index);
    out
}

/// 2026-09-26: One citation per http(s) URL in `content`, in document order:
/// - a markdown link `[title](url)` gives the URL with the `[...]` text as
///   title;
/// - a bare URL outside any link gives the URL as its own title;
/// - URLs inside fenced code (triple backticks) or inline code
///   (`` `...` ``) are skipped.
pub fn extract_url_citations(content: &str) -> Vec<Citation> {
    let mut out: Vec<Citation> = Vec::new();
    let masked = mask_code_spans(content);
    // 2026-09-26: Markdown links first. `masked` has the byte offsets of
    // `content`, so an index found in one slices the other.
    let mut scan = 0usize;
    while scan < masked.len() {
        let rest = &masked[scan..];
        let Some(lb_rel) = rest.find('[') else { break };
        let lb = scan + lb_rel;
        let after_lb = &masked[lb + 1..];
        let Some(rb_rel) = after_lb.find(']') else {
            scan = lb + 1;
            continue;
        };
        let rb = lb + 1 + rb_rel;
        if masked.as_bytes().get(rb + 1) != Some(&b'(') {
            scan = rb + 1;
            continue;
        }
        // 2026-09-26: The balancing `)`, not the first one, so a URL with
        // `(...)` in it (`https://en.wikipedia.org/wiki/Foo_(bar)`) is kept
        // whole.
        let after_paren = &masked[rb + 2..];
        let Some(rparen_rel) = balanced_paren_close(after_paren) else {
            scan = rb + 2;
            continue;
        };
        let rparen = rb + 2 + rparen_rel;
        let title = &content[lb + 1..rb];
        let target = content[rb + 2..rparen].trim();
        if (target.starts_with("http://") || target.starts_with("https://"))
            && target.len() > "https://".len()
        {
            out.push(Citation {
                start_index: rb + 2,
                end_index: rparen,
                url: target.to_string(),
                title: title.to_string(),
            });
        }
        scan = rparen + 1;
    }

    // 2026-09-26: Then bare URLs outside code and outside the markdown links
    // already found.
    let covered: Vec<(usize, usize)> = out.iter().map(|c| (c.start_index, c.end_index)).collect();
    let mut i = 0usize;
    while i < masked.len() {
        let rest = &masked[i..];
        let Some(off) = rest.find("http") else { break };
        let abs_start = i + off;
        let tail_masked = &masked[abs_start..];
        let is_url = tail_masked.starts_with("http://") || tail_masked.starts_with("https://");
        if !is_url {
            i = abs_start + 4;
            continue;
        }
        let tail = &content[abs_start..];
        let end_rel = tail
            .find(|c: char| {
                c.is_whitespace() || matches!(c, ']' | '}' | '"' | '<' | '>' | '`' | '\\')
            })
            .unwrap_or(tail.len());
        let mut raw = &tail[..end_rel];
        // 2026-09-26: Strip trailing sentence punctuation, emphasis markers
        // (`*`, `_`) and a `)` with no matching `(`, so
        // `https://en.wikipedia.org/wiki/Foo_(bar)` keeps its `)`.
        while let Some(last) = raw.chars().last() {
            let strip = match last {
                '.' | ',' | ';' | ':' | '!' | '?' | '*' | '_' => true,
                ')' => {
                    let opens = raw.matches('(').count();
                    let closes = raw.matches(')').count();
                    closes > opens
                }
                _ => false,
            };
            if !strip {
                break;
            }
            raw = &raw[..raw.len() - last.len_utf8()];
        }
        if raw.len() > "https://".len() {
            let start = abs_start;
            let end = abs_start + raw.len();
            let overlaps = covered.iter().any(|(s, e)| start < *e && end > *s);
            if !overlaps {
                out.push(Citation {
                    start_index: start,
                    end_index: end,
                    url: raw.to_string(),
                    title: raw.to_string(),
                });
            }
        }
        i = abs_start + end_rel.max(1);
    }
    out.sort_by_key(|c| c.start_index);
    out
}

/// 2026-09-26: Every `primary` entry, plus each `secondary` entry whose URL
/// is not already in `primary` or in an earlier `secondary` entry, sorted by
/// `start_index`. Repeated URLs within `primary` are all kept.
pub fn merge_dedupe(mut primary: Vec<Citation>, secondary: Vec<Citation>) -> Vec<Citation> {
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    for c in &primary {
        seen.insert(c.url.clone());
    }
    for c in secondary {
        if seen.insert(c.url.clone()) {
            primary.push(c);
        }
    }
    primary.sort_by_key(|c| c.start_index);
    primary
}

/// 2026-09-26: The byte offset of the `)` that balances an implied `(` just
/// before `s`, counting nested `()` pairs; `None` when there is none.
fn balanced_paren_close(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// 2026-09-26: A copy of `content` with every fenced code block (```) and
/// inline code span (`), delimiters included, replaced by ASCII spaces. An
/// unclosed fence or backtick masks to the end. Each masked character becomes
/// as many spaces as its UTF-8 length and newlines are kept, so byte offsets
/// and UTF-8 validity are preserved.
fn mask_code_spans(content: &str) -> String {
    let bytes = content.as_bytes();
    let mut out: Vec<u8> = bytes.to_vec();

    fn blank(out: &mut [u8], content: &str, start: usize, end: usize) {
        let region = &content[start..end];
        let mut cursor = start;
        for ch in region.chars() {
            let len = ch.len_utf8();
            if ch != '\n' {
                for b in out.iter_mut().skip(cursor).take(len) {
                    *b = b' ';
                }
            }
            cursor += len;
        }
    }

    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"```") {
            let after = i + 3;
            let rest = &content[after..];
            let end = match rest.find("```") {
                Some(r) => after + r + 3,
                None => bytes.len(),
            };
            blank(&mut out, content, i, end);
            i = end;
            continue;
        }
        if bytes[i] == b'`' {
            let after = i + 1;
            let rest = &content[after..];
            let end = match rest.find('`') {
                Some(r) => after + r + 1,
                None => bytes.len(),
            };
            blank(&mut out, content, i, end);
            i = end;
            continue;
        }
        let step = match bytes[i] {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf7 => 4,
            _ => 1,
        };
        i += step;
    }
    String::from_utf8(out).expect("mask preserves UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_citations_returns_empty() {
        let got = extract("plain text with no citations");
        assert!(got.is_empty());
    }

    #[test]
    fn merge_dedupe_by_url() {
        let a = vec![Citation {
            start_index: 0,
            end_index: 20,
            url: "https://example.com".into(),
            title: "first".into(),
        }];
        let b = vec![Citation {
            start_index: 50,
            end_index: 70,
            url: "https://example.com".into(),
            title: "second".into(),
        }];
        let merged = merge_dedupe(a, b);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn bare_url_extracted() {
        let got = extract_url_citations("see https://example.com/foo for more");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://example.com/foo");
        assert_eq!(got[0].title, "https://example.com/foo");
    }

    #[test]
    fn trailing_punctuation_stripped() {
        let got = extract_url_citations("go to https://example.com.");
        assert_eq!(got[0].url, "https://example.com");
    }

    #[test]
    fn wikipedia_parens_survive() {
        let got = extract_url_citations("see https://en.wikipedia.org/wiki/Foo_(bar) now");
        assert_eq!(got[0].url, "https://en.wikipedia.org/wiki/Foo_(bar)");
    }

    #[test]
    fn markdown_link_title_used() {
        let got = extract_url_citations("read [the docs](https://example.com/api) today");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://example.com/api");
        assert_eq!(got[0].title, "the docs");
    }

    #[test]
    fn markdown_link_with_parens_in_url() {
        let got =
            extract_url_citations("see [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) here");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://en.wikipedia.org/wiki/Foo_(bar)");
    }

    #[test]
    fn code_spans_skipped() {
        assert!(extract_url_citations("run `curl https://example.com` locally").is_empty());
        assert!(extract_url_citations("```\ncurl https://example.com\n```\nno cite").is_empty());
    }

    #[test]
    fn non_http_schemes_ignored() {
        assert!(extract_url_citations("ftp://example.com not a citation").is_empty());
    }

    #[test]
    fn empty_and_plain_content() {
        assert!(extract_url_citations("").is_empty());
        assert!(extract_url_citations("no URLs here").is_empty());
    }

    #[test]
    fn query_and_fragment_kept() {
        let got = extract_url_citations("see https://example.com/p?q=1&r=2#frag here");
        assert_eq!(got[0].url, "https://example.com/p?q=1&r=2#frag");
    }

    #[test]
    fn merged_citations_combines_and_dedupes() {
        let input = "Intro https://a.example.com text.\n\nSources:\n- https://a.example.com\n- https://b.example.com\n";
        let got = merged_citations(input).unwrap();
        let mut u: Vec<&str> = got.iter().map(|c| c.url.as_str()).collect();
        u.dedup();
        assert_eq!(u.len(), 2);
    }
}
