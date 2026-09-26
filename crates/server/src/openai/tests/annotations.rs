// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for URL-annotation extraction (`extract_url_annotations`).
//!
//! Owner: server (OpenAI API layer) tests.
//! Invariants: none beyond the types.

use crate::openai::*;

fn url_of(a: &Annotation) -> (usize, usize, &str, &str) {
    match a {
        Annotation::UrlCitation {
            url_citation:
                UrlCitation {
                    start_index,
                    end_index,
                    url,
                    title,
                },
        } => (*start_index, *end_index, url.as_str(), title.as_str()),
    }
}

#[test]
fn bare_url_extracted() {
    let got = extract_url_annotations("see https://example.com/foo for more").unwrap();
    assert_eq!(got.len(), 1);
    let (s, e, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/foo");
    assert_eq!(t, "https://example.com/foo");
    assert_eq!(s, 4);
    assert_eq!(e, 4 + "https://example.com/foo".len());
}

#[test]
fn trailing_sentence_punct_stripped() {
    let got = extract_url_annotations("go to https://example.com.").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com");
}

#[test]
fn wikipedia_parens_preserved() {
    let got = extract_url_annotations("see https://en.wikipedia.org/wiki/Foo_(bar) now").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
}

#[test]
fn markdown_link_uses_title() {
    let got = extract_url_annotations("read [the docs](https://example.com/api) today").unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/api");
    assert_eq!(t, "the docs");
}

#[test]
fn markdown_link_with_parens_in_url_preserved() {
    // 2026-09-26: The link target itself contains `(...)`; the nesting-aware close-paren
    // scan must return the whole URL.
    let got =
        extract_url_annotations("see [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) here")
            .unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
    assert_eq!(t, "Foo (bar)");
}

#[test]
fn url_in_fenced_code_skipped() {
    let input = "run this:\n```bash\ncurl https://example.com\n```\ndone";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn url_in_inline_code_skipped() {
    let input = "use `curl https://example.com` to fetch";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn multiple_urls_sorted_by_position() {
    let input = "first https://a.example.com and [second](https://b.example.com)";
    let got = extract_url_annotations(input).unwrap();
    assert_eq!(got.len(), 2);
    let (s0, _, u0, _) = url_of(&got[0]);
    let (s1, _, u1, _) = url_of(&got[1]);
    assert!(s0 < s1);
    assert_eq!(u0, "https://a.example.com");
    assert_eq!(u1, "https://b.example.com");
}

#[test]
fn non_http_ignored() {
    assert!(extract_url_annotations("ftp://example.com not a citation").is_none());
}

#[test]
fn empty_input_returns_none() {
    assert!(extract_url_annotations("").is_none());
    assert!(extract_url_annotations("no URLs here").is_none());
}

#[test]
fn query_and_fragment_preserved() {
    let got = extract_url_annotations("see https://example.com/p?q=1&r=2#frag here").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/p?q=1&r=2#frag");
}
