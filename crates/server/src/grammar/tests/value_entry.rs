// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the qwen3_coder parameter-value grammar at value
//! entry: the content tokens `Reykjavik` and `3 days` need no contextual
//! trials there, and factoring out `nonempty_value` changes no mask,
//! verdict or rollback.
//!
//! Owner: server (grammar).
//! Invariants: none beyond the types.

use super::super::compile_tools::xml_param_value_body_ebnf_opts;
use metrale_grammar::compiler::{CompiledGrammar, GrammarCompiler};
use metrale_grammar::matcher::GrammarMatcher;
use metrale_grammar::tokenizer::{TokenizerInfo, VocabType};

fn vocab() -> Vec<String> {
    let mut vocab = super::test_vocab();
    vocab.extend(
        [
            "Reykjavik",
            "3 days",
            "<script>",
            ">=",
            ">a",
            "</parameter>",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    vocab
}

fn compile(ebnf: &str) -> CompiledGrammar {
    let info = TokenizerInfo::new(&vocab(), VocabType::Raw, None, Some(vec![130]), false);
    GrammarCompiler::new(info, 1, false, -1)
        .compile_grammar_from_ebnf(ebnf, "root")
        .unwrap()
}

fn fill(matcher: &mut GrammarMatcher) -> Vec<i32> {
    let mut mask = vec![0; vocab().len().div_ceil(32)];
    matcher
        .fill_next_token_bitmask(&mut mask, 0, false)
        .unwrap();
    mask
}

#[test]
fn value_entry_content_tokens_need_no_contextual_trials() {
    for allow_empty in [false, true] {
        for force_close in [false, true] {
            let ebnf = xml_param_value_body_ebnf_opts(
                "</parameter>",
                Some(&["city".to_owned()]),
                allow_empty,
                force_close,
            );
            let compiled = compile(&ebnf);
            let mut matcher = GrammarMatcher::new(compiled.clone(), None, false, -1);
            assert!(matcher.accept_string("<parameter=city>\n", false));
            let mask = fill(&mut matcher);
            for content in ["Reykjavik", "3 days"] {
                let sorted = compiled.tokenizer_info().sorted_decoded_vocab();
                let index = sorted
                    .iter()
                    .position(|(_, s)| s == content.as_bytes())
                    .unwrap();
                let id = sorted[index].0 as usize;
                assert_ne!(mask[id / 32] & (1 << (id % 32)), 0);
                let trials = compiled
                    .inner()
                    .mask_cache
                    .lock()
                    .unwrap()
                    .values()
                    .filter(|m| m.uncertain_indices.contains(&(index as i32)))
                    .count();
                assert_eq!(
                    trials, 0,
                    "{content}: empty={allow_empty}, close={force_close}"
                );
            }
        }
    }
}

#[test]
fn value_entry_factoring_preserves_masks_rejections_and_rollback() {
    for allow_empty in [false, true] {
        for force_close in [false, true] {
            let ebnf = xml_param_value_body_ebnf_opts(
                "</parameter>",
                Some(&["city".to_owned()]),
                allow_empty,
                force_close,
            );
            // 2026-09-26: Reference grammar: the same value rule with
            // `nonempty_value` written inline.
            let reference = ebnf
                .replace(
                    "value ::= leading_ws nonempty_value?",
                    "value ::= leading_ws (first_content rest)?",
                )
                .replace(
                    "value ::= leading_ws nonempty_value",
                    "value ::= leading_ws first_content rest",
                )
                .replace("nonempty_value ::= first_content rest\n", "");
            let actual = compile(&ebnf);
            let old = compile(&reference);
            for (input, expected) in [
                ("<parameter=city>\nReykjavik\n</parameter>", true),
                ("<parameter=city><script></parameter>", true),
                ("<parameter=city>3 days</parameter>", true),
                ("<parameter=city></parameter>", allow_empty),
                ("<parameter=city> \n</parameter>", allow_empty),
                ("<parameter=wrong>Reykjavik</parameter>", false),
                ("<parameter=city>=wrong</parameter>", false),
                ("<parameter=city>>wrong</parameter>", false),
                ("<parameter=city></parameterX</parameter>", !force_close),
            ] {
                let mut a = GrammarMatcher::new(actual.clone(), None, true, -1);
                let mut b = GrammarMatcher::new(old.clone(), None, true, -1);
                let mut accepted = true;
                for byte in input.bytes() {
                    let before = fill(&mut a);
                    assert_eq!(before, fill(&mut b), "mask: {input:?}");
                    let aa = a.accept_token(i32::from(byte), false);
                    let bb = b.accept_token(i32::from(byte), false);
                    assert_eq!(aa, bb, "accept: {input:?}");
                    if !aa {
                        accepted = false;
                        break;
                    }
                    a.rollback(1);
                    b.rollback(1);
                    assert_eq!(before, fill(&mut a), "actual rollback: {input:?}");
                    assert_eq!(before, fill(&mut b), "reference rollback: {input:?}");
                    assert!(a.accept_token(i32::from(byte), false));
                    assert!(b.accept_token(i32::from(byte), false));
                }
                assert_eq!(
                    accepted && a.is_terminated(),
                    expected,
                    "{input:?}: empty={allow_empty}, close={force_close}"
                );
                assert_eq!(a.is_terminated(), b.is_terminated());
            }
            // 2026-09-26: A token can merge the key's closing `>` with the
            // first value byte: `>=` must be refused and `>a` accepted.
            for (prefix, token, expected) in [
                ("<parameter=city", ">=", false),
                ("<parameter=city", ">a", true),
                ("<parameter=city>", "</parameter>", allow_empty),
            ] {
                let mut a = GrammarMatcher::new(actual.clone(), None, false, -1);
                let mut b = GrammarMatcher::new(old.clone(), None, false, -1);
                assert!(a.accept_string(prefix, false));
                assert!(b.accept_string(prefix, false));
                assert_eq!(fill(&mut a), fill(&mut b));
                let id = vocab().iter().position(|s| s == token).unwrap() as i32;
                assert_eq!(a.accept_token(id, false), expected);
                assert_eq!(b.accept_token(id, false), expected);
            }
        }
    }
}
