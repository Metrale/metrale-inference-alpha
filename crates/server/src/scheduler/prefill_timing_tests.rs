// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that `request_start`, the TTFT origin, is taken before
//! grammar compilation. A parser records when its compile begins; the
//! deferred `start_chunked_prefill` runs twice with the same grammar
//! engine, and no GPU work or sleep is involved.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::api::InferenceRequest;
use crate::api::inference_types::GrammarSpec;
use crate::grammar::{GrammarEngine, GrammarError};
use crate::tool_parser::{
    IncomingToolCall, PromptLevers, ToolCallParser, ToolChoice, ToolDefinition,
};
use metrale_grammar::CompiledGrammar;

struct ObservedParser(Arc<Mutex<Option<Instant>>>);

impl ToolCallParser for ObservedParser {
    fn name(&self) -> &str {
        "observed-grammar"
    }
    fn system_prompt(&self, _: &[ToolDefinition], _: &ToolChoice, _: &PromptLevers) -> String {
        unreachable!("request is already tokenized")
    }
    fn format_tool_calls(&self, _: &[IncomingToolCall]) -> String {
        unreachable!("request is already tokenized")
    }
    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        _: &[ToolDefinition],
        _: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        *self.0.lock().unwrap() = Some(Instant::now());
        Some(engine.compile_ebnf("root ::= \"x\"", "root"))
    }
}

fn request(grammar_spec: Option<GrammarSpec>) -> InferenceRequest {
    super::test_support::blocking_request(grammar_spec)
}

#[test]
fn deferred_prefill_clock_includes_cold_and_cached_grammar_preparation() {
    let marker = Arc::new(Mutex::new(None));
    let mut engine = Some(GrammarEngine::new(&["x".to_string()], &[]).unwrap());
    for cached in [false, true] {
        let spec = GrammarSpec::ToolCall {
            tools: vec![],
            parser: Arc::new(ObservedParser(Arc::clone(&marker))),
            use_triggers: true,
        };
        let result = super::prefill_a_step::start_chunked_prefill(
            &super::sched_ctx::SchedCtx::for_test(),
            None,
            None,
            None,
            None,
            &super::lifecycle_tests::StubModel,
            request(Some(spec)),
            &[],
            1,
            0,
            0,
            &mut engine,
            0,
            true,
            None,
            None,
        )
        .unwrap();
        let super::emit_step::StartPrefillResult::InProgress(prefill) = result else {
            panic!("deferred prefill unexpectedly performed inference");
        };
        assert!(
            prefill.grammar_state.is_some(),
            "real grammar must be prepared"
        );
        let compiled_at = marker.lock().unwrap().unwrap();
        assert!(
            prefill.request_start <= compiled_at,
            "TTFT origin excluded grammar preparation (cached={cached})"
        );
    }
}
