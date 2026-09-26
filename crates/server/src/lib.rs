// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! 2026-09-26: Library target of metrale-server: the modules that the
//! integration tests, the `k3_tokenizer_check` example and
//! `cargo test -p metrale-server --lib` use without building the `met` binary.
//!
//! Owner: server.
//! Invariants: every module declared here is also declared in `main.rs`.

pub mod tokenizer;

// 2026-09-26: The tokenizer tests use `reasoning_parser::ReasoningFormat`, so
// a `--lib` build needs it; its only in-crate dependency is `tokenizer`.
pub mod reasoning_parser;

// 2026-09-26: Declared here as well as in `main.rs` so their unit tests run
// under `--lib`. `rate_limiter` calls `env_config::parse_min`, so
// `env_config` has to be here too.
#[path = "auth.rs"]
pub mod auth;
#[path = "env_config.rs"]
pub mod env_config;
#[path = "rate_limiter.rs"]
pub mod rate_limiter;
#[path = "refusal.rs"]
pub mod refusal;
