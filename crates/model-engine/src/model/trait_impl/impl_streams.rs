// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelStreams for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::ModelStreams;

impl ModelStreams for TransformerModel {
    fn default_stream(&self) -> u64 {
        self.default_stream_dispatch()
    }

    fn create_stream(&self) -> Result<u64> {
        self.create_stream_dispatch()
    }

    fn create_event(&self) -> Result<u64> {
        self.create_event_dispatch()
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_dispatch(event, stream)
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_dispatch(stream, event)
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.synchronize_dispatch(stream)
    }
}
