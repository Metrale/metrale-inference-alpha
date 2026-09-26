// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Drain of the chat stream's `return_token_ids` buffer.
//!
//! Owner: server chat streaming.
//! Invariants: none beyond the types.

use super::state::StreamState;

impl StreamState {
    /// 2026-09-26: When `on`, take the IDs buffered in `pending_token_ids`
    /// since the last take, leaving the buffer empty. Otherwise return an
    /// empty vec and leave the buffer untouched. An empty vec puts no
    /// `token_ids` field on the chunk (`ChatCompletionChunk` skips it when
    /// empty).
    pub(super) fn take_ids_if(&mut self, on: bool) -> Vec<u32> {
        if on {
            std::mem::take(&mut self.pending_token_ids)
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StreamState;

    #[test]
    fn take_ids_if_drains_only_when_on() {
        let mut st = StreamState::new(
            false,
            false,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            Vec::new(),
        );
        st.pending_token_ids = vec![7, 8, 9];

        assert!(st.take_ids_if(false).is_empty());
        assert_eq!(st.pending_token_ids, vec![7, 8, 9]);

        assert_eq!(st.take_ids_if(true), vec![7, 8, 9]);
        assert!(st.pending_token_ids.is_empty());
        assert!(st.take_ids_if(true).is_empty());
    }
}
