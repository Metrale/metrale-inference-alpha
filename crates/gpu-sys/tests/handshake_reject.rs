// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A peer that rejects a client closes the connection mid-handshake,
//! and the wire has no error frame. These tests pin how `read_rw_server_params`
//! words each truncation: a close before the rail echo is a rejection that names
//! the peer and `--max-blade-gb`, not the raw io error.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use std::io::{Read, Result as IoResult};

use metrale_gpu_sys::handshake::read_rw_server_params;

/// 2026-09-26: A peer that sends `remaining` and then closes the connection.
struct TruncatedPeer {
    remaining: Vec<u8>,
}

impl Read for TruncatedPeer {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if self.remaining.is_empty() {
            return Ok(0);
        }
        let n = buf.len().min(self.remaining.len());
        buf[..n].copy_from_slice(&self.remaining[..n]);
        self.remaining.drain(..n);
        Ok(n)
    }
}

#[test]
fn peer_hangup_before_rail_params_names_the_rejection_not_a_buffer_error() {
    let mut s = TruncatedPeer { remaining: vec![] };
    let err = read_rw_server_params(&mut s, 1, "test-peer").unwrap_err();
    let msg = format!("{err:#}");

    assert!(
        msg.contains("closed the connection during the rail handshake"),
        "must say the peer hung up: {msg}"
    );
    assert!(
        msg.contains("REJECTED"),
        "must name this as a rejection, not a transport fault: {msg}"
    );
    assert!(
        msg.contains("--max-blade-gb"),
        "must point at the most common cause (arena exceeds the peer cap): {msg}"
    );
    assert!(
        msg.contains("test-peer"),
        "must name WHICH peer rejected us: {msg}"
    );
    assert!(
        !msg.contains("failed to fill whole buffer"),
        "must not surface the raw io error as the headline: {msg}"
    );
}

#[test]
fn a_truncated_params_body_is_still_a_read_error_not_a_rejection() {
    // 2026-09-26: The peer sent its rail echo and closed inside the params body:
    // a read error, not a rejection.
    let mut s = TruncatedPeer {
        remaining: vec![1u8, 0xde, 0xad],
    };
    let err = read_rw_server_params(&mut s, 1, "test-peer").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        !msg.contains("REJECTED"),
        "a mid-body truncation is not a rejection: {msg}"
    );
    assert!(
        msg.contains("params"),
        "should blame the params read: {msg}"
    );
}

#[test]
fn a_rail_count_mismatch_is_reported_as_a_grant_mismatch() {
    // 2026-09-26: An echo with a different rail count has its own message.
    let mut s = TruncatedPeer {
        remaining: vec![3u8],
    };
    let err = read_rw_server_params(&mut s, 1, "test-peer").unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("granted 3 rails, wanted 1"), "{msg}");
}
