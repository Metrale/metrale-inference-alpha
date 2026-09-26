// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Round-trip and validation tests for the `wire` codecs: reader and
//! writer agreement, the status and mode byte values, and the count bounds.
//! No `cfg` gate, so they run on builds without the verbs shim.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use metrale_gpu_sys::wire::{
    CacheServerParams, MODE_TCP, MODE_VERBS, STATUS_ERR, STATUS_OK, VerbsClientParams,
    VerbsServerParams, read_server_rails, write_server_rails,
};

/// 2026-09-26: Pins the status and transport-mode byte values; the transcript
/// goldens do not name these constants.
#[test]
fn frozen_wire_constants() {
    assert_eq!(STATUS_OK, 0, "STATUS_OK is a frozen wire constant");
    assert_eq!(STATUS_ERR, 1, "STATUS_ERR is a frozen wire constant");
    assert_eq!(MODE_TCP, 0, "MODE_TCP is a frozen wire constant");
    assert_eq!(MODE_VERBS, 1, "MODE_VERBS is a frozen wire constant");
}

#[test]
fn verbs_server_params_round_trip() {
    let sp = VerbsServerParams {
        qpn: 0x1234,
        psn: 0x00ab_cdef & 0xff_ffff,
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
        layers: vec![(0x7f00_0000_0000, 1001), (0x7f00_0100_0000, 1002)],
    };
    let mut buf = Vec::new();
    sp.write_to(&mut buf).unwrap();
    let back = VerbsServerParams::read_from(&mut &buf[..]).unwrap();
    assert_eq!(sp, back);
}

#[test]
fn verbs_client_params_round_trip() {
    let cp = VerbsClientParams {
        qpn: 0x9999,
        psn: 0x0055_5555,
        gid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    };
    let mut buf = Vec::new();
    cp.write_to(&mut buf).unwrap();
    let back = VerbsClientParams::read_from(&mut &buf[..]).unwrap();
    assert_eq!(cp, back);
}

#[test]
fn server_rails_round_trip() {
    let mk = |qpn| VerbsServerParams {
        qpn,
        psn: 0x0012_3456 & 0xff_ffff,
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
        // 2026-09-26: As the peers send it: one base address on every rail, a
        // distinct rkey per rail.
        layers: vec![
            (0x7f00_0000_0000, 1001 + qpn),
            (0x7f00_0100_0000, 1002 + qpn),
        ],
    };
    let rails = vec![mk(0x1111), mk(0x2222)];
    let mut buf = Vec::new();
    write_server_rails(&mut buf, &rails).unwrap();
    let back = read_server_rails(&mut &buf[..], 2).unwrap();
    assert_eq!(rails, back);
}

#[test]
fn single_rail_framing_round_trips() {
    let sp = VerbsServerParams {
        qpn: 7,
        psn: 9,
        gid: [1u8; 16],
        layers: vec![(0x1000, 42)],
    };
    let mut buf = Vec::new();
    write_server_rails(&mut buf, std::slice::from_ref(&sp)).unwrap();
    assert_eq!(buf[0], 1);
    let back = read_server_rails(&mut &buf[..], 1).unwrap();
    assert_eq!(back, vec![sp]);
}

#[test]
fn read_server_rails_rejects_mismatch() {
    let sp = VerbsServerParams {
        qpn: 1,
        psn: 2,
        gid: [0u8; 16],
        layers: vec![(0x1000, 7)],
    };
    let mut buf = Vec::new();
    write_server_rails(&mut buf, std::slice::from_ref(&sp)).unwrap();
    assert!(read_server_rails(&mut &buf[..], 2).is_err());
}

#[test]
fn read_server_rails_rejects_implausible_counts_before_the_body() {
    for (count, want) in [(0u8, 1usize), (9, 9)] {
        let err = read_server_rails(&mut &[count][..], want).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("implausible server rail count: {count}")
        );
    }
}

#[test]
fn write_server_rails_enforces_count_bounds_before_writing() {
    let sp = VerbsServerParams {
        qpn: 1,
        psn: 2,
        gid: [3u8; 16],
        layers: vec![(4, 5)],
    };
    let mut buf = Vec::new();
    let err = write_server_rails(&mut buf, &[]).unwrap_err();
    assert_eq!(err.to_string(), "implausible server rail count: 0");
    assert!(buf.is_empty(), "rejected empty input writes no frame");

    write_server_rails(&mut buf, &vec![sp.clone(); 8]).expect("eight rails is the maximum");
    assert_eq!(buf[0], 8);

    buf.clear();
    let err = write_server_rails(&mut buf, &vec![sp; 9]).unwrap_err();
    assert_eq!(err.to_string(), "implausible server rail count: 9");
    assert!(buf.is_empty(), "rejected excess input writes no frame");
}

#[test]
fn verbs_server_params_enforces_layer_count_bounds() {
    let encoded = |count: u32, body_layers: usize| {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend(std::iter::repeat_n(0u8, body_layers * 12));
        buf
    };

    let zero = encoded(0, 0);
    let err = VerbsServerParams::read_from(&mut &zero[..]).unwrap_err();
    assert_eq!(err.to_string(), "implausible verbs layer count: 0");

    let maximum = encoded(4096, 4096);
    let parsed = VerbsServerParams::read_from(&mut &maximum[..]).expect("4096 is accepted");
    assert_eq!(parsed.layers.len(), 4096);

    let excess = encoded(4097, 0);
    let err = VerbsServerParams::read_from(&mut &excess[..]).unwrap_err();
    assert_eq!(err.to_string(), "implausible verbs layer count: 4097");
}

#[test]
fn kv_server_params_round_trip() {
    let sp = CacheServerParams {
        qpn: 0x4242,
        psn: 0x0012_3456 & 0xff_ffff,
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
        base_addr: 0x7f00_1234_0000,
        rkey: 0xdead_beef,
    };
    let mut buf = Vec::new();
    sp.write_to(&mut buf).unwrap();
    assert_eq!(CacheServerParams::read_from(&mut &buf[..]).unwrap(), sp);
}
