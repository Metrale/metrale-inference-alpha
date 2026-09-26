// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the paging protocol: header parse and golden bytes, the
//! stripe plan, dispatch, the loop, the client calls, and the connection read pin.
//!
//! Owner: storage, SSM snapshot tier.
//! Invariants: none beyond the types.

use super::super::{MemSwapStore, VecSlotArena};
use super::*;

type TestResidency = Residency<VecSlotArena, MemSwapStore>;

const B: usize = 8;

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

/// 2026-09-25: A PUT straight on the residency: alloc, write the slot, commit.
fn put(r: &mut TestResidency, key: u64, tag: u8) {
    let slot = r.alloc(key).unwrap();
    r.arena_mut().write_slot(slot, &blob(tag)).unwrap();
    r.commit(key).unwrap();
}
fn get(r: &mut TestResidency, key: u64) -> Option<Vec<u8>> {
    r.locate(key).unwrap().map(|slot| {
        let mut out = vec![0u8; B];
        r.arena().read_slot(slot, &mut out).unwrap();
        out
    })
}

fn residency(slots: usize) -> TestResidency {
    Residency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
}

/// 2026-09-25: `handle_paging_op` pins a GET hit until the same connection's next op;
/// meanwhile the slot survives an ALLOC from another connection.
#[test]
fn handle_paging_op_pins_get_and_auto_releases() {
    let mut r = residency(2);
    put(&mut r, 0, 0);
    put(&mut r, 1, 1);
    let mut pinned: Option<u64> = None;

    let reply = handle_paging_op(&mut r, OP_GET, 0, &mut pinned);
    assert!(matches!(reply, PagingReply::Located(_)));
    assert_eq!(pinned, Some(0));
    assert_eq!(r.read_pin_count(0), 1);

    // 2026-09-25: Both slots are full and key 0 is pinned, so this evicts key 1.
    put(&mut r, 2, 2);
    assert_eq!(
        get(&mut r, 0),
        Some(blob(0)),
        "pinned GET slot survived the ALLOC"
    );

    handle_paging_op(&mut r, OP_REMOVE, 99, &mut pinned);
    assert_eq!(pinned, None);
    assert_eq!(r.read_pin_count(0), 0, "pin auto-released on next op");
}

#[test]
fn stripe_plan_covers_every_byte_once() {
    for (blob, chunk, rails) in [
        (64usize, 16usize, 2usize),
        (70, 16, 2),
        (64, 16, 1),
        (10, 64, 2),
        (64, 64, 2),
        (66846720, 1048576, 2),
    ] {
        let plan = stripe_plan(blob, chunk, rails);
        assert_eq!(plan.len(), rails.max(1));
        let mut covered = vec![0u8; blob];
        for rail in &plan {
            for &(off, len) in rail {
                assert!(
                    len <= chunk && off + len <= blob,
                    "chunk oob {off}+{len}>{blob}"
                );
                for b in &mut covered[off..off + len] {
                    assert_eq!(*b, 0, "byte {off} double-covered");
                    *b = 1;
                }
            }
        }
        assert!(
            covered.iter().all(|&b| b == 1),
            "gap in coverage blob={blob}"
        );
    }
    assert!(stripe_plan(0, 16, 2).iter().all(|r| r.is_empty()));
}

/// 2026-09-25: SSM, KV and raw-mode (`blob_bytes == 0`) headers parse; a wrong first
/// u64 or an unsupported kind is refused.
#[test]
fn paging_header_v2_parse_and_reject() {
    let mut body = vec![PagingKind::SSM.0];
    body.extend_from_slice(&0x1000u64.to_le_bytes());
    body.extend_from_slice(&0x40u64.to_le_bytes());
    let mut c = std::io::Cursor::new(body);
    assert_eq!(
        parse_paging_header(PAGING_MAGIC_V2, &mut c).unwrap(),
        (PagingKind::SSM, 0x1000, 0x40)
    );
    let mut body = vec![PagingKind::KV.0];
    body.extend_from_slice(&0x2000u64.to_le_bytes());
    body.extend_from_slice(&0x80u64.to_le_bytes());
    let mut c = std::io::Cursor::new(body);
    assert_eq!(
        parse_paging_header(PAGING_MAGIC_V2, &mut c).unwrap(),
        (PagingKind::KV, 0x2000, 0x80)
    );
    // 2026-09-25: `blob_bytes == 0` parses; the peer, not the parser, gives it a
    // private arena.
    let mut body = vec![PagingKind::KV.0];
    body.extend_from_slice(&0x3000u64.to_le_bytes());
    body.extend_from_slice(&0u64.to_le_bytes());
    let mut c = std::io::Cursor::new(body);
    assert_eq!(
        parse_paging_header(PAGING_MAGIC_V2, &mut c).unwrap(),
        (PagingKind::KV, 0x3000, 0)
    );
    let mut c = std::io::Cursor::new(Vec::new());
    assert!(parse_paging_header(12345, &mut c).is_err());
    let mut body = vec![3u8];
    body.extend_from_slice(&[0u8; 16]);
    let mut c = std::io::Cursor::new(body);
    assert!(parse_paging_header(PAGING_MAGIC_V2, &mut c).is_err());
}

/// 2026-09-25: The "PAGE" + 1 first u64 is refused with its own message.
#[test]
fn v1_magic_is_affirmatively_rejected() {
    let mut c = std::io::Cursor::new(Vec::new());
    let err = parse_paging_header(0x5041_4745_0000_0001, &mut c).unwrap_err();
    assert!(
        err.to_string().contains("v1 client no longer supported"),
        "v1 rejection must be the dedicated diagnostic, got: {err}"
    );
    assert_eq!(PAGING_MAGIC_V1_RETIRED, 0x5041_4745_0000_0001);
}

/// 2026-09-25: Golden bytes of a raw-mode header, as `RdmaKvBackend::connect` and
/// `RdmaSnapshotArena::connect` send it.
#[test]
fn v2_raw_mode_wire_golden() {
    assert_eq!(
        encode_paging_v2_header(PagingKind::KV, 0x4_0000_0000, 0).to_vec(),
        vec![
            0x02, 0x00, 0x00, 0x00, 0x45, 0x47, 0x41, 0x50, // 2026-09-25: PAGING_MAGIC_V2 LE
            0x01, // 2026-09-25: kind KV
            0x00, 0x00, 0x00, 0x00, 0x04, 0, 0, 0, // 2026-09-25: arena 16 GiB
            0x00, 0, 0, 0, 0, 0, 0, 0, // 2026-09-25: blob 0: raw mode
        ],
        "v2 RAW-mode handshake bytes are frozen"
    );
}

/// 2026-09-25: Golden bytes of `encode_paging_v2_header`, one vector per kind. Every
/// field has a distinct value, so swapping two fields in both writer and reader
/// still fails.
#[test]
fn v2_handshake_wire_golden() {
    assert_eq!(
        encode_paging_v2_header(PagingKind::KV, 0x2000, 0x80).to_vec(),
        vec![
            0x02, 0x00, 0x00, 0x00, 0x45, 0x47, 0x41, 0x50, // 2026-09-25: PAGING_MAGIC_V2 LE
            0x01, // 2026-09-25: kind KV
            0x00, 0x20, 0, 0, 0, 0, 0, 0, // 2026-09-25: arena 0x2000
            0x80, 0, 0, 0, 0, 0, 0, 0, // 2026-09-25: blob 0x80
        ],
        "v2 KV handshake bytes are frozen (the deployed fleet peer parses them)"
    );
    assert_eq!(
        encode_paging_v2_header(PagingKind::SSM, 0x1000, 0x40).to_vec(),
        vec![
            0x02, 0x00, 0x00, 0x00, 0x45, 0x47, 0x41, 0x50, // 2026-09-25: PAGING_MAGIC_V2 LE
            0x00, // 2026-09-25: kind SSM
            0x00, 0x10, 0, 0, 0, 0, 0, 0, // 2026-09-25: arena 0x1000
            0x40, 0, 0, 0, 0, 0, 0, 0, // 2026-09-25: blob 0x40
        ],
        "v2 SSM handshake bytes are frozen"
    );
}

/// 2026-09-25: Every encoded header parses back to its `(kind, arena, blob)`.
#[test]
fn v2_encode_parses_back() {
    for (kind, arena, blob) in [
        (PagingKind::KV, 0x40_0000u64, 0x1_0000u64),
        (PagingKind::SSM, 0x1000, 0x40),
        (PagingKind::KV, 0x3000, 0),
        (PagingKind::SSM, 0x2000, 0),
    ] {
        let w = encode_paging_v2_header(kind, arena, blob);
        let first = u64::from_le_bytes(w[0..8].try_into().unwrap());
        let mut c = std::io::Cursor::new(w[8..].to_vec());
        assert_eq!(
            parse_paging_header(first, &mut c).unwrap(),
            (kind, arena, blob)
        );
    }
}

/// 2026-09-25: A PUT then a GET through `dispatch`; the client's RDMA WRITE is a
/// direct write of the returned slot.
#[test]
fn dispatch_put_then_get_roundtrips() {
    let mut r = residency(4);
    let PagingReply::Located(off) = dispatch(&mut r, OP_ALLOC, 7) else {
        panic!("alloc reply")
    };
    let slot = (off as usize) / B;
    r.arena_mut().write_slot(slot, &blob(0xAB)).unwrap();
    assert_eq!(dispatch(&mut r, OP_COMMIT, 7), PagingReply::Ok);
    let PagingReply::Located(goff) = dispatch(&mut r, OP_GET, 7) else {
        panic!("get reply")
    };
    let mut out = vec![0u8; B];
    r.arena().read_slot((goff as usize) / B, &mut out).unwrap();
    assert_eq!(out, blob(0xAB));
    assert_eq!(dispatch(&mut r, OP_GET, 999), PagingReply::Miss);
}

/// 2026-09-25: Fake duplex stream: scripted input, captured output.
struct Duplex {
    inp: std::io::Cursor<Vec<u8>>,
    out: Vec<u8>,
}
impl Read for Duplex {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.inp.read(b)
    }
}
impl Write for Duplex {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.out.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn req(op: u8, key: u64) -> Vec<u8> {
    let mut v = vec![op];
    v.extend_from_slice(&key.to_le_bytes());
    v
}

/// 2026-09-25: `run_paging_loop` over a scripted request stream writes the expected
/// reply bytes.
#[test]
fn run_paging_loop_scripts_ok() {
    let mut r = residency(4);
    // 2026-09-25: The script is fixed before the loop runs, so the ALLOC goes
    // through `dispatch` first to learn the slot to write.
    let PagingReply::Located(off) = dispatch(&mut r, OP_ALLOC, 1) else {
        panic!()
    };
    r.arena_mut()
        .write_slot((off as usize) / B, &blob(0x5A))
        .unwrap();

    let mut script = Vec::new();
    script.extend(req(OP_COMMIT, 1));
    script.extend(req(OP_GET, 1));
    script.extend(req(OP_GET, 42));
    script.extend(req(OP_REMOVE, 1));
    script.extend(req(OP_BYE, 0));
    let mut dx = Duplex {
        inp: std::io::Cursor::new(script),
        out: Vec::new(),
    };
    run_paging_loop(&mut dx, &mut r).unwrap();

    // 2026-09-25: COMMIT, GET 1, GET 42 (a miss), REMOVE; BYE gets no reply.
    let mut exp = Vec::new();
    exp.push(ST_OK);
    exp.push(ST_OK);
    exp.extend_from_slice(&off.to_le_bytes());
    exp.push(ST_MISS);
    exp.push(ST_OK);
    assert_eq!(dx.out, exp);
}

/// 2026-09-25: The client calls' request bytes and reply decoding.
#[test]
fn client_codec_alloc_get_miss() {
    let mut reply = vec![ST_OK];
    reply.extend_from_slice(&0x40u64.to_le_bytes());
    let mut dx = Duplex {
        inp: std::io::Cursor::new(reply),
        out: Vec::new(),
    };
    let off = client_alloc(&mut dx, 0xAB).unwrap();
    assert_eq!(off, 0x40);
    assert_eq!(dx.out, req(OP_ALLOC, 0xAB), "request bytes on the wire");

    let mut dx = Duplex {
        inp: std::io::Cursor::new(vec![ST_MISS]),
        out: Vec::new(),
    };
    assert_eq!(client_get(&mut dx, 7).unwrap(), None);

    let mut dx = Duplex {
        inp: std::io::Cursor::new(vec![ST_OK]),
        out: Vec::new(),
    };
    client_commit(&mut dx, 9).unwrap();
    assert_eq!(dx.out, req(OP_COMMIT, 9));
}

/// 2026-09-25: Client request bytes feed the peer's `dispatch`, and its reply bytes
/// feed the client decoding; a PUT then GET returns the same blob. The RDMA WRITE is
/// a direct write of the slot at the returned offset.
#[test]
fn client_peer_loopback_roundtrip() {
    let mut r = residency(4);
    // 2026-09-25: One request through the peer; returns a cursor over its reply.
    fn peer_roundtrip(r: &mut TestResidency, req_bytes: &[u8]) -> std::io::Cursor<Vec<u8>> {
        let op = req_bytes[0];
        let key = u64::from_le_bytes(req_bytes[1..9].try_into().unwrap());
        let mut reply = Vec::new();
        write_reply(&mut reply, &dispatch(r, op, key)).unwrap();
        std::io::Cursor::new(reply)
    }

    let mut wire = Vec::new();
    send_req(&mut wire, OP_ALLOC, 3).unwrap();
    let mut rep = peer_roundtrip(&mut r, &wire);
    assert_eq!(read_status(&mut rep).unwrap(), ST_OK);
    let off = read_offset(&mut rep).unwrap();
    r.arena_mut()
        .write_slot((off as usize) / B, &blob(0x77))
        .unwrap();

    wire.clear();
    send_req(&mut wire, OP_COMMIT, 3).unwrap();
    assert_eq!(
        read_status(&mut peer_roundtrip(&mut r, &wire)).unwrap(),
        ST_OK
    );

    wire.clear();
    send_req(&mut wire, OP_GET, 3).unwrap();
    let mut rep = peer_roundtrip(&mut r, &wire);
    assert_eq!(read_status(&mut rep).unwrap(), ST_OK);
    let goff = read_offset(&mut rep).unwrap();
    let mut out = vec![0u8; B];
    r.arena().read_slot((goff as usize) / B, &mut out).unwrap();
    assert_eq!(
        out,
        blob(0x77),
        "PUT→GET round-trips byte-identical over the protocol"
    );
}
