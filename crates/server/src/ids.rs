// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Ids (`uuid_v4`) and wire `created` timestamps
//! (`unix_timestamp`). Callers add their own wire prefixes (`chatcmpl-`,
//! `msg_`, `resp_`, `conv_`, `req_`).
//!
//! Owner: server API.
//! Invariants: `uuid_v4` always returns 32 lowercase hex digits in 8-4-4-4-12
//! groups.

/// 2026-09-26: Fill `buf` from `/dev/urandom`; an open or read error is `Err(())`.
pub(crate) fn getrandom(buf: &mut [u8]) -> Result<(), ()> {
    use std::fs::File;
    use std::io::Read;
    File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(buf))
        .map_err(|_| ())
}

/// 2026-09-26: A random UUID v4 from `/dev/urandom`, with the version (4) and
/// RFC 4122 variant bits set. When the read fails the 16 bytes are the Unix
/// time in nanoseconds, little-endian, with no version or variant bits: not
/// random, and not a valid v4 UUID.
pub(crate) fn uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    if let Ok(()) = getrandom(&mut bytes) {
        bytes[6] = (bytes[6] & 0x0F) | 0x40;
        bytes[8] = (bytes[8] & 0x3F) | 0x80;
    } else {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        bytes = t.to_le_bytes();
    }
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

/// 2026-09-26: Whole seconds since the Unix epoch, or 0 if the clock reads
/// earlier; the wire `created` and `created_at` fields use it.
pub(crate) fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
