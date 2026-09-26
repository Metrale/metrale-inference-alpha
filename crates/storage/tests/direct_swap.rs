// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Integration tests of [`DirectSwapFile`] on real files under
//! `target/metrale-tier-tests`. A test returns early when `o_direct_file`
//! reports that the filesystem refused `O_DIRECT`.
//!
//! Owner: storage (tests).
//! Invariants: none beyond the types.

use std::path::Path;

use metrale_storage::tier::{DirectSwapFile, Residency, SwapStore, VecSlotArena};

/// 2026-09-25: A new swap file under `target/metrale-tier-tests`, or `None` when
/// the create failed with `EINVAL` or `EOPNOTSUPP` found in the error chain.
/// Any other failure panics.
fn o_direct_file(record_bytes: usize, tag: &str) -> Option<(DirectSwapFile, std::path::PathBuf)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/metrale-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("dsf-{tag}-{}.swap", std::process::id()));
    match DirectSwapFile::create(&path, record_bytes) {
        Ok(f) => Some((f, path)),
        Err(e) => {
            let unsupported = e.chain().any(|cause| {
                cause
                    .downcast_ref::<std::io::Error>()
                    .and_then(std::io::Error::raw_os_error)
                    .is_some_and(|code| code == libc::EINVAL || code == libc::EOPNOTSUPP)
            });
            // 2026-09-25: With `METRALE_TIER_REQUIRE_O_DIRECT` set, a refused
            // `O_DIRECT` panics instead of skipping.
            if std::env::var_os("METRALE_TIER_REQUIRE_O_DIRECT").is_some() || !unsupported {
                panic!(
                    "DirectSwapFile setup failed instead of reporting unsupported O_DIRECT: {e:#}"
                );
            }
            eprintln!("skipping O_DIRECT test (filesystem refused O_DIRECT): {e:#}");
            None
        }
    }
}

/// 2026-09-25: A 4 KiB-aligned sub-slice of `len` bytes; `storage` must be at
/// least `len + 4095` bytes long.
fn page_aligned(storage: &mut [u8], len: usize) -> &mut [u8] {
    let pad = (4096 - (storage.as_ptr() as usize & 0xfff)) & 0xfff;
    &mut storage[pad..pad + len]
}

/// 2026-09-25: A sub-slice of `len` bytes that is not 4 KiB-aligned; `storage`
/// must be at least `len + 1` bytes long.
fn page_unaligned(storage: &mut [u8], len: usize) -> &mut [u8] {
    let offset = usize::from(storage.as_ptr() as usize & 0xfff == 0);
    &mut storage[offset..offset + len]
}

#[test]
fn direct_swap_file_rejects_bad_record_bytes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/metrale-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dsf-bad.swap");
    assert!(
        DirectSwapFile::create(&path, 0).is_err(),
        "zero record_bytes rejected"
    );
    assert!(
        DirectSwapFile::create(&path, 1000).is_err(),
        "non-4KiB multiple rejected"
    );
}

/// 2026-09-25: Records written and read through buffers that are not
/// page-aligned come back byte for byte.
#[test]
fn direct_swap_file_roundtrips_unaligned_records() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "rt") else {
        return;
    };
    assert_eq!(f.record_bytes(), rb);
    let mut pat_storage = vec![0u8; rb + 1];
    let pat = page_unaligned(&mut pat_storage, rb);
    assert_ne!(pat.as_ptr() as usize & 0xfff, 0, "write uses bounce");
    for (i, b) in pat.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    f.write_record(3, pat).unwrap();
    f.write_record(0, &vec![0xEE; rb]).unwrap();
    let mut out_storage = vec![0u8; rb + 1];
    let out = page_unaligned(&mut out_storage, rb);
    assert_ne!(out.as_ptr() as usize & 0xfff, 0, "read uses bounce");
    f.read_record(3, out).unwrap();
    assert_eq!(out, pat, "record 3 byte-identical");
    f.read_record(0, out).unwrap();
    assert_eq!(out, vec![0xEE; rb], "record 0 byte-identical");
    assert!(f.write_record(1, &pat[..100]).is_err());
    let mut short = vec![0u8; 100];
    assert!(f.read_record(0, &mut short).is_err());
    let _ = std::fs::remove_file(path);
}

/// 2026-09-25: A two-slot `Residency` over a `DirectSwapFile` spills and faults
/// back eight keys byte for byte.
#[test]
fn residency_over_o_direct_swap_byte_identical() {
    let rb = 4096usize;
    let Some((f, path)) = o_direct_file(rb, "resid") else {
        return;
    };
    let mut r = Residency::new(VecSlotArena::new(rb, 2), f).unwrap();
    for k in 0..8u64 {
        r.put_blob(k, &vec![k as u8; rb]).unwrap();
    }
    assert_eq!(r.total_keys(), 8);
    assert!(
        r.stats().spills_to_disk >= 6,
        "cold keys spilled to the O_DIRECT file"
    );
    let mut out = vec![0u8; rb];
    for k in 0..8u64 {
        assert!(r.get_blob(k, &mut out).unwrap(), "key {k}");
        assert_eq!(
            out,
            vec![k as u8; rb],
            "key {k} byte-identical through O_DIRECT"
        );
    }
    let _ = std::fs::remove_file(path);
}

/// 2026-09-25: Page-aligned buffers round-trip. The test does not observe which
/// internal branch ran.
#[test]
fn direct_swap_aligned_buffers_roundtrip() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "aligned") else {
        return;
    };
    let mut wstore = vec![0u8; rb + 4096];
    let w = page_aligned(&mut wstore, rb);
    assert_eq!(
        w.as_ptr() as usize & 0xfff,
        0,
        "write buffer is page-aligned → fast-path"
    );
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    f.write_record(7, w).unwrap();

    let mut rstore = vec![0u8; rb + 4096];
    let rbuf = page_aligned(&mut rstore, rb);
    assert_eq!(
        rbuf.as_ptr() as usize & 0xfff,
        0,
        "read buffer is page-aligned → fast-path"
    );
    f.read_record(7, rbuf).unwrap();
    for (i, b) in rbuf.iter().enumerate() {
        assert_eq!(*b, (i % 251) as u8, "aligned fast-path byte {i}");
    }
    let _ = std::fs::remove_file(path);
}

/// 2026-09-25: A new swap file is empty, and each `write_record` extends it to
/// `(disk_slot + 1) * record_bytes` when that is past its end.
#[test]
fn direct_swap_file_grows_on_first_write() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "grow") else {
        return;
    };
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        0,
        "freshly created (truncated) swap file starts empty"
    );
    f.write_record(0, &vec![0x7C; rb]).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        rb as u64,
        "one record written ⇒ file is exactly one record long"
    );
    f.write_record(4, &vec![0x7D; rb]).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 5 * rb as u64);
    let _ = std::fs::remove_file(path);
}
