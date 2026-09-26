// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Live smoke test of the peer paging tier over RDMA: it writes
//! `SMOKE_KEYS` records of `SMOKE_BLOB` bytes through a `SMOKE_SLOTS`-slot peer
//! arena, reads every one back and compares the bytes. `SMOKE_MODE` picks the
//! test: `putget` (default), `put` or `get` for SSM blobs through
//! `RdmaSnapshotArena`; `kv` or `kv-isolation` for KV blocks through
//! `KvPagingBackend`; `decode-isolation` for two SSM clients with different
//! salts. The peer address is `METRALE_SNAP_PEER` (default 127.0.0.1:9918).
//! Without the `cuda` feature and `cfg(metrale_rdma_verbs)`, `main` exits 1.
//!
//! Owner: metrale-storage.
//! Invariants: none beyond the types.

#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
fn main() -> anyhow::Result<()> {
    use metrale_storage::RdmaSnapshotArena;

    // 2026-09-25: The pinned bounce buffers (`cuMemAllocHost`) need a current
    // CUDA context.
    let _cuda = metrale_storage::cuda_min::CudaCtx::new(0)?;

    let addr = std::env::var("METRALE_SNAP_PEER").unwrap_or_else(|_| "127.0.0.1:9918".into());
    let blob: usize = std::env::var("SMOKE_BLOB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);
    let slots: usize = std::env::var("SMOKE_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n: u64 = std::env::var("SMOKE_KEYS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let mode = std::env::var("SMOKE_MODE").unwrap_or_else(|_| "putget".into());
    if mode == "kv" || mode == "kv-isolation" {
        return kv_main(&addr, blob, slots, n, &mode);
    }
    if mode == "decode-isolation" {
        return decode_isolation_main(&addr, blob, slots, n);
    }

    let arena_bytes = (slots * blob) as u64;
    println!(
        "connecting paging tier @ {addr} [{mode}]: {slots}-slot RAM arena × {blob} B, {n} keys \
         (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    let arena = RdmaSnapshotArena::connect_paging(&addr, arena_bytes, blob)?;

    let pat = |k: u64| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (k as u8) ^ (i as u8).wrapping_mul(31);
        }
        v
    };

    let mut put_ms: Option<f64> = None;
    if mode != "get" {
        let t0 = std::time::Instant::now();
        for k in 0..n {
            arena.paging_put(k, &pat(k))?;
        }
        put_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
    }
    if mode == "put" {
        println!("PUT-only done: {n} keys left resident+spilled in the shared peer arena");
        return Ok(());
    }

    let mut out = vec![0u8; blob];
    let t1 = std::time::Instant::now();
    for k in 0..n {
        let hit = arena.paging_get(k, &mut out)?;
        anyhow::ensure!(
            hit,
            "key {k} MISSING — peer dropped it, or (mode=get) not shared across connections"
        );
        anyhow::ensure!(
            out == pat(k),
            "key {k} CORRUPTED — spill/fault not byte-identical"
        );
    }
    let get_ms = t1.elapsed().as_secs_f64() * 1e3;
    if mode == "get" {
        println!(
            "CROSS-CONNECTION SHARING OK ✅  a SEPARATE client GET all {n} keys a prior `put` \
             run left in the shared peer — {:.1}ms/blob",
            get_ms / n as f64
        );
        return Ok(());
    }
    let put_ms = put_ms.unwrap_or(0.0);

    let t2 = std::time::Instant::now();
    let _ = arena.paging_get(0, &mut out)?;
    let one_get_us = t2.elapsed().as_micros();

    println!(
        "PAGING SMOKE OK ✅  {n} blobs through {slots}-slot arena, ALL byte-identical after NVMe \
         spill+fault.  put {put_ms:.0}ms ({:.1}/blob)  get {get_ms:.0}ms ({:.1}/blob)  \
         single fault-from-disk {one_get_us}us",
        put_ms / n as f64,
        get_ms / n as f64,
    );
    Ok(())
}

/// 2026-09-25: `SMOKE_MODE=decode-isolation`: two SSM decode clients with the
/// same fingerprint and the same keys `0..n`, but different salts, on one peer.
/// Client A (salt 1) writes every key. Client B (salt 2) must miss every key,
/// and removes each one. A third connection with salt 1 must then read every
/// key back intact, which shows B's misses were not a dead connection and B's
/// removes did not reach A's records.
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
fn decode_isolation_main(addr: &str, blob: usize, slots: usize, n: u64) -> anyhow::Result<()> {
    use metrale_storage::RdmaSnapshotArena;
    use metrale_storage::tier::hash::mix64;

    // 2026-09-25: The formula of `derive_decode_ns_salted`
    // (metrale-model-engine ssm_tier/fingerprint.rs) without its zero fallbacks:
    // ns = mix64(mix64(fp, DECODE_DOMAIN), salt), wire key = mix64(key, ns).
    // The literal equals `metrale_kernels::DECODE_DOMAIN`, which this crate
    // cannot import (no metrale-kernels dependency).
    const DECODE_DOMAIN_LIT: u64 = 0xD3C0_DE12_A5B6_C7D8;
    const FP: u64 = 0xFEED_FACE_CAFE_BEEF;
    let ns = |salt: u64| mix64(mix64(FP, DECODE_DOMAIN_LIT), salt);
    let (ns_a, ns_b) = (ns(1), ns(2));

    let arena_bytes = (slots * blob) as u64;
    let pat = |k: u64| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (k as u8) ^ (i as u8).wrapping_mul(37) ^ 0x5A;
        }
        v
    };
    println!(
        "connecting SSM decode-isolation @ {addr}: {slots}-slot arena × {blob} B, {n} keys \
         (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    let a = RdmaSnapshotArena::connect_paging(addr, arena_bytes, blob)?;
    for k in 0..n {
        a.paging_put(mix64(k, ns_a), &pat(k))?;
    }
    let b = RdmaSnapshotArena::connect_paging(addr, arena_bytes, blob)?;
    let mut out = vec![0u8; blob];
    for k in 0..n {
        let hit = b.paging_get(mix64(k, ns_b), &mut out)?;
        anyhow::ensure!(
            !hit,
            "DECODE ISOLATION BROKEN: client B was served client A's rollback blob (key {k})"
        );
        b.paging_remove(mix64(k, ns_b))?;
    }
    let a2 = RdmaSnapshotArena::connect_paging(addr, arena_bytes, blob)?;
    for k in 0..n {
        let hit = a2.paging_get(mix64(k, ns_a), &mut out)?;
        anyhow::ensure!(
            hit,
            "control MISS on key {k}: blob lost — dead connection, or B's removes \
             leaked across namespaces"
        );
        anyhow::ensure!(
            out == pat(k),
            "control CORRUPTED: key {k} not byte-identical"
        );
    }
    println!(
        "DECODE ISOLATION OK ✅  two client salts on one shared peer arena: B missed all \
         {n} of A's slot-coordinate keys (and B's removes did not leak); a same-salt \
         control connection restored all {n} byte-identical after peer NVMe spills"
    );
    Ok(())
}

/// 2026-09-25: `SMOKE_MODE=kv`: write blocks `0..n` through `KvPagingBackend`,
/// read each back into a device buffer and compare, then overwrite block 0 and
/// read it again.
///
/// `SMOKE_MODE=kv-isolation`: two clients with the same fingerprint and
/// different salts on one peer. After A writes every block, B's read of block
/// 0 must fail with the miss error ("unrecoverable"); then each client writes
/// or keeps its own block 0 and reads it back intact.
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
fn kv_main(addr: &str, blob: usize, slots: usize, n: u64, mode: &str) -> anyhow::Result<()> {
    use metrale_storage::backend::BlockReadRequest;
    use metrale_storage::backend::StorageBackend;
    use metrale_storage::cuda_min::{DeviceBuffer, copy_d_to_h_async, stream_sync};
    use metrale_storage::group::{GroupKey, GroupLayout, KvKind};
    use metrale_storage::kv_paging::ns::derive_kv_ns;
    use metrale_storage::kv_paging::{KvPagingBackend, KvPagingConnect};

    const NKV: u16 = 8;
    anyhow::ensure!(
        blob.is_multiple_of(2 * NKV as usize),
        "SMOKE_BLOB must be a multiple of {} (2·num_kv_heads)",
        2 * NKV
    );
    // 2026-09-25: Built from the public fields so that `block_bytes()` equals
    // `SMOKE_BLOB`.
    let layout = GroupLayout {
        num_layers: 1,
        num_blocks: n as u32,
        num_kv_heads: NKV,
        group_stride: (blob / (2 * NKV as usize)) as u64,
        fs_block_size: 4096,
    };
    assert_eq!(layout.block_bytes() as usize, blob);
    let arena_bytes = (slots * blob) as u64;
    let fp: u64 = 0xFEED_FACE_CAFE_BEEF;
    let connect = |salt: u64| -> anyhow::Result<KvPagingBackend> {
        KvPagingBackend::connect(
            addr,
            layout,
            KvPagingConnect {
                arena_bytes,
                ns: derive_kv_ns(fp, &layout, 2, 16, 128, salt),
            },
        )
    };
    let pat = |b: u32, tag: u8| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (b as u8) ^ (i as u8).wrapping_mul(29) ^ tag;
        }
        v
    };
    let key = |b: u32| GroupKey::new(0, b, 0, KvKind::K);
    let dev = DeviceBuffer::new(blob)?;
    let mut host = vec![0u8; blob];
    let read_block = |be: &mut KvPagingBackend, b: u32, host: &mut [u8]| -> anyhow::Result<()> {
        be.read_blocks(
            &[BlockReadRequest {
                base_key: key(b),
                dst_dev_ptr: dev.ptr,
            }],
            0,
        )?;
        copy_d_to_h_async(host.as_mut_ptr() as *mut _, dev.ptr, blob, 0)?;
        stream_sync(0)
    };

    println!(
        "connecting KV paging tier @ {addr} [{mode}]: {slots}-slot arena × {blob} B blocks, \
         {n} blocks (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    if mode == "kv-isolation" {
        let mut a = connect(0x0000_0000_0000_0001)?;
        let mut b = connect(0x0000_0000_0000_0002)?;
        for blk in 0..n as u32 {
            a.write_block_from_host(key(blk), &pat(blk, 0xA0))?;
        }
        let miss = read_block(&mut b, 0, &mut host);
        anyhow::ensure!(
            miss.is_err(),
            "ISOLATION BROKEN: client B was served client A's KV block"
        );
        let msg = format!("{:#}", miss.unwrap_err());
        anyhow::ensure!(
            msg.contains("unrecoverable"),
            "unexpected miss error: {msg}"
        );
        b.write_block_from_host(key(0), &pat(0, 0xB0))?;
        read_block(&mut b, 0, &mut host)?;
        anyhow::ensure!(host == pat(0, 0xB0), "client B corrupted");
        read_block(&mut a, 0, &mut host)?;
        anyhow::ensure!(host == pat(0, 0xA0), "client A corrupted by B's write");
        println!(
            "KV ISOLATION OK ✅  two salts on one shared peer arena: B missed A's blocks \
             (hard error), both round-tripped their own byte-identical"
        );
        return Ok(());
    }
    let mut be = connect(0x5EED)?;
    let t0 = std::time::Instant::now();
    for blk in 0..n as u32 {
        be.write_block_from_host(key(blk), &pat(blk, 0))?;
    }
    let put_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    for blk in 0..n as u32 {
        read_block(&mut be, blk, &mut host)?;
        anyhow::ensure!(
            host == pat(blk, 0),
            "block {blk} CORRUPTED — spill/fault not byte-identical"
        );
    }
    let get_ms = t1.elapsed().as_secs_f64() * 1e3;
    be.write_block_from_host(key(0), &pat(0, 0x77))?;
    read_block(&mut be, 0, &mut host)?;
    anyhow::ensure!(host == pat(0, 0x77), "overwrite-in-place corrupted");
    println!(
        "KV PAGING SMOKE OK ✅  {n} blocks through {slots}-slot arena, ALL byte-identical \
         after NVMe spill+fault (+ overwrite-in-place).  put {put_ms:.0}ms ({:.1}/blk)  \
         get {get_ms:.0}ms ({:.1}/blk)",
        put_ms / n as f64,
        get_ms / n as f64,
    );
    Ok(())
}

#[cfg(not(all(feature = "cuda", metrale_rdma_verbs)))]
fn main() {
    eprintln!("snapshot_paging_smoke needs --features cuda + rdma-core (metrale_rdma_verbs)");
    std::process::exit(1);
}
