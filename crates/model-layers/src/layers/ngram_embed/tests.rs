// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the n-gram embedding: the ids against the Python
//! fixture, the GPU output against checkpoint goldens, and the NVMe row
//! cache against a resident table.
//!
//! Owner: model-layers (n-gram embedding).
//! Invariants: none beyond the types.

use crate::layers::ngram_embed::*;
use crate::weight_map::DenseWeight;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

#[derive(serde::Deserialize)]
struct Fixture {
    name: String,
    vocab_size: u64,
    hidden_size: usize,
    ngram_vocab_size_ratio: u64,
    emb_neighbor_num: usize,
    emb_split_num: usize,
    eos_token_id: u32,
    tokens: Vec<u32>,
    expected_ids: std::collections::HashMap<String, Vec<u64>>,
}

#[test]
fn ids_match_python_reference() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/ngram_ref/ngram_id_fixtures.json"
    );
    let fixtures: Vec<Fixture> = serde_json::from_str(
        &std::fs::read_to_string(path)
            .expect("run bench/ngram_ref/make_fixtures.py to generate fixtures"),
    )
    .unwrap();
    assert!(!fixtures.is_empty());
    for f in fixtures {
        let dims = NgramDims {
            vocab_size: f.vocab_size,
            ratio: f.ngram_vocab_size_ratio,
            neighbor_num: f.emb_neighbor_num,
            split_num: f.emb_split_num,
            eos_token_id: f.eos_token_id,
            hidden_size: f.hidden_size,
        };
        let got = ngram_ids(&dims, &f.tokens);
        assert_eq!(got.len(), dims.num_tables(), "{}", f.name);
        for (index, ids) in got.iter().enumerate() {
            let want = &f.expected_ids[&index.to_string()];
            assert_eq!(ids, want, "{} table {index}", f.name);
        }
    }
}

/// 2026-09-25: GPU output against the golden vectors, on tables compacted to
/// the rows the fixture touches. The data directory comes from
/// `METRALE_NGRAM_TEST_DATA` (written by
/// `bench/ngram_ref/make_gpu_testdata.py`). Ignored by default: it needs a
/// GPU. Run with `METRALE_NGRAM_TEST_DATA=<dir> cargo test -p
/// metrale-model-layers --release ngram_gpu -- --ignored`.
#[test]
#[ignore]
fn ngram_gpu_matches_golden() {
    let dir = std::env::var("METRALE_NGRAM_TEST_DATA")
        .expect("set METRALE_NGRAM_TEST_DATA (see make_gpu_testdata.py)");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
            .unwrap();
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(
        0,
        &metrale_kernels::ptx_modules(),
    )
    .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;

    let toks: Vec<u32> = meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let hidden = meta["hidden_size"].as_u64().unwrap() as usize;
    let dims = NgramDims {
        vocab_size: meta["vocab_size"].as_u64().unwrap(),
        ratio: meta["ngram_vocab_size_ratio"].as_u64().unwrap(),
        neighbor_num: meta["emb_neighbor_num"].as_u64().unwrap() as usize,
        split_num: meta["emb_split_num"].as_u64().unwrap() as usize,
        eos_token_id: meta["eos_token_id"].as_u64().unwrap() as u32,
        hidden_size: hidden,
    };

    let upload = |bytes: &[u8]| -> DevicePtr {
        let p = g.alloc(bytes.len()).unwrap();
        g.copy_h2d_async(bytes, p, g.default_stream()).unwrap();
        p
    };
    let load = |path: String| std::fs::read(path).unwrap();

    // 2026-09-25: Compacted word table and remapped base token ids.
    let word_ids: Vec<u64> = meta["word_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let word = DenseWeight {
        weight: upload(&load(format!("{dir}/word_rows.bin"))),
    };
    let remap = |ids: &[u64], all: &[u64]| -> Vec<u32> {
        ids.iter()
            .map(|id| all.binary_search(id).unwrap() as u32)
            .collect()
    };
    let compact_toks = remap(
        &toks.iter().map(|&t| t as u64).collect::<Vec<_>>(),
        &word_ids,
    );

    // 2026-09-25: Compacted per-table rows and full projections. The ids
    // come from `ngram_ids` and are remapped to the compact row order, so
    // the id math is exercised too.
    let host_ids = ngram_ids(&dims, &toks);
    let mut tables = Vec::new();
    let mut projs = Vec::new();
    let mut compact_ids = Vec::new();
    let mut table_rows_n = Vec::new();
    for t in meta["tables"].as_array().unwrap() {
        let index = t["index"].as_u64().unwrap() as usize;
        assert_eq!(index, tables.len());
        let ids: Vec<u64> = t["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        tables.push(NgramTable::Bf16(DenseWeight {
            weight: upload(&load(format!("{dir}/table{index}_rows.bin"))),
        }));
        table_rows_n.push(ids.len());
        projs.push(DenseWeight {
            weight: upload(&load(format!("{dir}/proj{index}.bin"))),
        });
        compact_ids.push(remap(&host_ids[index], &ids));
    }

    // 2026-09-25: `embed()` computes the real ids itself and cannot take
    // compact ones, so `run` drives the same op sequence by hand on the
    // module's kernels and buffers.
    let ng = NgramEmbedding::new(dims, word, tables, projs, 32, g).unwrap();
    let stream = g.default_stream();
    let m = toks.len();
    let out = g.alloc(m * hidden * 2).unwrap();
    let inv = 1.0f32 / (1 + ng.dims.num_tables()) as f32;
    let td = ng.dims.table_dim();
    let put_ids = |ids: &[u32]| {
        let b: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d_async(&b, ng.ids_dev, stream).unwrap();
    };
    let run = |tables: &[NgramTable]| -> Vec<u8> {
        g.memset(out, 0, m * hidden * 2).unwrap();
        put_ids(&compact_toks);
        crate::layers::ops::batched_embed(
            g,
            ng.batched_embed_k,
            ng.ids_dev,
            ng.word.weight,
            ng.proj_buf,
            m as u32,
            hidden as u32,
            stream,
        )
        .unwrap();
        crate::layers::ops::scaled_add(
            g,
            ng.scaled_add_k,
            out,
            ng.proj_buf,
            inv,
            (m * hidden) as u32,
            stream,
        )
        .unwrap();
        for index in 0..ng.dims.num_tables() {
            put_ids(&compact_ids[index]);
            match &tables[index] {
                NgramTable::Bf16(w) => crate::layers::ops::batched_embed(
                    g,
                    ng.batched_embed_k,
                    ng.ids_dev,
                    w.weight,
                    ng.gather_buf,
                    m as u32,
                    td as u32,
                    stream,
                )
                .unwrap(),
                NgramTable::Fp8(w) => crate::layers::ops::batched_embed_fp8(
                    g,
                    ng.batched_embed_fp8_k,
                    ng.ids_dev,
                    w.weight,
                    w.row_scale,
                    ng.gather_buf,
                    m as u32,
                    td as u32,
                    stream,
                )
                .unwrap(),
                // 2026-09-25: The NVMe-backed variant has its own test
                // (`cached_table_matches_resident_table`).
                NgramTable::Cached(_) => unreachable!("test builds resident tables"),
            }
            crate::layers::ops::dense_gemm_bf16_pipelined(
                g,
                ng.gemm_k,
                ng.gather_buf,
                &ng.projs[index],
                ng.proj_buf,
                m as u32,
                hidden as u32,
                td as u32,
                stream,
            )
            .unwrap();
            crate::layers::ops::scaled_add(
                g,
                ng.scaled_add_k,
                out,
                ng.proj_buf,
                inv,
                (m * hidden) as u32,
                stream,
            )
            .unwrap();
        }
        g.synchronize(stream).unwrap();
        let mut got = vec![0u8; m * hidden * 2];
        g.copy_d2h(out, &mut got).unwrap();
        got
    };

    // 2026-09-25: The criterion is the worst per-row relative Frobenius
    // error against the f32 golden, not an elementwise bound: the GPU path
    // sums BF16-rounded contributions and the golden sums in f32.
    let golden = load(format!("{dir}/golden_f32.bin"));
    let worst_row = |got_bf16: &[u8]| -> f32 {
        let mut worst = 0f32;
        for r in 0..m {
            let mut err2 = 0f64;
            let mut ref2 = 0f64;
            for c in 0..hidden {
                let i = r * hidden + c;
                let bits = u16::from_le_bytes([got_bf16[i * 2], got_bf16[i * 2 + 1]]);
                let got = f32::from_bits((bits as u32) << 16) as f64;
                let want = f32::from_le_bytes([
                    golden[i * 4],
                    golden[i * 4 + 1],
                    golden[i * 4 + 2],
                    golden[i * 4 + 3],
                ]) as f64;
                err2 += (got - want) * (got - want);
                ref2 += want * want;
            }
            worst = worst.max((err2 / ref2).sqrt() as f32);
        }
        worst
    };

    let bf16_worst = worst_row(&run(&ng.tables));
    assert!(
        bf16_worst < 0.02,
        "BF16 fused embedding diverges from golden: worst row Frobenius \
         rel = {bf16_worst}"
    );
    println!("ngram GPU parity (BF16 tables): worst row Frobenius rel = {bf16_worst:.4}");

    // 2026-09-25: FP8 leg: quantize the compacted tables with
    // `NgramTable::quantize_bf16` and rerun, with a looser bound.
    let fp8_tables: Vec<NgramTable> = ng
        .tables
        .iter()
        .enumerate()
        .map(|(index, t)| {
            let NgramTable::Bf16(w) = t else {
                panic!("test built BF16 tables")
            };
            NgramTable::quantize_bf16(w, table_rows_n[index], td, g, stream).unwrap()
        })
        .collect();
    let fp8_worst = worst_row(&run(&fp8_tables));
    assert!(
        fp8_worst < 0.05,
        "FP8-quantized fused embedding diverges from golden: worst row \
         Frobenius rel = {fp8_worst}"
    );
    println!("ngram GPU parity (FP8 tables): worst row Frobenius rel = {fp8_worst:.4}");
}

/// 2026-09-25: A gather through the bounded NVMe row cache is byte-identical
/// to one from the resident table, with fewer slots than rows so that rows
/// are evicted and faulted in again. Uses the resident test's data; the
/// backing file is the table written row-major.
#[test]
#[ignore]
fn cached_table_matches_resident_table() {
    let dir = std::env::var("METRALE_NGRAM_TEST_DATA")
        .expect("set METRALE_NGRAM_TEST_DATA (see make_gpu_testdata.py)");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
            .unwrap();
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(
        0,
        &metrale_kernels::ptx_modules(),
    )
    .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    // 2026-09-25: Table 0's compacted rows, [n_rows, dim] BF16; the row id
    // is the compact index.
    let t0 = &meta["tables"].as_array().unwrap()[0];
    let n_rows = t0["ids"].as_array().unwrap().len();
    let dim = t0["dim"].as_u64().unwrap() as usize;
    let bytes = std::fs::read(format!("{dir}/table0_rows.bin")).unwrap();
    assert_eq!(bytes.len(), n_rows * dim * 2);
    let row_stride = dim * 2;

    let resident = g.alloc(bytes.len()).unwrap();
    g.copy_h2d_async(&bytes, resident, stream).unwrap();

    let tmp = std::env::temp_dir().join(format!("metrale_ngram_cache_{}.bin", std::process::id()));
    std::fs::write(&tmp, &bytes).unwrap();

    // 2026-09-25: Fewer slots than rows, so rows are evicted and faulted in
    // again.
    let slots = (n_rows / 3).max(4);
    let mut cache =
        metrale_storage::NgramRowCache::open(&tmp, None, n_rows as u64, row_stride, slots)
            .expect("open cache");

    // 2026-09-25: Every row twice, in batches of `slots / 2`: `resolve` pins
    // a batch's slots until `end_batch`, so a batch must fit in the slots.
    let ids: Vec<u64> = (0..n_rows as u64).chain(0..n_rows as u64).collect();
    let m = ids.len();
    let batch = slots / 2;
    let ids_dev = g.alloc(m * 4).unwrap();
    let out_res = g.alloc(m * dim * 2).unwrap();
    let out_cache = g.alloc(m * dim * 2).unwrap();

    let embed_k = g.kernel("embed_from_argmax", "batched_embed").unwrap();
    // 2026-09-25: Resident: gather by row id.
    let idb: Vec<u8> = ids.iter().flat_map(|v| (*v as u32).to_le_bytes()).collect();
    g.copy_h2d_async(&idb, ids_dev, stream).unwrap();
    crate::layers::ops::batched_embed(
        g, embed_k, ids_dev, resident, out_res, m as u32, dim as u32, stream,
    )
    .unwrap();

    // 2026-09-25: Cached: per batch, resolve to slots, then gather from the
    // arena into that batch's slice of the output.
    let table = DevicePtr(cache.table_dev_va().unwrap());
    let mut slot_ids = Vec::new();
    for (bi, chunk) in ids.chunks(batch).enumerate() {
        cache.resolve(chunk, &mut slot_ids).unwrap();
        assert_eq!(slot_ids.len(), chunk.len());
        let sb: Vec<u8> = slot_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d_async(&sb, ids_dev, stream).unwrap();
        let dst = out_cache.offset(bi * batch * dim * 2);
        crate::layers::ops::batched_embed(
            g,
            embed_k,
            ids_dev,
            table,
            dst,
            chunk.len() as u32,
            dim as u32,
            stream,
        )
        .unwrap();
        // 2026-09-25: Synchronize before `end_batch` releases the pins, so a
        // later fault cannot overwrite a slot this gather still reads.
        g.synchronize(stream).unwrap();
        cache.end_batch();
    }

    let mut a = vec![0u8; m * dim * 2];
    let mut b = vec![0u8; m * dim * 2];
    g.copy_d2h(out_res, &mut a).unwrap();
    g.copy_d2h(out_cache, &mut b).unwrap();
    let _ = std::fs::remove_file(&tmp);
    let (hits, misses, evictions) = cache.stats();
    assert!(
        evictions > 0,
        "test must exercise eviction (slots={slots}, rows={n_rows}); \
         hits={hits} misses={misses} evictions={evictions}"
    );
    assert_eq!(
        a, b,
        "cached gather diverged from the resident table \
         (hits={hits} misses={misses} evictions={evictions})"
    );
    println!(
        "ngram row cache: BYTE-IDENTICAL over {m} lookups with {slots}/{n_rows} slots \
         (hits={hits} misses={misses} evictions={evictions})"
    );
}

#[test]
fn decode_context_window_suffices() {
    // 2026-09-25: The last token's ids over the full sequence equal its ids
    // over only the trailing (n-1)-token window plus the new token.
    let dims = NgramDims {
        vocab_size: 997,
        ratio: 5,
        neighbor_num: 4,
        split_num: 2,
        eos_token_id: 2,
        hidden_size: 48,
    };
    let seq: Vec<u32> = vec![901, 15, 371, 2, 88, 990, 41, 7, 640, 3, 55];
    let full = ngram_ids(&dims, &seq);
    let keep = dims.neighbor_num - 1;
    let tail: Vec<u32> = seq[seq.len() - 1 - keep..].to_vec();
    let win = ngram_ids(&dims, &tail);
    for (index, ids) in full.iter().enumerate() {
        assert_eq!(
            ids.last(),
            win[index].last(),
            "window-decode diverges at table {index}"
        );
    }
}
