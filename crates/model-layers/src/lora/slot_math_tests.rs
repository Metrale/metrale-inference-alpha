// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the LoRA slot math: pool layout and offsets, the
//! per-step `seq_slot` build, scale-table values, victim selection, and
//! routed-prefill slot and pair selection.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use metrale_config::{LayerType, PeftAdapterConfig};
use metrale_model_weights::weights::WeightStore;

use crate::lora::test_support::*;
use crate::lora::*;

#[test]
fn slot_base_is_k_times_slot_bytes() {
    let cfg = cfg();
    let mr = 16;
    let sb = pool_slot_bytes(&cfg, mr);
    for k in 0..8 {
        assert_eq!(slot_base_offset(k, &cfg, mr), k * sb);
    }
}

#[test]
fn pool_slot_bytes_absolute_golden() {
    // 2026-09-25: An absolute byte count, so a change to any module's dims or
    // padding fails here. On the factory config at max_rank 16 it is q/k/v/o
    // on the 12 full-attention layers plus out_proj on the other 36; gate/up/
    // down reserve nothing because the config has routed experts.
    let cfg = cfg();
    assert_eq!(pool_slot_bytes(&cfg, 16), 15_335_424);
}

#[test]
fn module_offsets_walk_matches_pack_loop_and_fill_exactly_one_slot() {
    // 2026-09-25: Repeats `pack_slot`'s A-then-B walk and checks
    // `module_slot_offsets` at every step, and that the walk ends exactly at
    // `pool_slot_bytes`.
    let cfg = cfg();
    let mr = 16;
    let mut off = 0usize;
    for layer in 0..cfg.num_hidden_layers {
        for module in LoraModule::ALL {
            if !module.applies_to_layer(&cfg, layer) {
                assert_eq!(
                    module_slot_offsets(&cfg, mr, layer, module),
                    None,
                    "layer {layer} {module:?} is not applicable and must have no slot"
                );
                continue;
            }
            let (out, inp) = module.dims(&cfg);
            let a_off = off;
            let b_off = off + mr * inp * BF16_BYTES;
            off = b_off + out * mr * BF16_BYTES;
            assert_eq!(
                module_slot_offsets(&cfg, mr, layer, module),
                Some((a_off, b_off)),
                "layer {layer} {module:?}"
            );
            assert!(a_off < b_off, "A precedes B within a module region");
        }
    }
    assert_eq!(
        off,
        pool_slot_bytes(&cfg, mr),
        "one pass fills exactly one slot"
    );
}

#[test]
fn module_offsets_none_for_non_full_attention_layer() {
    let cfg = cfg();
    assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
    assert_eq!(module_slot_offsets(&cfg, 16, 0, LoraModule::KProj), None);
}

#[test]
fn slot_boundaries_do_not_overlap() {
    let cfg = cfg();
    let mr = 16;
    let sb = pool_slot_bytes(&cfg, mr);
    // 2026-09-25: The last packed module ends flush against slot 1's base.
    // Which module is last depends on the last layer's type and the config,
    // so it is derived, not hardcoded.
    let (last_layer, last_module) = (0..cfg.num_hidden_layers)
        .flat_map(|l| LoraModule::ALL.iter().map(move |m| (l, *m)))
        .rfind(|(l, m)| m.applies_to_layer(&cfg, *l))
        .expect("some module is packed");
    let (_, b_off) = module_slot_offsets(&cfg, mr, last_layer, last_module).unwrap();
    let (out, _) = last_module.dims(&cfg);
    assert_eq!(b_off + out * mr * BF16_BYTES, sb);
    assert_eq!(slot_base_offset(1, &cfg, mr), sb);
}

#[test]
fn scale_table_values_per_slot_and_padded() {
    // 2026-09-25: One f32 per slot in slot order: alpha/r, or alpha/sqrt(r)
    // with rsLoRA, and 0.0 for slots with no adapter.
    let store = WeightStore::empty();
    let mk = |alpha: f64, r: usize, rslora: bool| LoraAdapterInput {
        name: String::new(),
        store: &store,
        peft: PeftAdapterConfig {
            r,
            lora_alpha: alpha,
            target_modules: vec!["k_proj".into()],
            target_modules_pattern: None,
            use_rslora: rslora,
            layers_to_transform: None,
            trainable_token_indices: Vec::new(),
            modules_to_save: Vec::new(),
            lora_embedding: false,
        },
    };
    let adapters = [mk(16.0, 8, false), mk(16.0, 4, true)];
    let v = scale_table_values(&adapters, 8);
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], (16.0_f64 / 8.0) as f32);
    assert_eq!(v[1], (16.0_f64 / (4.0_f64).sqrt()) as f32);
    assert!(v[2..].iter().all(|&s| s == 0.0));
}

#[test]
fn seq_slot_host_defers_negatives_and_pads() {
    // 2026-09-25: Rows on slots 1 and 0, one row deferring (-1 resolves to the
    // active slot 2), padded to 4 rows with -1.
    let slots = [1i32, -1, 0];
    let v = build_seq_slot_host(&slots, 4, 2);
    assert_eq!(v, vec![1, 2, 0, -1]);
}

#[test]
fn seq_slot_uniform_prefill_fills_and_resolves() {
    // 2026-09-25: The model's `upload_seq_slot_uniform` passes `count` copies
    // of one request's slot; every row must resolve the same way.
    for &count in &[1usize, 4, 32] {
        let v = build_seq_slot_host(&vec![3i32; count], count, 7);
        assert_eq!(v, vec![3i32; count], "count={count} explicit slot B");
        let v = build_seq_slot_host(&vec![-1i32; count], count, 5);
        assert_eq!(v, vec![5i32; count], "count={count} deferred → active");
        let v = build_seq_slot_host(&vec![0i32; count], count, 2);
        assert_eq!(v, vec![0i32; count], "count={count} slot 0");
    }
}

#[test]
fn victim_free_first_before_lru() {
    // 2026-09-25: The placeholder (slot 3) is chosen ahead of idle filled
    // slots with older ticks.
    let cache = vec![
        (2, view(true, 0, 1)),
        (3, view(false, 0, 99)),
        (4, view(true, 0, 9)),
    ];
    assert_eq!(select_victim_slot(&cache), Ok(3));
}

#[test]
fn victim_lru_idle_when_all_filled() {
    // 2026-09-25: With no placeholder, the idle slot with the smallest tick
    // (slot 4) is chosen; the busy slot 3 is skipped although its tick is
    // older.
    let cache = vec![
        (2, view(true, 0, 50)),
        (3, view(true, 1, 5)),
        (4, view(true, 0, 12)),
    ];
    assert_eq!(select_victim_slot(&cache), Ok(4));
}

#[test]
fn victim_pool_full_when_all_busy() {
    let cache = vec![(2, view(true, 1, 1)), (3, view(true, 2, 2))];
    assert_eq!(select_victim_slot(&cache), Err(VictimError::PoolFull));
}

#[test]
fn routed_prefill_slot_predicate() {
    assert_eq!(
        routed_prefill_slot_of(-1, 0, 2),
        None,
        "-1 defers to active"
    );
    assert_eq!(
        routed_prefill_slot_of(0, 0, 2),
        None,
        "names the active slot"
    );
    assert_eq!(
        routed_prefill_slot_of(1, 0, 2),
        Some(1),
        "routes to a non-active slot"
    );
    assert_eq!(routed_prefill_slot_of(5, 0, 2), None, "out of range");
    assert_eq!(routed_prefill_slot_of(0, 1, 2), Some(0));
    assert_eq!(routed_prefill_slot_of(1, 1, 2), None);
    assert_eq!(routed_prefill_slot_of(-1, 1, 2), None);
}

#[test]
fn select_routed_pair_by_global_index_and_module() {
    // 2026-09-25: Only global layer 3 is adapted, with a distinct pair per
    // module, so a wrong layer or module selection returns another pair or
    // none.
    let mut layers: Vec<Option<LoraLayerWeights>> = (0..8).map(|_| None).collect();
    let k = dummy_pair(100, 1024, 512);
    let v = dummy_pair(200, 1024, 512);
    let o = dummy_pair(300, 2048, 1024);
    layers[3] = Some(LoraLayerWeights {
        k_proj: Some(k),
        v_proj: Some(v),
        o_proj: Some(o),
        ..LoraLayerWeights::empty(3)
    });

    assert_eq!(
        select_routed_pair(&layers, 3, LoraModule::KProj).map(|p| p.a.weight.0),
        Some(100)
    );
    assert_eq!(
        select_routed_pair(&layers, 3, LoraModule::VProj).map(|p| p.a.weight.0),
        Some(200)
    );
    assert_eq!(
        select_routed_pair(&layers, 3, LoraModule::OProj).map(|p| p.a.weight.0),
        Some(300)
    );

    assert!(select_routed_pair(&layers, 3, LoraModule::GateProj).is_none());
    assert!(select_routed_pair(&layers, 2, LoraModule::KProj).is_none());
    assert!(select_routed_pair(&layers, 99, LoraModule::KProj).is_none());
}
