// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Allocation tests for the two `METRALE_FP8_ROWWISE` GDN prefill
//! arms (`trait_prefill_proj.rs`'s `in_proj_qkvz`, `trait_prefill_helper.rs`'s
//! `out_proj`): their BF16 weights come from the arena slab, and no call
//! allocates.
//!
//! The tests stop at the dequant step (`rowwise_qkvz_bf16`,
//! `rowwise_out_proj_bf16`) and do not reach the cuBLASLt matmul after it:
//! other tests in this binary create real CUDA backends, and cuBLASLt given
//! the mock backend's fabricated device pointers could fault a live context.
//! `the_arms_refuse_to_run_when_the_ledger_entry_is_absent` runs both arms up
//! to their slab check.
//!
//! Owner: model-layers (qwen3_ssm tests).
//! Invariants: none beyond the types.

use super::tests::native_fp8_gdn_layer;
use super::*;
use crate::weight_map::Fp8Weight;
use crate::weight_map::WeightQuantFormat;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::{BufferArena, BufferSizes, ssm_rowwise_w_bf16_bytes_for};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

const M: u32 = 28;

/// 2026-09-25: A small GDN config: 4 layers (3 linear-attention, 1 full
/// attention), hidden 512, 4 key heads and 8 value heads of 64. Small because
/// `MockGpuBackend::alloc` backs every allocation with host memory; the
/// full-size slab arithmetic is checked without allocating in
/// `metrale_gpu_runtime::buffers::tests::rowwise_bf16_slab_is_sized_only_when_the_lever_is_armed`.
fn compact_gdn() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 512;
    c.num_hidden_layers = 4;
    c.linear_num_key_heads = 4;
    c.linear_key_head_dim = 64;
    c.linear_num_value_heads = 8;
    c.linear_value_head_dim = 64;
    c.full_attention_interval = 4;
    c.layer_types = (0..4)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c
}

/// 2026-09-25: A per-row FP8 weight, tagged `Fp8PerRow`, the only tag
/// `set_fp8_rowwise_prefill_weights` accepts.
fn fp8_per_row(gpu: &MockGpuBackend, n: usize, k: usize) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc(n * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    }
}

/// 2026-09-25: An arena sized as with `METRALE_FP8_ROWWISE=1`, without setting
/// the variable: the slab size comes from
/// `ssm_rowwise_w_bf16_bytes_for(config, true)`.
fn armed_arena(config: &ModelConfig, gpu: &MockGpuBackend) -> BufferArena {
    let mut sizes = BufferSizes::from_config(config, 64, 4096, 16, 32);
    sizes.ssm_rowwise_w_bf16 = ssm_rowwise_w_bf16_bytes_for(config, true);
    BufferArena::from_sizes(config, sizes, 64, 32, gpu).unwrap()
}

/// 2026-09-25: An arena with a zero-sized slab, as without the lever.
fn unarmed_arena(config: &ModelConfig, gpu: &MockGpuBackend) -> BufferArena {
    let mut sizes = BufferSizes::from_config(config, 64, 4096, 16, 32);
    sizes.ssm_rowwise_w_bf16 = 0;
    BufferArena::from_sizes(config, sizes, 64, 32, gpu).unwrap()
}

macro_rules! fwd_ctx {
    ($buffers:expr, $gpu:expr, $config:expr, $dispatch:expr, $derived:expr, $levers:expr, $stats:expr) => {
        ForwardContext {
            buffers: $buffers,
            hc_row_offset: 0,
            gpu: $gpu,
            config: $config,
            dispatch: $dispatch,
            derived: $derived,
            levers: $levers,
            stats: $stats,
            attn_metadata: None,
            decode_step: false,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Fold,
        }
    };
}

struct Fixture {
    config: ModelConfig,
    qkvz: Fp8Weight,
    out_proj: Fp8Weight,
}

fn fixture(gpu: &MockGpuBackend) -> (Fixture, Qwen3SsmLayer) {
    let config = compact_gdn();
    let mut layer = native_fp8_gdn_layer(gpu, &config, true, true);
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkvz = fp8_per_row(gpu, config.ssm_qkvz_size(), config.hidden_size);
    let out_proj = fp8_per_row(gpu, config.hidden_size, value_dim);
    layer.set_fp8_rowwise_prefill_weights(Some(qkvz), Some(out_proj));
    (
        Fixture {
            config,
            qkvz,
            out_proj,
        },
        layer,
    )
}

/// 2026-09-25: The first `rowwise_qkvz_bf16` call allocates nothing and
/// launches one dequant; the second returns the same slice and launches
/// nothing.
#[test]
fn rowwise_qkvz_bf16_weight_comes_from_the_ledger_not_a_fresh_alloc() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = armed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );

    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    let first = layer
        .rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0)
        .expect("first row-wise QKVZ bind");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the row-wise QKVZ arm allocated on its FIRST call — that is the \
         per-layer BF16 weight dequant from the #917 H100 receipt (167772160 B \
         on the 27B this fixture shrinks)"
    );
    // 2026-09-25: The launch count shows the call did work, so "allocated
    // nothing" cannot pass by doing nothing.
    assert_eq!(
        gpu.launch_count() - launches,
        1,
        "expected one dequant_fp8_blockscaled_bf16 into the ledgered slab"
    );
    let second = layer
        .rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0)
        .expect("second row-wise QKVZ bind");
    assert_eq!(first, second, "the slice is per layer and must not move");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes(), gpu.launch_count()),
        (allocs, bytes, launches + 1),
        "the second call must be a load, not a second dequant"
    );
}

/// 2026-09-25: The same for `rowwise_out_proj_bf16`, plus: the layer's two
/// slices do not overlap.
#[test]
fn rowwise_out_proj_bf16_weight_comes_from_the_ledger_not_a_fresh_alloc() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = armed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );

    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    let out = layer
        .rowwise_out_proj_bf16(&ctx, &fx.out_proj, 0)
        .expect("first row-wise out_proj bind");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the row-wise out_proj arm allocated on its FIRST call — the same \
         off-ledger BF16 weight dequant, [hidden, value_dim] x 2 B"
    );
    assert_eq!(gpu.launch_count() - launches, 1);
    layer
        .rowwise_out_proj_bf16(&ctx, &fx.out_proj, 0)
        .expect("second row-wise out_proj bind");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes(), gpu.launch_count()),
        (allocs, bytes, launches + 1)
    );

    // 2026-09-25: `out_proj` was carved first, so the QKVZ slice starts right
    // after it.
    let qkvz = layer
        .rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0)
        .expect("row-wise QKVZ bind");
    let out_bytes = ops::dequant_fp8_bf16_bytes(&fx.out_proj);
    assert_eq!(
        out_bytes,
        fx.out_proj.n as usize * fx.out_proj.k as usize * 2
    );
    assert_eq!(qkvz.0, out.0 + out_bytes as u64, "slices must not overlap");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "neither arm may allocate"
    );
}

/// 2026-09-25: The slab holds exactly `num_ssm_layers()` layer pairs; carving
/// one more returns an "exhausted" error.
#[test]
fn the_ledgered_slab_holds_exactly_every_gdn_layer_and_refuses_the_next() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = armed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );

    let layer_bytes =
        ops::dequant_fp8_bf16_bytes(&fx.qkvz) + ops::dequant_fp8_bf16_bytes(&fx.out_proj);
    assert_eq!(
        buffers.ssm_rowwise_w_bf16_bytes(),
        fx.config.num_ssm_layers() * layer_bytes
    );
    // 2026-09-25: The first pair is `layer`'s; the other layers' slices are
    // carved directly rather than through one layer each.
    let _ = layer.rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0).unwrap();
    let _ = layer.rowwise_out_proj_bf16(&ctx, &fx.out_proj, 0).unwrap();
    for _ in 1..fx.config.num_ssm_layers() {
        buffers.take_ssm_rowwise_w_bf16(layer_bytes).unwrap();
    }
    let over = buffers.take_ssm_rowwise_w_bf16(layer_bytes);
    assert!(
        over.is_err(),
        "one GDN layer past the model's count must be refused, not served \
         from a fresh allocation"
    );
    assert!(format!("{:#}", over.unwrap_err()).contains("exhausted"));
}

/// 2026-09-25: With no slab in the arena, both arms (`prefill_qkvz_proj`,
/// `prefill_out_proj_dispatch`) return the "slab is absent" error before the
/// matmul and allocate nothing.
#[test]
fn the_arms_refuse_to_run_when_the_ledger_entry_is_absent() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = unarmed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );
    let c = &fx.config;
    let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;

    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let qkvz = layer.prefill_qkvz_proj(
        buffers.norm_output(),
        buffers.ssm_deinterleaved(),
        M,
        c.ssm_qkvz_size(),
        c.hidden_size,
        c.linear_num_key_heads,
        c.linear_key_head_dim,
        c.linear_num_value_heads / c.linear_num_key_heads.max(1),
        c.linear_value_head_dim,
        &ctx,
        0,
    );
    let out = layer.prefill_out_proj_dispatch(
        &ctx,
        buffers.norm_output(),
        buffers.hidden_states(),
        M,
        c.hidden_size,
        value_dim,
        0,
    );
    for (what, r) in [("in_proj_qkvz", qkvz), ("out_proj", out)] {
        let e = format!("{:#}", r.expect_err(what));
        assert!(
            e.contains("row-wise") && e.contains("slab is absent"),
            "{what}: expected the missing-ledger-entry refusal, got: {e}"
        );
    }
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "a missing ledger entry must not be papered over with a fresh allocation"
    );
}
