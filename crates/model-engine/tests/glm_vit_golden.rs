// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM-5.3-Flash vision tower (`GlmVit`) against the reference's
//! float32 golden outputs.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Runs `GlmVit::forward_batched` on the golden `pixel_values` of three images
//! (one non-square) and compares the merged embeddings per token. With
//! `METRALE_DUMP_GLM_VIT` set it also compares the patch-embed, block-0 and
//! block-23 taps, so a mismatch points at a stage.
//!
//! The tower runs in BF16 and the golden is float32. Measured 2026-09-22 on
//! GB10: the reference itself in bfloat16 reaches a min per-token cosine of only
//! 0.47-0.68 against its own float32 run at the last block. So the grading is
//! split:
//!
//! * The early taps (patch embed, block 0) must reach cosine >= `EARLY_MIN_COSINE`
//!   against float32.
//! * The last block and the merged output are compared with the `golden-bf16/`
//!   control (the reference run in bfloat16): the mean relative error against
//!   float32 may be at most `CONTROL_SLACK_REL` times the control's, and the mean
//!   cosine at most `CONTROL_SLACK_COS` below it. Without the control directory
//!   these stages are printed but not gated.
//!
//! Needs a CUDA GPU, the `glm-5.3-flash / nvfp4` kernel target and the extracted
//! vision weights. Run:
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! LIBRARY_PATH=$HOME/metrale-scratch/.ncclstub LD_LIBRARY_PATH=$LIBRARY_PATH \
//! METRALE_GLM_VISION=1 GLM_VIT_GOLDEN_GPU_ORDINAL=0 \
//! GLM_VIT_GOLDEN_DIR=/path/to/ws2-vision \
//! METRALE_DUMP_GLM_VIT=/tmp/glmvit-taps \
//! cargo test -p metrale-model-engine --test glm_vit_golden -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use metrale_config::VisionConfig;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::glm_vit::{GlmVit, GlmVitBlock, GlmVitGeometry, GlmVitMerger};

#[path = "glm_vit_golden/fixtures.rs"]
mod fixtures;
use fixtures::{Stats, bf16_bytes_to_f32, compare, ptr, read_npy_f32, upload_tower};

/// 2026-09-25: Minimum per-token cosine against float32 for the early taps.
/// Measured 2026-09-22: patch embed 0.99998, block 0 0.99995.
const EARLY_MIN_COSINE: f64 = 0.9999;

/// 2026-09-25: On the late stages, the mean relative error against float32 may be
/// at most this multiple of the bfloat16 control's.
const CONTROL_SLACK_REL: f64 = 1.30;
/// 2026-09-25: On the late stages, the mean per-token cosine may sit at most this
/// far below the control's.
const CONTROL_SLACK_COS: f64 = 0.02;

struct Image {
    name: &'static str,
    grid_h: usize,
    grid_w: usize,
}

const IMAGES: [Image; 3] = [
    Image {
        name: "a_noise_448x448",
        grid_h: 32,
        grid_w: 32,
    },
    Image {
        name: "b_geometric_448x448",
        grid_h: 32,
        grid_w: 32,
    },
    Image {
        name: "c_gradient_640x360",
        grid_h: 26,
        grid_w: 46,
    },
];

fn build_encoder(
    w: &HashMap<String, DevicePtr>,
    v: &VisionConfig,
    p_max_patches: usize,
    gpu: &dyn GpuBackend,
) -> Result<GlmVit> {
    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        let b = format!("blocks.{i}");
        blocks.push(GlmVitBlock {
            norm1_w: ptr(w, &format!("{b}.norm1.weight"))?,
            qkv_w: ptr(w, &format!("{b}.attn.qkv.weight"))?,
            qkv_b: ptr(w, &format!("{b}.attn.qkv.bias"))?,
            q_norm_w: ptr(w, &format!("{b}.attn.q_norm.weight"))?,
            k_norm_w: ptr(w, &format!("{b}.attn.k_norm.weight"))?,
            proj_w: ptr(w, &format!("{b}.attn.proj.weight"))?,
            proj_b: ptr(w, &format!("{b}.attn.proj.bias"))?,
            norm2_w: ptr(w, &format!("{b}.norm2.weight"))?,
            gate_w: ptr(w, &format!("{b}.mlp.gate_proj.weight"))?,
            gate_b: ptr(w, &format!("{b}.mlp.gate_proj.bias"))?,
            up_w: ptr(w, &format!("{b}.mlp.up_proj.weight"))?,
            up_b: ptr(w, &format!("{b}.mlp.up_proj.bias"))?,
            down_w: ptr(w, &format!("{b}.mlp.down_proj.weight"))?,
            down_b: ptr(w, &format!("{b}.mlp.down_proj.bias"))?,
        });
    }
    let merger = GlmVitMerger {
        post_layernorm_w: ptr(w, "post_layernorm.weight")?,
        downsample_w: ptr(w, "downsample.weight")?,
        downsample_b: ptr(w, "downsample.bias")?,
        proj_w: ptr(w, "merger.proj.weight")?,
        norm_w: ptr(w, "merger.post_projection_norm.weight")?,
        norm_b: ptr(w, "merger.post_projection_norm.bias")?,
        gate_w: ptr(w, "merger.gate_proj.weight")?,
        up_w: ptr(w, "merger.up_proj.weight")?,
        down_w: ptr(w, "merger.down_proj.weight")?,
    };
    let geo = GlmVitGeometry {
        hidden_size: v.hidden_size,
        num_heads: v.num_heads,
        intermediate_size: v.intermediate_size,
        spatial_merge_size: v.spatial_merge_size,
        out_hidden_size: v.out_hidden_size,
        projection_intermediate_size: v.projection_intermediate_size,
        patch_size: v.patch_size,
        temporal_patch_size: v.temporal_patch_size,
        rms_norm_eps: v.rms_norm_eps,
        swiglu_limit: v.swiglu_limit,
        rope_theta: v.rope_theta,
        max_pixels: Some(p_max_patches * v.patch_size * v.patch_size),
    };
    GlmVit::new(
        ptr(w, "patch_embed.proj.weight")?,
        ptr(w, "patch_embed.proj.bias")?,
        blocks,
        merger,
        &geo,
        gpu,
    )
}

fn golden_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("GLM_VIT_GOLDEN_DIR").expect("set GLM_VIT_GOLDEN_DIR (see module doc)"),
    )
}

#[test]
#[ignore = "requires an idle CUDA device, the compiled glm-5.3-flash target, and the extracted vision weights"]
fn glm_vision_tower_matches_the_transformers_golden() -> Result<()> {
    let ordinal: usize = std::env::var("GLM_VIT_GOLDEN_GPU_ORDINAL")
        .context("set GLM_VIT_GOLDEN_GPU_ORDINAL explicitly")?
        .parse()?;
    let root = golden_dir();
    let gold = root.join("golden");

    let cfg = metrale_config::parse_config(&std::fs::read_to_string(root.join("config.json"))?)
        .context("parse the checkpoint's own config.json")?;
    let v = cfg.vision.clone().context(
        "config.vision is None. Either METRALE_GLM_VISION=1 is unset — the tower is opt-in, and \
         off is the default so a certified text serve keeps its footprint — or the parser \
         branch regressed, which is under test here as much as the kernels are",
    )?;
    println!(
        "vision_config: depth={} hidden={} heads={} patch={} merge={} out={} proj_inter={} \
         eps={} swiglu_limit={} theta={} mean={:?} std={:?} block_major={}",
        v.depth,
        v.hidden_size,
        v.num_heads,
        v.patch_size,
        v.spatial_merge_size,
        v.out_hidden_size,
        v.projection_intermediate_size,
        v.rms_norm_eps,
        v.swiglu_limit,
        v.rope_theta,
        v.image_mean,
        v.image_std,
        v.block_major_patches
    );

    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("compile the glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;

    let weights = upload_tower(&root.join("shards/vision_combined.safetensors"), &gpu)?;
    // 2026-09-25: The three images have 3244 patches together. 3600 leaves room
    // without allocating for `FALLBACK_MAX_PATCHES` (6400); the score buffer
    // grows with the square of the patch count.
    let encoder = build_encoder(&weights, &v, 3600, &gpu)?;

    // 2026-09-25: Encode all three images in one `forward_batched` call, as the
    // prefill vision path does for several images.
    let mut pixels = Vec::new();
    for img in &IMAGES {
        let (shape, data) = read_npy_f32(&gold.join(format!("{}_pixel_values.npy", img.name)))?;
        ensure!(
            shape == vec![img.grid_h * img.grid_w, 1176],
            "{}: pixel_values shape {shape:?} does not match grid {}x{}",
            img.name,
            img.grid_h,
            img.grid_w
        );
        pixels.push(data);
    }
    let batch: Vec<(&[f32], usize, usize)> = IMAGES
        .iter()
        .zip(&pixels)
        .map(|(i, p)| (p.as_slice(), i.grid_h, i.grid_w))
        .collect();
    let per_image = encoder.forward_batched(&batch, &gpu, stream)?;
    gpu.synchronize(stream)?;

    let control_dir = root.join("golden-bf16");
    let have_control = control_dir.is_dir();
    if !have_control {
        println!(
            "\nWARNING: no golden-bf16/ control at {} — the compounded stages will be REPORTED \
             but not gated. See golden-bf16/README.md for how it is produced.",
            control_dir.display()
        );
    }

    let mut failures = Vec::new();
    let mut mp_off = 0usize;
    let mut p_off = 0usize;
    println!(
        "\n| image | stage | pair | rows | max-abs | mean-abs | mean-rel | min cos | mean cos |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");

    /// 2026-09-25: One stage of one image: print the pairings, then gate.
    macro_rules! grade {
        ($img:expr, $stage:expr, $early:expr, $actual:expr, $fp32:expr, $ctrl:expr, $rows:expr, $cols:expr) => {{
            let row = |pair: &str, s: &Stats| {
                println!(
                    "| {} | {} | {pair} | {} | {:.4e} | {:.4e} | {:.4e} | {:.6} | {:.6} |",
                    $img, $stage, $rows, s.max_abs, s.mean_abs, s.mean_rel, s.min_cos, s.mean_cos
                );
            };
            let a_vs_f32 = compare($actual, $fp32, $rows, $cols)?;
            row("metrale vs f32", &a_vs_f32);
            if $early {
                // 2026-09-25: Early stage: absolute gate against float32.
                if a_vs_f32.min_cos < EARLY_MIN_COSINE {
                    failures.push(format!(
                        "{} {}: min cosine {:.6} < {EARLY_MIN_COSINE} vs the fp32 reference",
                        $img, $stage, a_vs_f32.min_cos
                    ));
                }
            } else if let Some(ctrl) = $ctrl {
                let c_vs_f32 = compare(&ctrl, $fp32, $rows, $cols)?;
                let a_vs_c = compare($actual, &ctrl, $rows, $cols)?;
                row("torch-bf16 vs f32", &c_vs_f32);
                row("metrale vs torch-bf16", &a_vs_c);
                // 2026-09-25: Late stage: gate relative to the bfloat16 control.
                if a_vs_f32.mean_rel > c_vs_f32.mean_rel * CONTROL_SLACK_REL {
                    failures.push(format!(
                        "{} {}: mean relative error {:.4} exceeds {CONTROL_SLACK_REL}x the \
                         torch-bf16 control's {:.4}",
                        $img, $stage, a_vs_f32.mean_rel, c_vs_f32.mean_rel
                    ));
                }
                if a_vs_f32.mean_cos < c_vs_f32.mean_cos - CONTROL_SLACK_COS {
                    failures.push(format!(
                        "{} {}: mean cosine {:.6} is more than {CONTROL_SLACK_COS} below the \
                         torch-bf16 control's {:.6}",
                        $img, $stage, a_vs_f32.mean_cos, c_vs_f32.mean_cos
                    ));
                }
            }
        }};
    }

    for (i, img) in IMAGES.iter().enumerate() {
        let (post_h, post_w, merged_p) = per_image[i];
        ensure!(
            (post_h, post_w) == (img.grid_h / 2, img.grid_w / 2),
            "{}: post-merge grid {post_h}x{post_w}",
            img.name
        );

        // 2026-09-25: Intermediate taps first, so failures read in pipeline order.
        if let Ok(dir) = std::env::var("METRALE_DUMP_GLM_VIT") {
            let p = img.grid_h * img.grid_w;
            for (label, gfile, early) in [
                ("patch_embed", "tap_patch_embed", true),
                ("block00", "tap_block0", true),
                ("block23", "tap_block23", false),
            ] {
                let raw = std::fs::read(Path::new(&dir).join(format!("{label}.bin")))
                    .with_context(|| format!("tap dump {label}.bin"))?;
                let all = bf16_bytes_to_f32(&raw);
                let slice = &all[p_off * v.hidden_size..(p_off + p) * v.hidden_size];
                let (shape, expect) =
                    read_npy_f32(&gold.join(format!("{}_{gfile}.npy", img.name)))?;
                ensure!(
                    shape == vec![p, v.hidden_size],
                    "{} {label}: shape {shape:?}",
                    img.name
                );
                let ctrl = match have_control {
                    true => Some(
                        read_npy_f32(&control_dir.join(format!("{}_{label}.npy", img.name)))?.1,
                    ),
                    false => None,
                };
                grade!(
                    img.name,
                    label,
                    early,
                    slice,
                    &expect,
                    ctrl,
                    p,
                    v.hidden_size
                );
            }
        }

        // 2026-09-25: The merged output rows.
        let mut bytes = vec![0u8; merged_p * v.out_hidden_size * 2];
        gpu.copy_d2h(encoder.out_row(mp_off), &mut bytes)?;
        let actual = bf16_bytes_to_f32(&bytes);
        let (shape, expect) = read_npy_f32(&gold.join(format!("{}_merged_embeds.npy", img.name)))?;
        ensure!(
            shape == vec![merged_p, v.out_hidden_size],
            "{}: golden merged shape {shape:?} vs {merged_p}x{}",
            img.name,
            v.out_hidden_size
        );
        let ctrl = match have_control {
            true => Some(read_npy_f32(&control_dir.join(format!("{}_merged.npy", img.name)))?.1),
            false => None,
        };
        grade!(
            img.name,
            "merged",
            false,
            &actual,
            &expect,
            ctrl,
            merged_p,
            v.out_hidden_size
        );

        mp_off += merged_p;
        p_off += img.grid_h * img.grid_w;
    }

    if !failures.is_empty() {
        bail!(
            "GLM vision tower does not match the reference:\n  {}",
            failures.join("\n  ")
        );
    }
    println!(
        "\nPASS: 3 images, {mp_off} merged tokens. Early taps >= {EARLY_MIN_COSINE} vs fp32; \
         compounded stages within the torch-bf16 control."
    );
    Ok(())
}
