// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: End-to-end gate for the GLM-5.3 KDA layer (`glm5next_kda::Glm5NextKdaLayer`),
//! against a CPU reference and the golden `gen_kda_layer_golden.py` writes.
//!
//! Owner: model-arch examples (GLM-5.3 KDA layer).
//! Invariants:
//! - The run fails if a packet does not classify as a KDA block, if the KDA blocks do not share
//!   one (name, dtype, shape) signature, if binding does not account for every tensor, or if
//!   any suite fails (exit status 1).
//!
//! Three parts:
//! 1. Synthetic: LCG weights at the golden's geometry through the regimes `decode1`,
//!    `prefill4`, `prefill7` and `prefill7_decode1`, GPU against the CPU reference only.
//! 2. Binding: every KDA block listed in `$KDA_PACKET_DIR/kda_family_audit.json` is bound
//!    with `binding::bind_kda_weights`.
//! 3. Execution: the layers in `EXEC_LAYERS`, on their checkpoint weights, against the golden.
//!
//! Each suite ends with a pad-corruption check: a ragged prefill whose padded tail is filled
//! with 7.5, then a decode. The outputs, the carried state, the conv state and the next decode
//! must all equal the clean run's.
//!
//! Floors are reported per stage, so a residual is never read as a rounding budget:
//! * A: the CPU reference in FP32 against the FP32 golden;
//! * B: the BF16 golden against the FP32 golden;
//! * D: the GPU against the CPU reference that follows the GPU's dtype ladder;
//! * E: the final output, GPU against the BF16 golden, gated with the carried state at 8x
//!   floor B or 1% of the stage's magnitude.
//!
//! The GPU writes the L2-normalised q|k as BF16 (fused in the decode conv, `l2_norm_bf16` on
//! prefill), so the floor-D reference rounds there as well.
//!
//!   KDA_PACKET_DIR=/path/to/packets \
//!   cargo run -p metrale-model-arch --release --example kda_layer_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::binding::{self, AttnBlockKind, KdaTensorSource};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace,
};
use serde_json::Value;

#[path = "common/kda_layer_cpu.rs"]
pub(crate) mod kda_layer_cpu;
use kda_layer_cpu::*;

#[path = "common/kda_layer_harness.rs"]
pub(crate) mod kda_layer_harness;
use kda_layer_harness::*;

#[path = "common/kda_layer_report.rs"]
pub(crate) mod kda_layer_report;
use kda_layer_report::*;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_kda_ref/kda_layer_golden.json",
        "gen_kda_layer_golden.py",
    )
});

/// 2026-09-25: Chunk width for the prefill scan. At D = 128, `C = 64` needs 81,920 B of shared
/// memory for the scan, above the `SMEM_CEILING` that `Glm5NextKdaConfig` refuses past, so 32
/// is the widest chunk it accepts.
pub(crate) const CHUNK: usize = 32;

pub(crate) fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

pub(crate) struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
    fn scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.u() * s).collect()
    }
}

pub(crate) fn r(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
/// 2026-09-25: BF16 rounding of an intermediate value, switched off in the `pure` pass that
/// measures floor A. The CPU reference applies it only to intermediates, not to weights.
pub(crate) fn rq(x: f32, pure: bool) -> f32 {
    if pure { x } else { r(x) }
}
pub(crate) fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| r(*x)).collect()
}
pub(crate) fn sample(v: &[f32], stride: usize) -> Vec<f32> {
    v.iter().step_by(stride).copied().collect()
}
pub(crate) fn maxabs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}
pub(crate) fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let v: Value = serde_json::from_str(&GOLDEN)?;
    let f = &v["fixture"];
    let dm = Dims {
        hid: f["hidden"].as_u64().unwrap() as usize,
        h: f["heads"].as_u64().unwrap() as usize,
        d: f["head_dim"].as_u64().unwrap() as usize,
        ks: f["kernel"].as_u64().unwrap() as usize,
    };
    let cfg = Glm5NextKdaConfig {
        hidden: dm.hid,
        heads: dm.h,
        head_dim: dm.d,
        conv_kernel: dm.ks,
        gate_lower_bound: f["lower_bound"].as_f64().unwrap() as f32,
        rms_norm_eps: f["rms_eps"].as_f64().unwrap() as f32,
        l2_eps: f["l2_eps"].as_f64().unwrap() as f32,
        chunk: CHUNK,
    };

    println!("GLM-5.3-Flash KDA layer family — Metrale Engine vs HF transformers 5.16.1");
    println!("  checkpoint {}", f["checkpoint"]);
    println!(
        "  hidden={} heads={} head_dim={} conv_dim={} kernel={} act={} o_norm_act={}",
        dm.hid,
        dm.h,
        dm.d,
        dm.conv_dim(),
        dm.ks,
        f["hidden_act"],
        f["o_norm_act"]
    );
    println!(
        "  READ from config: gate_lower_bound={} rms_norm_eps={:e} (never defaulted)",
        cfg.gate_lower_bound, cfg.rms_norm_eps
    );
    println!(
        "  Metrale Engine chunk C={CHUNK} (smem ceiling), HF chunk C={}",
        f["hf_chunk"]
    );

    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    if Lcg(0x5EED_1A70)
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("LCG mismatch with the generator");
    }
    println!("  LCG parity with the generator: ok");

    let kernels = Glm5NextKdaKernels::resolve(gpu)?;
    println!(
        "  {} kernel entry points resolved (no fallback path)",
        Glm5NextKdaKernels::ENTRY_POINTS
    );
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, 8)?;

    let mut ok = true;

    println!("\n=== PART 1 — synthetic full-layer oracle (LCG weights, production geometry) ===");
    let mut wr = Lcg(0xA11A_5000);
    let wts = Wts {
        q: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        k: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        v: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        conv: round_bf16(&wr.scaled(dm.conv_dim() * dm.ks, 0.5)),
        f_a: round_bf16(&wr.scaled(dm.d * dm.hid, 0.02)),
        f_b: round_bf16(&wr.scaled(dm.qkv() * dm.d, 0.05)),
        dt_bias: wr.scaled(dm.qkv(), 0.5),
        a_log: wr.scaled(dm.h, 0.5),
        b: round_bf16(&wr.scaled(dm.h * dm.hid, 0.02)),
        g_a: round_bf16(&wr.scaled(dm.d * dm.hid, 0.02)),
        g_b: round_bf16(&wr.scaled(dm.qkv() * dm.d, 0.05)),
        o_norm: round_bf16(&wr.scaled(dm.d, 1.0)),
        o: round_bf16(&wr.scaled(dm.hid * dm.qkv(), 0.02)),
    };
    let syn = Glm5NextKdaWeights {
        q_proj: dwt(gpu, &wts.q)?,
        k_proj: dwt(gpu, &wts.k)?,
        v_proj: dwt(gpu, &wts.v)?,
        conv: dwt(gpu, &wts.conv)?,
        f_a: dwt(gpu, &wts.f_a)?,
        f_b: dwt(gpu, &wts.f_b)?,
        dt_bias: up_f32(gpu, &wts.dt_bias)?,
        a_log: up_f32(gpu, &wts.a_log)?,
        b_proj: dwt(gpu, &wts.b)?,
        g_a: dwt(gpu, &wts.g_a)?,
        g_b: dwt(gpu, &wts.g_b)?,
        o_norm: dwt(gpu, &wts.o_norm)?,
        o_proj: dwt(gpu, &wts.o)?,
    };
    ok &= run_suite(
        gpu,
        Glm5NextKdaLayer::new(usize::MAX, cfg, syn, kernels)?,
        &ws,
        dm,
        cfg,
        &wts,
        None,
        "synthetic",
    )?;

    let dir = std::env::var("KDA_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/metrale-scratch/kda-family".to_string());
    let audit: Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{dir}/kda_family_audit.json"
    ))?)?;
    let kda_layers: Vec<usize> = audit["kda_layers"]
        .as_array()
        .context("audit has no kda_layers")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();

    println!(
        "\n=== PART 2 — typed binding of ALL {} KDA blocks ===",
        kda_layers.len()
    );
    println!("  packets {dir}");
    let mut family: Vec<(usize, Glm5NextKdaLayer)> = Vec::new();
    let (mut tot_bound, mut tot_unknown, mut tot_bytes, mut tot_nonattn) =
        (0usize, 0usize, 0usize, 0usize);
    let mut sig: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for &l in &kda_layers {
        let pkt = Packet::open(&format!("{dir}/layer{l}.safetensors"))?;
        let kind = binding::classify_attn_block(&pkt.names());
        if kind != AttnBlockKind::Kda {
            bail!("layer {l} classifies as {kind:?}, not Kda — refusing to bind it as KDA");
        }
        // 2026-09-25: The (name, dtype, shape) signature of each block, so uniformity across
        // the blocks is checked rather than assumed.
        let mut names: Vec<String> = pkt
            .names()
            .into_iter()
            .filter(|n| n.starts_with("self_attn."))
            .map(|n| {
                let t = pkt.get(&n).unwrap();
                format!("{n}:{}:{:?}", t.dtype.name(), t.shape)
            })
            .collect();
        names.sort();
        sig.entry(names.join("|")).or_default().push(l);

        let (w, rep) = binding::bind_kda_weights(gpu, &cfg, l, &pkt)?;
        tot_bound += rep.bound;
        tot_unknown += rep.unknown_self_attn.len();
        tot_bytes += rep.bytes;
        tot_nonattn += rep.non_attn_seen;
        family.push((l, Glm5NextKdaLayer::new(l, cfg, w, kernels)?));
    }
    println!(
        "  bound {tot_bound}/{} self_attn tensors across {} layers · UNKNOWN {tot_unknown} · \
         silent skips 0 · {:.2} GiB · {tot_nonattn} non-attn tensors seen and deliberately NOT bound",
        binding::KDA_TENSORS.len() * kda_layers.len(),
        kda_layers.len(),
        tot_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  distinct (name, dtype, shape) signatures across the family: {}",
        sig.len()
    );
    for ls in sig.values() {
        println!("    one signature covers {} layers: {:?}", ls.len(), ls);
    }
    if sig.len() != 1 {
        bail!(
            "the KDA family is NOT uniform — {} distinct signatures; STOP",
            sig.len()
        );
    }
    if tot_bound != binding::KDA_TENSORS.len() * kda_layers.len() || tot_unknown != 0 {
        bail!("binding accounting failed");
    }

    println!("\n=== PART 3 — real-weight execution: KDA layers {EXEC_LAYERS:?} ===");
    for &l in EXEC_LAYERS {
        if !kda_layers.contains(&l) {
            bail!("layer {l} is not a KDA layer");
        }
        let pkt = Packet::open(&format!("{dir}/layer{l}.safetensors"))?;
        let (w, _) = binding::bind_kda_weights(gpu, &cfg, l, &pkt)?;
        let host = host_weights(&pkt);
        // 2026-09-25: The golden holds one set of regimes per layer; pass `run_suite` only
        // this layer's.
        let lv = serde_json::json!({ "regimes": v["by_layer"][l.to_string()] });
        if lv["regimes"].is_null() {
            bail!("golden has no block for layer {l}");
        }
        ok &= run_suite(
            gpu,
            Glm5NextKdaLayer::new(l, cfg, w, kernels)?,
            &ws,
            dm,
            cfg,
            &host,
            Some(&lv),
            &format!("layer{l}"),
        )?;
    }

    println!(
        "\n  family instantiated: {} bound KDA blocks share one workspace and one kernel set",
        family.len()
    );
    println!(
        "\n{}",
        if ok {
            "RESULT: PASS — the reusable KDA layer executes every tested block at the bf16 floor"
        } else {
            "RESULT: FAIL"
        }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
