#!/usr/bin/env python3

# 2026-09-26: Final-logit overlap of the L39 residual, Metrale Engine FP8 versus HF BF16: top-k Jaccard, KL, argmax.
#
# Owner: bench, FP8 drift investigation.
# Invariants: none beyond the types.
"""C1 (2026-05-26) — top-K final-logit overlap, Metrale Engine FP8 vs HF BF16.

Uses the existing per-layer hidden-state dumps at
/workspace/metrale-dumps/fp8native_dgx2/ to compute the FINAL token-level
logit distribution under two precision regimes and quantify divergence
at the level that actually drives generation (token selection), not
just hidden-state cosine.

Pipeline:
  h_L39 (from each source)
  -> apply final RMSNorm with HF BF16 norm weights
  -> matmul against HF BF16 lm_head.weight
  -> softmax in float32
  -> compare top-K (K=1, 5, 10, 50, 200) by Jaccard
  -> compute KL(BF16 || FP8) and top-1 agreement

Single-prompt summary (the canonical 10382-token chat probe, last-token slice).

Outputs to bench/fp8_dgx2_drift/c1_final_logit_overlap.json and prints a
short report.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

SNAP = pathlib.Path(
    "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DUMP_DIR = pathlib.Path("/workspace/metrale-dumps/fp8native_dgx2")
OUT = pathlib.Path(__file__).resolve().parent / "c1_final_logit_overlap.json"

# 2026-09-26: L39 is the last of 40 layers. Only the dumped hidden vectors and the
# final-norm and lm_head weights are read; no model is loaded.
HIDDEN_SIZE = 2048
LAST_LAYER = 39
RMS_NORM_EPS = 1.0e-6


def load_config_rms_eps() -> float:
    cfg = json.load(open(SNAP / "config.json"))
    tc = cfg.get("text_config", cfg)
    eps = tc.get("rms_norm_eps", 1e-6)
    return float(eps)


def load_final_norm_weight() -> np.ndarray:
    """RMS-norm gamma for the final norm. Stored in shard 26."""
    shard = SNAP / "model-00026-of-00026.safetensors"
    # 2026-09-26: NumPy has no bfloat16 dtype, so read with torch and cast to f32.
    with safe_open(str(shard), framework="pt") as f:
        for key in f.keys():
            if key.endswith("model.language_model.norm.weight"):
                t = f.get_tensor(key)
                return t.to(torch.float32).cpu().numpy()
    raise RuntimeError("final norm weight not found in shard 26")


def load_lm_head_weight() -> np.ndarray:
    """lm_head [vocab, hidden] in BF16 → cast to f32 for stable matmul."""
    shard = SNAP / "model-00026-of-00026.safetensors"
    with safe_open(str(shard), framework="pt") as f:
        for key in f.keys():
            if key == "lm_head.weight":
                t = f.get_tensor(key)
                return t.to(torch.float32).cpu().numpy()
    raise RuntimeError("lm_head.weight not found in shard 26")


def rms_norm(h: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    h = h.astype(np.float32)
    rms = np.sqrt(np.mean(h * h) + eps)
    return (h / rms) * gamma.astype(np.float32)


def load_hidden(path: pathlib.Path) -> np.ndarray:
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype="<f4").copy()
    assert arr.size == HIDDEN_SIZE, f"{path}: expected {HIDDEN_SIZE}, got {arr.size}"
    return arr


def topk(logits: np.ndarray, k: int) -> set[int]:
    if k >= logits.size:
        return set(range(logits.size))
    idx = np.argpartition(-logits, k - 1)[:k]
    return set(idx.tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def softmax_f64(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    # 2026-09-26: KL(p || q) in nats, over the entries where p > 1e-20.
    mask = p > 1e-20
    pp = p[mask]
    qq = np.clip(q[mask], 1e-30, None)
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def main() -> None:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading rms eps + final-norm gamma + lm_head", flush=True)
    eps = load_config_rms_eps()
    gamma = load_final_norm_weight()
    lm_head = load_lm_head_weight()
    print(f"  eps={eps}", flush=True)
    print(f"  gamma.shape={gamma.shape}", flush=True)
    print(f"  lm_head.shape={lm_head.shape} (vocab x hidden, f32)", flush=True)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    h_metrale = load_hidden(DUMP_DIR / f"metrale_L{LAST_LAYER}.bin")
    h_bf16 = load_hidden(DUMP_DIR / f"hf_bf16_L{LAST_LAYER}.bin")
    print(f"\nh_metrale: norm={np.linalg.norm(h_metrale):.4f}  max={h_metrale.max():.4f}", flush=True)
    print(f"h_bf16:  norm={np.linalg.norm(h_bf16):.4f}  max={h_bf16.max():.4f}", flush=True)

    cos = float(np.dot(h_metrale, h_bf16) / (np.linalg.norm(h_metrale) * np.linalg.norm(h_bf16)))
    print(f"residual cos(metrale, bf16) at L{LAST_LAYER}: {cos:.5f}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] computing logits via lm_head matmul ...", flush=True)
    z_metrale = rms_norm(h_metrale, gamma, eps)
    z_bf16 = rms_norm(h_bf16, gamma, eps)
    logits_metrale = lm_head @ z_metrale.astype(np.float32)
    logits_bf16 = lm_head @ z_bf16.astype(np.float32)
    print(f"  logits.shape={logits_metrale.shape}", flush=True)

    logit_cos = float(
        np.dot(logits_metrale, logits_bf16)
        / (np.linalg.norm(logits_metrale) * np.linalg.norm(logits_bf16))
    )

    arg_metrale = int(np.argmax(logits_metrale))
    arg_bf16 = int(np.argmax(logits_bf16))
    top1_agree = arg_metrale == arg_bf16

    results = {
        "residual_cos_L39": cos,
        "logit_cos": logit_cos,
        "argmax_metrale_token": arg_metrale,
        "argmax_bf16_token": arg_bf16,
        "top1_agree": top1_agree,
        "topk_jaccard": {},
        "kl_bf16_vs_metrale": None,
        "kl_metrale_vs_bf16": None,
    }

    for k in (1, 5, 10, 50, 200, 1000):
        a = topk(logits_metrale, k)
        b = topk(logits_bf16, k)
        j = jaccard(a, b)
        results["topk_jaccard"][str(k)] = j
        print(f"  top-{k:<5d} jaccard(metrale, bf16): {j:.4f}", flush=True)

    p_metrale = softmax_f64(logits_metrale)
    p_bf16 = softmax_f64(logits_bf16)
    kl_metrale_vs_bf16 = kl_div(p_metrale, p_bf16)
    kl_bf16_vs_metrale = kl_div(p_bf16, p_metrale)
    results["kl_metrale_vs_bf16"] = kl_metrale_vs_bf16
    results["kl_bf16_vs_metrale"] = kl_bf16_vs_metrale

    print(f"\n=== summary ===", flush=True)
    print(f"  residual cos L39       : {cos:.5f}", flush=True)
    print(f"  logit cos              : {logit_cos:.5f}", flush=True)
    print(f"  argmax(metrale)          : {arg_metrale}", flush=True)
    print(f"  argmax(bf16 ref)       : {arg_bf16}", flush=True)
    print(f"  top1 agree             : {top1_agree}", flush=True)
    print(f"  KL(metrale || bf16)      : {kl_metrale_vs_bf16:.4f} nats", flush=True)
    print(f"  KL(bf16  || metrale)     : {kl_bf16_vs_metrale:.4f} nats", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
