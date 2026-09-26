#!/usr/bin/env python3

# 2026-09-26: Per-layer cosine of last-token residuals, Metrale Engine FP8 versus vLLM FP8.
#
# Owner: bench, FP8 drift investigation.
# Invariants: none beyond the types.
"""Apples-to-apples per-layer cosine: Metrale-FP8 vs vLLM-FP8.

Both engines run the SAME FP8 model (Qwen3.6-35B-A3B-FP8) on the SAME ~10378-token
prompt. vLLM passes the opencode harness 10/10; Metrale Engine drifts. This finds the layer
where Metrale Engine's residual stream first diverges from vLLM's (the FP8 implementation gap,
isolated from FP8 quant noise itself — both engines have the same quant noise).

Inputs in /workspace/metrale-dumps/fp8native_dgx2/:
  vllm_L{0..39}.bin   - vLLM-FP8 per-layer last-token residual (f32 LE, 2048)
  metrale_L{0..39}.bin  - Metrale-FP8 per-layer last-token residual (f32 LE, 2048)
  (optional) vllm_logits.bin / metrale_logits.bin - final logits over vocab
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

OUT = pathlib.Path("/workspace/metrale-dumps/fp8native_dgx2")
N_LAYERS = 40
# 2026-09-26: Qwen3.6-35B-A3B-FP8's full-attention layers, from config.json
# layer_types: 3, 7, ..., 39. The other 30 are labelled "ssm".
ATTN_LAYERS = set(range(3, 40, 4))

def load(p):
    b = p.read_bytes()
    return np.frombuffer(b, dtype="<f4").astype(np.float64)

def cmp(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return (float(a @ b / (na*nb + 1e-30)),
            float(np.linalg.norm(a-b)/(nb+1e-30)),
            float(na), float(nb))

def main():
    have = lambda pre: all((OUT/f"{pre}_L{i}.bin").exists() for i in range(N_LAYERS))
    if not have("vllm"):
        print("MISSING vllm_L*.bin — run the vLLM dump first"); sys.exit(1)
    if not have("metrale"):
        print("MISSING metrale_L*.bin — run the Metrale Engine dump (METRALE_NEMO_DUMP) next"); sys.exit(1)
    print(f"{'layer':>5} {'type':>5} {'cos':>9} {'rel_l2':>9} {'|vllm|':>9} {'|metrale|':>9}")
    print("-"*52)
    onset = None
    rows = []
    for i in range(N_LAYERS):
        a = load(OUT/f"metrale_L{i}.bin"); v = load(OUT/f"vllm_L{i}.bin")
        if a.size != v.size:
            print(f"{i:>5}  SIZE MISMATCH metrale={a.size} vllm={v.size}"); continue
        cos, rel, nv, na = cmp(a, v)
        typ = "attn" if i in ATTN_LAYERS else "ssm"
        rows.append((i, typ, cos, rel, nv, na))
        flag = ""
        if cos < 0.999 and onset is None:
            onset = i; flag = "  <-- divergence onset (cos<0.999)"
        print(f"{i:>5} {typ:>5} {cos:>9.5f} {rel:>9.5f} {nv:>9.1f} {na:>9.1f}{flag}")
    print("-"*52)
    worst = min(rows, key=lambda r: r[2])
    print(f"worst layer: L{worst[0]} ({worst[1]}) cos={worst[2]:.5f} rel_l2={worst[3]:.4f}")
    print(f"final-layer L39 cos={rows[-1][2]:.5f}")
    if onset is not None:
        print(f"DIVERGENCE ONSET: L{onset} — inspect this layer's ops (attn/SSM/MoE/norm) next")
    else:
        print("No layer below cos 0.999 — Metrale-FP8 matches vLLM-FP8; gap is NON-numerical (sampler/parser/scheduler)")
    if (OUT/"vllm_logits.bin").exists() and (OUT/"metrale_logits.bin").exists():
        vl = load(OUT/"vllm_logits.bin"); al = load(OUT/"metrale_logits.bin")
        if vl.size == al.size:
            cos,rel,_,_ = cmp(al, vl)
            k=20
            vtop=set(np.argsort(vl)[-k:]); atop=set(np.argsort(al)[-k:])
            print(f"\nfinal logits: cos={cos:.5f} rel_l2={rel:.4f} top1_match={np.argmax(vl)==np.argmax(al)} top{k}_overlap={len(vtop&atop)}/{k}")

if __name__ == "__main__":
    main()
