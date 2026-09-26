#!/usr/bin/env python3

# 2026-09-26: Per-op cosine and abs-diff between Metrale Engine and HF op dumps; writes JSON rows.
#
# Owner: bench, FP8 drift investigation.
# Invariants: none beyond the types.
"""Compute per-op cosine similarity and abs-diff between Metrale Engine dumps and
HF reference dumps for the master drift table.

Inputs:
  --metrale-dir   : directory with metrale_op_L{i}_{op}.bin (full-attn ops)
                  + gdnsub_step0_L{ssm_idx}_{stage}.bin (SSM stages)
                  + metrale_L{i}.bin (per-layer hidden)
  --hf-dir      : directory with hf_op_L{i}_{op}.bin (all hooks)
                  + hf_bf16_L{i}.bin (per-layer hidden — legacy name)
  --out         : JSON output file

Metrale-to-HF op-name mapping is defined in OP_MAP below. SSM stage names use
the SSM-relative layer index in the Metrale Engine filename but the absolute layer
index in the HF filename — we translate via the layer_types list.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def load_bin(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype="<f4")
    if data.size == 0:
        return None
    return data


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# 2026-09-26: Qwen3.6-35B-A3B's full-attention layers (config.json layer_types).
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
SSM_LAYERS = [i for i in range(40) if i not in FULL_ATTN_LAYERS]

# 2026-09-26: Absolute SSM layer index to SSM-relative index, as gdnsub file names use.
SSM_REL = {abs_i: rel_i for rel_i, abs_i in enumerate(SSM_LAYERS)}


def compare_ssm_stages(metrale_dir: pathlib.Path, hf_dir: pathlib.Path, results: list):
    """Compare SSM stages.

    Metrale Engine filename:  gdnsub_step0_L{ssm_rel}_{stage}.bin  (BF16 raw bytes)
    HF filename:     hf_op_L{abs}_{ssm_*}.bin  (f32)

    Stage mapping (Metrale Engine → HF op):
      pre_norm         → input_norm_in
      post_norm        → input_norm_out
      qkvz             → ssm_in_proj_qkvz
      conv             → ssm_conv1d (approx — post-silu)
      gnorm            → ssm_norm (gated RMSNorm)
      out_proj         → ssm_out_proj
      moe_out          → moe_out
    """
    stage_map = [
        ("pre_norm", "input_norm_in"),
        ("post_norm", "input_norm_out"),
        ("post_qkvz", "ssm_in_proj_qkvz"),
        ("conv", "ssm_conv1d"),
        ("l2", None),
        ("gdn", None),
        ("gnorm", "ssm_norm"),
        ("out_proj", "ssm_out_proj"),
        ("moe_out", "moe_out"),
    ]

    for abs_i in SSM_LAYERS:
        rel_i = SSM_REL[abs_i]
        for metrale_stage, hf_op in stage_map:
            metrale_path = metrale_dir / f"gdnsub_step0_L{rel_i}_{metrale_stage}.bin"
            if hf_op is None:
                # 2026-09-26: No HF counterpart: record the Metrale Engine file,
                # and its size and norm when it exists.
                row = {
                    "layer": abs_i,
                    "op": f"ssm.{metrale_stage}",
                    "metrale_file": metrale_path.name,
                    "hf_file": None,
                    "status": "metrale_only_no_hf_ref",
                }
                if metrale_path.exists():
                    raw = np.fromfile(str(metrale_path), dtype=np.uint16)
                    metrale_arr = np.frombuffer(
                        np.left_shift(raw.astype(np.uint32), 16).tobytes(),
                        dtype="<f4",
                    )
                    row["metrale_shape"] = int(metrale_arr.size)
                    row["metrale_norm"] = float(np.linalg.norm(metrale_arr))
                results.append(row)
                continue
            hf_path = hf_dir / f"hf_op_L{abs_i}_{hf_op}.bin"
            # 2026-09-26: SSM stage dumps are raw BF16; widen to f32.
            if metrale_path.exists():
                raw = np.fromfile(str(metrale_path), dtype=np.uint16)
                metrale_arr = np.frombuffer(
                    np.left_shift(raw.astype(np.uint32), 16).tobytes(),
                    dtype="<f4",
                )
            else:
                metrale_arr = None
            hf_arr = load_bin(hf_path) if hf_path.exists() else None
            row = {
                "layer": abs_i,
                "op": f"ssm.{metrale_stage}",
                "metrale_file": metrale_path.name,
                "hf_file": hf_path.name,
            }
            if metrale_arr is None or hf_arr is None:
                row["status"] = "missing"
                row["metrale_present"] = metrale_arr is not None
                row["hf_present"] = hf_arr is not None
            elif metrale_arr.size != hf_arr.size:
                # 2026-09-26: Sizes differ: a cosine over the common prefix only.
                n = min(metrale_arr.size, hf_arr.size)
                row["status"] = "shape_mismatch"
                row["metrale_shape"] = metrale_arr.size
                row["hf_shape"] = hf_arr.size
                if n > 0:
                    row["cos_sim_prefix"] = cos_sim(metrale_arr[:n], hf_arr[:n])
            else:
                row["status"] = "ok"
                row["shape"] = int(metrale_arr.size)
                row["cos_sim"] = cos_sim(metrale_arr, hf_arr)
                row["max_abs"] = max_abs(metrale_arr, hf_arr)
                row["mean_abs"] = mean_abs(metrale_arr, hf_arr)
            results.append(row)


def compare_attention_ops(metrale_dir: pathlib.Path, hf_dir: pathlib.Path, results: list):
    """Compare full-attention layer ops.

    Metrale Engine uses full-attn-RELATIVE index 0..9 (Qwen3AttentionLayer.attn_layer_idx).
    HF uses ABSOLUTE layer index (L3, L7, ..., L39).
    """
    full_attn_ops = [
        "input_norm_in",
        "input_norm_out",
        "q_proj_full",
        "k_proj",
        "v_proj",
        "o_proj",
        "post_attn_norm_out",
        "moe_out",
    ]
    hf_only_ops = ["router_gate", "shared_expert", "q_after_norm", "k_after_norm"]
    for rel_i, abs_i in enumerate(FULL_ATTN_LAYERS):
        for op in full_attn_ops:
            metrale_path = metrale_dir / f"metrale_op_L{rel_i}_{op}.bin"
            hf_path = hf_dir / f"hf_op_L{abs_i}_{op}.bin"
            metrale_arr = load_bin(metrale_path)
            hf_arr = load_bin(hf_path)
            row = {
                "layer": abs_i,
                "op": f"attn.{op}",
                "metrale_file": metrale_path.name,
                "hf_file": hf_path.name,
            }
            if metrale_arr is None or hf_arr is None:
                row["status"] = "missing"
                row["metrale_present"] = metrale_arr is not None
                row["hf_present"] = hf_arr is not None
                results.append(row)
                continue
            # 2026-09-26: q_proj_full is compared over its full length and marked
            # ok_warn_layout, with a note that the two sides lay out Q and gate
            # differently.
            if op == "q_proj_full":
                if metrale_arr.size != hf_arr.size:
                    row["status"] = "shape_mismatch"
                    row["metrale_shape"] = int(metrale_arr.size)
                    row["hf_shape"] = int(hf_arr.size)
                    results.append(row)
                    continue
                n = metrale_arr.size
                half = n // 2
                row["status"] = "ok_warn_layout"
                row["shape"] = int(n)
                row["cos_sim"] = cos_sim(metrale_arr, hf_arr)
                row["max_abs"] = max_abs(metrale_arr, hf_arr)
                row["mean_abs"] = mean_abs(metrale_arr, hf_arr)
                row["note"] = (
                    "Q+gate interleaved (metrale) vs Q,gate split (hf); "
                    "full-length cosine likely <1.0 even with byte-exact compute."
                )
            else:
                if metrale_arr.size != hf_arr.size:
                    row["status"] = "shape_mismatch"
                    row["metrale_shape"] = int(metrale_arr.size)
                    row["hf_shape"] = int(hf_arr.size)
                    results.append(row)
                    continue
                row["status"] = "ok"
                row["shape"] = int(metrale_arr.size)
                row["cos_sim"] = cos_sim(metrale_arr, hf_arr)
                row["max_abs"] = max_abs(metrale_arr, hf_arr)
                row["mean_abs"] = mean_abs(metrale_arr, hf_arr)
            results.append(row)


def compare_layer_hidden(metrale_dir: pathlib.Path, hf_dir: pathlib.Path, results: list):
    """End-of-layer hidden state (= residual stream) for all 40 layers."""
    for abs_i in range(40):
        metrale_path = metrale_dir / f"metrale_L{abs_i}.bin"
        # 2026-09-26: Prefer hf_op_L{i}_layer_out.bin, else hf_bf16_L{i}.bin.
        hf_path = hf_dir / f"hf_op_L{abs_i}_layer_out.bin"
        if not hf_path.exists():
            hf_path = hf_dir / f"hf_bf16_L{abs_i}.bin"
        metrale_arr = load_bin(metrale_path)
        hf_arr = load_bin(hf_path)
        row = {
            "layer": abs_i,
            "op": "layer.hidden_out",
            "metrale_file": metrale_path.name,
            "hf_file": hf_path.name,
        }
        if metrale_arr is None or hf_arr is None:
            row["status"] = "missing"
            row["metrale_present"] = metrale_arr is not None
            row["hf_present"] = hf_arr is not None
            results.append(row)
            continue
        if metrale_arr.size != hf_arr.size:
            row["status"] = "shape_mismatch"
            row["metrale_shape"] = int(metrale_arr.size)
            row["hf_shape"] = int(hf_arr.size)
            results.append(row)
            continue
        row["status"] = "ok"
        row["shape"] = int(metrale_arr.size)
        row["cos_sim"] = cos_sim(metrale_arr, hf_arr)
        row["max_abs"] = max_abs(metrale_arr, hf_arr)
        row["mean_abs"] = mean_abs(metrale_arr, hf_arr)
        results.append(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrale-dir", required=True)
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrale_dir = pathlib.Path(args.metrale_dir)
    hf_dir = pathlib.Path(args.hf_dir)

    results: list = []
    compare_layer_hidden(metrale_dir, hf_dir, results)
    compare_attention_ops(metrale_dir, hf_dir, results)
    compare_ssm_stages(metrale_dir, hf_dir, results)

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} comparison rows to {out}")

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"  status counts: {by_status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
