#!/usr/bin/env python3

# 2026-09-25: Writes the KDA conv golden (kda_conv_golden.json) at conv_dim 24576.
#
# Owner: model-arch (GLM-5.3-Flash KDA reference).
# Invariants:
# - Writes only /w/kda_conv_golden.json.
"""Conv contract at GLM-5.3-Flash PRODUCTION geometry, against HF transformers 5.16.1.

Slice 6 gate. Metrale Engine classes `causal_conv1d_update_l2norm` as REUSE for KDA, but it has never
been driven at KDA geometry and it carries hardcoded assumptions (BLOCK=256, head_dim=128 ->
2 heads/block, `qk_channels % 256 == 0`). This binds it to HF before any layer integration.

Geometry: conv_dim = 3*64*128 = 24576, qk_channels = 2*64*128 = 16384, head_dim = 128,
kernel = 4, activation = silu (config `hidden_act`).

Two paths are captured because Metrale Engine has two and they are NOT the same kernel:
  DECODE  — one token, conv + SiLU + L2 fused (`causal_conv1d_update_l2norm`)
  PREFILL — N tokens, conv + SiLU only (`causal_conv1d_update_prefill`), L2 applied
            separately (`l2_norm_bf16`). L2 must therefore happen EXACTLY ONCE on this path.

STATE WIDTH: HF keeps `kernel_size - 1` = 3 slots; Metrale Engine keeps 4 and shifts left before
convolving, so the oldest slot is shifted out and never participates. The mapping is
`HF_state[0..3] == Metrale_state[1..4]` pre-shift. The golden records HF's 3-wide state; the
Rust side widens it.

Inputs come from the integer LCG the microtest reproduces bit for bit; only outputs are
committed, and the bulky ones as a prime-strided sample plus an fp64 index-weighted checksum.
"""

import hashlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    causal_conv1d_fn,
    causal_conv1d_update,
    l2norm,
)

H, D = 64, 128
CONV_DIM = 3 * H * D
QK = 2 * H * D
KS = 4
T_PRE = 4
STRIDE = 251


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0

    def t(self, *shape):
        n = 1
        for s in shape:
            n *= s
        return torch.tensor([self.u() for _ in range(n)], dtype=torch.float32).reshape(shape)


probe = Lcg(0x5EED_C0F0)
lcg_probe = [probe.u() for _ in range(8)]

rng = Lcg(0x5EED_C0F0)
weight = rng.t(CONV_DIM, KS) * 0.5
state0 = rng.t(CONV_DIM, KS - 1) * 0.5
tok = rng.t(CONV_DIM)
pre = rng.t(T_PRE, CONV_DIM)

# 2026-09-25: Decode: causal_conv1d_update, then l2norm on the q|k channels only.
st = state0.clone().unsqueeze(0)
dec = causal_conv1d_update(
    tok.reshape(1, CONV_DIM, 1), st, weight, None, activation="silu"
).reshape(CONV_DIM)
dec_state = st.reshape(CONV_DIM, KS - 1).clone()

dec_out = dec.clone()
qk = dec_out[:QK].reshape(2 * H, D)
dec_out[:QK] = l2norm(qk, dim=-1, eps=1e-6).reshape(-1)

# 2026-09-25: Prefill: causal_conv1d_fn over T_PRE tokens from a zero state, then l2norm on q|k.
pre_ct = pre.t().contiguous().reshape(1, CONV_DIM, T_PRE)
pre_conv = causal_conv1d_fn(pre_ct, weight, None, activation="silu").reshape(CONV_DIM, T_PRE)
pre_out = pre_conv.t().contiguous().clone()
pre_qk = pre_out[:, :QK].reshape(T_PRE * 2 * H, D)
pre_out[:, :QK] = l2norm(pre_qk, dim=-1, eps=1e-6).reshape(T_PRE, QK)
# 2026-09-25: The conv state after a zero-started prefill is the last KS - 1 raw inputs.
pre_state = pre[-(KS - 1) :, :].t().contiguous()


def arr(t):
    return "[" + ",".join(f"{x:.9g}" for x in t.flatten().tolist()) + "]"


def entry(t):
    return f'{{"shape":{list(t.shape)},"data":{arr(t)}}}'


def ck(t):
    f = t.flatten().to(torch.float64)
    i = torch.arange(f.numel(), dtype=torch.float64)
    return float((f * (i + 1.0)).sum())


body = (
    "{\n"
    f' "fixture":{{"conv_dim":{CONV_DIM},"qk_channels":{QK},"head_dim":{D},"heads":{H},'
    f'"kernel":{KS},"t_prefill":{T_PRE},"sample_stride":{STRIDE},"seed":"0x5EEDC0F0",'
    f'"hf_state_width":{KS - 1},"activation":"silu","l2_eps":1e-06}},\n'
    f' "lcg_probe":[{",".join(f"{x:.9g}" for x in lcg_probe)}],\n'
    ' "outputs":{\n'
    f'  "decode_out":{entry(dec_out)},\n'
    f'  "decode_state_sample":{entry(dec_state.flatten()[::STRIDE].contiguous())},\n'
    f'  "decode_conv_presilu_sample":{entry(dec[::STRIDE].contiguous())},\n'
    f'  "prefill_out_sample":{entry(pre_out.flatten()[::STRIDE].contiguous())},\n'
    f'  "prefill_state_sample":{entry(pre_state.flatten()[::STRIDE].contiguous())}\n'
    " },\n"
    f' "checksums":{{"prefill_out":{ck(pre_out)!r},"prefill_state":{ck(pre_state)!r},'
    f'"decode_out":{ck(dec_out)!r},"decode_state":{ck(dec_state)!r}}}\n'
    "}\n"
)

with open("/w/kda_conv_golden.json", "w") as fh:
    fh.write(body)

print("transformers", __import__("transformers").__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
print("decode_out |max|", float(dec_out.abs().max()))
print("prefill_out |max|", float(pre_out.abs().max()))
qn = dec_out[:QK].reshape(2 * H, D).norm(dim=-1)
print("decode q|k row norms: min", float(qn.min()), "max", float(qn.max()), "(must be ~1)")
