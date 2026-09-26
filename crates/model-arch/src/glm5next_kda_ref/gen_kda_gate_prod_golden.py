#!/usr/bin/env python3

# 2026-09-25: Writes the KDA forget-gate golden (kda_gate_prod_golden.json) at H=64, D=128.
#
# Owner: model-arch (GLM-5.3-Flash KDA reference).
# Invariants:
# - Writes only /w/kda_gate_prod_golden.json.
"""GLM-5.3-Flash KDA gate golden at PRODUCTION geometry (H=64, D=128).

Slice 2's `kda_golden.json` binds every KDA sub-op to HuggingFace, but at a toy
geometry (H=2, D=4). The gate kernel's remaining risk at production scale is purely
INDEXING — `dt_bias` is per (head, channel) `[H*D]` while `A_log` is per head `[H]`,
and a kernel that collapsed either axis would still pass the toy fixture if H and D
were small enough to alias. This file removes that gap.

Source of truth: the real `transformers` 5.16.1 `Glm5NextTextForgetGate`. No equation
is re-derived here.

Determinism: inputs come from an integer LCG mapped to float32 by exact division by
2^24, so there is no transcendental in the generator and no RNG. Inputs are stored in
the file regardless, so the consumer never reproduces this arithmetic.

Adversarial by construction:
  * `dt_bias` varies across BOTH d and h, with a large per-channel ramp, so a kernel
    that broadcast one bias per head produces visibly wrong numbers.
  * `A_log` is distinct per head and spans a wide range, so a kernel that used head 0's
    decay everywhere is caught.
  * `g_raw` is scaled per head so that some rows sit deep in sigmoid saturation
    (both tails) while others stay in the linear region.
"""

import hashlib
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextForgetGate,
)

H = 64
D = 128
T = 2  # 2026-09-25: > 1, so token-major row indexing (row = t*H + h) is exercised
HIDDEN = 8  # 2026-09-25: only builds the module; f_a_proj and f_b_proj are replaced below
LOWER_BOUND = -5.0


class Lcg:
    """Integer LCG -> float32 in [-1, 1). No transcendentals, no platform libm."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next_unit(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0

    def tensor(self, *shape):
        n = 1
        for s in shape:
            n *= s
        return torch.tensor([self.next_unit() for _ in range(n)], dtype=torch.float32).reshape(shape)


rng = Lcg(0x5EED_0053)

# 2026-09-25: g_raw is scaled by 0.5 + 1.6 * h per head, so low heads stay in the sigmoid's
# linear region and high heads saturate.
g_raw = rng.tensor(T, H, D)
head_amp = torch.tensor([0.5 + 1.6 * h for h in range(H)], dtype=torch.float32)
g_raw = g_raw * head_amp.view(1, H, 1)

# 2026-09-25: dt_bias is per (head, channel) and ramps across both d and h, so neither axis can
# be collapsed.
dt_bias = rng.tensor(H, D) * 0.3
dt_bias = dt_bias + torch.tensor(
    [[(d - D / 2) * 0.02 + h * 0.01 for d in range(D)] for h in range(H)], dtype=torch.float32
)

# 2026-09-25: A_log is distinct per head, from -1 to 1, so exp(A_log) spans about [0.37, 2.72].
A_log = torch.tensor([-1.0 + 2.0 * h / (H - 1) for h in range(H)], dtype=torch.float32)

cfg = SimpleNamespace(
    hidden_size=HIDDEN,
    linear_head_dim=D,
    linear_num_heads=H,
    linear_lower_bound=LOWER_BOUND,
)

# 2026-09-25: Glm5NextTextForgetGate computes g from hidden_states through f_a_proj and f_b_proj.
# Both are replaced by an identity below, so the module's own forward applies the gate law to
# g_raw.
fg = Glm5NextTextForgetGate(cfg)
with torch.no_grad():
    fg.dt_bias.copy_(dt_bias.reshape(-1))
    fg.A_log.copy_(A_log)


class _IdentityFb(torch.nn.Module):
    """Makes `f_b_proj(f_a_proj(x))` return x unchanged, so `forward` receives g_raw."""

    def forward(self, x):
        return x


with torch.no_grad():
    fg.f_a_proj = _IdentityFb()
    fg.f_b_proj = _IdentityFb()
    gate = fg(g_raw.reshape(T, 1, H * D)).reshape(T, H, D)

# 2026-09-25: The gate law written out directly, to check the identity shim above.
check = LOWER_BOUND * torch.sigmoid(
    torch.exp(A_log).view(1, H, 1) * (g_raw.float() + dt_bias.view(1, H, D))
)
shim_err = float((gate - check).abs().max())
assert shim_err == 0.0, f"identity-projection shim changed the result: {shim_err}"


def arr(t):
    vals = t.flatten().tolist()
    return "[" + ",".join(f"{v:.9g}" for v in vals) + "]"


def entry(t):
    return f'{{"shape":{list(t.shape)},"data":{arr(t)}}}'


sat_lo = float((gate <= LOWER_BOUND * 0.999999).sum())
sat_hi = float((gate >= -1e-30).sum())

body = (
    "{\n"
    f' "fixture":{{"heads":{H},"head_dim":{D},"tokens":{T},"lower_bound":{LOWER_BOUND}}},\n'
    f' "coverage":{{"saturated_at_lower_bound":{int(sat_lo)},"saturated_at_zero":{int(sat_hi)},'
    f'"total":{T * H * D}}},\n'
    ' "inputs":{\n'
    f'  "g_raw":{entry(g_raw)},\n'
    f'  "dt_bias":{entry(dt_bias)},\n'
    f'  "A_log":{entry(A_log)}\n'
    " },\n"
    ' "outputs":{\n'
    f'  "gate":{entry(gate)}\n'
    " }\n"
    "}\n"
)

with open("/w/kda_gate_prod_golden.json", "w") as fh:
    fh.write(body)

print("transformers", __import__("transformers").__version__)
print("torch", torch.__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"saturated at lower_bound: {int(sat_lo)} / {T * H * D}")
print(f"saturated at zero:        {int(sat_hi)} / {T * H * D}")
print("gate min/max", float(gate.min()), float(gate.max()))
