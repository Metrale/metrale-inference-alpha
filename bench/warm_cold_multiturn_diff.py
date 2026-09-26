#!/usr/bin/env python3
# 2026-09-26: For each case, generates a turn-1 response once, then sends the turn-1
# prompt, response and a fixed follow-up twice as a greedy completion and reports the
# first token where the two continuations differ. Exits 1 if any case differs.
#
# Owner: bench, warm/cold prefix-cache diffs.
# Invariants: none beyond the types.
"""Multi-turn WARM-vs-COLD greedy diff — exercises decode-era Marconi snapshots.

The shipped warning is about WARM restores of SSM state that was checkpointed
DURING DECODE (decode_marconi_checkpoint, every 64 tokens), then restored when
the next turn's prefix covers those generated tokens. That is the agentic
multi-turn pattern: turn N's (prompt+response) becomes turn N+1's cached prefix.

Protocol per case (prefix-caching ON, MTP forced ON):
  1. COLD reference: build the full turn-2 prompt = T1_prompt + T1_response +
     T2_suffix and send it with a UNIQUE salt prefix so the radix lookup
     misses entirely → pure recompute (the proven-correct cache-off path).
  2. WARM: send T1_prompt (generates T1_response, saving decode-era Marconi
     snapshots), then send T1_prompt + T1_response + T2_suffix. The lookup now
     HITS deep into the generated region and Marconi restores a decode-era
     intermediate SSM snapshot mid-sequence, then suffix-prefills T2_suffix.
  3. Greedy: WARM continuation MUST equal COLD continuation. First divergence
     localizes corruption.

To maximize the chance of an INTERMEDIATE (non-leaf, block-straddling) snapshot
restore, T1 generates a long response (many 64-token checkpoint boundaries) and
T2 shares a deep prefix.
"""
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("METRALE_URL", "http://localhost:8888")
MODEL = os.environ.get("METRALE_MODEL", "model")
T1_MAX = int(os.environ.get("T1_MAX", "320"))
T2_MAX = int(os.environ.get("T2_MAX", "128"))

T1_PROMPTS = [
    "Write a long, detailed essay (at least 300 words) about the history of "
    "the Roman Empire, covering its founding, expansion, key emperors, and "
    "eventual decline. Be thorough and specific with dates and names.",
    "Explain in extensive detail how the internet works, from physical cables "
    "and routers through TCP/IP, DNS, HTTP, and TLS, with concrete examples at "
    "every layer. Write at least 300 words.",
    "Describe the complete lifecycle of a star, from a molecular cloud through "
    "main sequence to its final state, covering low-mass and high-mass stars "
    "separately. Be detailed and at least 300 words.",
    "Write a comprehensive guide to training a neural network: data prep, "
    "architecture choice, loss functions, optimizers, regularization, and "
    "evaluation. At least 300 words with specifics.",
    "Give a detailed chronological account of the major events of the 20th "
    "century, decade by decade, naming key figures, wars, and inventions. "
    "Write at least 300 words.",
]

# 2026-09-26: Appended after the turn-1 prompt and response to form the turn-2 prompt.
T2_SUFFIX = (
    "\n\nNow, based on everything above, produce a concise 5-bullet summary "
    "of the single most important point from each major section."
)


def gen(prompt, max_tokens):
    t0 = time.perf_counter()
    body = {
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "logprobs": 1,
    }
    req = urllib.request.Request(
        f"{URL}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    c = r["choices"][0]
    u = r.get("usage", {})
    lp = c.get("logprobs") or {}
    return {
        "text": c["text"],
        "tokens": lp.get("tokens"),
        "finish_reason": c.get("finish_reason"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def first_div(a, b):
    if a.get("tokens") and b.get("tokens"):
        ta, tb = a["tokens"], b["tokens"]
        for i in range(min(len(ta), len(tb))):
            if ta[i] != tb[i]:
                return i
        return -1 if len(ta) == len(tb) else min(len(ta), len(tb))
    return -1 if a["text"] == b["text"] else 0


def run_case(name, t1_prompt):
    # 2026-09-26: Turn 1 runs behind a unique salt line, so its prompt shares no leading
    # text with the unsalted turn-2 prompt; its response is reused verbatim.
    salt_t1 = f"[gen#{name}-{time.time_ns()}]\n"
    t1 = gen(salt_t1 + t1_prompt, T1_MAX)
    t1_resp = t1["text"]
    t2_prompt = t1_prompt + t1_resp + T2_SUFFIX

    # 2026-09-26: t2_prompt is sent twice: the first send is the reference ("cold"),
    # the second the replay ("warm").
    cold = gen(t2_prompt, T2_MAX)

    warm = gen(t2_prompt, T2_MAX)

    div = first_div(cold, warm)
    ok = div == -1
    return {
        "name": name, "ok": ok, "div": div,
        "t1_completion": t1["completion_tokens"],
        "warm_cached": warm["cached"], "warm_pt": warm["prompt_tokens"],
        "cold_cached": cold["cached"],
        "cold_text": cold["text"], "warm_text": warm["text"],
        "cold_fr": cold["finish_reason"], "warm_fr": warm["finish_reason"],
    }


def main():
    n = int(os.environ.get("N", "10"))
    prompts = (T1_PROMPTS * ((n // len(T1_PROMPTS)) + 1))[:n]
    print(f"=== Multi-turn WARM-vs-COLD greedy diff: {n} cases "
          f"(T1_MAX={T1_MAX}, T2_MAX={T2_MAX}) ===", flush=True)
    fails = 0
    for i, p in enumerate(prompts):
        r = run_case(f"C{i}", p)
        st = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            fails += 1
        deep = r["warm_cached"] > 64
        print(f"  [{st}] {r['name']}: t1_gen={r['t1_completion']} "
              f"warm_cached={r['warm_cached']}/{r['warm_pt']} "
              f"(deep_hit={deep}) div_idx={r['div']} "
              f"cold_fr={r['cold_fr']} warm_fr={r['warm_fr']}", flush=True)
        if not r["ok"]:
            print(f"      COLD: {r['cold_text']!r}", flush=True)
            print(f"      WARM: {r['warm_text']!r}", flush=True)
    print(f"\n=== {n - fails}/{n} PASS, {fails} FAIL ===", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
