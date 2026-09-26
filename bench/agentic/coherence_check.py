#!/usr/bin/env python3

# 2026-09-26: Coherence probe: checkable questions asked cold, then warm through the prefix cache.
#
# Owner: bench, agentic probes.
# Invariants: none beyond the types.
"""Post-stress coherence check.

Aliased KV blocks corrupt output rather than crashing, so a clean refcount
log is necessary but not sufficient. Ask questions with checkable answers,
twice each (cold, then warm through the prefix cache), and verify both.
"""
import os, json, os, sys, time, urllib.request

URL = os.environ.get("METRALE_URL", "http://localhost:8888/v1/chat/completions")
MODEL = os.environ.get("COHERENCE_MODEL", "laguna-s-2.1")

# 2026-09-26: The system prompt is sent with every case (see `ask`), so every run
# is judged under the same instructions unless COHERENCE_SYSTEM overrides them.
SYSTEM = os.environ.get(
    "COHERENCE_SYSTEM",
    "You are a careful, precise assistant. Answer the question directly and "
    "correctly. Work step by step when a question needs it, and state the final "
    "answer explicitly. Do not refuse or hedge on simple factual or arithmetic "
    "questions.",
)
CASES = [
    ("What is 17 * 23? Reply with the number, then explain your working.", "391"),
    ("Name the capital city of Japan, then describe it in a few sentences.", "Tokyo"),
    ("Spell the word 'refrigerator' backwards, then explain how you did it.", "rotaregirfer"),
]


def ask(prompt):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_tokens": 300, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d["choices"][0]["message"]["content"]


if __name__ == "__main__":
    fails = 0
    for pas in ("cold", "warm"):
        for prompt, expect in CASES:
            try:
                out = ask(prompt)
                # 2026-09-26: Judge the stated answer: the expected value must be in
                # the first or last non-empty line. A value found only in the working
                # is reported as WORKING-ONLY and still counts as a failure.
                lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
                head = lines[0].lower() if lines else ""
                tail = lines[-1].lower() if lines else ""
                e = expect.lower()
                ok = e in head or e in tail
                buried = (not ok) and e in out.lower()
                # 2026-09-26: Empty output, or output with 2% or more unprintable
                # characters, fails even when the expected value is present.
                printable = sum(c.isprintable() or c.isspace() for c in out)
                clean = len(out) > 0 and printable / len(out) > 0.98
                status = "OK " if (ok and clean) else ("WORKING-ONLY" if buried else "FAIL")
                if not (ok and clean):
                    fails += 1
                print(f"  [{pas}] {status} expect={expect!r} -> {out[:90]!r}", flush=True)
            except Exception as e:
                fails += 1
                print(f"  [{pas}] FAIL {expect!r}: {e}", flush=True)
            time.sleep(1)
    print(f"\n  coherence: {6-fails}/6 passed")
    sys.exit(1 if fails else 0)
