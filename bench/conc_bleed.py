#!/usr/bin/env python3

# 2026-09-26: Cross-request bleed detector: per-worker canaries, a solo baseline, then concurrent rounds.
#
# Owner: bench.
# Invariants: none beyond the types.
"""
Cross-request BLEED + corruption detector for a running Metrale Engine server.

WHY THIS EXISTS
---------------
Single-request probes cannot see the failure mode that matters most in
production: one request's content surfacing in ANOTHER request's response.
That was observed live (an agentic client received text from an unrelated
concurrent request), and no existing harness could reproduce it on demand.

HOW IT WORKS
------------
Each concurrent worker owns a UNIQUE canary token and a UNIQUE topic, and its
prompt asks for its own canary back. Verdicts per response:

  OK       answer contains ITS OWN canary and number
  TYPO     canary MISSPELLED but number correct — a model spelling artefact on an
           invented token, explicitly NOT corruption (see `_near`)
  PARTIAL  own canary, wrong number
  WRONG    no canary / wrong value        -> model weakness OR corruption
  BLEED    contains ANOTHER worker's canary or topic
           -> cross-request contamination. This is the smoking gun.

A SOLO reference pass runs every prompt sequentially first. That separates
"the model is weak on this prompt" from "concurrency corrupts it": a prompt
that is OK alone and WRONG/BLEED under load is an engine bug, not the model.

USAGE
-----
  python3 bench/conc_bleed.py <model> [concurrency] [rounds]

  PORT=8888          server port
  THINK=1            enable thinking (default OFF — see below)
  THINK_BUDGET=2048  per-request thinking budget, Anthropic-style
                     {"thinking":{"budget_tokens":N}}; implies THINK=1
  MAXTOK=200         max_tokens per request
  TOOLS=1            attach a tool schema and scan tool-call arguments too
  EXACT=1            deterministic-answer mode; the SENSITIVE bleed detector
  STREAM=1           use the SSE streaming endpoint — a DIFFERENT code path from
                     blocking, and the one agentic clients actually use
  PREFIX_WORDS=6000  shared-prefix length; MUST exceed the Marconi checkpoint
                     interval (4096 tokens) or the SSM snapshot path is never hit

Run it BOTH ways. Thinking ON and OFF exercise different code paths, and
thinking materially changes the result: with thinking ON and a small
max_tokens, several models spend the entire budget reasoning and return EMPTY
content, which scores WRONG and would MASK real bleed. Measured on Holo 3.1:
0/8 solo with thinking on, 8/8 solo with it off, same build, same prompts.
That is why thinking defaults to OFF here — a harness that cannot establish a
clean solo baseline cannot detect anything.

THE TRIGGER IS A SHARED PREFIX
------------------------------
Back-to-back on ONE Laguna-XS container, same detector, only prompt shape differs:

    DISTINCT prompts (PREFIX_WORDS=0)  ->  32/32 CLEAN
    SHARED 6000-word prefix            ->  4 BLEED, e.g. VORPAL -> "VORGYR 19"
                                           (VORPAL's "VOR" + GYRE's "GYR")

This RECONCILES this harness with bench/agentic/conc_harness.py rather than either
being wrong. That harness passes 7/8 at C=8 with clean KV health — and its own
source says it uses DISTINCT prompts deliberately, "otherwise the prefix cache
would dedupe". It therefore cannot trigger this. It is not a weak test; it tests
a different shape.

The shared-prefix shape is the PRODUCTION one: every agentic client sends a long
common system prompt and differs only in the tail. So "the agentic harness passes"
is a statement about distinct-prompt traffic ONLY, and says nothing about
shared-prefix concurrency — which is what real clients do.

Sequential is clean at BOTH shapes, so it needs concurrency AND sharing together.

COUNT BLEED, NOT WRONG
----------------------
EXACT mode scores a correct-but-prose answer ("The project codename is MIMSY...
MIMSY 31") as WRONG. Use BLEED events for the corruption rate: 4/32 = 12.5%.

INTERPRETING RESULTS
--------------------
  solo OK == N and concurrent all OK      -> clean
  solo OK == N but concurrent WRONG/BLEED -> CONCURRENCY BUG (the point of this)
  solo already failing                    -> fix the probe or the model config
                                             FIRST; the run tells you nothing
"""
import json, os, sys, urllib.request, concurrent.futures, collections

MODEL = sys.argv[1] if len(sys.argv) > 1 else "puzzle"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

PORT = os.environ.get("PORT", "8888")
URL = f"http://localhost:{PORT}/v1/chat/completions"
MAXTOK = int(os.environ.get("MAXTOK", "200"))
# 2026-09-26: Words of shared prefix prepended to every worker's prompt, so the
# concurrent requests match each other deep in the prefix cache and differ only in
# the tail. PREFIX_WORDS=0 gives each worker a distinct prompt.
#
# On a hybrid SSM model the shared part must be longer than the SSM checkpoint
# interval (--ssm-checkpoint-interval, default 256 blocks = 4096 tokens at
# block_size 16). The other snapshots sit at each prompt's own end, so a shorter
# shared prefix finds no snapshot and the server recomputes all KV.
PREFIX_WORDS = int(os.environ.get("PREFIX_WORDS", "6000"))
BUDGET = os.environ.get("THINK_BUDGET")
THINK = os.environ.get("THINK") == "1" or BUDGET is not None

# 2026-09-26: A unique canary and topic per worker. In prose mode another worker's
# canary, or the first word of its topic, in a response is scored BLEED.
WORKERS = [
    ("ZANTHOR", "quartz mining", "7"),
    ("BRILLIG", "harbour dredging", "12"),
    ("VORPAL", "orchard grafting", "19"),
    ("SLITHY", "kiln firing", "23"),
    ("MIMSY", "rope splicing", "31"),
    ("GYRE", "lamp trimming", "44"),
    ("TULGEY", "salt panning", "58"),
    ("JUBJUB", "clock regulating", "63"),
]
ALL_CANARIES = [w[0] for w in WORKERS]
ALL_TOPICS = [w[1].split()[0] for w in WORKERS]


# 2026-09-26: TOOLS=1 sends TOOL_SCHEMA with tool_choice auto; tool-call names and
# arguments are scanned along with the content.
TOOLS = os.environ.get("TOOLS") == "1"
STREAM = os.environ.get("STREAM") == "1"
# 2026-09-26: EXACT=1 asks for "<CANARY><NUM>" and nothing else and compares its
# alphanumeric characters exactly, so a fragment of another canary shows.
EXACT = os.environ.get("EXACT") == "1"
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "record_site_entry",
        "description": "Record the project codename and entry count for a site log.",
        "parameters": {
            "type": "object",
            "properties": {
                "codename": {"type": "string", "description": "The project codename"},
                "entries": {"type": "integer", "description": "Number of log entries"},
            },
            "required": ["codename", "entries"],
        },
    },
}]


def _ask_stream(body, req_headers):
    """Accumulate SSE deltas, including tool-call argument fragments."""
    body["stream"] = True
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers=req_headers
    )
    text = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                text += d.get("content") or ""
                for tc in (d.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    text += " " + str(fn.get("name", "")) + " " + str(fn.get("arguments", ""))
    return text


def ask(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": THINK},
    }
    if TOOLS:
        body["tools"] = TOOL_SCHEMA
        body["tool_choice"] = "auto"
    if BUDGET:
        body["thinking"] = {"budget_tokens": int(BUDGET)}
    headers = {"Content-Type": "application/json"}
    try:
        if STREAM:
            return _ask_stream(body, headers)
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        text = msg.get("content") or ""
        # 2026-09-26: Scan tool-call names and arguments too: with TOOLS=1 the
        # answer may be only in the arguments.
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            text += " " + str(fn.get("name", "")) + " " + str(fn.get("arguments", ""))
        return text
    except Exception as e:
        return f"__ERR__ {type(e).__name__}"


# 2026-09-26: Built once from a fixed seed and reused by every worker, the solo
# pass and every round, so repeated requests can hit the cache at the shared depth.
_rng = __import__("random").Random(20260728)
_VOCAB = ("ledger harbour quarry lantern trellis cistern meadow gantry pallet "
          "furrow beacon cobble thicket parapet spindle mortar wicker bramble "
          "conduit rafter").split()
SHARED_PREFIX = " ".join(_rng.choice(_VOCAB) for _ in range(PREFIX_WORDS))


def make_prompt(canary, topic, num, _pad=None):
    # 2026-09-26: The shared prefix first, identical for all workers, then the
    # per-worker section that carries the canary.
    head = f"Reference log:\n{SHARED_PREFIX}\n\n" if PREFIX_WORDS else ""
    body = (f"{head}"
            f"Project codename is {canary}. The project concerns {topic}. "
            f"The site log records exactly {num} entries.\n\n")
    if EXACT:
        return body + ("Question: Reply with EXACTLY the codename followed by the "
                       "number, nothing else. Start your reply with the codename.")
    return body + ("Question: What is the project codename, and how many entries "
                   "does the site log record? Answer in one short sentence.")


def _near(a, b, tol=2):
    """True if some token in `b` is within `tol` edits of `a` (same length only).

    Models misspell INVENTED tokens: Holo returned "VORRAL" for "VORPAL" with the
    number correct. Scoring that WRONG produced a false concurrency signal that
    cost real debugging time — the substance was right every time. A near-miss on
    the canary with the correct number is a spelling artefact, NOT corruption.
    """
    for w in b.replace(",", " ").replace(".", " ").split():
        if len(w) == len(a) and sum(x != y for x, y in zip(w, a)) <= tol:
            return True
    return False


def classify(idx, out):
    mine, num = WORKERS[idx][0], WORKERS[idx][2]
    up = out.upper()
    if EXACT:
        clean = "".join(ch for ch in up if ch.isalnum())
        if clean == f"{mine}{num}":
            return "OK", ""
        # 2026-09-26: A deviation is BLEED when, after removing this worker's
        # canary and number, it still contains a prefix of 3 or more characters of
        # another canary; otherwise it is WRONG.
        for j, c in enumerate(ALL_CANARIES):
            if j == idx:
                continue
            for k in range(len(c), 2, -1):
                if c[:k] in clean.replace(mine, "").replace(num, ""):
                    return "BLEED", f"fragment {c[:k]!r} of {c}"
        return "WRONG", f"expected {mine}{num}"
    # 2026-09-26: In prose mode a foreign canary or topic word outranks every
    # other verdict.
    others = [c for j, c in enumerate(ALL_CANARIES) if j != idx and c in up]
    otop = [t for j, t in enumerate(ALL_TOPICS) if j != idx and t.upper() in up]
    if others or otop:
        return "BLEED", f"saw {others + otop}"
    if mine in up and num in out:
        return "OK", ""
    if mine in up:
        return "PARTIAL", "canary ok, number wrong"
    if num in out and _near(mine, up):
        return "TYPO", "canary misspelled, number correct — not corruption"
    return "WRONG", ""


def run_round(rnd):
    # 2026-09-26: The same prompts as the solo pass and every other round, so the
    # concurrent rounds can reuse the prefix cache.
    prompts = [make_prompt(c, t, n) for (c, t, n) in WORKERS[:CONC]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
        outs = list(ex.map(ask, prompts))
    return [(classify(i, o), o) for i, o in enumerate(outs)]


mode = (f"thinking={'ON' if THINK else 'OFF'} tools={'ON' if TOOLS else 'OFF'} "
        f"api={'STREAM' if STREAM else 'blocking'} exact={'ON' if EXACT else 'OFF'}")
if BUDGET:
    mode += f" budget={BUDGET}"
print(f"model={MODEL} port={PORT} concurrency={CONC} rounds={ROUNDS} {mode} "
      f"max_tokens={MAXTOK} shared_prefix_words={PREFIX_WORDS}")

print("=== SOLO reference (sequential — establishes the baseline) ===")
solo_ok = 0
for i, (c, t, n) in enumerate(WORKERS[:CONC]):
    v, _ = classify(i, ask(make_prompt(c, t, n)))
    solo_ok += v == "OK"
    print(f"  {c:<8} {v}")
print(f"  solo OK: {solo_ok}/{CONC}")
if solo_ok < CONC:
    print("  !! solo baseline is not clean — fix the probe/model config first;")
    print("     concurrent results below cannot distinguish model weakness from bleed.")

tally = collections.Counter()
bleeds = []
# 2026-09-26: Keep the text of non-OK responses: another request's answer,
# rephrased, may carry no canary and so scores WRONG, not BLEED.
wrongs = []
print("\n=== CONCURRENT passes ===")
for r in range(1, ROUNDS + 1):
    line = []
    for i, ((v, why), out) in enumerate(run_round(r)):
        tally[v] += 1
        line.append(f"{WORKERS[i][0][:4]}:{v[0]}")
        if v == "BLEED":
            bleeds.append((WORKERS[i][0], why, out[:150]))
        elif v not in ("OK", "TYPO"):
            wrongs.append((r, WORKERS[i][0], v, out[:220]))
    print(f"  round {r}: " + " ".join(line))

print(f"\n=== TOTALS over {ROUNDS * CONC} concurrent requests ===")
for k, v in tally.most_common():
    print(f"  {k:<8} {v}")
if wrongs:
    print(f"\n--- {len(wrongs)} non-OK responses (inspect for disguised bleed) ---")
    for r, name, v, txt in wrongs[:8]:
        print(f"  round {r} [{name}] {v}: {txt!r}")
if bleeds:
    print(f"\n*** {len(bleeds)} CROSS-REQUEST BLEED EVENTS ***")
    for name, why, txt in bleeds[:6]:
        print(f"  [{name}] {why}\n      {txt!r}")
    sys.exit(1)
print("\n  no cross-request bleed detected")
