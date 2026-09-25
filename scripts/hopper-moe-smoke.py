#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Save raw correctness and streaming receipts before a Hopper timing run."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def check_sse(raw):
    events = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    assert events, "No SSE events"
    assert "data: [DONE]" in raw, "Missing stream terminator"
    choices = [c for e in events for c in e.get("choices", [])]
    assert any(c.get("finish_reason") for c in choices), "Missing finish reason"
    usage = [e["usage"] for e in events if not e.get("choices") and e.get("usage")]
    assert usage, "Missing usage-only event needed by stock benchmark client"
    return {"usage": usage[-1], "finish_reasons": [c["finish_reason"]
            for c in choices if c.get("finish_reason")]}


def self_test():
    try:
        check_sse('data: {"choices":[]}\n\ndata: [DONE]\n')
    except AssertionError:
        pass
    else:
        raise AssertionError("Invalid stream accepted")
    raw = ('data: {"choices":[{"finish_reason":"length"}]}\n\n'
           'data: {"choices":[],"usage":{"completion_tokens":32}}\n\n'
           'data: [DONE]\n')
    assert check_sse(raw)["usage"]["completion_tokens"] == 32


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    self_test()
    a.out.mkdir(parents=True, exist_ok=True)

    def request(name, body, endpoint="chat/completions"):
        body = {"model": a.model, "temperature": 0, "seed": 42, **body}
        (a.out / f"{name}.request.json").write_text(json.dumps(body, indent=2))
        start = time.monotonic()
        req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/" + endpoint,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read().decode()
        (a.out / f"{name}.response.txt").write_text(raw)
        (a.out / f"{name}.seconds.txt").write_text(str(time.monotonic() - start))
        return raw

    def arithmetic(i):
        raw = request(f"arithmetic-{i}", {
            "messages": [{"role": "user", "content": "What is 2+2? Reply with only the digit."}],
            "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
        response = json.loads(raw)
        answer = response["choices"][0]["message"]["content"].strip()
        assert answer == "4", f"Weak known-answer oracle failed: {answer!r}"
        return answer

    arithmetic(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(arithmetic, range(1, 5)))
    raw = request("stream-min32", {"prompt": "Continue the list: one, two, three,",
                  "stream": True, "stream_options": {"include_usage": True},
                  "min_tokens": 32, "max_tokens": 32}, "completions")
    parsed = check_sse(raw)
    assert parsed["usage"]["completion_tokens"] == 32, parsed
    (a.out / "summary.json").write_text(json.dumps({"status": "passed",
        "limits": "Only short arithmetic, concurrent requests and 32-token streaming contract; not broad model correctness",
        "stream": parsed}, indent=2))
    print(json.dumps(parsed))


if __name__ == "__main__":
    main()
