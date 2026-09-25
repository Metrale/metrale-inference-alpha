#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reject stock-client successes that contain errors or shortened requests."""
import argparse
import json
import math
from pathlib import Path


def validate(data, requests, concurrency):
    problems = []
    for key, expected in (("num_prompts", requests), ("completed", requests),
                          ("failed", 0), ("max_concurrency", concurrency),
                          ("total_input_tokens", requests * 128),
                          ("total_output_tokens", requests * 1024)):
        if data.get(key) != expected:
            problems.append(f"{key}: expected {expected}, got {data.get(key)!r}")
    for key, expected in (("input_lens", 128), ("output_lens", 1024)):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != requests or any(
                value != expected for value in values):
            problems.append(f"{key}: require {requests} entries all equal to {expected}")
    errors = data.get("errors")
    if not isinstance(errors, list) or len(errors) != requests or any(errors):
        problems.append("errors: require one empty error per request")
    for key in ("duration", "output_throughput"):
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            problems.append(f"{key}: require a finite positive number")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    problems = validate(json.loads(args.result.read_text()), args.requests, args.concurrency)
    print(json.dumps({"valid": not problems, "problems": problems}, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
