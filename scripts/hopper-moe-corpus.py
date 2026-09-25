#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build immutable, matched Hopper benchmark inputs with the pinned tokenizer.

Run inside the pinned vLLM client image; no GPU is required. Prompts are synthetic
load-test inputs, not a quality evaluation. Pass their already-rendered strings
to vllm bench serve's custom dataset with --skip-chat-template --disable-shuffle.
"""

import argparse
import hashlib
import json
from pathlib import Path


MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
INPUT_LENGTH = 128
OUTPUT_LENGTH = 1024
CONCURRENCIES = (1, 8, 32, 128)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def render_exact(tokenizer, request_id):
    """Adjust only user content; never slice rendered text or template tokens."""
    nonce = sha256(request_id.encode())[:20]
    content = (
        f"Request {nonce}. Write a detailed practical guide to planning, testing, "
        "and maintaining a community garden. Cover soil, water, seasons, tools, "
        "accessibility, pests, records, and volunteer coordination. Give concrete "
        "examples and continue with useful details until the response budget ends."
    )

    def render(user_content):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # The stock custom dataset counts with tokenizer(prompt), whose default
        # adds special tokens. Require both counts to match rather than silently
        # relying on the server or client to repair a mismatched prompt.
        exact = len(tokenizer.encode(prompt, add_special_tokens=False))
        client = len(tokenizer.encode(prompt, add_special_tokens=True))
        return prompt, exact, client

    _, base_length, _ = render(content)
    estimate = max(0, INPUT_LENGTH - base_length)
    # A leading-space word is normally one token. Bounded fallback handles
    # tokenizer merges while retaining the entire official assistant prefix.
    counts = sorted(range(INPUT_LENGTH + 1), key=lambda n: (abs(n - estimate), n))
    for suffix in ("", ".", "!", " More", "\n"):
        for count in counts:
            prompt, exact, client = render(content + " Detail" * count + suffix)
            if exact == INPUT_LENGTH and client == INPUT_LENGTH:
                return prompt
    raise RuntimeError(
        f"Cannot construct {INPUT_LENGTH}-token prompt for {request_id}; "
        f"un-padded rendered length={base_length}. No template truncation allowed."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned tokenizer has no official chat template")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        args.out_dir / f"c{c}-r{rep}.jsonl"
        for c in CONCURRENCIES
        for rep in range(1, 4)
    ] + [args.out_dir / "manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Output files already exist; use a fresh output directory")

    seen = set()
    prepared = []
    cells = []
    for concurrency in CONCURRENCIES:
        count = max(16, 4 * concurrency)
        for repetition in range(1, 4):
            rows = []
            for index in range(count):
                request_id = f"hopper-moe-v1-c{concurrency}-r{repetition}-i{index}"
                prompt = render_exact(tokenizer, request_id)
                prompt_hash = sha256(prompt.encode())
                if prompt_hash in seen:
                    raise RuntimeError(f"Duplicate rendered prompt: {request_id}")
                seen.add(prompt_hash)
                rows.append({"prompt": prompt, "output_tokens": OUTPUT_LENGTH})
            data = "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ).encode()
            filename = f"c{concurrency}-r{repetition}.jsonl"
            prepared.append((args.out_dir / filename, data))
            cells.append({
                "file": filename,
                "concurrency": concurrency,
                "repetition": repetition,
                "requests": count,
                "sha256": sha256(data),
                "bytes": len(data),
                "min_input_tokens": INPUT_LENGTH,
                "max_input_tokens": INPUT_LENGTH,
            })

    tokenizer_hashes = {
        path.name: sha256(path.read_bytes())
        for path in sorted(args.model_dir.iterdir())
        if path.is_file() and (
            path.name.startswith(("tokenizer", "chat_template", "special_tokens"))
            or path.name in ("vocab.json", "merges.txt", "added_tokens.json")
        )
    }
    manifest = {
        "schema": "hopper-moe-corpus-v1",
        "model": MODEL,
        "expected_model_revision": REVISION,
        "revision_note": "Caller must stage the pinned snapshot; file hashes record actual tokenizer inputs.",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files_sha256": tokenizer_hashes,
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "input_tokens": INPUT_LENGTH,
        "requested_output_tokens": OUTPUT_LENGTH,
        "enable_thinking": False,
        "add_generation_prompt": True,
        "client_skip_chat_template": True,
        "client_disable_shuffle": True,
        "unique_rendered_prompts": len(seen),
        "request_count_policy": "max(16, 4 * concurrency)",
        "cache_policy": "Distinct corpus per cell/repetition; reuse identical files across engines. Use separate warmup prompts.",
        "output_length_note": "Request min_tokens=1024 on both engines and verify actual usage; output_tokens alone sets only the maximum.",
        "cells": cells,
    }
    # Validate everything before publishing any file; exclusive writes protect
    # existing experiment inputs from accidental replacement.
    for path, data in prepared:
        with path.open("xb") as output:
            output.write(data)
    with (args.out_dir / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps({
        "manifest": str(args.out_dir / "manifest.json"),
        "cells": len(cells),
        "unique_prompts": len(seen),
        "input_tokens": INPUT_LENGTH,
    }))


if __name__ == "__main__":
    main()
