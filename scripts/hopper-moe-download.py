#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Download the comparison's pinned checkpoint; never embed credentials."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
lock = json.loads(Path(__file__).with_name("hopper-moe-model.json").read_text())
snapshot_download(
    repo_id=lock["model"],
    revision=lock["revision"],
    local_dir=str(args.destination),
    max_workers=8,
)
index = json.loads((args.destination / "model.safetensors.index.json").read_text())
missing = [name for name in set(index["weight_map"].values())
           if not (args.destination / name).is_file()]
if missing:
    raise SystemExit(f"Missing weight shards: {missing}")
(args.destination / "comparison-model-lock.json").write_text(
    json.dumps(lock, indent=2) + "\n"
)
print(f"Pinned model staged at {args.destination}")
