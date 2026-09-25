# Private Hopper MoE rehearsal

No remote Git writes. Receipts belong outside the checkout. This is an
internal rehearsal until a third party actually operates the client.

1. Run `bash scripts/hopper-moe-preflight.sh /workspace/hopper-receipts/preflight`.
   Review processes, driver and disk before starting GPU work. Do not kill
   unrelated processes. Confirm model + Docker/build headroom, not just weights.
2. In a dedicated Python environment install `huggingface_hub`, recording its
   version. Run `python scripts/hopper-moe-download.py /workspace/models/qwen36`.
   This public model does not require copying stored credentials.
3. Build only the required target (can overlap downloading, not GPU timing):

```sh
docker build -f docker/hopper/Dockerfile \
  --build-arg METRALE_TARGET_MODEL=qwen3.6-35b-a3b \
  --build-arg METRALE_TARGET_QUANT=nvfp4 \
  --build-arg METRALE_GIT_SHA="$(git rev-parse HEAD)" \
  -t atlas-hopper-moe:private .
```

The `nvfp4` build directory is not permission to requantize the FP8 checkpoint.
Record the source diff, image ID, compiler versions and binary hash separately;
the Dockerfile currently installs Rust stable. Freeze the resulting image for
all timed runs. Resolve `vllm/vllm-openai:v0.30.0` to an immutable image digest
and record its actual version/help before selecting client flags.

4. Run `spark serve /models --check-kernels --no-tui` inside the built image,
   mounting the pinned model at `/models` and granting the one Hopper GPU.
   Audit success alone is not decode success. Candidate Atlas serve arguments:

```sh
spark serve /models --port 8888 --max-seq-len 2048 \
  --max-num-seqs 128 --max-batch-size 128 \
  --kv-cache-dtype fp8 --kv-high-precision-layers 0 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --lm-head-dtype bf16 --disable-thinking --no-tui
```

No speculative flag: speculation must remain off. Inspect startup logs for
actual quantization, cache dtype, capacity and fallback behavior. Qualify
known-answer prompts, streaming and repeated requests before measurement.
Use the same checkpoint and effective settings for vLLM, sequentially on
the same physical GPU. Do not silently adjust only one engine to fit.

5. Freeze stock `vllm bench serve` version/help and request corpus on the
   client machine. Protocol: ISL128, OSL1024, C1/8/32/128, three repetitions,
   temperature0, seed42 where supported, thinking off, neutral penalties.
   Fix warm-up/cache state and request count before timing. Save each command,
   raw JSON, actual input/output counts, failures and finish reasons.
6. Report tok/s, TTFT p50/p90, ITL p50 and variability. Keep crashes and losing
   cells. Short outputs do not qualify as OSL1024 wins. No H200 inference from
   H100 data. Copy receipts off rental storage before shutdown.
