# Metrale Engine

Metrale Engine is an LLM inference engine written in Rust and CUDA. Every
`(hardware, model, quantization)` target gets its own hand-tuned kernel set,
compiled to PTX at build time and embedded in one binary, `met`, which serves
an OpenAI- and Anthropic-compatible HTTP API. There is no Python in the
serving path.

NVIDIA GB10 (DGX Spark) is the primary target. Kernel sets also build for
Hopper (H100/H200, `sm_90a`) and B200 (`sm_100a`), with a Kimi K3 bring-up set
for B300; AMD Strix Halo (gfx1151, through SCALE) and Apple Metal build from
source. [`docs/HARDWARE.md`](docs/HARDWARE.md) lists what each target covers.

- Website: [metrale.ai/engine](https://metrale.ai/engine)
- Book: [book.dev.metrale.ai](https://book.dev.metrale.ai) (source in [`book/`](book/src/SUMMARY.md))
- API reference: [docs.dev.metrale.ai](https://docs.dev.metrale.ai)

## Install

`metralectl` launches validated recipes (image, checkpoint and serve settings)
on your machine:

```bash
curl -fsSL https://metrale.ai/install.sh | sh    # Linux, macOS
irm https://metrale.ai/install.ps1 | iex         # Windows (PowerShell)
uvx metralectl list                              # or run it without installing

metralectl list                                  # the recipes
metralectl run qwen3.6-35b-a3b-fp8-mtp           # serve one
```

The installer puts `metralectl` in `~/.local/bin`, verifies the download
against the release's `SHA256SUMS`, and installs its background agent.
`metralectl run` needs Docker with the NVIDIA Container Toolkit.

## Build from source

CUDA 13.0 or newer with `nvcc` on `PATH`:

```bash
git clone https://github.com/Metrale/metrale-inference-alpha.git
cd metrale-inference-alpha
export PATH=/usr/local/cuda/bin:$PATH
# the GB10 NVFP4 targets (the defaults); narrow with METRALE_TARGET_MODEL=<dir under kernels/gb10/>
cargo build --release -p metrale-server --bin met
```

`METRALE_TARGET_HW` picks another hardware set under `kernels/`. To lint and
test without a GPU or `nvcc`, set `METRALE_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000`
([`CONTRIBUTING.md`](CONTRIBUTING.md)). Docker images build from
[`docker/`](docker/docker-guide.md).

## Quick start

```bash
target/release/met serve Qwen/Qwen3.6-35B-A3B-FP8 --max-seq-len 16384

curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

`met serve` listens on `127.0.0.1:8888`; pass `--bind 0.0.0.0` with
`--require-auth` to expose it. [`QUICKSTART.md`](QUICKSTART.md) has per-model
commands, and `met serve --help` lists every flag.

## Benchmarks

Qwen3.8-27B NVFP4 on one GB10, aggregate decode tok/s against vLLM 0.27.1
running its own MTP speculative decoding, every workload axis matched
(ISL 128, OSL 1024, temperature 0, fp8 KV and MTP K=4 on both):

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Metrale Engine | 23.59 | 41.02 | 74.21 | 125.95 | 203.36 | 291.01 | 386.63 | 478.11 |
| vLLM + MTP | 19.72 | 37.11 | 71.61 | 124.48 | 197.03 | 283.48 | 361.39 | 358.57 |
| ratio | 1.20x | 1.11x | 1.04x | 1.01x | 1.03x | 1.03x | 1.07x | 1.33x |

Source: [`bench/ladder38/published.json`](bench/ladder38/published.json) and
the per-rung measurements it names; the campaign log is
[`bench/ladder38/RESULTS.md`](bench/ladder38/RESULTS.md).

Certification gates (BFCL accuracy, agentic coding, concurrency, TTFT, state
fidelity) run on GPU boxes and commit signed records to
[`.benchmarks/`](.benchmarks); the pull-request gate check verifies each
record's signature against the keys in `.github/record-signers/`. The book's
[Certification](book/src/operations/certify.md) chapter and
[`docs/provable-benchmark-work.md`](docs/provable-benchmark-work.md) explain
what a record proves and how to check one.

## License

Metrale Engine is licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option. Third-party code keeps its own
licence; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The public history starts on 2026-09-26, when the project was relicensed MIT
OR Apache-2.0; certified benchmark records in `.benchmarks/` are signed and
dated.
