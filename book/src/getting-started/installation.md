# Installation

Metrale Engine ships as a single Docker image that contains the release binary plus every compiled `(GB10, model, quant)` PTX module — one target set per `kernels/gb10/<model>/<quant>/` directory. There is no "install Metrale Engine + download kernels" step — the kernels are baked in.

## Hardware prerequisites

Metrale Engine is designed for broad hardware support — the engine is vendor-agnostic above the kernel layer (`ComputeTarget` at build time, `GpuBackend` at runtime, `CommBackend` for collectives) and new hardware plugs in at the trait layer. The first shipped target is **NVIDIA GB10 (SM121)** — the Grace-Blackwell Superchip in the NVIDIA DGX Spark workstation. To run the shipped image you need:

- A DGX Spark (or any GB10-based system) with 119.7 GB of unified GPU memory
- NVIDIA driver supporting CUDA 13.0 or later
- `docker` with `--gpus all` support (recent `nvidia-container-toolkit`)
- Internet access for the first model download; models are cached under `~/.cache/huggingface` after that

H100/H200 and B200 images build from `docker/hopper/Dockerfile` and `docker/b200/Dockerfile` (see `docker/docker-guide.md`); AMD and Apple targets build from source. The GB10 image's PTX is compiled with `-arch=sm_121f` using SM121-specific tile shapes and a software E2M1 conversion — none of that is architectural, it's just the first target we hyperoptimized. Adding a new hardware target is two trait impls plus kernel source; see [Adding a new hardware target](https://github.com/Metrale/metrale-inference-alpha/blob/main/docs/HARDWARE.md#adding-a-new-hardware-target).

## Install metralectl

`metralectl` launches Metrale Engine recipes: it picks the image, the checkpoint and the serve settings a recipe was validated under and runs the `docker run` they imply.

```bash
curl -fsSL https://metrale.ai/install.sh | sh       # Linux, macOS
irm https://metrale.ai/install.ps1 | iex            # Windows (PowerShell)
uvx metralectl list                                 # or run it with no install
```

The installer puts the binary in `~/.local/bin` (`METRALECTL_INSTALL_DIR` overrides it), refuses a download whose SHA-256 is not in the release's `SHA256SUMS`, and installs the background agent (`METRALECTL_NO_AGENT=1` skips that). `metralectl list`, `metralectl show` and `metralectl run --print` work without Docker; `metralectl run` needs it. If something stops a launch, see [Troubleshooting](./troubleshooting.md).

The rest of this page runs the image directly with `docker`.

## Pull the image

```bash
docker pull metrale/metrale-inference-gb10:latest
```

The image contains the Rust release binary, every GB10 PTX module set, tokenizer dependencies, and the `nvidia-container-runtime` library surfaces. No Python, no CUDA toolkit.

## Bring your own weights

Metrale Engine loads HuggingFace `safetensors` directly. The image does **not** ship model weights. On first run, the binary resolves a HuggingFace model ID (e.g. `Sehyo/Qwen3.5-35B-A3B-NVFP4`) against `~/.cache/huggingface/hub` — download the weights once with the `hf` CLI or let the server download-on-miss:

```bash
pip install -U huggingface_hub
hf download Sehyo/Qwen3.5-35B-A3B-NVFP4
```

The command is `hf`, not `huggingface-cli`: `huggingface_hub` 1.0 renamed the
binary, and on 1.16+ the old name is gone entirely. Older docs and scripts still
say `huggingface-cli download …`, which now fails with "command not found".

Mount the cache directory into the container:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```

## Build from source (optional)

You only need to build from source if you are modifying Metrale Engine. `rust-toolchain.toml` pins the Rust release; CUDA 13.0+ with `nvcc` on `PATH` (or `CUDA_HOME` set) is required for a real build. Clippy and fmt can run without CUDA via `METRALE_SKIP_BUILD=1`.

```bash
git clone https://github.com/Metrale/metrale-inference-alpha.git
cd metrale-inference-alpha

# Full build — compiles every (gb10, model, quant) target (~6 min)
docker build -f docker/gb10/Dockerfile -t metrale-inference-gb10 .

# Rust-only check (no CUDA). CUDARC_CUDA_VERSION is needed alongside
# METRALE_SKIP_BUILD: without it cudarc's build script shells out to
# `nvcc --version` and panics on a host that has no CUDA toolkit.
# This pair is exactly what ci.yml exports. Deny-warnings comes from
# [workspace.lints], so `-- -Dwarnings` is not needed and CI does not pass it.
METRALE_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
cargo fmt --all -- --check

# Unit tests (uses MockGpuBackend; no GPU required)
cargo test --release

# Integration tests (require GPU + weights)
cargo test -p metrale-server --release -- --ignored
```

The build system reads `kernels/gb10/HARDWARE.toml` for architecture flags, enumerates every `(model, quant)` subdirectory that matches the `METRALE_TARGET_*` wildcards, compiles each `.cu` source file through `nvcc`, and emits a single `target_ptx.rs` that the `metrale-kernels` crate embeds in the final binary. Zero runtime compilation.

## Verify the install

```bash
docker run --rm --gpus all metrale/metrale-inference-gb10:latest --version
# → met 1.0.0-beta-preview   (the workspace version in Cargo.toml)
docker run --rm --gpus all metrale/metrale-inference-gb10:latest --help | head -20
```

If `--version` errors with "no compatible GPU", the `nvidia-container-toolkit` is not picking up the device. Check `docker info | grep -i runtime` and `nvidia-smi` on the host.

You are now ready for the [Quickstart](./quickstart.md).
