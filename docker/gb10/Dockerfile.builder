# Metrale Engine reproducible BUILD environment — pins every native/FFI dependency so a
# `cargo build` "just works" without remembering CUTLASS_HOME / FLASHINFER_HOME /
# CUDA-13.2. This is the env behind the hand-built
# `metrale-holo:cuda13.2-fp4test` image, captured as code.
#
# The recurring failures this fixes:
#   • "CUTLASS support was not built; set CUTLASS_HOME" — build.rs silently drops
#     the native NVFP4 GEMM (`metrale_cutlass` cfg) when CUTLASS_HOME is unset.
#   • FlashInfer FA2 ragged-prefill wrapper needs FLASHINFER_HOME + its PINNED CCCL.
#
# Two ways to use it:
#   1. As a BUILD SANDBOX (mount the repo, run any cargo cmd — all env preset):
#        docker build -f docker/gb10/Dockerfile.builder --target builder -t metrale-gb10:build .
#        docker run --rm --gpus all -v "$PWD":/build -w /build metrale-gb10:build \
#          cargo build --release -p metrale-model-arch --example nvfp4_gemm_bench \
#            --no-default-features --features "cuda gpu-examples"
#   2. As a full SERVE image (compiles metrale-server):
#        docker build -f docker/gb10/Dockerfile.builder -t metrale-gb10:cuda13.2-fp4 .
#
# Pinned versions (override with --build-arg):
ARG CUDA_VER=13.2.0
ARG CUTLASS_SHA=cf064d2e6bad2886238ac565b3b49007764f4939
ARG FLASHINFER_SHA=a671c02ee2fbcdde7cc991f5a01c7cf5eb4a8972

# ── Build stage: toolchain + all pinned native deps ──────────────────────────
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04 AS builder
ARG CUTLASS_SHA
ARG FLASHINFER_SHA

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      curl ca-certificates build-essential pkg-config git cmake libclang-dev \
      libibverbs-dev libnccl2 libnccl-dev \
      python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Rust (stable — overrides rust-toolchain.toml's 1.85 pin; libloading 0.9 needs >=1.88).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=stable
ENV CUDA_HOME=/usr/local/cuda

# CUTLASS (header-only; build.rs compiles cutlass_nvfp4_gemm.cu against it).
ENV CUTLASS_HOME=/opt/cutlass
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git ${CUTLASS_HOME} && \
    git -C ${CUTLASS_HOME} checkout ${CUTLASS_SHA}

# FlashInfer + its PINNED CCCL (libcudacxx/cub) for the FA2 ragged-prefill wrapper.
# build.rs puts $FLASHINFER_HOME/3rdparty/cccl/{libcudacxx/include,cub} ahead of
# the CUDA-13 toolkit CCCL via -isystem (toolkit CCCL lacks cuda::fast_mod_div).
ENV FLASHINFER_HOME=/opt/flashinfer
RUN git clone --filter=blob:none https://github.com/flashinfer-ai/flashinfer.git ${FLASHINFER_HOME} && \
    git -C ${FLASHINFER_HOME} checkout ${FLASHINFER_SHA} && \
    git -C ${FLASHINFER_HOME} submodule update --init --depth 1 3rdparty/cccl

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/compat:${LD_LIBRARY_PATH}

WORKDIR /build

# ── Optional: compile a release metrale-server (skip when used as a build sandbox) ─
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/

ENV METRALE_TARGET_HW=gb10
ENV METRALE_TARGET_MODEL=*
ENV METRALE_TARGET_QUANT=*
# Native FP4 GEMM + cuBLASLt BF16 prefill projections on by default (matches prod).
ENV METRALE_CUTLASS_NVFP4_GEMM=1

# CUDARC_CUDA_VERSION=13000: the vendored cudarc 0.19.2 tops out at 13.1 in its
# nvcc-version table, so `nvcc --version` (13.2) panics without the pin. Same
# value every CI workflow pins.
RUN CUDARC_CUDA_VERSION=13000 cargo build --release -p metrale-server

# ── Runtime stage: serve image on CUDA 13.2 ──────────────────────────────────
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu24.04
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# NCCL >= 2.28 (ncclMemAlloc symmetric-memory windows) + RDMA userspace.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28" >&2; exit 1; }

COPY --from=builder /build/target/release/met /usr/local/bin/met
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE-MIT /LICENSE-MIT
COPY LICENSE-APACHE /LICENSE-APACHE
COPY THIRD_PARTY_NOTICES.md /THIRD_PARTY_NOTICES.md
COPY LICENSES/ /LICENSES/
COPY README.md /README.md

ENV RUST_LOG=info
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64
EXPOSE 8888
ENTRYPOINT ["met"]
