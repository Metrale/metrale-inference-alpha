# Metrale Engine — Qwen3.5-35B-A3B-NVFP4 on GB10
#
# Build from repo root:
#   docker build -f docker/gb10/qwen3.5-35b-a3b/nvfp4/Dockerfile -t metrale-35b .
#
# Run:
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     metrale-35b serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
# metrale-bench include_str!s its benchmark prompts from tests/fixtures;
# without them the release build fails with "couldn't read".
COPY tests/fixtures/ tests/fixtures/

ENV METRALE_TARGET_HW=gb10
ENV METRALE_TARGET_MODEL=qwen3.5-35b-a3b
ENV METRALE_TARGET_QUANT=nvfp4

RUN cargo build --release -p metrale-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      (echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" && exit 1)

COPY --from=builder /build/target/release/met /usr/local/bin/met
COPY LICENSE-MIT /LICENSE-MIT
COPY LICENSE-APACHE /LICENSE-APACHE
COPY THIRD_PARTY_NOTICES.md /THIRD_PARTY_NOTICES.md
COPY LICENSES/ /LICENSES/
COPY README.md /README.md

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["met"]
