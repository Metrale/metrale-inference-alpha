# metrale-comm

**Role:** the multi-GPU collective-ops abstraction. One trait, two impls: NCCL for real distributed runs and a no-op backend for single-GPU.
**Path:** `crates/comm/`
**Key files:** `src/lib.rs` (`CommBackend` trait, `SingleGpuBackend`), `src/nccl_backend.rs` (the `NcclBackend` impl), `crates/gpu-sys/src/nccl.rs` (raw NCCL FFI).

## Why this is its own crate

Multi-GPU in Metrale Engine is **Expert Parallelism (EP)** — the MoE experts of models beyond one GB10's weight budget (122B, 119B, 229B) are split across two nodes connected via RoCEv2. Token dispatch between ranks goes through NCCL all-reduces and send/recv.

`metrale-comm` isolates the NCCL surface so that:

1. Single-GPU deployments never link against NCCL (the binary loads a `SingleGpuBackend`).
2. Tests for the scheduler and the layer code can run against the no-op impl on CI.
3. Porting to a different collective-ops library (RCCL for AMD, Metal MPS, oneCCL) is a new `CommBackend` impl, nothing else changes.

## The trait

```rust
pub trait CommBackend: Send + Sync {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()>;
    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;
    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()>;
    fn barrier(&self) -> Result<()>;
    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()>;
    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()>;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    // default methods: all_reduce_async, group_start/group_end, buffer
    // registration, symmetric memory, is_healthy/attempt_reconnect, …
}
```

All pointer arguments are `u64` (matching CUDA's `CUdeviceptr`) to avoid coupling the crate to `metrale-gpu-runtime`'s `DevicePtr`. Point-to-point transfers and `all_reduce_async` take the stream they run on, so collectives and kernel launches can be pipelined.

## `SingleGpuBackend` — the no-op

```rust
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> { Ok(()) }
    fn all_gather(&self, _s: u64, _r: u64, _b: usize) -> Result<()> { Ok(()) }
    // ...
    fn rank(&self) -> usize { 0 }
    fn world_size(&self) -> usize { 1 }
}
```

This is what runs in single-GPU serving. The expert-parallel code paths in `metrale-model-layers::layers::moe` still call `comm.all_reduce(...)` — the op is a no-op under `SingleGpuBackend` and an actual NCCL call under `NcclBackend`. The caller never branches on `world_size`.

## `NcclBackend` — the real impl

In `nccl_backend.rs`. Uses the unsafe NCCL FFI in `crates/gpu-sys/src/nccl.rs`. Construction flow:

1. The `master` rank (0) calls `ncclGetUniqueId` and serves the id over a TCP rendezvous on `--master-addr`/`--master-port` (default 29500).
2. Every other rank connects to the rendezvous and receives the id.
3. All ranks call `ncclCommInitRank(world_size, id, rank)` in parallel; the call is collective and blocks until every rank has joined.

The NCCL env layer is fussy on GB10 — `scripts/start-ep2.sh` and `scripts/start-minimax-ep2.sh` pin the critical vars (the script's `NCCL_ENV` block is the full list):

| Variable | Value | Reason |
|---|---|---|
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` (or `enp1s0f1np1`, whichever has an IPv4 address) | Forces the RoCE interface, not the mgmt ethernet |
| `NCCL_IB_DISABLE` | `0` | IB transport enabled |
| `NCCL_IB_HCA` | `rocep1s0f0` | The RoCE HCA device |
| `NCCL_IB_ROCE_VERSION_NUM` / `NCCL_IB_ADDR_FAMILY` | `2` / `AF_INET` | Force RoCEv2 over IPv4 |
| `NCCL_NET_GDR_LEVEL` / `NCCL_NET_GDR_C2C` / `NCCL_DMABUF_ENABLE` | `0` | GPUDirect RDMA off: `nvidia_peermem` does not work on the GB10 kernel |
| `NCCL_NVLS_ENABLE` | `0` | NVLink-SHARP crashes on aarch64 Blackwell; force off |

These are worth the paragraph — a mis-set `NCCL_SOCKET_IFNAME` on GB10 will silently fall back to the 1 GbE management interface and drop EP=2 throughput by an order of magnitude.

## The EP=2 throughput path

For Qwen3.5-122B-A10B NVFP4 at EP=2:

- 128 experts per rank (256 total).
- Token dispatch: the gate runs on every rank, top-k expert IDs are selected, tokens destined for remote experts are `reduce_scatter`'d to the owning rank.
- Expert compute happens locally.
- Expert outputs are `all_gather`'d back.
- Result: ~46 tok/s sustained on 600-token decodes (see [Multi-GPU](../operations/multi-gpu.md)).

The bandwidth pressure is all in the dispatch + gather, which is why RoCEv2 with GDR matters. A plain TCP NCCL falls off by 3×.

## The critical MTP-flag symmetry rule

A subtle footgun: when the head (rank 0) runs with `--speculative --mtp-quantization nvfp4 --num-drafts N`, the worker **must** be started with the same flags. If not, the MTP verify command from the head lands in the worker's SSM layer without intermediate buffers allocated and you get an SSM intermediate-buffer error. `scripts/start-ep2.sh` handles this; a manual two-command launch does not, and it has bit multiple contributors. See the [Multi-GPU chapter](../operations/multi-gpu.md).

## NCCL safety in tests

The unit tests for the expert-parallel layer code do not instantiate `NcclBackend`. They hold a `Box<dyn CommBackend> = Box::new(SingleGpuBackend)` and verify the code path by checking that the layer calls `all_reduce` at the right moment — the launch recorder in `MockGpuBackend` plus a trace in `SingleGpuBackend` is enough. The real NCCL path is validated by `scripts/test-minimax-ep2.sh` against a live two-node cluster.

## What's explicitly not here

- **No kernel code.** The EP=2 token-dispatch logic lives in Rust at `crates/model-layers/src/layers/moe/forward_ep.rs`, and the routed grouped-GEMM kernel in `kernels/gb10/common/moe_w4a16_grouped_gemm.cu`.
- **No scheduler logic.** That's `metrale-server::scheduler`.
- **No RDMA-specific code.** Metrale Engine talks through NCCL; NCCL talks through `libibverbs`/`librdmacm`. We do not bypass.

Adding a new collective-ops library is a single `impl CommBackend` in a new module here plus a selection arm in `metrale-server::main` that picks the right backend given the vendor.
