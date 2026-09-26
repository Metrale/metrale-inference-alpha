# metrale-gpu-sys

**Path:** `crates/gpu-sys/`

Raw FFI only: the cuFile (GDS) and NVML dlopen wrappers, the NCCL bindings and the RDMA verbs C shim. The one-sided RDMA verbs primitive here is shared by every RDMA tier (experts, KV overflow, weight staging, LoRA, SSM snapshots); its wire constants are a frozen external contract. The crate is CUDA-free by constraint, so both the non-CUDA peer daemons and the CUDA client tiers link it.
