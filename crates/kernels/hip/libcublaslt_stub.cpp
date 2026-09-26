// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: cuBLASLt stub for the HIP target, built as `libcublasLt.so`
// (Linux) or into the HIP `cuda.dll` (Windows), so the `-lcublasLt` that
// metrale-gpu-runtime's build script emits resolves. Every entry point
// returns 1, a non-success status, so `cublasLtCreate` fails and
// metrale-gpu-runtime's `cublaslt.rs` returns an error instead of running a
// GEMM.
//
// Owner: HIP shims (kernels crate).
// Invariants: every stub returns 1. The stubs declare no parameters; C
// linkage resolves them by name alone.
extern "C" {

int cublasLtCreate() { return 1; }
int cublasLtDestroy() { return 1; }
int cublasLtMatmul() { return 1; }
int cublasLtMatmulAlgoGetHeuristic() { return 1; }
int cublasLtMatmulDescCreate() { return 1; }
int cublasLtMatmulDescDestroy() { return 1; }
int cublasLtMatmulDescSetAttribute() { return 1; }
int cublasLtMatmulPreferenceCreate() { return 1; }
int cublasLtMatmulPreferenceDestroy() { return 1; }
int cublasLtMatmulPreferenceSetAttribute() { return 1; }
int cublasLtMatrixLayoutCreate() { return 1; }
int cublasLtMatrixLayoutDestroy() { return 1; }

}
