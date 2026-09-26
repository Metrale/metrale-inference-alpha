// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Gate self-test, positive fixture: a kernel that compiles for every arch the
// gate runs at (measured 2026-09-26, CUDA 13.0.88: sm_90a, sm_100a, sm_120a, sm_121a, sm_121f).
// If it fails, the toolchain or the arch string is wrong, and hopper_ptx_gate.sh refuses.

extern "C" __global__ void metrale_gate_selftest_good(const float *in, float *out,
                                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * 2.0f + 1.0f;
  }
}
