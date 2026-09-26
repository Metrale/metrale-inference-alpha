// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Gate self-test, negative fixture that hopper_ptx_gate.sh uses for sm_9* and
// sm_12*: a kernel that must not assemble there.
//
// `redux.sync.max.abs.f32` is a floating-point warp reduction that, of the arches below,
// only sm_100a accepts. It is inline PTX, so the rejection comes from ptxas, the stage that
// decides whether a kernel can be assembled for a device.
//
// Measured 2026-09-26 with CUDA 13.0.88 (aarch64), `nvcc --ptx -O3` then `ptxas`:
//
//   -arch=sm_90a   FAIL  "Instruction 'redux.f32' not supported on .target 'sm_90a'"
//   -arch=sm_100a  PASS
//   -arch=sm_120a  FAIL  (same message)
//   -arch=sm_121a  FAIL  (same message)
//   -arch=sm_121f  FAIL  (same message)
//
// If this file starts to pass for an arch the gate uses it for, the gate has no failure
// path there, and its self-test refuses to report.








extern "C" __global__ void metrale_gate_selftest_bad(float *inout) {
  float v = inout[0];
  float r;
  asm volatile("redux.sync.max.abs.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(v));
  inout[0] = r;
}
