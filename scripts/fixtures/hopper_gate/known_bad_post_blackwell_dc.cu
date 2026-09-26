// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Gate self-test, negative fixture that hopper_ptx_gate.sh uses for sm_100*: a
// kernel that must not assemble there.
//
// `mma.sync ... .kind::mxf4nvf4.block_scale` is the warp-level block-scaled NVFP4 MMA, the
// instruction kernels/gb10/common/w4a4_gemv_mx.cu issues. ptxas accepts it for sm_120a and
// sm_121a/f and rejects it for sm_90a and sm_100a, so this fixture and
// known_bad_post_hopper.cu are mirror images: each fails where the other passes, except at
// sm_90a, where both fail.
//
// It is inline PTX, so the rejection comes from ptxas, the stage that decides whether a
// kernel can be assembled for a device.
//
// Measured 2026-09-26 with CUDA 13.0.88 (aarch64), `nvcc --ptx -O3` then `ptxas`:
//
//   -arch=sm_90a   FAIL  "Instruction 'mma with block scale' not supported on .target 'sm_90a'"
//   -arch=sm_100a  FAIL  "Instruction 'mma with block scale' not supported on .target 'sm_100a'"
//   -arch=sm_120a  PASS
//   -arch=sm_121a  PASS
//   -arch=sm_121f  PASS
//
// If this file starts to pass for sm_100a, the gate has no failure path there, and its
// self-test refuses to report.






















extern "C" __global__ void metrale_gate_selftest_bad_sm100(const unsigned int *in,
                                                         float *out) {
  unsigned int a0 = in[0], a1 = in[1], a2 = in[2], a3 = in[3];
  unsigned int b0 = in[4], b1 = in[5];
  unsigned int sfa = in[6], sfb = in[7];
  unsigned short bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},"
      "{%4,%5,%6,%7},"
      "{%8,%9},"
      "{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(acc0), "=f"(acc1), "=f"(acc2), "=f"(acc3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(acc0),
        "f"(acc1), "f"(acc2), "f"(acc3), "r"(sfa), "h"(bid_a), "h"(tid_a),
        "r"(sfb), "h"(bid_b), "h"(tid_b));
  out[0] = acc0 + acc1 + acc2 + acc3;
}
