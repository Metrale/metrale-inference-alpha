// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Smoke kernel for the Metal build and launch path: zeroes the first `n` floats
// of `out`. `metal_alloc_copy_launch_roundtrip` (crates/gpu-runtime/src/metal_backend/
// tests/parity_basic.rs) launches it.


#include <metal_stdlib>
using namespace metal;

kernel void noop_smoke(
    device float *out [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        out[gid] = 0.0f;
    }
}
