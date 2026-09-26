// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Instantiates the header-only NVTX3 API so Rust can call it: `nvtx.rs`
// binds these two functions. `build.rs` compiles this file only with the `nvtx`
// cargo feature.
//
// Owner: metrale-gpu-sys.
// Invariants: none beyond the types.
#include <nvtx3/nvToolsExt.h>

int metrale_nvtx_range_push(const char *name) { return nvtxRangePushA(name); }
int metrale_nvtx_range_pop(void) { return nvtxRangePop(); }
