// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Tensor-core prefill attention for gemma-4-31b's global layers
// (`global_head_dim = 512` in MODEL.toml): kernels/gb10/common/attn_prefill.cu built at
// HDIM 512 and BC 16 as `attn_prefill_512tc`, which `ops::wide_prefill_kernel` prefers to
// the scalar `attn_prefill_512` unless METRALE_ATTN_512_TC=0. The build is the same as
// kernels/gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512tc.cu, which gives the reason for
// BC 16.





#define HDIM 512
#define BC 16
#define METRALE_PREFILL_ENTRY attn_prefill_512tc
#define METRALE_SKIP_PREFILL_64 1
#include "../../common/attn_prefill.cu"
