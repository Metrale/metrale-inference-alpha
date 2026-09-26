// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/gemma-4-31b/nvfp4/attn_prefill_512tc.cu (2026-09-24; 18 of 27 lines differ, see kernels/FORKS.md)

// 2026-09-26: Tensor-core prefill attention for gemma-4-26b-a4b's global layers
// (`global_head_dim = 512` in MODEL.toml): kernels/gb10/common/attn_prefill.cu built at
// HDIM 512 and BC 16 as `attn_prefill_512tc`, which `ops::wide_prefill_kernel` prefers to
// the scalar `attn_prefill_512` unless METRALE_ATTN_512_TC=0.
//
// BC is 16 because at BC 32 the kernel's static shared memory is 135,936 B, more than the
// 101,376 B a GB10 block can have (`MAX_DYNAMIC_SMEM`, model-layers ops/ssm_ssd.rs); at
// BC 16 it is 84,992 B. METRALE_SKIP_PREFILL_64 leaves out `attn_prefill_64`, which needs
// 120,064 B at this shape.










#define HDIM 512
#define BC 16
#define METRALE_PREFILL_ENTRY attn_prefill_512tc
#define METRALE_SKIP_PREFILL_64 1
#include "../../common/attn_prefill.cu"
