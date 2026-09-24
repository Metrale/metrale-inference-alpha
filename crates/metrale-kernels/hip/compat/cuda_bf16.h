#pragma once
#include <hip/hip_bf16.h>
typedef __hip_bfloat16  __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#ifndef METRALE_CVTA_COMPAT
#define METRALE_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
