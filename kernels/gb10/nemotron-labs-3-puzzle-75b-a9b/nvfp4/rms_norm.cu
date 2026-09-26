// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/rms_norm.cu (2026-09-24; 1630 of 1433 lines differ, see kernels/FORKS.md)

// 2026-09-25: RMS norm kernels for the nemotron-labs-3-puzzle-75b-a9b tree (module norm). Every kernel here scales
// by the weight itself, x x rsqrt(mean(x^2) + eps) x w; common/rms_norm.cu's rms_norm uses (1 + w) instead.
//
// Owner: gb10 kernels (nemotron-labs-3-puzzle-75b-a9b).
// Invariants: none beyond the types.
//
// The row kernels run one block per row, sum squares in FP32, and hold one partial per warp in warp_sums[32], so
// blockDim.x is at most 1024 (the ops launchers use min(hidden_size, 1024)).

#include <cuda_bf16.h>


__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}


__device__ __forceinline__ unsigned int pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}






// 2026-09-25: out = x x rsqrt(mean(x^2) + eps) x w per row; BF16 in and out.
extern "C" __global__ void rms_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;


    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }


    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();


    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);


    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}






// 2026-09-25: rms_norm, and residual = input.
extern "C" __global__ void rms_norm_residual(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    __nv_bfloat16* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);


    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    unsigned int* res32 = (unsigned int*)res;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int x_packed = x32[i];
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x_packed, xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res32[i] = x_packed;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = x[hidden_size - 1];
    }
}







// 2026-09-25: hidden += src (stored as BF16), then out = rms_norm(hidden) and residual = hidden.
extern "C" __global__ void residual_add_rms_norm(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    __nv_bfloat16* res = residual + token * hidden_size;


    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    unsigned int* h32 = (unsigned int*)h;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float hv0, hv1, sv0, sv1;
        unpack_bf16x2(h32[i], hv0, hv1);
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = hv0 + sv0;
        float new1 = hv1 + sv1;
        h32[i] = pack_bf16x2(new0, new1);
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = __bfloat162float(h[hidden_size - 1]);
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = __float2bfloat16(nv);
        sum_sq += nv * nv;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);


    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    unsigned int* res32 = (unsigned int*)res;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int h_packed = h32[i];
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(h_packed, xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res32[i] = h_packed;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(h[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = h[hidden_size - 1];
    }
}
















// 2026-09-25: Gate first, then a norm per group: t = x x silu(g), each group_size slice of t is multiplied by
// its own rsqrt(mean(t^2) + eps), then by w. Token t's gate row starts at t x gate_stride. Assumes
// hidden_size / group_size <= 8 (group_rms[8]), group_size % 128 == 0 so that the 32 quads a warp handles
// in one iteration share a group, and hidden_size <= 16 x blockDim.x (temp_cache holds 4 quads per thread).
extern "C" __global__ void gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;

    const unsigned int quad_size = hidden_size / 4;
    const unsigned int group_quads = group_size / 4;
    const unsigned int num_groups = hidden_size / group_size;

    const unsigned long long* x64 = (const unsigned long long*)x;
    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int lane_id = tid % 32;











    // 2026-09-25: Each (group, warp) pair has its own slot, written only by that warp's lane 0, and the slots
    // are summed in ascending warp order, so the result does not depend on warp scheduling.
    __shared__ float group_rms[8];
    __shared__ float group_warp_sums[8][32];
    const unsigned int warp_id  = tid / 32;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    for (unsigned int z = tid; z < 8u * 32u; z += blockDim.x)
        ((float*)group_warp_sums)[z] = 0.0f;
    __syncthreads();


    float temp_cache[16];
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long xv = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)xv, f0, f1);
        unpack_bf16x2((unsigned int)(xv >> 32), f2, f3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);


        float s0 = g0 / (1.0f + __expf(-g0));
        float s1 = g1 / (1.0f + __expf(-g1));
        float s2 = g2 / (1.0f + __expf(-g2));
        float s3 = g3 / (1.0f + __expf(-g3));


        float t0 = f0 * s0, t1 = f1 * s1, t2 = f2 * s2, t3 = f3 * s3;
        temp_cache[n_cached]     = t0;
        temp_cache[n_cached + 1] = t1;
        temp_cache[n_cached + 2] = t2;
        temp_cache[n_cached + 3] = t3;
        n_cached += 4;




        float sq = t0*t0 + t1*t1 + t2*t2 + t3*t3;
        float warp_sq = warp_reduce_sum(sq);
        if (lane_id == 0) {
            unsigned int grp = i / group_quads;



            group_warp_sums[grp][warp_id] += warp_sq;
        }
    }
    __syncthreads();



    if (tid < num_groups) {
        float s = 0.0f;
        for (unsigned int w = 0; w < num_warps; ++w) s += group_warp_sums[tid][w];
        group_rms[tid] = rsqrtf(s / (float)group_size + eps);
    }
    __syncthreads();


    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float rms = group_rms[i / group_quads];

        float t0 = temp_cache[ci];
        float t1 = temp_cache[ci + 1];
        float t2 = temp_cache[ci + 2];
        float t3 = temp_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned int lo = pack_bf16x2(t0 * rms * w0, t1 * rms * w1);
        unsigned int hi = pack_bf16x2(t2 * rms * w2, t3 * rms * w3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}





// 2026-09-25: In place, x /= sqrt(sum(x^2) + eps) over each head_dim slice; head h of token t starts at
// t x stride + h x head_dim. Grid (num_heads, num_tokens), block min(head_dim, 1024) (ops::l2_norm).
extern "C" __global__ void l2_norm_bf16(
    __nv_bfloat16* __restrict__ data,
    unsigned int head_dim,
    float eps,
    unsigned int stride
) {
    unsigned int head = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* x = data + (unsigned long long)token * stride + head * head_dim;

    float sum_sq = 0.0f;
    const unsigned int half_size = head_dim / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((head_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[head_dim - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float inv_norm = rsqrtf(warp_sums[0] + eps);

    unsigned int* out32 = (unsigned int*)x;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        out32[i] = pack_bf16x2(v0 * inv_norm, v1 * inv_norm);
    }
    if ((head_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[head_dim - 1]);
        x[head_dim - 1] = __float2bfloat16(val * inv_norm);
    }
}



















// 2026-09-25: residual[i] += src[i] for i < n, FP32 += BF16, one thread per element.
extern "C" __global__ void f32_residual_add(
    float* __restrict__ residual,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual[i] += __bfloat162float(src[i]);
    }
}


// 2026-09-25: FP32 input: out = x x rsqrt(mean(x^2) + eps) x w x silu(g) over the whole row, norm before the
// gate. group_size is ignored.
extern "C" __global__ void gated_rms_norm_f32_input(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size
) {
    (void)group_size;
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;


    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float f = x[i];
        sum_sq += f * f;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);


    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    const unsigned int quad_size = hidden_size / 4;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x[base];
        float f1 = x[base + 1];
        float f2 = x[base + 2];
        float f3 = x[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}


// 2026-09-25: out = x x rsqrt(mean(x^2) + eps) x w x silu(g) per (head, token) row of head_dim values. Token
// t's input and output rows start at t x input_token_stride and its gate row at t x gate_token_stride, head h
// at h x head_dim within them. Grid (heads_per_token, num_actual_tokens), block min(head_dim, 1024)
// (ops::gated_rms_norm_prefill); x_cache holds 4 quads per thread, so head_dim <= 16 x blockDim.x.
extern "C" __global__ void gated_rms_norm_prefill(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps,
    unsigned int input_token_stride,
    unsigned int gate_token_stride
) {
    unsigned int head = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + (unsigned long long)token * input_token_stride + head * head_dim;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_token_stride + head * head_dim;
    __nv_bfloat16* out = output + (unsigned long long)token * input_token_stride + head * head_dim;

    const unsigned int quad_size = head_dim / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)v, f0, f1);
        unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);

    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}








// 2026-09-25: hidden (FP32) += src (BF16), then out = rms_norm(hidden) as BF16 and residual = hidden in FP32.
extern "C" __global__ void residual_add_rms_norm_f32(
    float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;


    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float sv0, sv1;
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = h[base]     + sv0;
        float new1 = h[base + 1] + sv1;
        h[base]     = new0;
        h[base + 1] = new1;
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = h[hidden_size - 1];
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = nv;
        sum_sq += nv * nv;
    }


    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);



    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = h[base];
        float xv1 = h[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = h[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}

// 2026-09-25: The same computation as residual_add_rms_norm_f32 above.
extern "C" __global__ void residual_add_rms_norm_f32_abs(
    float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float sv0, sv1;
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = h[base]     + sv0;
        float new1 = h[base + 1] + sv1;
        h[base]     = new0;
        h[base + 1] = new1;
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = h[hidden_size - 1];
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = nv;
        sum_sq += nv * nv;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = h[base];
        float xv1 = h[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = h[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}






// 2026-09-25: residual_add_rms_norm that also writes the normed row in FP32 to output_f32, the MoE router input
// under METRALE_FP32_ROUTING (ops::residual_add_rms_norm_gatef32). output and residual are computed as in
// residual_add_rms_norm.
extern "C" __global__ void residual_add_rms_norm_gatef32(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ output_f32,
    __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* outf = output_f32 + token * hidden_size;
    __nv_bfloat16* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    unsigned int* h32 = (unsigned int*)h;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float hv0, hv1, sv0, sv1;
        unpack_bf16x2(h32[i], hv0, hv1);
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = hv0 + sv0;
        float new1 = hv1 + sv1;
        h32[i] = pack_bf16x2(new0, new1);
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = __bfloat162float(h[hidden_size - 1]);
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = __float2bfloat16(nv);
        sum_sq += nv * nv;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    unsigned int* res32 = (unsigned int*)res;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int h_packed = h32[i];
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(h_packed, xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        float n0 = xv0 * rms * wv0;
        float n1 = xv1 * rms * wv1;
        out32[i] = pack_bf16x2(n0, n1);
        outf[i * 2]     = n0;
        outf[i * 2 + 1] = n1;
        res32[i] = h_packed;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(h[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        float n = val * rms * w;
        out[hidden_size - 1] = __float2bfloat16(n);
        outf[hidden_size - 1] = n;
        res[hidden_size - 1] = h[hidden_size - 1];
    }
}
// 2026-09-25: rms_norm with FP32 input.
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base], xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}




// 2026-09-25: The same computation as rms_norm_f32 above.
extern "C" __global__ void rms_norm_f32_in_abs(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}








// 2026-09-25: FP32 input: out = rms_norm(x) as BF16, and residual = x in FP32.
extern "C" __global__ void rms_norm_residual_f32(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;


    float sum_sq = 0.0f;

    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }


    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();


    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);



    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}

// 2026-09-25: The same computation as rms_norm_residual_f32 above.
extern "C" __global__ void rms_norm_residual_f32_abs(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}














// 2026-09-25: rms_norm over gridDim.y groups of gridDim.x rows in one launch: row (x, y) starts at
// y x row_stride + x x hidden_size, and each block runs rms_norm's per-row body on its row
// (ops::rms_norm_strided: grid (rows_per_group, num_groups)).
extern "C" __global__ void rms_norm_strided(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps,
    unsigned int row_stride
) {
    unsigned int tid = threadIdx.x;
    unsigned long long base =
        (unsigned long long)blockIdx.y * row_stride +
        (unsigned long long)blockIdx.x * hidden_size;

    const __nv_bfloat16* x = input + base;
    __nv_bfloat16* out = output + base;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}
