// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: DeepSeek-V4.1 Flash projections on raw K-quant blocks read in place, with no BF16 expansion
// and no requantisation: Q2_K and Q3_K (routed experts, attention) and Q6_K (the output head).
//
// Owner: gb10 kernels (deepseek-v4-flash, built for deepseek-v4.1-flash through kernel_source).
// Invariants: none beyond the types.
//
// Decode (m <= KQ_MAX_M = 8 activation rows): kquant_q8_1_rows_bf16 quantises the activation to plain
// block_q8_1 (32 values, d and sum), and the kquant_mmvq_* GEMVs score each 256-value super-block with the
// vendored vec_dot_q{2,3,6}_K_q8_1 (dp4a on the packed codes). Prefill: metrale_q{2,3}_k_mmq128_{nc,wc}
// run the vendored MMQ tile with the type set to Q2_K or Q3_K, as qwen3.6-27b's q2_0_mmq.cu does for Q2_0,
// on activations quantised by metrale_q8_1_quantize_d2s6_bf16 (Q2_K) or metrale_q8_1_quantize_d4_bf16 (Q3_K).
// The CPU oracle is crates/model-layers/src/layers/ops/kquant_mmq_tests.rs; this tree's KERNEL.toml adds --fmad=false.









#include <cuda_bf16.h>
// 2026-09-25: build.rs compiles staged copies of the sources and passes the directory that resolved this
// file as -I (CompileJob::include_dir), so this path to the vendored headers in the qwen3.6-27b tree resolves.



#include "../../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"
#include "../../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/quantize_impl.cuh"



template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void metrale_kq_tile(
        const char * __restrict__ x, const int * __restrict__ y, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int ncols_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) break;
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int it = blockIdx.x;
    const int jt = blockIdx.y;

    const int offset_y   = jt*mmq_x*(int)(sizeof(block_q8_1_mmq)/sizeof(int));
    const int offset_dst = jt*mmq_x*stride_col_dst + it*mmq_y;
    const int tile_x_max_i = nrows_x   - it*mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt*mmq_x - 1;
    const int offset_x = it*mmq_y*stride_row_x;
    const int kb0_stop = ncols_x / qk;

    mul_mat_q_process_tile<type, mmq_x, need_check, false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

#define KQ_MMQ_ENTRY(NAME, TYPE, CHECK)                                                              \
extern "C" __global__ void __launch_bounds__(256, 1) NAME(                                           \
        const char* x, const int* y, __nv_bfloat16* dst,                                             \
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {\
    metrale_kq_tile<TYPE, 128, CHECK>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst); \
}
KQ_MMQ_ENTRY(metrale_q2_k_mmq128_nc, GGML_TYPE_Q2_K, false)
KQ_MMQ_ENTRY(metrale_q2_k_mmq128_wc, GGML_TYPE_Q2_K, true)
KQ_MMQ_ENTRY(metrale_q3_k_mmq128_nc, GGML_TYPE_Q3_K, false)
KQ_MMQ_ENTRY(metrale_q3_k_mmq128_wc, GGML_TYPE_Q3_K, true)

// 2026-09-25: MMQ activation quantisers, BF16 in: the D2S6 q8_1 layout for Q2_K and the D4 layout for Q3_K.
extern "C" __global__ void metrale_q8_1_quantize_d2s6_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_D2S6, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}
extern "C" __global__ void metrale_q8_1_quantize_d4_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_D4, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}



// 2026-09-25: bf16 [M, K] -> block_q8_1 [M, K/32]. One warp per 32-value block: d = amax/127,
// q = round(x/d), ds = (d, sum x). K must be a multiple of 32.
extern "C" __global__ void kquant_q8_1_rows_bf16(
        const __nv_bfloat16* __restrict__ x, void* __restrict__ vy,
        unsigned int K, unsigned int M) {
    const unsigned int nblk = K / QK8_1;
    const unsigned int gw   = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    if (gw >= M * nblk) return;
    const unsigned int m  = gw / nblk;
    const unsigned int ib = gw % nblk;
    const float xi = __bfloat162float(x[(size_t)m * K + (size_t)ib * QK8_1 + lane]);
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    block_q8_1* y = (block_q8_1*)vy;
    y[gw].qs[lane] = q;
    if (lane == 0) y[gw].ds = make_half2(d, sum);
}

// 2026-09-25: moe_v41_swiglu and kquant_q8_1_rows_bf16 in one launch: h[i] = bf16(swiglu(gate, up) * w) with
// moe_v41_swiglu's expression, then the q8_1 block of the 32 BF16 values just written, with the rows
// kernel's reductions and rounding, so the bytes match the two launches. One warp per 32-value block;
// inter % 32 == 0.

extern "C" __global__ void kquant_swiglu_q8_1_rows_bf16(
        const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
        const float* __restrict__ w, __nv_bfloat16* __restrict__ h, void* __restrict__ vy,
        unsigned int rows, unsigned int inter, float limit) {
    const unsigned int nblk = inter / QK8_1;
    const unsigned int gw   = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    if (gw >= rows * nblk) return;
    const unsigned int i = gw * QK8_1 + lane;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    float v = (g / (1.0f + expf(-g))) * u;
    if (w != nullptr) v *= w[i / inter];
    const __nv_bfloat16 hb = __float2bfloat16(v);
    h[i] = hb;
    const float xi = __bfloat162float(hb);
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    block_q8_1* y = (block_q8_1*)vy;
    y[gw].qs[lane] = q;
    if (lane == 0) y[gw].ds = make_half2(d, sum);
}

#define KQ_NWARPS 4
#define KQ_MAX_M  8

template <ggml_type type>
static __device__ __forceinline__ float kq_vec_dot(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs);
template <> __device__ __forceinline__ float kq_vec_dot<GGML_TYPE_Q2_K>(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    return vec_dot_q2_K_q8_1(vbq, bq8_1, kbx, iqs);
}
template <> __device__ __forceinline__ float kq_vec_dot<GGML_TYPE_Q3_K>(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    return vec_dot_q3_K_q8_1(vbq, bq8_1, kbx, iqs);
}
template <> __device__ __forceinline__ float kq_vec_dot<GGML_TYPE_Q6_K>(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    return vec_dot_q6_K_q8_1(vbq, bq8_1, kbx, iqs);
}

// 2026-09-25: One block (32 x KQ_NWARPS threads) per output row blockIdx.x; m activation rows (<= KQ_MAX_M)
// share every weight read. dst[j * nrows_x + row], BF16.
template <ggml_type type>
static __device__ __forceinline__ void kq_mmvq(
        const void* __restrict__ x_row, const block_q8_1* __restrict__ y,
        __nv_bfloat16* __restrict__ dst, const int ncols_x, const int nrows_x, const int m) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = 1;
    constexpr int blocks_per_iter = vdr * KQ_NWARPS * 32 / qi;
    const int tid = 32 * threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = ncols_x / QK8_1;

    float tmp[KQ_MAX_M];
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) tmp[j] = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        for (int j = 0; j < m; ++j) {
            tmp[j] += kq_vec_dot<type>(x_row, &y[(size_t)j * blocks_per_col_y + kby], kbx, kqs);
        }
    }

    __shared__ float red[KQ_NWARPS][KQ_MAX_M];
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) {
        const float v = warp_reduce_sum(tmp[j]);
        if (threadIdx.x == 0) red[threadIdx.y][j] = v;
    }
    __syncthreads();
    if (threadIdx.y == 0 && (int)threadIdx.x < m) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < KQ_NWARPS; ++w) s += red[w][threadIdx.x];
        dst[(size_t)threadIdx.x * nrows_x + row] = __float2bfloat16(s);
    }
}

// 2026-09-25: vx [N rows][K/256 blocks] raw, vy block_q8_1 [m][K/32], dst BF16 [m][N]. Grid (N), block (32, KQ_NWARPS).
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    const char* x_row = (const char*)vx + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q2_K);
    kq_mmvq<GGML_TYPE_Q2_K>(x_row, (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    const char* x_row = (const char*)vx + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q3_K);
    kq_mmvq<GGML_TYPE_Q3_K>(x_row, (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}

// 2026-09-25: kq_mmvq batched over experts: blockIdx.y picks expert e, whose blocks come from vxs[e]; its
// activation is vy + e * y_stride_bytes (0 shares one activation) and its output dst + e * m * nrows_x.
// The per-row code is kq_mmvq's. Grid (nrows_x, n_experts), block (32, KQ_NWARPS).





extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_experts(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const char* x_row = (const char*)vxs[e] + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q2_K);
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq<GGML_TYPE_Q2_K>(x_row, y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k_experts(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const char* x_row = (const char*)vxs[e] + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q3_K);
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq<GGML_TYPE_Q3_K>(x_row, y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}

// 2026-09-25: Warp-per-row form: warp threadIdx.y of block blockIdx.x owns row blockIdx.x * NW + threadIdx.y;
// its 32 lanes take two super-blocks per step (qi = 16 lanes each) with kq_vec_dot, and one warp shuffle
// reduces them, with no __syncthreads. The products are kq_mmvq's; only the order they are summed in differs.
// y_row_blocks is the q8_1 blocks between activation rows (ncols_x / 32, or more when the input is a column
// slice of a wider row); dst_row_stride is the BF16 columns between output rows.









template <ggml_type type, int NW = KQ_NWARPS>
static __device__ __forceinline__ void kq_mmvq_warp_s(
        const char* __restrict__ x_rows, const size_t row_bytes, const block_q8_1* __restrict__ y,
        const int y_row_blocks, __nv_bfloat16* __restrict__ dst, const int dst_row_stride,
        const int ncols_x, const int nrows_x, const int m) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = 1;
    constexpr int blocks_per_iter = vdr * 32 / qi;
    const int lane = threadIdx.x;
    const int row = blockIdx.x * NW + threadIdx.y;
    if (row >= nrows_x) return;
    const char* x_row = x_rows + (size_t)row * row_bytes;
    const int blocks_per_row_x = ncols_x / qk;
    float tmp[KQ_MAX_M];
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) tmp[j] = 0.0f;
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (lane % (qi / vdr));
        for (int j = 0; j < m; ++j) {
            tmp[j] += kq_vec_dot<type>(x_row, &y[(size_t)j * y_row_blocks + kby], kbx, kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) {
        const float v = warp_reduce_sum(tmp[j]);
        if (lane == 0 && j < m) dst[(size_t)j * dst_row_stride + row] = __float2bfloat16(v);
    }
}
template <ggml_type type>
static __device__ __forceinline__ void kq_mmvq_warp(
        const char* __restrict__ x_rows, const size_t row_bytes, const block_q8_1* __restrict__ y,
        __nv_bfloat16* __restrict__ dst, const int ncols_x, const int nrows_x, const int m) {
    kq_mmvq_warp_s<type>(x_rows, row_bytes, y, ncols_x / QK8_1, dst, nrows_x, ncols_x, nrows_x, m);
}

// 2026-09-25: Grid (ceil(nrows_x / KQ_NWARPS)), block (32, KQ_NWARPS).
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_w(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    kq_mmvq_warp<GGML_TYPE_Q2_K>((const char*)vx, (size_t)(ncols_x / QK_K) * sizeof(block_q2_K),
                                 (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k_w(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    kq_mmvq_warp<GGML_TYPE_Q3_K>((const char*)vx, (size_t)(ncols_x / QK_K) * sizeof(block_q3_K),
                                 (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}
// 2026-09-25: n_groups projections in one launch: group g = blockIdx.y multiplies weight rows [g * nrows_x,
// (g + 1) * nrows_x) with columns [g * ncols_x, (g + 1) * ncols_x) of one activation row quantised whole
// (q8_1 blocks are 32 values and independent, so a slice's blocks are the ones the slice alone would give),
// into columns [g * nrows_x, (g + 1) * nrows_x) of an output row n_groups * nrows_x wide. The per-row code is
// kq_mmvq_warp_s's. Grid (ceil(nrows_x / KQ_NWARPS), n_groups), block (32, KQ_NWARPS).




extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_groups_w(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int n_groups) {
    const unsigned int g = blockIdx.y;
    const size_t row_bytes = (size_t)(ncols_x / QK_K) * sizeof(block_q2_K);
    const char* x_rows = (const char*)vx + (size_t)g * nrows_x * row_bytes;
    const block_q8_1* y = (const block_q8_1*)vy + (size_t)g * (ncols_x / QK8_1);
    kq_mmvq_warp_s<GGML_TYPE_Q2_K>(x_rows, row_bytes, y, (int)(n_groups * ncols_x / QK8_1),
                                   dst + (size_t)g * nrows_x, (int)(n_groups * nrows_x),
                                   (int)ncols_x, (int)nrows_x, (int)m);
}

// 2026-09-25: Two Q2_K projections of one activation in one launch: blockIdx.y picks tensor 0 or 1 (its
// blocks, row count and output), both read the one q8_1 activation, and the per-row code is kq_mmvq_warp's.
// Grid (ceil(max(nrows0, nrows1) / KQ_NWARPS), 2), block (32, KQ_NWARPS).





extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_pair_w(
        const void* __restrict__ vx0, __nv_bfloat16* __restrict__ dst0, unsigned int nrows0,
        const void* __restrict__ vx1, __nv_bfloat16* __restrict__ dst1, unsigned int nrows1,
        const void* __restrict__ vy, unsigned int ncols_x, unsigned int m) {
    const bool second = blockIdx.y != 0;
    const void* vx = second ? vx1 : vx0;
    __nv_bfloat16* dst = second ? dst1 : dst0;
    const unsigned int nrows_x = second ? nrows1 : nrows0;
    kq_mmvq_warp<GGML_TYPE_Q2_K>((const char*)vx, (size_t)(ncols_x / QK_K) * sizeof(block_q2_K),
                                 (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}

// 2026-09-25: The Q6_K output head on raw 210-byte super-blocks through the vendored vec_dot_q6_K_q8_1
// (qi = 32: one super-block per warp step). Grid (ceil(nrows_x / KQ_NWARPS)), block (32, KQ_NWARPS).

extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q6_k_w(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    kq_mmvq_warp<GGML_TYPE_Q6_K>((const char*)vx, (size_t)(ncols_x / QK_K) * sizeof(block_q6_K),
                                 (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}

// 2026-09-25: Grid (ceil(nrows_x / KQ_NWARPS), n_experts), block (32, KQ_NWARPS).
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_experts_w(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq_warp<GGML_TYPE_Q2_K>((const char*)vxs[e], (size_t)(ncols_x / QK_K) * sizeof(block_q2_K),
                                 y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k_experts_w(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq_warp<GGML_TYPE_Q3_K>((const char*)vxs[e], (size_t)(ncols_x / QK_K) * sizeof(block_q3_K),
                                 y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}

// 2026-09-25: The expert batch at 2 and 8 warps per block, per-row code unchanged. Grid (ceil(nrows_x / NW),
// n_experts), block (32, NW). METRALE_DS41_EXPERT_WARPS picks 2, 4 (the _w entries) or 8; the default is 8.



#define KQ_EXPERTS_W_ENTRY(TYPE, BLOCK, SUFFIX, NW)                                                     \
    extern "C" __global__ void __launch_bounds__(32 * NW) kquant_mmvq_##SUFFIX(                         \
            const void* const* __restrict__ vxs, const void* __restrict__ vy,                             \
            __nv_bfloat16* __restrict__ dst, unsigned int ncols_x, unsigned int nrows_x, unsigned int m,  \
            unsigned int y_stride_bytes) {                                                                \
        const unsigned int e = blockIdx.y;                                                                \
        const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);          \
        kq_mmvq_warp_s<TYPE, NW>((const char*)vxs[e], (size_t)(ncols_x / QK_K) * sizeof(BLOCK), y,        \
                                 (int)(ncols_x / QK8_1), dst + (size_t)e * m * nrows_x, (int)nrows_x,    \
                                 (int)ncols_x, (int)nrows_x, (int)m);                                     \
    }
KQ_EXPERTS_W_ENTRY(GGML_TYPE_Q2_K, block_q2_K, q2_k_experts_w2, 2)
KQ_EXPERTS_W_ENTRY(GGML_TYPE_Q2_K, block_q2_K, q2_k_experts_w8, 8)
KQ_EXPERTS_W_ENTRY(GGML_TYPE_Q3_K, block_q3_K, q3_k_experts_w2, 2)
KQ_EXPERTS_W_ENTRY(GGML_TYPE_Q3_K, block_q3_K, q3_k_experts_w8, 8)
