// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4 W4A4 GEMM for the dense FFN on the vendored MMQ in q4k_vendor/ (block-scale MMA over E2M1
// values with UE4M3 scales), launched from crates/model-layers/src/layers/ops/nvfp4_mmq.rs; plus the weight repack,
// the activation quantizers and the scale2 folds. Conventional 2D tiling: no MoE ids, one channel. dst is BF16
// [M, N]. Grid (ceil(N / 128), ceil(M / mmq_x), 1), Block (32, 8, 1).
//
// Weights: metrale_nvfp4_repack turns the checkpoint layout (E2M1 [N, K/2], low nibble = even k, and E4M3 [N, K/16]
// scales) into per-row [qs: K/64 * 32 bytes][d: K/64 * 4 bytes], a bit shuffle with no requantization. Byte j of
// sub-block s holds values 16s + j (low nibble) and 16s + 8 + j (high nibble). The MMA applies only the per-16
// scales; the caller folds in the per-tensor FP32 scale2 (see ops/nvfp4_mmq.rs).
//
// Owner: gb10 kernels (qwen3.6-27b).
// Invariants: none beyond the types. Every metrale_nvfp4_* export is defined only inside the
// BLACKWELL_MMA_AVAILABLE guard (crates/kernels/tests/nvfp4_mmq_capability.rs).


#include <cuda_bf16.h>
#include "q4k_vendor/mmq.cuh"
#include "q4k_vendor/quantize_impl.cuh"

// 2026-09-25: No export may resolve to the vendor's NO_DEVICE_CODE trap, so all of them sit behind the vendor's
// capability macro (compute capability 12.x). Without them DenseFfn's MMQ arm does not engage and it keeps its
// W4A16 path and transposed weights (crates/model-layers/src/layers/dense_ffn.rs).
#if defined(BLACKWELL_MMA_AVAILABLE) // Metrale Engine optional module

// 2026-09-25: One (N tile, M tile) of the GEMM: identity ids, one channel and sample; the vendored
// mul_mat_q_process_tile does the work.
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void metrale_nvfp4_tile(
        const char * __restrict__ x, const int * __restrict__ y, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int ncols_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst) {
    constexpr ggml_type type = GGML_TYPE_NVFP4;
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

    // 2026-09-25: block_fp4_mmq has the size of block_q8_1_mmq (static_assert in q4k_vendor/mmq.cuh), so this is
    // q4k_mmq.cu's y offset.
    const int offset_y   = jt*mmq_x*(int)(sizeof(block_fp4_mmq)/sizeof(int));
    const int offset_dst = jt*mmq_x*stride_col_dst + it*mmq_y;
    const int tile_x_max_i = nrows_x   - it*mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt*mmq_x - 1;
    const int offset_x = it*mmq_y*stride_row_x;
    const int kb0_stop = ncols_x / qk;

    mul_mat_q_process_tile<type, mmq_x, need_check, false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

// 2026-09-25: M tile 128. The `_wc` entries instantiate need_check; the launcher uses them when N % 128 != 0.
extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq128_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<128, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq128_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<128, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}

// 2026-09-25: Small M tiles. The vendored MMA path's granularity is 8 columns (mmq_get_granularity_device), so any
// multiple of 8 is a legal mmq_x; this file instantiates 16, 32, 64 and 128. A 128 tile issues MMAs for all 128
// columns whatever M is. grid.y = ceil(M / mmq_x), and each M tile reads the whole weight again, which is why
// ops::nvfp4_mmq_gemm_tiled requires M <= mmq_x.





extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq16_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<16, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq16_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<16, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq32_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<32, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq32_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<32, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}











extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq64_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<64, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) metrale_nvfp4_mmq64_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    metrale_nvfp4_tile<64, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}

// 2026-09-25: BF16 [ne1 = M rows, ne00 = K] to block_fp4_mmq: E2M1 values with one UE4M3 scale per 16, the scale
// chosen by trying the first estimate and +-1, +-2 codes. One thread per 16 values; grid (ne1, ceil(ne0 / 2048), 1),
// block (128), with ne0 = K rounded up to 256.
extern "C" __global__ void metrale_nvfp4_quantize_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_nvfp4_worker<__nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}

// 2026-09-25: Weight repack to the per-row layout described at the top of this file: E2M1 codes and E4M3 scale
// bytes are copied, not requantized. One thread per 64-value block; grid ceil(N * K / 64 / 256), block 256.



extern "C" __global__ void metrale_nvfp4_repack(
        const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales,
        block_nvfp4* __restrict__ out, int n_rows, int k) {
    const int64_t nblocks = (int64_t) n_rows * (k / QK_NVFP4);
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;

    const int blocks_per_row = k / QK_NVFP4;
    const int row = (int)(b / blocks_per_row);
    const int kb  = (int)(b % blocks_per_row);

    const uint8_t* prow = packed + (int64_t) row * (k / 2);
    const uint8_t* srow = scales + (int64_t) row * (k / 16);
    block_nvfp4 dst;

    // 2026-09-25: Row r of the output is [qs: blocks_per_row * 32 bytes][d: blocks_per_row * 4 bytes], 36 bytes per
    // block as in block_nvfp4, but with each row's nibble bytes contiguous so the tile loader
    // (load_tiles_nvfp4_nvfp4 in q4k_vendor/mmq.cuh) reads them with 16-byte loads. A row is 16-byte aligned only
    // when blocks_per_row % 4 == 0, i.e. K % 256 == 0; nothing here checks it. The qwen3.6-27b FFN dims (K 5120
    // and 17408) satisfy it.













    const int qs_region = blocks_per_row * 32;
    uint8_t* row_out = (uint8_t*) out + (int64_t) row * blocks_per_row * 36;
    uint8_t* qs_out  = row_out + kb * 32;
    uint8_t* d_out   = row_out + qs_region + kb * 4;

#pragma unroll
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
        const int k0 = kb * QK_NVFP4 + s * QK_NVFP4_SUB;
        dst.d[s] = srow[k0 / 16];
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const int ka = k0 + j;
            const int kb2 = k0 + 8 + j;
            const uint8_t na = (prow[ka >> 1] >> ((ka & 1) * 4)) & 0xF;
            const uint8_t nb = (prow[kb2 >> 1] >> ((kb2 & 1) * 4)) & 0xF;
            dst.qs[s * 8 + j] = (int8_t)(na | (nb << 4));
        }
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) qs_out[j] = (uint8_t) dst.qs[j];
#pragma unroll
    for (int s2 = 0; s2 < 4; ++s2) d_out[s2] = dst.d[s2];
}

// 2026-09-25: The MMQ output lacks the per-tensor FP32 scale2, so these two kernels apply it.
// metrale_nvfp4_scale_bf16 multiplies the down projection's output in place. metrale_nvfp4_silu_mul_scaled
// computes silu(gate * gate_scale) * (up * up_scale), moe_silu_mul's formula with the scales folded in and, like
// gb10/common/moe_silu_mul.cu, no clamp. Both: grid ceil(total / 256), block 256.











extern "C" __global__ void metrale_nvfp4_scale_bf16(
    __nv_bfloat16* __restrict__ data, float scale, unsigned int total_elements) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * scale);
}

extern "C" __global__ void metrale_nvfp4_silu_mul_scaled(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output, float gate_scale, float up_scale,
    unsigned int total_elements) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = __bfloat162float(gate[idx]) * gate_scale;
    float u = __bfloat162float(up[idx]) * up_scale;
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    output[idx] = __float2bfloat16(g * sigmoid_g * u);
}

// 2026-09-25: silu(gate * gate_scale) * (up * up_scale), with no clamp, quantized straight to block_fp4_mmq for the
// down GEMM, so no BF16 [M, inter] intermediate is written. The thread mapping and the scale search are those of
// quantize_mmq_nvfp4_worker, one thread per 16 values. ne00 is the intermediate width, ne0 = ne00 rounded up to
// 256, ne1 = M. Grid (M, ceil(ne0 / 2048)), block 128.





extern "C" __global__ void metrale_nvfp4_silu_mul_quant(
        const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
        void* __restrict__ vy, float gate_scale, float up_scale,
        long ne00 , long ne0 , int ne1 ) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) return;
    const int64_t i1 = blockIdx.x;
    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) return;

    const int64_t ib = k_block * ne1 + i1;
    block_fp4_mmq* yb = (block_fp4_mmq*) vy + ib;
    const int sub = (int) ((i0_base % QK_K) / QK_NVFP4_SUB);

    float vals_raw[QK_NVFP4_SUB];
    float amax_raw = 0.0f;
    const int64_t base_idx = i1 * ne00;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        const int64_t i00 = i0_base + k;
        float v = 0.0f;
        if (i00 < ne00) {
            float g = __bfloat162float(gate[base_idx + i00]) * gate_scale;
            float u = __bfloat162float(up[base_idx + i00]) * up_scale;
            v = g * (1.0f / (1.0f + __expf(-g))) * u;
        }
        vals_raw[k] = v;
        amax_raw = fmaxf(amax_raw, fabsf(v));
    }

    static constexpr int test_offsets[5] = {0, -1, 1, -2, 2};
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_raw / 6.0f);
    float best_err = FLT_MAX;
    uint8_t fp8_code = 0;
    float subblock_scale = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) continue;
        const uint8_t code = (uint8_t) test_code;
        const float test_scale = ggml_cuda_ue4m3_to_fp32(code);
        const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals_raw[k];
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
            const float err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }
        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = code;
            subblock_scale = test_scale;
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t q0 = 0, q1 = 0;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 0], inv_scale) << (8 * k);
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 8], inv_scale) << (8 * k + 4);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 4], inv_scale) << (8 * k);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 12], inv_scale) << (8 * k + 4);
    }
    uint32_t* yqs = reinterpret_cast<uint32_t*>(yb->qs);
    yqs[2 * sub + 0] = q0;
    yqs[2 * sub + 1] = q1;
    reinterpret_cast<uint8_t*>(yb->d4)[sub] = fp8_code;
#else
    NO_DEVICE_CODE;
#endif
}

#endif // Metrale Engine optional module
