// 2026-09-25: C-ABI wrapper around FlashInfer's ragged batched-prefill attention: BF16 Q/K/V/O,
// head_dim 128 or 256, GQA, causal or not, with a sliding window on the head_dim-128 entry.
// It plans with flashinfer::PrefillPlan and launches
// flashinfer::BatchPrefillWithRaggedKVCacheDispatched, following FlashInfer's own
// csrc/batch_prefill.cu caller without its torch wrapping.
//
// Owner: gpu-runtime (FlashInfer reference object).
// Invariants:
// - No persistent state: every scratch buffer is passed in.
// - No C++ exception crosses the C ABI: run_ragged_prefill returns -4 for any caught one.
// - Every entry returns 0 on success and a nonzero status otherwise.






#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstddef>
#include <vector>

#include <flashinfer/allocator.h>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

namespace flashinfer {


// 2026-09-25: Declaration of the dispatched entry, which prefill.cuh (included above) defines.
template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);
}

using flashinfer::BatchPrefillRaggedParams;
using flashinfer::DefaultAttention;
using flashinfer::GetPtrFromBaseOffset;
using flashinfer::MaskMode;
using flashinfer::PosEncodingMode;
using flashinfer::PrefillPlan;
using flashinfer::PrefillPlanInfo;




// 2026-09-25: head_dim is a template parameter; each extern "C" entry pins one value, 128
// or 256.
namespace {
constexpr PosEncodingMode kPosEnc = PosEncodingMode::kNone;
constexpr bool kUseFp16QkReduction = false;





// 2026-09-25: No custom mask, sliding window, logits soft-cap or ALiBi. Used by the
// head_dim-256 entry.
using StandardAttention = DefaultAttention< false,
                                           false,
                                           false,
                                           false>;










// 2026-09-25: The sliding-window variant, used by the head_dim-128 entry for every layer:
// window_left = -1 for full attention, sliding_window - 1 for a windowed layer. The
// engine's own prefill kernel masks key k for query q when q - k >= sliding_window
// (kernels/gb10/common/attn_prefill.cu), hence the - 1; the caller converts
// (flashinfer/hd128.rs).
using WindowedAttention = DefaultAttention< false,
                                           true,
                                           false,
                                           false>;

using Params = BatchPrefillRaggedParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
}















// 2026-09-25: Workspace sizes for metrale_fi_ragged_prefill_*, from over-estimated terms
// rather than from a plan. The batch term is
// max(2 * SM count (512 without a device), max_batch, max_total_qo_rows) + 256. The int
// and pinned-int workspaces get the same size; the float workspace (split-KV tmp_v and
// tmp_s) is sized for cta_tile_q 128. Returns -1 for a null out pointer and -2 for a
// head_dim other than 128 or 256.
// The int terms, in order, are the arrays apply_plan_info reads: request_indices,
// qo_tile_indices, kv_tile_indices ([padded_batch] i32 each), o_indptr [max_batch + 1],
// kv_chunk_size, total_num_rows, merge_indptr [max_total_qo_rows + 1] and
// block_valid_mask [padded_batch] bool, each with 16 bytes of slack.
extern "C" int metrale_fi_ragged_prefill_workspace_sizes(
    uint32_t max_batch, uint32_t max_total_qo_rows, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t head_dim,
    size_t* float_ws_bytes_out, size_t* int_ws_bytes_out, size_t* pinned_int_ws_bytes_out) {
  if (float_ws_bytes_out == nullptr || int_ws_bytes_out == nullptr ||
      pinned_int_ws_bytes_out == nullptr) {
    return -1;
  }
  if (head_dim != 128 && head_dim != 256) {
    return -2;
  }




  int num_sm = 0;
  int dev_id = 0;
  if (cudaGetDevice(&dev_id) == cudaSuccess) {
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
  }
  const uint32_t grid_ceiling = (num_sm > 0) ? static_cast<uint32_t>(2 * num_sm) : 512u;


  uint32_t padded_batch = grid_ceiling;
  if (max_batch > padded_batch) padded_batch = max_batch;
  if (max_total_qo_rows > padded_batch) padded_batch = max_total_qo_rows;

  padded_batch += 256;

  const size_t align_slack = 16;



  size_t int_bytes = 0;
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(int32_t) + align_slack;
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(int32_t) + align_slack;
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(int32_t) + align_slack;
  int_bytes += (static_cast<size_t>(max_batch) + 1) * sizeof(int32_t) + align_slack;
  int_bytes += sizeof(int32_t) + align_slack;
  int_bytes += sizeof(uint32_t) + align_slack;

  int_bytes += (static_cast<size_t>(max_total_qo_rows) + 1) * sizeof(int32_t) + align_slack;
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(bool) + align_slack;



  const uint32_t cta_tile_q_max = 128;
  size_t float_bytes = 0;
  float_bytes += static_cast<size_t>(num_qo_heads) * padded_batch * cta_tile_q_max *
                     head_dim * sizeof(float) + align_slack;
  float_bytes += static_cast<size_t>(num_qo_heads) * padded_batch * cta_tile_q_max *
                     sizeof(float) + align_slack;

  *int_ws_bytes_out = int_bytes;
  *pinned_int_ws_bytes_out = int_bytes;
  *float_ws_bytes_out = float_bytes;
  return 0;
}




namespace {


// 2026-09-25: Point the plan-derived params fields at int_buffer_ptr, and return the
// split-KV scratch (tmp_v, tmp_s) in float_buffer_ptr, or null without split-KV.
inline void apply_plan_info(Params& params, const PrefillPlanInfo& plan_info,
                            void* int_buffer_ptr, void* float_buffer_ptr,
                            __nv_bfloat16** tmp_v_out, float** tmp_s_out) {
  params.request_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);

  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    params.merge_indptr =
        GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.merge_indptr_offset);
    tmp_v = GetPtrFromBaseOffset<__nv_bfloat16>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;
  params.max_total_num_rows = plan_info.total_num_rows;
  if (plan_info.enable_cuda_graph) {
    params.total_num_rows =
        GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
  }
  *tmp_v_out = tmp_v;
  *tmp_s_out = tmp_s;
}







// 2026-09-25: Launch for a fixed head_dim, variant and mask mode; cta_tile_q (from the
// plan) selects the instantiation.
template <uint32_t HEAD_DIM, typename Variant, MaskMode MASK_MODE>
inline cudaError_t run_dispatched(Params& params, __nv_bfloat16* tmp_v, float* tmp_s,
                                  int64_t cta_tile_q, cudaStream_t stream) {
  cudaError_t status = cudaSuccess;
  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
        CTA_TILE_Q, HEAD_DIM, HEAD_DIM, kPosEnc, kUseFp16QkReduction, MASK_MODE,
        Variant, Params>(params, tmp_v, tmp_s, false, stream);
  });
  return status;
}

}

// 2026-09-25: Plan, fill params, launch. window_left is -1 (no window) or
// sliding_window - 1. Status: -2 head_dim mismatch, -3 num_kv_heads 0 or not dividing
// num_qo_heads, -4 a caught C++ exception, else a cudaError_t.
template <uint32_t HEAD_DIM, typename Variant>
static int run_ragged_prefill(
    const void* q, const void* k, const void* v, void* o,
    const int32_t* qo_indptr_h, const int32_t* kv_indptr_h,
    const int32_t* qo_indptr_d, const int32_t* kv_indptr_d,
    uint32_t batch, uint32_t total_qo_rows, uint32_t total_kv_rows,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    float sm_scale, int causal, int32_t window_left,
    void* float_ws, size_t float_ws_bytes,
    void* int_ws, size_t int_ws_bytes,
    void* pinned_int_ws, size_t pinned_int_ws_bytes,
    void* stream_raw) {
  if (head_dim != HEAD_DIM) return -2;
  if (num_kv_heads == 0 || num_qo_heads % num_kv_heads != 0) return -3;



  // 2026-09-25: A C++ exception must not unwind across the C ABI into Rust.
  try {

  cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);




  // 2026-09-25: Plan from the host indptr arrays into int_ws / pinned_int_ws.
  PrefillPlanInfo plan_info;
  cudaError_t status = PrefillPlan<int32_t>(
      float_ws, float_ws_bytes,
      int_ws, pinned_int_ws, int_ws_bytes,
      plan_info,
      const_cast<int32_t*>(qo_indptr_h), const_cast<int32_t*>(kv_indptr_h),
      total_qo_rows,
      batch,
      num_qo_heads, num_kv_heads,
      head_dim, head_dim,
      1,
      false,
      sizeof(__nv_bfloat16),



      window_left,
      -1,
      false,
      0,
      stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }




  // 2026-09-25: Q, K, V and O are contiguous [rows, heads, head_dim].
  Params params;

  params.q = static_cast<__nv_bfloat16*>(const_cast<void*>(q));
  params.k = static_cast<__nv_bfloat16*>(const_cast<void*>(k));
  params.v = static_cast<__nv_bfloat16*>(const_cast<void*>(v));
  params.o = static_cast<__nv_bfloat16*>(o);
  params.lse = nullptr;

  // 2026-09-25: The kernel reads the device indptr copies.
  params.q_indptr = const_cast<int32_t*>(qo_indptr_d);
  params.kv_indptr = const_cast<int32_t*>(kv_indptr_d);

  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.group_size = flashinfer::uint_fastdiv(num_qo_heads / num_kv_heads);

  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.k_stride_n = num_kv_heads * head_dim;
  params.k_stride_h = head_dim;
  params.v_stride_n = num_kv_heads * head_dim;
  params.v_stride_h = head_dim;

  params.window_left = window_left;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;



  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  apply_plan_info(params, plan_info, int_ws, float_ws, &tmp_v, &tmp_s);





  if (causal) {
    status = run_dispatched<HEAD_DIM, Variant, MaskMode::kCausal>(
        params, tmp_v, tmp_s, plan_info.cta_tile_q, stream);
  } else {
    status = run_dispatched<HEAD_DIM, Variant, MaskMode::kNone>(
        params, tmp_v, tmp_s, plan_info.cta_tile_q, stream);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return 0;

  } catch (...) {
    return -4;
  }
}




// 2026-09-25: head_dim 256, no sliding window.
extern "C" int metrale_fi_ragged_prefill_bf16_hd256(
    const void* q, const void* k, const void* v, void* o,
    const int32_t* qo_indptr_h, const int32_t* kv_indptr_h,
    const int32_t* qo_indptr_d, const int32_t* kv_indptr_d,
    uint32_t batch, uint32_t total_qo_rows, uint32_t total_kv_rows,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    float sm_scale, int causal,
    void* float_ws, size_t float_ws_bytes,
    void* int_ws, size_t int_ws_bytes,
    void* pinned_int_ws, size_t pinned_int_ws_bytes,
    void* stream_raw) {
  return run_ragged_prefill<256, StandardAttention>(
      q, k, v, o, qo_indptr_h, kv_indptr_h, qo_indptr_d, kv_indptr_d,
      batch, total_qo_rows, total_kv_rows, num_qo_heads, num_kv_heads, head_dim,
      sm_scale, causal, -1,
      float_ws, float_ws_bytes, int_ws, int_ws_bytes,
      pinned_int_ws, pinned_int_ws_bytes, stream_raw);
}

// 2026-09-25: head_dim 128. The hd256 ABI plus window_left after causal: -1 for full
// attention, sliding_window - 1 for a windowed layer.
extern "C" int metrale_fi_ragged_prefill_bf16_hd128(
    const void* q, const void* k, const void* v, void* o,
    const int32_t* qo_indptr_h, const int32_t* kv_indptr_h,
    const int32_t* qo_indptr_d, const int32_t* kv_indptr_d,
    uint32_t batch, uint32_t total_qo_rows, uint32_t total_kv_rows,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    float sm_scale, int causal, int32_t window_left,
    void* float_ws, size_t float_ws_bytes,
    void* int_ws, size_t int_ws_bytes,
    void* pinned_int_ws, size_t pinned_int_ws_bytes,
    void* stream_raw) {
  return run_ragged_prefill<128, WindowedAttention>(
      q, k, v, o, qo_indptr_h, kv_indptr_h, qo_indptr_d, kv_indptr_d,
      batch, total_qo_rows, total_kv_rows, num_qo_heads, num_kv_heads, head_dim,
      sm_scale, causal, window_left,
      float_ws, float_ws_bytes, int_ws, int_ws_bytes,
      pinned_int_ws, pinned_int_ws_bytes, stream_raw);
}
