// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Scratch footprint of the kernel-batched prefill staging, shared by the arena sizing and the admission check.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types; the functions are pure.

/// 2026-09-25: Streams `scratch` is provisioned for in the kernel-batched
/// prefill staging. A batch whose footprint does not fit is refused by
/// `check_kernel_batched_eligible` (metrale-model-engine) and runs per
/// stream, so this sets how often the batched path is available, not safety.
pub const Q12_SIZING_STREAMS: usize = 8;

/// 2026-09-25: Scratch bytes of the kernel-batched prefill staging for `n`
/// streams of `chunk_len` tokens each. Both `BufferSizes::from_config` and
/// `check_kernel_batched_eligible` call it (or the varlen form), so they agree
/// on whether a batch fits. The terms follow `prefill_b/batch_kernel.rs` (MoE
/// top-K area, `n` per-stream metadata blocks), `prefill_b/stage_batched.rs`
/// (positions, three streams under MRoPE, slots, the block-table and seq_len
/// pointer arrays, `cu_seqlens`, `kv_lens`) and the `h_state_ptrs` table.
pub fn q12_batched_scratch_bytes(n: usize, chunk_len: usize, top_k: usize, mrope: bool) -> usize {
    q12_batched_scratch_bytes_varlen(n, n * chunk_len, chunk_len, top_k, mrope)
}

/// 2026-09-25: [`q12_batched_scratch_bytes`] for a ragged batch: `total_tokens`
/// is the packed total and `max_chunk_len` sizes the per-stream metadata
/// slots, so a varlen batch is not charged the longest stream's length for
/// every stream.
pub fn q12_batched_scratch_bytes_varlen(
    n: usize,
    total_tokens: usize,
    max_chunk_len: usize,
    top_k: usize,
    mrope: bool,
) -> usize {
    let moe = ((total_tokens * top_k * 4 * 2) + 63) & !63;
    let per_stream_meta = ((max_chunk_len * 16) + 64).max(4096);
    let pos = (total_tokens * 4 + 7) & !7;
    let pos_streams = if mrope { 3 } else { 1 };
    let slot = (total_tokens * 8 + 7) & !7;
    let ptrs = ((n * std::mem::size_of::<u64>()) + 7) & !7;
    let cu_seqlens = (((n + 1) * 4) + 7) & !7;
    let kv_lens = ((n * 4) + 7) & !7;
    let stage_meta = pos_streams * pos + slot + 2 * ptrs + cu_seqlens + kv_lens;
    let h_state_ptrs = n * std::mem::size_of::<u64>();
    moe + n * per_stream_meta + stage_meta + h_state_ptrs
}
