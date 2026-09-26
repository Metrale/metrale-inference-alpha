// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Token embedding gather: out[t, :] = embed_table[token_ids[t], :],
// one thread per (hidden, token) element on grid (hidden_size, num_tokens).
// A token id >= vocab_size writes a zero row.
//
// Layout:
//   token_ids   : uint32 [num_tokens]
//   embed_table : bfloat [vocab_size, hidden_size]
//   out         : bfloat [num_tokens, hidden_size]
//
// Owner: metal kernels. Invariants: none beyond the types.

#include <metal_stdlib>
using namespace metal;

kernel void embed_lookup(
    constant uint &num_tokens   [[buffer(0)]],
    constant uint &hidden_size  [[buffer(1)]],
    constant uint &vocab_size   [[buffer(2)]],
    device const uint   *token_ids   [[buffer(3)]],
    device const bfloat *embed_table [[buffer(4)]],
    device bfloat       *out         [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint tok_idx = gid.y;
    uint hid_idx = gid.x;
    if (tok_idx >= num_tokens || hid_idx >= hidden_size) {
        return;
    }
    uint v = token_ids[tok_idx];
    if (v >= vocab_size) {



        out[tok_idx * hidden_size + hid_idx] = bfloat(0.0);
        return;
    }
    out[tok_idx * hidden_size + hid_idx] =
        embed_table[v * hidden_size + hid_idx];
}
