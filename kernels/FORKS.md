# Kernel forks

Generated 2026-09-24 by the reformation's kernel dedup (tools/c/forks.py). A fork is a regular
source whose stem exists elsewhere in `kernels/` with different bytes. Its origin is the
`gb10/common/` file of that stem when one exists, otherwise the stem's oldest file. Each fork
carries a `// forked-from:` header naming the origin and the distance below. Byte-identical
copies no longer exist: they were collapsed into `[sources] use` and `[hardware] inherits`.

Distance = lines difflib reports as changed / lines of the larger file.

| stem | fork | origin | distance | fork lines |
|---|---|---|---|---|
| `dense_gemm_bf16.cu` | `strix-hip/common/dense_gemm_bf16.cu` | `gb10/common/dense_gemm_bf16.cu` | 452/629 (71%) | 629 |
| `dense_gemm_tc.cu` | `strix-hip/common/dense_gemm_tc.cu` | `gb10/common/dense_gemm_tc.cu` | 154/207 (74%) | 207 |
| `dsa_indexer.cu` | `b300/common/dsa_indexer.cu` | `gb10/common/dsa_indexer.cu` | 70/627 (11%) | 627 |
| `embed_scale.cu` | `gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu` | `gb10/gemma-4-31b/nvfp4/embed_scale.cu` | 4/36 (11%) | 36 |
| `gated_delta_rule.cu` | `gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu` | `gb10/common/gated_delta_rule.cu` | 2069/1769 (116%) | 1466 |
| `gated_delta_rule.cu` | `gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu` | `gb10/common/gated_delta_rule.cu` | 1878/1580 (118%) | 1466 |
| `gated_delta_rule.cu` | `gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu` | `gb10/common/gated_delta_rule.cu` | 1928/1630 (118%) | 1466 |
| `gated_delta_rule.cu` | `gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu` | `gb10/common/gated_delta_rule.cu` | 2213/2514 (88%) | 1466 |
| `gated_delta_rule.cu` | `gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu` | `gb10/common/gated_delta_rule.cu` | 2339/1713 (136%) | 1466 |
| `hyper_connection.cu` | `gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu` | `gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu` | 628/591 (106%) | 288 |
| `attn_prefill.cu` | `strix-hip/common/attn_prefill.cu` | `gb10/common/attn_prefill.cu` | 982/990 (99%) | 990 |
| `attn_prefill_512.cu` | `gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu` | `gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu` | 38/141 (26%) | 114 |
| `attn_prefill_512.cu` | `gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu` | `gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu` | 1/114 (0%) | 114 |
| `attn_prefill_512tc.cu` | `gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512tc.cu` | `gb10/gemma-4-31b/nvfp4/attn_prefill_512tc.cu` | 18/27 (66%) | 18 |
| `attn_prefill_fp8kv.cu` | `strix-hip/common/attn_prefill_fp8kv.cu` | `gb10/common/attn_prefill_fp8kv.cu` | 437/493 (88%) | 493 |
| `attn_prefill_h128.cu` | `strix-hip/common/attn_prefill_h128.cu` | `gb10/common/attn_prefill_h128.cu` | 1045/928 (112%) | 928 |
| `attn_prefill_paged_indirect.cu` | `gb10/qwen3.6-27b/nvfp4/attn_prefill_paged_indirect.cu` | `gb10/common/attn_prefill_paged_indirect.cu` | 64/66 (96%) | 66 |
| `attn_prefill_v47.cu` | `strix-hip/common/attn_prefill_v47.cu` | `gb10/common/attn_prefill_v47.cu` | 518/498 (104%) | 498 |
| `logit_softcap.cu` | `gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu` | `gb10/gemma-4-31b/nvfp4/logit_softcap.cu` | 17/41 (41%) | 41 |
| `mla_absorbed.cu` | `gb10/mistral-small-4/nvfp4/mla_absorbed.cu` | `gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu` | 13/377 (3%) | 377 |
| `mla_fused_prefill.cu` | `gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu` | `gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu` | 79/213 (37%) | 213 |
| `mla_prefill_attn.cu` | `gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu` | `gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu` | 17/143 (11%) | 143 |
| `moe_fp8_grouped_gemm.cu` | `strix-hip/common/moe_fp8_grouped_gemm.cu` | `gb10/common/moe_fp8_grouped_gemm.cu` | 652/524 (124%) | 524 |
| `moe_shared_expert_fused.cu` | `b300/common/moe_shared_expert_fused.cu` | `gb10/common/moe_shared_expert_fused.cu` | 41/361 (11%) | 361 |
| `moe_shared_expert_fused.cu` | `gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu` | `gb10/common/moe_shared_expert_fused.cu` | 68/361 (18%) | 361 |
| `moe_shared_expert_fused_batch2.cu` | `gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu` | `gb10/common/moe_shared_expert_fused_batch2.cu` | 449/584 (76%) | 584 |
| `moe_shared_expert_fused_batch3.cu` | `gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu` | `gb10/common/moe_shared_expert_fused_batch3.cu` | 37/420 (8%) | 420 |
| `moe_silu_mul.cu` | `gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu` | `gb10/common/moe_silu_mul.cu` | 179/169 (105%) | 169 |
| `moe_silu_mul.cu` | `gb10/step3p7-flash/nvfp4/moe_silu_mul.cu` | `gb10/common/moe_silu_mul.cu` | 170/169 (100%) | 169 |
| `moe_w4a16_grouped_gemm.cu` | `b300/common/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 442/1031 (42%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 1893/1494 (126%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 1627/1369 (118%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 1900/1642 (115%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 840/1031 (81%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 1680/1412 (118%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 2613/2355 (110%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `strix-hip/common/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 1001/1031 (97%) | 1031 |
| `moe_w4a16_grouped_gemm.cu` | `strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu` | `gb10/common/moe_w4a16_grouped_gemm.cu` | 1632/1031 (158%) | 1031 |
| `paged_decode_attn_512.cu` | `gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu` | `gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu` | 2/527 (0%) | 527 |
| `paged_decode_attn_fp8_mla.cu` | `gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu` | `gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu` | 2/567 (0%) | 567 |
| `paged_decode_attn_mla.cu` | `gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu` | `gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu` | 3/538 (0%) | 538 |
| `paged_decode_attn_nvfp4.cu` | `gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu` | `gb10/common/paged_decode_attn_nvfp4.cu` | 71/581 (12%) | 581 |
| `prefill_paged_compute.cuh` | `strix-hip/common/prefill_paged_compute.cuh` | `gb10/common/prefill_paged_compute.cuh` | 1331/1062 (125%) | 1062 |
| `prefill_paged_compute_512.cuh` | `strix-hip/common/prefill_paged_compute_512.cuh` | `gb10/common/prefill_paged_compute_512.cuh` | 333/363 (91%) | 363 |
| `rms_norm.cu` | `gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu` | `gb10/common/rms_norm.cu` | 1111/1433 (77%) | 1433 |
| `rms_norm.cu` | `gb10/gemma-4-31b/nvfp4/rms_norm.cu` | `gb10/common/rms_norm.cu` | 1102/1433 (76%) | 1433 |
| `rms_norm.cu` | `gb10/minimax-m2-229b/nvfp4/rms_norm.cu` | `gb10/common/rms_norm.cu` | 1622/1433 (113%) | 1433 |
| `rms_norm.cu` | `gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu` | `gb10/common/rms_norm.cu` | 1630/1433 (113%) | 1433 |
| `rms_norm.cu` | `gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu` | `gb10/common/rms_norm.cu` | 1517/1433 (105%) | 1433 |
| `rope.cu` | `gb10/mistral-small-4/nvfp4/rope.cu` | `gb10/common/rope.cu` | 428/590 (72%) | 590 |
| `vision_encoder.cu` | `gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu` | `gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu` | 149/462 (32%) | 313 |
| `w4a16_gemm.cu` | `gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 1391/1410 (98%) | 327 |
| `w4a16_gemm.cu` | `gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 1671/1688 (98%) | 327 |
| `w4a16_gemm.cu` | `gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 7317/7349 (99%) | 327 |
| `w4a16_gemm.cu` | `gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 1915/1937 (98%) | 327 |
| `w4a16_gemm.cu` | `strix-hip/common/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 145/327 (44%) | 327 |
| `w4a16_gemm.cu` | `strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 1082/1095 (98%) | 327 |
| `w4a16_gemm.cu` | `strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu` | `gb10/common/w4a16_gemm.cu` | 1078/1094 (98%) | 327 |
| `w4a16_gemm_v2.cu` | `gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu` | `gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu` | 37/284 (13%) | 252 |
| `w4a4_gemm.cu` | `gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu` | `gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu` | 157/258 (60%) | 258 |
| `w8a16_gemm.cu` | `strix-hip/common/w8a16_gemm.cu` | `gb10/common/w8a16_gemm.cu` | 364/319 (114%) | 247 |
| `w8a16_gemm_t.cu` | `strix-hip/common/w8a16_gemm_t.cu` | `gb10/common/w8a16_gemm_t.cu` | 485/637 (76%) | 637 |
| `w8a16_gemv.cu` | `hopper/common/w8a16_gemv.cu` | `gb10/common/w8a16_gemv.cu` | 214/223 (95%) | 223 |
| `w8a16_gemv_batch4.cu` | `b300/common/w8a16_gemv_batch4.cu` | `gb10/common/w8a16_gemv_batch4.cu` | 24/303 (7%) | 303 |
| `w8a16_gemv_fused.cu` | `hopper/common/w8a16_gemv_fused.cu` | `gb10/common/w8a16_gemv_fused.cu` | 346/389 (88%) | 389 |

## Fork counts by stem

| stem | forks |
|---|---|
| `moe_w4a16_grouped_gemm.cu` | 9 |
| `w4a16_gemm.cu` | 7 |
| `gated_delta_rule.cu` | 5 |
| `rms_norm.cu` | 5 |
| `attn_prefill_512.cu` | 2 |
| `moe_shared_expert_fused.cu` | 2 |
| `moe_silu_mul.cu` | 2 |
| `dense_gemm_bf16.cu` | 1 |
| `dense_gemm_tc.cu` | 1 |
| `dsa_indexer.cu` | 1 |
| `embed_scale.cu` | 1 |
| `hyper_connection.cu` | 1 |
| `attn_prefill.cu` | 1 |
| `attn_prefill_512tc.cu` | 1 |
| `attn_prefill_fp8kv.cu` | 1 |
| `attn_prefill_h128.cu` | 1 |
| `attn_prefill_paged_indirect.cu` | 1 |
| `attn_prefill_v47.cu` | 1 |
| `logit_softcap.cu` | 1 |
| `mla_absorbed.cu` | 1 |
| `mla_fused_prefill.cu` | 1 |
| `mla_prefill_attn.cu` | 1 |
| `moe_fp8_grouped_gemm.cu` | 1 |
| `moe_shared_expert_fused_batch2.cu` | 1 |
| `moe_shared_expert_fused_batch3.cu` | 1 |
| `paged_decode_attn_512.cu` | 1 |
| `paged_decode_attn_fp8_mla.cu` | 1 |
| `paged_decode_attn_mla.cu` | 1 |
| `paged_decode_attn_nvfp4.cu` | 1 |
| `prefill_paged_compute.cuh` | 1 |
| `prefill_paged_compute_512.cuh` | 1 |
| `rope.cu` | 1 |
| `vision_encoder.cu` | 1 |
| `w4a16_gemm_v2.cu` | 1 |
| `w4a4_gemm.cu` | 1 |
| `w8a16_gemm.cu` | 1 |
| `w8a16_gemm_t.cu` | 1 |
| `w8a16_gemv.cu` | 1 |
| `w8a16_gemv_batch4.cu` | 1 |
| `w8a16_gemv_fused.cu` | 1 |
