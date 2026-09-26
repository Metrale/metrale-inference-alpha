# K3 serving foundation

This layer connects the separate host reference graph, tensor-parallel plan and
rank-local safetensors loader to model assembly and serving. It is development
bring-up, not a declaration of full-model or chat support.

The K3 loader binds BF16/FP32 weights and opt-in packed E8M0 experts. Packed
multi-rank weights must carry the rank-aware loader's partition marker; the
loader refuses to reinterpret an unmarked upload as an already sliced model.
FP32 engine projection conversions are adopted by the derived-weight owner.
The shared tensor plan controls slicing instead of a second model-local plan.

Bound layers copy activations to the host reference graph and back. Existing
KDA and MLA CUDA mixer callbacks can be disabled with `K3_CUDA_KDA=0` and
`K3_CUDA_MLA=0`. Packed experts require `K3_ALLOW_MXFP4=1` and their E8M0 kernel.
This foundation retains the original per-call KDA state transfer. Resident
recurrence and optional resident dense/shared MLP execution are a later slice;
this implementation sets the core dense callback to `None`.

Serving selects the rank-local safetensors loader before device allocation,
refuses expert parallelism and GGUF for that path, and pins packed K3 to an
actually compiled MXFP4 kernel target. K3 per-token prefill keeps the scheduler's
chunk budget; it does not inherit the unrelated single-chunk MLA restriction.

## Validation limits

The host K3 tests pass with 11 fixture-dependent tests ignored
(`cargo test -p metrale-model-weights --lib kimi_k3`), and Metal-feature
library checks for model/server pass on macOS. These do not execute CUDA.
Linux/CUDA tests, launch, shutdown and numerical regressions are still
required. Full K3 weights, B300, TP8, multiple hosts and XTML chat/tools are
not validated here.
