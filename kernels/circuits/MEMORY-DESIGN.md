# The circuit memory model: design and the reserve wiring

Status: 2026-10-02, with `met circuit memory` (crates/circuit/src/memory/, crates/server/src/cli/circuit_memory*.rs).

Owner directive (2026-10-02): "being able to see and define explicitly (for metadata and
tracking), for each node, the memory used. That way, we can define the base amount of memory
expected based on initial ISL's. We need to take into account the KV, SSM, and now with the
additions from [the new speculative-decoding techniques], other caches (e.g., caching of
acceptance tokens). We want to be able to use this as a utility for the circuit."

## 1. What the model is

One pure evaluator, `metrale_circuit::memory::evaluate`, over the instantiated circuit and
explicit counts. Nothing in the crate reads a file, the environment or an engine global.

| Term | Source of truth | Per node |
|---|---|---|
| Stored weights | the checkpoint's declared formats (`Format::weight_bytes`, scales included), `out x k` from the node's edges, routed experts x `experts` | the node |
| Derived copies | `kernels/circuits/COPIES.toml`: one rule per loader allocation (requantized serving copies, transposed twins, MMQ repacks, widened scales, leaked intermediates), matched by op, declared format, served format and serve settings | the node |
| Outside the circuit | the checkpoint's safetensors headers (norms, conv, gates, a vision tower), measured by the CLI | totals only |
| Activations | the buffer planner's arena for each fused run given (decode C, verify C x K, prefill chunk); the legacy arena (`BufferSizes::from_config` + the GDN two-phase buffers) is what the budget charges while the legacy forward is the default | the producing node, at the peak run |
| Workspace | `[[family.workspace]]` in KERNEL_FAMILIES.toml, sized at the widest run; `arena = true` marks scratch the legacy arena already holds | every node of the family |
| States | `[[block.<b>.state]]` recurrent and paged-KV declarations (`StatePlan`, M5) | the node that updates or writes it |
| Caches | `[[block.<b>.state]]` cache kinds (`StateKind::is_cache`): prefix and ring snapshots, the carry stash and tables, the WY tables, the accepted-hidden stash, the drafter capture, the prompt-lookup index (host), a token tree's mask and drafted ids | state table |
| Driver | `kernels/<class>/HARDWARE.toml [memory]` (#72's calibration: fixed + per mille of the util budget) plus measured chunk slack | totals only |

A cache whose count is absent sizes nothing; an unknown kind is refused at load with the list of
kinds. The counts are the engine's: the CLI computes them with the engine's own functions
(`ssm_reserve::pool_counts_with`, `mtp_state_slots_with`, `marconi_snapshot_slots`,
`decode_rollback_ring_slots`, `resolve_num_drafts`, `resolve_mtp_max_seqs`,
`resolve_kv_dtype_str`, `resolve_prefill_budget`, `BufferSizes::from_config`) from the
`ServeArgs` the serve's own parser builds out of a recipe and flags.

## 2. Accuracy (the ledger validation)

`crates/server/src/cli/circuit_memory_ledger_tests.rs` evaluates the model at five real boots'
settings and pool sizes and compares every term the boot's allocation ledger itemizes:

| Term | Tolerance | Worst error over the five boots |
|---|---:|---:|
| weights (stored + derived + outside) | 1% | -0.19% (Nemotron-3-Nano: the router is stored F32 and modelled 16-bit; the 52.8 MiB dequant scratch) |
| KV (target + MTP pools) | 0.1% | +0.01% |
| SSM pool | 0.1% | 0.00% |
| snapshots (Marconi, decode ring) | 0.5% | 0.00% |
| legacy arena | 1% | +0.02% |
| GDN two-phase buffers | 0.1% | -0.02% |

and #72's named reserve terms (SSM pool, Marconi, decode ring, carry stash, driver) equal the
model's to the logged MB.

## 3. Wiring #72's reserve to the model (proposal)

#72 (`pr/default-tier-kv`) plans the pre-load reserve from named terms
(`serve_phases/preflight/reserve_plan.rs` `ReservePlan::terms`). Today each term is computed by
its own function; the SSM pool already reads the circuit (`PoolPlan` over `StatePlan`, M5). The
minimal change that makes every term derive from the circuit model, in the order of risk:

1. **Marconi and the decode ring.** `marconi_bytes` and `ring_slots * slots * per_seq_blob`
   become `memory::caches::cache_terms` over the model's circuit states with
   `CacheInputs { prefix_snapshot_slots, ring }`. This also counts the last-hidden row the
   allocation adds (`ssm_snapshot_init.rs:88`, which the reserve leaves out today).
2. **The carry stash.** `runtime_headroom::carry_stash_bytes` becomes the `carry_stash` and
   `carry_table` caches with `CacheInputs { carry: (verify_slots, VERIFY_WY_TABLE_SEQS) }`;
   `GdnCarrySizes` stays the allocator's arithmetic and
   `cli/circuit_memory_parity_tests.rs` pins the two equal.
3. **The driver terms.** `DRIVER_FIXED_BYTES` and `DRIVER_BUDGET_PER_MILLE` move to
   `kernels/gb10/HARDWARE.toml [memory]` (already declared there), read through
   `memory::DriverTerms` at build time like the `[defaults]` levers
   (`metrale_kernels::TARGET_DEFAULTS`), so another class states its own calibration.
4. **The weights.** The post-load audit's `predicted_derived_bytes` (native-FP8 dense route
   only) is replaced by `memory::weights::node_weights` over COPIES.toml, which covers every
   default load path; `METRALE_DENSE_FP8`'s copies become rules with `when` keys.

Each step is a byte-identical refactor except (1)'s hidden row, whose size is printed. Steps
1-3 touch only `reserve_plan.rs` and `runtime_headroom.rs`; step 4 touches the post-load audit.
Until then the validation test above is the agreement proof.

## 4. Findings while validating

- **4.6 GiB of leaked load-time copies on the dense 27B** (both tiers; freed by #86, KV 4382 -> 8764 blocks on the recipe): the BF16 intermediates of
  the FP8 attention requant (3.1 GiB, `attn_arms.rs:143-152`), the GDN `out_proj` FP8 predequant
  that the cast overwrites (1.4 GiB, `gdn_dequant.rs:418-424`), and the first `in_proj_ba`
  interleave. COPIES.toml marks them `leaked`; freeing them is a loader fix, not a model change.
- **Derived copies are 58% of the dense 27B's resident weights** (30.5 GiB on 21.8 GiB stored).
- **The declared circuit gave nvidia/Qwen3.6-35B-A3B-NVFP4's routed experts BF16**: ModelOpt
  names the fused `…mlp.experts` module, not its projections. `declared_precision.rs` now lets a
  routed expert's projection inherit it.
- **Nemotron-3-Nano's router was declared NVFP4 by the global ModelOpt algo** but is stored F32
  (128 x 2688 per layer, 30 MiB over 23 layers): the router is a `NemotronHTopkRouter`
  parameter, which no ModelOpt config quantizes. The circuit now marks it `unquantized`
  (section 5 has the loader's BF16 cast of it).

## 5. Derived-copy backlog: inherent or avoidable

Every COPIES.toml rule, by what it costs on the validated serves (MiB, from `met circuit
memory`) and whether a serve could do without it. **Inherent** means a kernel that runs today
reads it and nothing else serves that read; **avoidable** names what would remove it. Owners are
the agents or areas whose code the change sits in.

| copy (rule) | serves | MiB | verdict | what removes it | owner |
|---|---|---:|---|---|---|
| BF16 intermediate of the FP8 attention requant (`dense-attn-bf16-dequant`) | dense 27B | 3200 | **leak** | freed after the requant (#86) | CIRCUIT-MEM |
| GDN `out_proj` FP8 predequant replaced by the cast (`dense-gdn-out-fp8-predequant`) | dense 27B | 1440 | **leak** | not made when the cast is (#86) | CIRCUIT-MEM |
| first `in_proj_ba` interleave (`dense-gdn-ba-interleave-first`) | dense 27B | 45 | **leak** | made only on the native-NVFP4 path (#86) | CIRCUIT-MEM |
| FP8 per-channel -> NVFP4 requant (`dense-fp8-to-nvfp4`) | dense 27B | 5018 | inherent under `--weight-quantization nvfp4` (the tier's definition); **avoidable** under `declared`, where the W8A8 decode already reads the FP8 store and the NVFP4 copy serves the routes W8A8 does not cover (prefill above its row ceiling, the batched paths) | W8A8 on every route of a declared-FP8 projection | engine (W8A8) |
| NVFP4 transposed twins of those requants (`dense-fp8-nvfp4-twin`, `dense-fp8-ffn-twin`) | dense 27B | 3870 (+1148 declared) | inherent while the prefill tile GEMMs read K-major NVFP4 | goes with the requant above; or a tile GEMM reading the row-major weight | kernels |
| unscaled FP8 casts of GDN qkvz / out_proj for `fp8_gemm_n128` (`dense-gdn-fp8-cast`) | dense 27B | 5280 | **avoidable**: the checkpoint's per-row FP8 is already resident and `METRALE_FP8_ROWWISE` prefills from it with no cast (opt-in today) | the per-row FP8 prefill as default, after a TTFT A/B | engine (prefill) |
| MMQ `block_nvfp4` repack of the NVFP4 FFN (`dense-ffn-mmq-repack`) | dense 27B | 8033-9180 | inherent: decode reads the row-major store, the MMQ prefill the repack | one layout both read (a kernel change) | kernels |
| BF16 dequant of the FP8 lm_head (`fp8-head-bf16-dequant`) | dense 27B | 2425 | inherent under `--lm-head-dtype bf16` (it is the served head; then the FP8 store's 1213 MiB is the avoidable one); **to verify** under `declared` (FP8 head on W8A8: likely read only to quantize the draft head) | free whichever of the FP8 store or the BF16 dequant no route reads | engine (head setup) |
| draft-only NVFP4 head (`draft-head-nvfp4-of-*`) | 27B, FP8 35B | 682 / 273 | inherent with a non-NVFP4 target head; avoidable if the draft read the target head | a draft propose over the target's head format | MTP |
| fused GDN `[qkv\|z]` FP8, sources kept (`moe-gdn-qkvz-concat*`) | 35B FP8 and NVFP4 | 720 | **avoidable** if no route reads the separate `in_proj_qkv` / `in_proj_z` store tensors (the dense loader prunes them, `prune.rs`); needs a reader audit | prune the sources after the concat | engine (MoE loader) |
| FP8 attention prefill twins (`moe-attn-fp8-prefill-twin*`) | 35B FP8 and NVFP4 | 260 | inherent while `w8a16_gemm_t` reads K-major | a GEMM reading row-major | kernels |
| NVFP4 routed and shared expert transposed tables (`moe-nvfp4-expert-twin`) | NVFP4 35B | 17348 | inherent: the grouped prefill GEMM reads K-major, the decode row-major | one layout both read | MOENVFP4 |
| NVFP4 shared expert -> FP8 predequant (`moe-nvfp4-shared-fp8-predequant`) | NVFP4 35B | 120 | **avoidable**: being removed (MOENVFP4 ae95f37a3, prefill W4A16 from the NVFP4 store) | in flight | MOENVFP4 |
| FP8 shared expert -> NVFP4 (`moe-shared-expert-nvfp4`) | FP8 35B | 68 | to verify which of the FP8 store and the NVFP4 copy the serve reads | free the unread one | engine (MoE loader) |
| Nemotron routed expert transposed tables (`nemotron-expert-twin`) | Nano | 15758 | inherent, as for the NVFP4 35B | one layout both read | MOENVFP4 / Nemotron |
| Nemotron Mamba2 NVFP4 -> FP8 predequant (`nemotron-mamba-fp8-predequant`) | Nano | 628 | **avoidable** with a W4A16 prefill from the NVFP4 store (as ae95f37a3 does for the shared expert) | W4A16 Mamba2 projections in prefill | Nemotron |
| NVFP4 lm_head transposed twin (`head-nvfp4-twin`) | NVFP4 heads | 189-682 | inherent while the head's tile GEMM reads K-major (`METRALE_NO_LMHEAD_TGEMM` trades it for speed) | - | kernels |
| FP32 block-scale grids, per-row scales, the BA interleave | all | < 50 each | inherent (the kernels' scale format) | - | - |

Not in COPIES.toml (outside the validated serves, from the loader survey): the Nemotron dequant
scratch (`nemotron.rs:68`, 53 MiB, never freed: **leak**); the MTP projections' and experts' BF16
dequants on the MoE when the serve does not speculate (`impl_a1_init.rs:88`, about 1.5 GiB, never
freed: **leak**); the same out_proj-predequant pattern in `qwen35/.../linear_attn_arms/nvfp4.rs:
173-180` for `Fp8Dequanted` GDN layers (**leak**); the 4-byte absmax buffer per quantize call
(`loaders_fp8.rs:171`, leaks one 2 MiB chunk's worth of driver slack in total).

**A precision finding, not a copy:** the Nemotron loader casts the F32 router of Nano and
Lightning to BF16 (`weight_map/nemotron.rs:228-231`), while HF computes the router logits in FP32
from the F32 weight. That lowers routing precision below the checkpoint's; LIFECYCLE-DESIGN.md
section 5.5 records the package's intent to keep F32. The declared circuit now marks the router
`unquantized` (it had resolved Nano's to NVFP4 from the checkpoint-wide ModelOpt `quant_algo`,
which does not reach a `NemotronHTopkRouter` parameter).

## 6. Follow-ups

- `hardware::estimate::footprint` (the `met circuit plan` memory fit) sizes weights with its own
  `out x k x experts` sum; it should read `memory::weights::node_weights` (which also stores the
  draft head's reused `lm_head` once). That changes the checked-in matrix reports, so it is left
  to its own change.
- FUSIONS.toml has no FP8-KV paged-attention rule on gb10, so an FP8-KV serve's attention plans as
  a placeholder group and its split-K workspace is not attributed.
- The routed experts' transposed NVFP4 tables:
  - `nemotron-expert-twin` is built only without a latent MoE (`prefill_weights.rs`); the rule's
    `when = { latent_moe = "off" }` models that, so Nemotron-3-Super (`moe_latent_size` 1024)
    carries none. At util 0.85, its recipe (65536 tokens, C=8) fits: 7363 MiB headroom, a KV pool
    of 150591 blocks against 32777 needed.
  - `moe-nvfp4-expert-twin` (qwen35 loader): until 2026-10-02 the loader built them whatever
    the memory (its 2 GiB free-memory gate bound only non-fast-MoE layers, and on the default
    levers every NVFP4 MoE layer is one), so Qwen3.5-122B-A10B could not fit on one GB10. They
    are now built only when this model says they fit: `met serve` plans the checkpoint under its
    own flags at its declared `--max-seq-len` and batch (`serve_phases/expert_tables.rs`, the
    embedded kernel tree of `metrale-kernel-tree`), through the function `met circuit memory`
    decides with (`cli/circuit_memory_tables.rs`), publishes `MoeExpertTables` to the loader, logs
    it, and discloses a skip in `serve_resolved` (`moe_expert_tables = skip`). The rule reads the
    decision (`when = { moe_expert_tables = "build" }`).
    - nvidia/Qwen3.6-35B-A3B-NVFP4, its recipe at 0.85 (8 x 262144 tokens): build, 17348 MiB of
      tables, 24636 MiB headroom with them.
    - Qwen3.5-122B-A10B, its single-GB10 recipe at 0.85 (1 x 16384 tokens): skip, 62451 MiB of
      tables; headroom -43643 MiB with them, 18808 MiB without, 101336 KV blocks for 1026 needed.
- The plan gives a Nemotron router the BF16 of an `unquantized` node. Nano and Lightning store it
  F32, and #90 serves it F32, as stored. The circuit cannot see a safetensors dtype, so the plan
  under-counts each of those routers by half of its 1.4 MB per layer (16 MB on Nano), inside the
  weight tolerance.
