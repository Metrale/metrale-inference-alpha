# B300 Kimi K3 bring-up target

Build with `METRALE_TARGET_HW=b300 METRALE_TARGET_MODEL=kimi-k3` and select
`METRALE_TARGET_QUANT=bf16` for the small twin or `mxfp4` for official packed
weights. The `nvfp4` directory retains the registry's default quant alias; it
does not convert MXFP4 weights into NVFP4.

This target uses `sm_103a` (CUDA 12.9 or newer compiler support; the repository
requires CUDA 13.0+). B200 `sm_100a` and DGX Spark `sm_121f` binaries must still
fail the B300 architecture preflight. Runtime device memory and SM count must
be checked on the actual rental. All 182 selected Kimi MXFP4 kernels compiled with CUDA 13.0.88 for
`sm_103a` on an ARM64 DGX Spark host. B300 module-load, inference and performance
remain unverified; cross-compilation does not execute on B300.

This tree compiles gb10's sources through explicit `[sources] use` lists:
`common/KERNEL.toml` names the 185 gb10/common files it compiles and each
`kimi-k3/<quant>/KERNEL.toml` names the Kimi leaf and the DeepSeek E8M0 GEMM.
The resolver (`crates/closure/src/layout.rs`) stages them into this
tree's directories, so their includes resolve against b300's own headers. It
does NOT inherit gb10: the module set is the bring-up snapshot's, and a gb10
file added later reaches B300 only when it is added to a list here. `common/`
holds only the four sources that diverged during bring-up. The snapshot's
byte-identical copies became `use` entries on 2026-09-24, so an edit to a listed gb10 file now
reaches this target. Keep future B300-tuned sources here as real files; never
edit gb10's file for B300.

Conservative serving defaults are explicit in `HARDWARE.toml`. The existing
W4A16 E8M0 path is the initial numerical baseline. Native datacentre FP4
block-scaled MMA needs a separate tcgen05 implementation and correctness and
performance receipts before enabling it. This target does not prove packed
TP8 loading or full Kimi inference; those are separate integration gates.

## Changes after the source snapshot

The four files B300 still holds are its divergences from the bring-up
snapshot, recorded by Git:

- `common/moe_shared_expert_fused.cu`: removed the inherited hardcoded
  DeepSeek-only activation clamp from generic SiLU decode. Routed and shared
  experts now use the same plain `silu(gate) * up` math. Kimi's LatentMoE calls
  the separate E8M0 GEMM and `situ_glu_vec` path; it does not dispatch this
  generic SiLU kernel. No GB10 source or clamp-scope exception was changed.

## Focused extraction

This target is extracted from draft #1150 at `bba13ef3a`. Historical compiler
and device observations above apply to the original integration workspace;
this extraction runs CPU-side structural/registration checks only. It does not
add the model runtime integration or qualify a complete serving build.
Existing Hopper/B200 scaffolding from merged #1045 is reused, and the GB10
expert alias changes are reviewed separately in #1165. No GPU or rental was
started for this extraction. Keep the review draft until its dependent runtime
work and required hardware qualification are complete.

The owned snapshot deliberately preserves source bytes, including inherited
extra blank lines at EOF in `common/gated_delta_rule_wy.cu` and
`common/gated_delta_rule_wy_f16.cu`. `git diff --check` reports those two
whitespace-only findings; no kernel arithmetic was changed to clean them up.
