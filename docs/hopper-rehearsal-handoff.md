# Hopper rehearsal: draft integration, DO NOT MERGE

This branch collects the September 25 H100 Qwen and H200 Nemotron work for continued development. It is based on Metrale main `26f1e45ffd8ec16bbee7ab1064e93d40055b28a3`. The combined renamed branch has not been built or run on a GPU. The measurements below belong to the original private candidates, not this PR tip. No certification or merge readiness is claimed.

## Included changes

- Persistent grouped-FP8 scratch, unique GPU argmax with conservative fallback, skipped unused margin scan, and reuse of processed greedy logits.
- Hopper exact decode top-k, adaptive short-prefill worklists, native-input M16/M128/M64 expert tiles, exact-order router, and shared projection M16/N32 tile.
- Nemotron Nano NoPE attention correction and FP32 router logits. The latter did not fix the remaining FP8 KV failure.
- Split token-parallel conv1d computation from state commit, with paired-symbol fallback and wrapper tests.
- Pinned model/corpus preparation, stock-client runner, request-length validation, and small kernel checks. Logs, credentials, model weights and certification records are excluded.

Metrale import changes are mechanical crate/environment/helper renames, retaining the new base's dense GEMV modules, and declaring ten Hopper-owned kernels in HARDWARE.toml. Existing AGPL-3.0-only headers and repository licensing are retained. Source provenance is below.

## Historical measurements

Internal same-host C1, TP1, no speculation, ISL128/OSL1024, temperature0/seed42, three repetitions of16 requests. These are not independent third-party measurements. TTFT includes scheduling, prefill and first-token overhead; isolated prefill throughput was not measured.

| GPU/model | Engine/candidate | TTFT p50 ms | TPOT p50 ms | Overall output tok/s |
|---|---|---:|---:|---:|
| H100 Qwen3.6-35B-A3B FP8 | vLLM frozen |40.429–40.653|4.13050–4.13096|239.962–240.038|
| H100 Qwen3.6-35B-A3B FP8 | private v11 |49.581–50.035|3.97263–3.97540|248.735–248.916|
| H200 Nemotron-3-Nano-30B-A3B NVFP4 | vLLM frozen, FP8 KV |34.224–34.918|2.92711–2.92737|337.838–338.069|
| H200 Nemotron | private engine |not scored|not scored|not scored|

Qwen improved from the initial200.676tok/s and223.187ms TTFT. v11 completed48/48 full requests with0errors and exact full-text agreement to v10. Confirmation20:46:38UTC,5h13m13s after explicit setup receipt15:33:25UTC. Decode is ahead of the saved reference; prefill remains behind. Atlas KV auto-calibration differs from saved vLLM default scales1.0 and is disclosed.

Qwen checkpoint `Qwen/Qwen3.6-35B-A3B-FP8`, revision `95a723d08a9490559dae23d0cff1d9466213d989`. v11 source `86b41396f7b7ed0aabfbdf77ec7a2ab47837691a`, image `sha256:b0c19d73c6d9ab10dd8d5f82c161abe82ea40574f2563201d26b8f8e2983133e`, binary `5c8f87b2b88ba649ce93844985f259ed0f36594e941eb63cfc9a1e3dd823270a`. Earlier native-input arithmetic changes in v7/v9 changed text and are not BF16-bit-equivalent. Ten semantic checks are not broad quality certification. Do not claim native FP8 machine instructions from the PTX alone.

Nemotron checkpoint `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, revision `6efb4a2a1c1fa277ce7b3df7a1416255011b1c99`. NoPE-only source `8b0af5fa3616342d126634e21cb560f1a478d492` with BF16 KV passes9/9 semantic checks and graph-enabled128/1024 decode. Output still echoes the nonce once before the substantive essay. This is unmatched to FP8-KV vLLM and not a scored performance baseline. FP8 KV yields incorrect arithmetic and CUDA716 on long graph decode; BF16 provides the working control. FP32 router and serial convolution did not fix those observed failures. The combined PR includes the later router and convolution fixes, so it is not the same candidate as that working control.

## Validation and remaining work

Original components: v11 GPU512 exact tile cases plus full-model checks; shared projection64 exact cases; convolution132 GPU chunk cases,264 graph replays and two forced old-fail/new-pass cases; router66 GPU cases and projection boundaries. These validate their original sources, not the combined branch.

Import checks: cargo fmt, kernel-shadow structure, cross-hardware reach, changed-source SPDX, Python compile, shell syntax and git diff whitespace pass. A full macOS cargo check is blocked by preexisting Linux-only libc references in spark-storage (O_DIRECT/posix_fadvise family). No Linux combined build, clippy/test/doc sweep or GPU campaign has run. The Docker-backed license checker is left to CI; the local SPDX check covers changed source files only.

Next owner should build the combined branch on Linux before relying on it, qualify Qwen and Nemotron independently, check non-Hopper reach of the shared conv change, and split reviewed fixes from numerical experiments. Preserve frozen references and failed cases. Keep exclusive GPU timing windows. Fix/qualify Nemotron FP8 attention/KV before a matched score. No MTP, wider concurrency, cache-hit restoration or broad model-quality claim follows from these C1 results.

The old runbook is a preparation recipe, not a record of final launch flags. Historical Qwen v11 also used ring slots2, no-tail-split and SSM cuBLAS; Metrale equivalents are METRALE_NO_TAIL_SPLIT=1 and METRALE_CUBLAS_GEMM=ssm. Freeze and record all effective flags before a new measurement. Preserve CUDA caches and persistent build outputs.

## Commit provenance

All imports use cherry-pick -x. Original private commits shared base `d822945614e64854978ab416e434f6e5eea8a975`. No private branch was rewritten. Later work by the takeover agent is outside this snapshot.

| Imported commit | Original commit | Change |
|---|---|---|
| `42bcfb5d9ec0bb4c159e534cadbda3fb6b0d5697` | `adc34b05e1a1d4512ac3501b1029f4c6ebdd8686` | moe: retain FP8 grouped scratch across CUDA graph replay |
| `023e3a66c146113f86edd0a7cfc090639ade11db` | `bc77ff98583429c5dcb36b8b50dccf6e319d03a4` | hopper: certify unique GPU argmax for minimum-token greedy decode |
| `6d29c2630098a2f0d4cf87e15b2171fbf74f0f28` | `64ec2bc7ab4d9d4a46776e946c2bd0f18197e8d2` | scheduler: skip unused margin scan outside parameter bodies |
| `4858feb9be4b1dd4849ca836069e9ce2c90ed5ff` | `5a45042573a5ae2d0205df12f2d96ae045c0043c` | scheduler: reuse processed logits for greedy host sampling |
| `d367794492014cd06a0b6073db22e33992a0eaf9` | `77c049b97c750e366b568665f4e670578ab802d2` | hopper: specialize exact single-token FP8 MoE top-k routing |
| `5c0a693e59faba0f872a6a7f113251e1c51800f5` | `446929f561e66fb3cd35d4bc4ef8631f2636c21d` | hopper: adapt short FP8 MoE prefill tiles to expert occupancy |
| `93d36b2856b3248add9b6c6a8ec60fefbde03990` | `f0754576244c698571c8d1c7fb43bf4a1b86d107` | hopper: trial native FP8 accumulation in small-expert prefill |
| `51f11e3833f9549fa5a852aa94ecda0f862a649b` | `2883ad41d5391aae928eb3e2adbe624322cc203c` | hopper: simplify packed FP8 NaN sanitization |
| `f0399c612c5cb99fccca91671ebba70e0fc46712` | `6ac5fe9c7a5983c8d523f7b1e8040132b9434cb1` | spark-model: specialize exact-order Hopper router at M128 |
| `cfbb2fb18133c29264a6be7613f88da8cefc50e8` | `46bdb19474ee5d1d2c4cf06fba5077d5452bd0a3` | hopper: trial native-input M128 in bounded MoE prefill |
| `74e8f981976bfed4fc464c7c5b658056401f2ca4` | `8defa8e4b740ad3e913c2cf7d17e194995bdaac5` | hopper: tile short shared FP8 projections across more SMs |
| `441d189286351e24fe450a3fc6e872cfe8f4dca2` | `86b41396f7b7ed0aabfbdf77ec7a2ab47837691a` | hopper: use exact native M64 virtual tiles for bounded MoE prefill |
| `db2dee4f5671b5a8cfaa361ffd480d240d05dcde` | `8b0af5fa3616342d126634e21cb560f1a478d492` | spark-model: disable rotary attention for Nemotron Nano NoPE |
| `cf8a384e88cd35d531353b7ad328cb5e58250b1e` | `36e48221668ccad6db236d782bd87a8a3e363f1b` | spark-model: preserve Hopper Nano FP32 logits through routing |
| `e36a92892192358b3ad08587b34c78f64e7b6f55` | `71351e86bb77248704c2c4d8427012e2d770bb8d` | spark-model: remove obsolete Nemotron prefill scale field |
| `f3fbf24e00c54398ab1b7dedc5939c99fa707148` | `1493adffe4de58d59038c04b581732632e6f151b` | spark-model: separate conv1d prefill state commit from output computation |
| `3895834b97c06c86e8b1df9d74867a9fe454e785` | `ccb872fcb0cbe2df240f0d3c2616590a0f008a21` | bench: prepare private Hopper MoE comparison receipts and smoke checks |
| `004951b7c83e54c346857dfcf8cdc9ba858f7282` | `20a066ba830e62cb619aca40d110d5bf8b9cbf60` | bench: freeze matched 128-token Hopper prompt corpora |
| `55f17326ade6a1d41881912b6346955e2513cb41` | `c90e13b32f40de10353b0d21bf324f091f94e8cd` | bench: record isolated stock-client Hopper cells without overwriting failures |
| `fad5dc069ddd18666650de3d70210516cca7b4ca` | `7f1f80cd0b3cdd1cbb39a2e6ca41b6572555018f` | bench: reject shortened or failed Hopper comparison cells |
| `3dbb1f9e87fc4f785176dec68e97afa8a2b1eb19` | `integration-only` | hopper: adapt rehearsal imports to Metrale names and kernel registry |

Authorship: AI-assisted implementation and import, directed by TheTom. Draft preservation only; DO NOT MERGE.
