# Kernel track: Hopper kernels for Muse Spark Open

We worked on all five kernel problems: dense projections, gated attention, routed NVFP4 experts, a complete decoder layer, and a complete prefill layer. The progression was from isolated GPU schedules to compositions that preserve the full layer's routing, normalization, cache writes, and residual updates.

## Results and ranking

The following standings were fetched on **20 November 2026 (UTC)** for `aditya3799`. Rank is computed from the official per-problem board as one plus the number of participant entries with a lower reported time; the stock SGLang row is excluded from participant counts.

| Problem | Rank / participants | Speedup vs SGLang | Scored job |
|---|---:|---:|---|
| [Dense projections](https://openkernelchallenge.dev/problems/muse_bf16_proj) | **72 / 1,740** | **1.5967x** | `ac25b659640245d1` |
| [Gated attention decode](https://openkernelchallenge.dev/problems/muse_attention_decode) | **126 / 1,379** | **2.6935x** | `fcf3d95f5f2046bc` |
| [Routed NVFP4 experts](https://openkernelchallenge.dev/problems/muse_moe_w4a16) | **110 / 1,422** | **2.4389x** | `5f5a8c8b72b24ee4` |
| [Whole decoder layer](https://openkernelchallenge.dev/problems/muse_decode_layer) | **194 / 1,296** | **2.2684x** | `c3d4cf6955bf4cdc` |
| [Whole prefill layer](https://openkernelchallenge.dev/problems/muse_prefill_layer) | **54 / 895** | **1.9360x** | `399cbda410c440ee` |

[Machine-readable standings](results/standings.json) retain the source endpoints and fetch times. The final overall endpoint publishes only its top 100; this account is not listed. An older development note recorded overall rank 173 during the contest, but that is **not a verified final overall rank**. Prefill has its own board and does not contribute to the overall decode-step ranking.

Each speedup is the problem's weighted, multi-load score against SGLang, not a full-model serving speedup. These values cannot be multiplied together or substituted for the serving-track result. The per-problem endpoints do not expose a final-review marker in this snapshot; the table reports their current published rows.

## What we implemented and investigated

### Dense projections

The workload includes QKV/gate, output, pooler, unpooler, and router matrices. Small token counts make weight movement and launch/reduction overhead important. The selected v221 implementation combines skinny GEMV/GEMM paths, lossless weight packing, register-direct access, split-K scheduling, and guarded per-shape route selection. The FP32 router uses a BF16 high/low representation with FP32 accumulation where supported by the problem's rules.

Our local work explored exact coefficient recovery, probe rows, FP64-reference checks, cold-cache preparation-time comparisons, and composition of independently validated routes. The scored v221 entry passed **9/9 tests**. Later experiments and the unrelated current working file are preserved separately.

### Gated attention decode

The attention kernel combines Q/K RMS normalization, sliding-layer RoPE, FP8 cache updates, causal attention, per-head RMS normalization, and gating. It uses online softmax without materializing the attention matrix. Its scheduling work covers split-KV, cluster/cooperative reduction, full-context streaming, and different decode/verification execution paths.

The documented winning continuation enabled the q_len=8 resident cooperative work split while retaining the q_len=1 path. It passed **16/16 tests**. Several apparently promising pipeline-depth, occupancy, and cache-policy combinations regressed at aggregate scale and remain research-only.

### Routed NVFP4 experts

The MoE work builds per-expert route lists on the GPU, streams packed NVFP4 weights, computes gate/up projections and SwiGLU, then down-projects and applies expert normalization and weighted combination. Experiments covered small-row paths, WGMMA schedules, exact packing/scale layouts, producer-consumer handoff, K permutations, register pressure, and finalization.

The best published job passed **5/5 tests** and achieved **2.4389x**. There is a source-provenance limit: the workspace's `submission.py` was edited after that run. We preserve the pre-experiment backup as `muse_moe_w4a16/baseline_snapshot.py` and the later file as `working_snapshot.py`; neither is presented as a cryptographically verified copy of the best scored payload.

### Complete decoder layer

This combines projections, attention, the FP32 router, local experts, normalization, and residual boundaries into a graph-capturable layer. The selected **S108** source integrates the attention and exact small-token projection work into the existing composite decoder. It passed **13/13 tests**.

An important finding was that isolated component wins often failed to transfer to the layer. Extra registers, longer live ranges, synchronization, and producer tails could outweigh a removed launch. Later S109-S128 experiments are retained, but the current working snapshot is not labelled as the 2.2684x scored S108 artifact.

### Complete prefill layer

Prefill uses cuBLAS dense projections, FA3 attention, fused Triton boundary/routing operations, and Hopper grouped expert GEMMs with NVFP4 weights reconstructed in registers. The work explored persistent expert scheduling, tile allocation, exact scale sharing, and the interaction between large and short prefills.

The selected source is explicitly named for job `399cbda410c440ee` and matches its documented frozen hash. It passed **5/5 tests**. Historical lower-precision weight-storage experiments are not the selected source; their earlier timing figures are not used as our final result.

## How we validated changes

The official evaluator used fresh inputs, CUDA graph replay, cold L2, and paired candidate/baseline timings. We inspected per-shape profiles and aggregate load scores, retained known-good baselines, and used unscored checks before promotion. Local work included static Hopper builds, register/spill inspection, graph replay with changing inputs, numerical checks, and rejected-experiment ledgers.

A passing tolerance check is not proof of bitwise equivalence, and an isolated local speedup is not an H200 full-layer result. GPU architecture, power state, input distributions, warm/cold caches, and selection overhead all matter. The research records include unsuccessful trials so these limits remain visible.

## Source map and reproduction

Start with the selected source in each problem directory and its README. [Source provenance](results/source_provenance.json) records the original filename, exported filename, SHA-256, scored job association, and the MoE caveat. Original source is copied byte-for-byte, with attribution retained.

These entries target the contest's Linux/CUDA environment and Hopper H200 hardware; many embed CUDA/C++ compiled with PyTorch's `load_inline`. Python syntax validation alone does not establish runtime compatibility. Use the official problem scaffolding for the `Inputs`, `Config`, `Weights`, reference functions, and NVFP4 helper rather than inventing replacement signatures.

```bash
# From this repository root: CPU-only source/hash checks; no GPU or submissions.
python kernel-track/scripts/verify_artifacts.py

# Obtain the official scaffolding in a separate directory.
mkdir -p /tmp/okc-kernel-repro
cd /tmp/okc-kernel-repro
okc init muse_bf16_proj
# Copy this repository's kernel-track/muse_bf16_proj/submission.py
# over the newly generated starter, then use the official reference locally.
```

The kernel submission window is closed. Historical `okc check` / `okc submit` commands in archived notes describe the contest workflow and are not a promise that new remote runs will be accepted.

## Research archive

`research/<problem>/` preserves source variants, build/analysis scripts, and written investigation records. Original relative names are retained. Some scripts reference the old workspace, omitted profiles, or separately installed dependencies; the archive is not advertised as a portable one-command benchmark suite. Failed, slower, superseded, and rule-incompatible experiments are historical records, not production recommendations.

The code includes public community kernels that we adapted, tuned, and composed. See [NOTICE](../NOTICE.md) and source headers for attribution rather than treating all embedded CUDA as newly authored work.
"# muse-spark-kernels" 
