# muse_decode_layer research notes

Snapshot: 2026-10-28 (Asia/Calcutta)

## 1. Problem model

One call executes everything one H200 does for a Muse Spark Open decoder layer under
TP8/EP8:

1. BF16 QKV/gate projection.
2. QK RMSNorm, optional interleaved RoPE, FP8 KV store, and paged GQA.
3. Per-head output RMSNorm/gate and BF16 output projection.
4. FP32 norm-preserving residual boundary and pre-FFN RMSNorm.
5. FP32 router, biased top-8 selection, and route renormalization.
6. BF16 pooler and pre-expert norm.
7. Local NVFP4 W4A16 experts, SiLU-and-multiply, post-expert norm, and weighted sum.
8. BF16 unpooler and the second FP32 residual/norm boundary.

The scored token counts are `T = B * L`:

| Load | B | L | T | Variants |
|---|---:|---:|---:|---|
| decode 1 | 1 | 1 | 1 | sliding, full |
| decode 16 | 16 | 1 | 16 | sliding, full |
| decode 32 | 32 | 1 | 32 | sliding, full |
| verify 1 | 1 | 8 | 8 | sliding, full |
| verify 16 | 16 | 8 | 128 | sliding, full |
| verify 32 | 32 | 8 | 256 | sliding, full |

This is not one uniform GEMM regime. Use at least three dispatch families:

- tiny: `T <= 8`, FP32 router input and only 1024 unpooler output columns;
- small decode: `T in {16, 32}`;
- verify: `T in {128, 256}`, full 8192-column unpooling and split BF16 router.

Correctness tests add `T=264`, page size 64, scaled KV, mixed request lengths,
outliers, and contexts up to 1,048,576 tokens. Shape specialization is allowed;
value specialization is not.

## 2. Measured local evidence

The component repositories give an unusually useful first-order budget. Their score
units are compatible with the whole-layer load mix closely enough for planning:

| Component | SGLang-equivalent | Best local | Local speedup | Evidence |
|---|---:|---:|---:|---|
| Dense projections + router | 4.8308 ms | 3.2003 ms | 1.5095x | `submission_nathen_v18` |
| Gated attention decode | 3.7663 ms | 2.4705 ms | 1.5245x | current `submission.py` |
| Routed NVFP4 experts | 13.4485 ms | 13.4833 ms | 0.9974x | current MoE reference work |

These sum to 22.0456 ms. Against the whole-layer baseline of 22.6203 ms, the residual
norm/routing/launch overhead is about 0.5747 ms.

If the attention and dense wins integrated perfectly while the expert path and boundary
overhead stayed unchanged, the projected layer time would be approximately 19.6940 ms,
or only 1.149x. This is a strong integration milestone but far from the live leader's
9.4308 ms. The MoE path and cross-component traffic are therefore the decisive work.

### Reusable local findings

- `../muse_bf16_proj/submission_nathen_v18.py` is the measured champion, not the newest
  file. It is strongest on the exact layer shapes, especially the router (2.2x-6.5x)
  and tiny unpooler (1.58x-1.74x).
- `../muse_attention_decode/submission.py` is most valuable for decode. It reaches
  1.63x-2.35x on decode attention but only 1.03x-1.39x on full verify attention.
  Dispatching verify back to FA3 is likely correct for some shapes.
- `../muse_prefill_layer/submission.py` demonstrates Humming for large routed expert
  batches and overlap with the independent router, but its 753-16384 token regime does
  not transfer directly to `T<=256`.
- `../prefill_moe_reference/submission.py` and its profiles show that the current custom
  MoE attempt does not beat Marlin at decode sizes; do not assume a Triton rewrite is an
  improvement without a remote profile.

## 3. Baseline architecture and bottlenecks

The official `baseline.py` is already well engineered:

- cuBLAS/dispatcher for four dense projections;
- fused QK norm/RoPE and FA3 for attention;
- fused residual/norm boundaries;
- router and pooler overlap on separate streams;
- Marlin W4A16 expert GEMMs with fused post-expert norm/weighted sum;
- sliced unpooling for `T<=8`.

The remaining opportunities are consequently structural, shape-specific, or both.
Replacing one fused SGLang primitive with several generic PyTorch/Triton launches is
unlikely to win.

The speed-of-light table reinforces the split:

- `T=1` floors are 40-44 us, dominated by reading dense and touched-expert weights;
- full attention matters increasingly at batched contexts;
- verify floors rise to 144-267 us, where useful tensor-core occupancy becomes possible.

## 4. Optimization roadmap

### Phase A: establish a profileable 1.00x baseline

Adapt the official `baseline.py` into `submission.py`, retaining the problem marker.
Run one check and save the generated profile. This proves the repository, imports,
weight preparation, and output packing before any component merge.

Acceptance gate: 13/13 tests and 0.98x-1.02x overall.

### Phase B: integrate known component wins independently

1. Integrate the dense dispatcher from `submission_nathen_v18.py` behind the whole-layer
   state object. Reuse its lossless packed weights and prepare-time selection.
2. Integrate the attention decode kernels from `muse_attention_decode/submission.py` for
   `q_len=1`; retain FA3 where verify measurements say it wins.
3. Keep SGLang Marlin experts and fused boundaries unchanged.

Do not merge the two large sources at once. Each integration needs its own correctness
check and profile so regressions remain attributable.

Acceptance gates:

- dense-only: approximately 1.07x whole-layer or better;
- dense + attention: target 1.12x-1.16x;
- no scored shape slower by more than 3% unless load-weighted time improves clearly.

### Phase C: specialize route preparation and MoE by token regime

This phase determines competitiveness.

For `T<=32`:

- keep Marlin GEMMs initially;
- fuse top-k postprocessing, global-to-local ID mapping, and small-route alignment;
- preallocate all workspaces in `prepare()` and avoid `contiguous()` copies whose source
  is already in the required layout;
- tune block size from actual fixed shape bounds while remaining input-value agnostic;
- investigate a persistent small-route kernel that consumes only valid local routes.

For `T in {128, 256, 264}`:

- benchmark the Humming expert path already used by the prefill work against Marlin;
- tune the first and second expert GEMMs separately—the routed M differs after padding;
- overlap router/routing with pooler exactly as the baseline does;
- preserve FP32 reduction and the required post-expert RMSNorm-before-weighting order.

High-value custom-kernel direction:

- consume packed E2M1 weights and FP8 block scales directly with Hopper tensor cores;
- fuse W13's epilogue with SiLU-and-multiply, avoiding the 8192-wide BF16 temporary;
- fuse W2's epilogue with post-expert RMS accumulation and route weighting where the
  reduction order remains within tolerance.

Avoid deliberately replacing strong local routes with near-tied remote experts. The
checker has slack for numerical variation, but systematic workload avoidance is not a
generalizable kernel speedup and risks final-review rejection.

### Phase D: fuse whole-layer boundaries

Once component kernels are stable, attack traffic and launch gaps unavailable to the
standalone challenges:

1. **Output projection epilogue:** fuse `o` projection output with post-attention RMS,
   residual coefficients, stream update, and pre-FFN normalization. For tiny T this can
   remove a full 8192-wide round trip and a launch.
2. **Routing handoff:** emit the BF16 FFN input and FP32 router representation from the
   same boundary kernel; feed router and pooler without extra casts/copies.
3. **Tiny unpool boundary:** for `T<=8`, compute only 1024 nonzero unpool columns, fold
   `post_ffn_norm` into the producer, accumulate the known zero-padded RMS correctly,
   then update the full FP32 stream and next-layer norm.
4. **Expert-to-unpool handoff:** evaluate fusing weighted expert finalize with unpooler
   input staging. The benefit is mainly launch removal; the 4096-wide tensor is small
   relative to expert weights, so profile before investing heavily.

PDL or stream overlap is useful only where dependencies permit it. CUDA-graph replay
and cold-L2 timing must be used; eager timing will overvalue launch-cache effects.

## 5. Numerical and judge constraints

- Residual update and next attention input have a shape-dependent 3%-15% relative-L2
  bound tied to FP8-query error, but the residual comparison subtracts the carried input.
  Small branch errors are therefore visible.
- KV writes must be within 8% and must honor page size and per-tensor scale.
- Router weights are FP32. Above 8 tokens the permitted fast representation is BF16
  high + low with FP32 accumulation; arbitrary BF16 downcasting is not valid.
- Chosen experts must be unique and near the reference's biased top-8, with mandatory
  preservation of clearly separated local experts.
- The MoE output is checked using the submitted route IDs, so routing tolerance does not
  permit skipping computation for routes that were actually returned.
- `prepare()` may losslessly repack weights and allocate reusable workspaces, but
  `kernel_fn` must consume fresh input data on every replay.

## 6. Experiment order

1. E001: SGLang-equivalent submission, obtain whole-layer profile. **Complete:**
   13/13 tests, 1.000045x, 22.6193 ms, job `01169a9a59374fcc`.
2. E002: dense/router integration only. **Complete:** 13/13 tests, 1.078919x,
   20.9657 ms, job `66881319d3044dcd`.
3. E003: attention decode integration only on top of E002. **Complete:** 13/13 tests,
   1.147103x, 19.7195 ms, job `2cad40d153f142aa`.
4. E004: `T=128/256` Humming-vs-Marlin expert dispatch. **Complete:** 13/13 tests,
   1.189263x, 19.0204 ms, job `92cdd3b87d3447a4`.
5. E005: specialized small-route alignment/finalize for `T<=32`. **Complete:** fused
   remap/alignment passed 13/13 tests at 1.203221x and 18.7998 ms, job
   `e053f1c2efcf4b66`.
6. E006: tiny sliced-unpool + FFN-boundary fusion. **Rejected:** passed 13/13 at
   1.205351x / 18.7666 ms, but the 0.18% overall delta was noise-level and verify-1
   sliding regressed. Reverted to E005; job `50b4cf7b6ad24157`.
7. E007: output-projection + attention-boundary fusion.
8. Only after a passing check with a clear confidence interval: scored submission.
   **Complete for the E005 checkpoint:** S001 passed 13/13 at 1.197188x / 18.8945 ms,
   job `dad63279b80446a3`, initially rank 715/841.

With 20 daily checks available, reserve at least four for integration/fallback validation
and use at most 16 for exploratory variants.

## 7. Sources

- Official task: https://openkernelchallenge.dev/problems/muse_decode_layer
- Official rules: https://openkernelchallenge.dev/rules
- SGLang Muse layer at the challenge commit:
  https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/muse_spark_layers.py
- SGLang model at the challenge commit:
  https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py
- SGLang Marlin MoE wrapper:
  https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/moe_wna16_marlin.py
