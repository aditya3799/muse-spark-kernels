# Independently implemented optimization branch

This branch explores kernels written specifically for this project. Public
submissions informed only high-level choices such as shape dispatch, splitting
the FP32 router weight into BF16 high/low components, and using tensor cores.
No Lee Penkman or Nathen Fernandes CUDA kernel source or route table is embedded
in `submission_original.py` or in the custom router inside
`submission_coldtune.py`.

The public-source leaderboard candidate remains in `submission.py`; it was not
changed by these experiments.

## Current original components

- `submission_original.py`: router research harness with independently written
  scalar split-K, vectorized FP32 GEMV, cooperative WMMA, cold-L2 autotuning,
  a packed high/low BF16 GEMM fallback, and Hopper-native WGMMA variants.  The
  newest route uses B128-swizzled operands, a two-stage `cp.async` pipeline,
  and direct FP32 accumulation of the BF16 high/low weight components.
- `submission_coldtune.py`: the earlier local dense cuBLASLt tuner combined with
  the new 512-thread vectorized router. The m=1 specialization uses non-caching
  `float4` loads; m=2..8 use cached vector loads.
- `compile_extension_local.py`: local compilation, correctness, and directional
  cold-cache tests for every custom route.

## H200 checks

| Candidate stage | Job | Tests | Speedup | Score |
|---|---|---:|---:|---:|
| One-warp WMMA prototype | `4eb50f0be0234b84` | 9/9 | 0.777555x | 6212.81 us |
| Cooperative shared-memory WMMA | `eb84838f481c4000` | 9/9 | 0.745001x | 6484.28 us |
| Cold-L2 route autotuner | `98155e5c6cab4391` | 9/9 | 1.072454x | 4504.44 us |
| Vectorized FP32 small router | `a6a4f59beac7457e` | 9/9 | 1.118813x | 4317.79 us |
| Non-caching loads + stable fallback | `dec94413b339447f` | 9/9 | 1.138212x | 4244.20 us |
| Original dense/router hybrid | `89686cda867b4842` | 9/9 | 1.157866x | 4172.16 us |
| First shared/shared WGMMA prototype | `65f888ef95c3493a` | 9/9 | 1.139209x | 4240.49 us |
| B128-swizzled WGMMA staging | `01a8b8a24bdd4649` | 9/9 | 1.149564x | 4202.29 us |
| Register-A WGMMA, K64 stage | `432cecb329c34afa` | 9/9 | 1.148x | 4209.70 us |
| Double-buffered `cp.async` WGMMA | `3fe0ba39ed1d4e02` | 9/9 | 1.149x | 4205.90 us |
| Forced WGMMA end-to-end validation | `534fb034d7f54d62` | 9/9 | 1.145x | 4219.00 us |
| Inactive-row-elided WGMMA | `e001086ac860499a` | 9/9 | 1.146x | 4214.70 us |
| Integrated TMA + WGMMA (scored) | `1bfbed3324da4cae` | 9/9 | 1.148x | 4208.70 us |

The earlier dense-autotuner check `4409b16f33fa4bc3` remains the highest
aggregate result for the fully local branch at 1.165664x. The new hybrid is
inside its run-to-run confidence range, while materially improving its m=1
router from 5.454 us to 4.361 us.

## Findings

1. A one-warp WMMA tile repeats too much activation/weight traffic at m=128 and
   m=256.
2. Cooperative shared-memory WMMA reduces traffic but synchronization and
   shared-memory layout costs overwhelm the benefit without Hopper WGMMA/TMA.
3. Four-way scalar split-K improves memory-level parallelism, but its memset and
   atomic merge limit the small shapes.
4. A 512-thread `float4` FP32 GEMV is the best independently written small
   router so far. On the H200 hybrid check it reached 4.361 us at m=1.
5. Synchronous shared/shared WGMMA was limited by operand staging.  B128
   swizzling alone did not help, and register-sourced A was slower because its
   global loads were less coalesced and increased register pressure.
6. A two-buffer `cp.async` pipeline was the first custom WGMMA route to beat the
   packed fallback in cold tuning at m=16 (14.3--14.6 us versus 14.8 us).  When
   forced through the judge it passed all nine tests, directly validating the
   descriptor, swizzle, async ordering, and accumulator mapping.  Its graph
   replay router time was 7.6 us, down from about 9.5 us for this branch's
   packed fallback.
7. Initializing inactive m=16 rows once instead of issuing zero-fill copies on
   every K stage was correct and gave a small graph-replay improvement.  The
   cold result was within noise, so further zero-fill micro-tuning is not worth
   another remote check.
8. The preserved public-source scored kernel is still faster at m=16 (about
   4.8 us), so the new route should not replace `submission.py` yet.  The next
   credible independent step is a wider-N WGMMA tile or persistent CTA design
   to reduce duplicated activation movement and split-K epilogue overhead.
9. The integrated TMA route now exists and was exercised directly in scored job
   `1bfbed3324da4cae`.  It uses three 2-D tensor maps (activation, BF16-high,
   BF16-low), B128 hardware swizzling, alternating 48 KiB shared stages,
   transaction mbarriers, and WGMMA high/low accumulation.  It passed 9/9 and
   improved the custom m=16 graph latency from 7.6 us to 7.2 us.  Cold tuning
   improved from 14.5 us (`cp.async`, variant 73) to 13.6 us (TMA, variant 83),
   a 6.2% route-level gain.  Aggregate performance remains below the preserved
   best because the rest of the independent branch is substantially slower.

The Hopper layouts and ordering rules were derived from NVIDIA's PTX ISA and
the official CUTLASS SM90 GMMA traits/wrappers, not from another contestant's
kernel implementation.
