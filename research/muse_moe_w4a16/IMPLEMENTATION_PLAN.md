# muse_moe_w4a16: implementation and next-kernel plan

> Update 2026-10-30: `submission_v99.py` replaces the v77/v96 kernel after
> official job `698ba06f353d47a4` passed 5/5 at **5,701.94 us / 2.358583x**
> (95% CI 2.323204-2.393962x). The exact scored source is active as
> `submission.py`, SHA256
> `B59196AC2365CBA30CC666CC74F38051FF8F5E19C8FBD9E771ADFD4696C40765`.
> It retains BF16 activation semantics and uses lossless NVFP4 repacking,
> DQ7 decoding, guarded exact PRMT scale conversion, shape-specific tiny/mid
> paths, and a fused persistent WGS kernel. The run moved the team from rank
> 482 to rank 131 of 1,180. `submission_v96.py` remains the prior recovery
> point. With the day's 20/20 checks exhausted, do not spend another scored
> attempt on an unvalidated compile-time knob change.

> Update 2026-10-29: v61 validated Hopper CTA clusters; v62's interleaved
> W13-to-W2 pipeline exposed scheduler dependency inversion, and v63's safe
> tail-only form was neutral. v64 then paired two K16 WGMMA instructions per
> async commit group, preserving four operations in flight while halving group
> control. Job `9c38c537be5744a1` passed 5/5; paired medians improved NCW16 from
> 256 to 253 us at M=128 and 305 to 301 us at M=256, where `WGPAIR` was
> selected. Scored job `09cc04a74eeb4016` then passed 5/5 at 7,151.1 us /
> 1.88061x and reproduced 297 versus 302 us at M=256. `submission.py` is
> identical to the new best `submission_v64.py`; board rank is 387/1,051. See
> `OPTIMIZATION_CHECKPOINT.md` for hashes and full job details.
>
> Follow-up 2026-10-29: v65's two 8-warp CTAs/SM and v66's two-CTA DSM
> full-K medium path both passed 5/5 but lost their paired tuner comparisons.
> v65 measured 127/178 us at M=16/32 versus 113/158 us for v64; v66 measured
> 179/289 us versus 111/157 us. More CTA residency and remote DSM activation
> reads are closed. The next bounded experiment is deeper WGMMA async flight
> plus `WGPAIR` on the M=128 CPB2 path; v64 remains active.
>
> Unscored follow-up: v71 implements the bounded experiment. Its `WG4` path
> commits four HGMMA instructions at once and double-buffers decoded A
> registers, producing eight non-serialized operations in flight at the same
> 96-register allocation as v64. Hopper SASS confirms four HGMMA instructions
> per group and alternating banks. A TPX16 CPB2+WGPAIR alternative also compiles
> without serialization at 56 registers; all TPX32 CPB2 forms were pruned after
> ptxas serialized them. A three-bank/12-op v72 was likewise rejected by ptxas.
> v71 is ready for one explicitly authorized check; `submission.py` remains
> byte-identical to scored v64 and no new check has been run.
>
> Scored follow-up: authorized job `c8cf0df0e3b74ab0` passed v71 5/5 but
> regressed to 7,202.4 us / 1.86723x; its new paths were not selected.  v73
> therefore returns to the larger structural lever and repairs v67's multicast
> hang: odd tails now allocate only half the row-block units, and a rank-1
> acknowledgement prevents the one-entry queue epoch from being overwritten.
> Exhaustive work-map validation and exact SM90a compilation pass for focused
> TPX16/TPX32 CPB2 cluster candidates, with multicast instructions confirmed in
> SASS.  v73 has not received any remote run; v64 remains active.
>
> Consumer-side follow-up: v74 replaces repeated fragment bit decode plus
> packed BF16 scale multiplication with an exact generic 128-KiB codebook while
> keeping model weights packed at four bits.  Exhaustive mapping validation
> passes.  Retained M1/M8, M16, and M128/M256 specializations compile without
> spills and remove 128, 256, and 512 `HMUL2` instructions respectively.  Full
> M32 lookup spilled; its replacement uses lookup only for W13 experts with at
> most 16 routes and is spill-free.  Original v64 kernels remain tuner fallbacks.
> Authorized scored job `523e782e654a42d3` passed 5/5 but all `gk`/`wx`
> candidates failed before launch because an int32 lookup tensor was requested
> through `data_ptr<uint32_t>()`; old fallbacks scored 1.68652x / 7,974.1 us.
> Both wrappers now isolate LUT access behind the LUT build and use an untyped
> pointer cast, while baseline builds never inspect the tensor.  The repaired
> v73 multicast scheduler is also combined with lookup at the proven TPX32 and
> NCW16/WGPAIR shapes.  Local SM90a builds confirm the retained candidates are
> zero-spill; NCW16 has no serialization warning, while TPX32 is tuner-gated
> because ptxas reports serialized WGMMA.  A static audit caught and fixed the
> required MC2 counter allocation (2,048 ints, not 64), verifies extension
> signatures, and exhaustively proves the two-rank map for np=1..128.  Active
> `submission.py` remained v64 throughout the experiment.
>
> Scored resolution: authorized job `aaf8d72c60624cae` passed 5/5 at
> 7,181.96 us / 1.87254x, but the tuner rejected every lookup and multicast
> candidate.  Lookup was 3-5x slower than arithmetic (for example M16 411 vs
> 114 us, M128 835 vs 246 us, M256 936 vs 299 us); combined multicast+lookup
> reached 881-1,494 us.  Arithmetic multicast also lost.  The apparent static
> SASS savings were overwhelmed by dependent table-load latency.  This direction
> is closed, and v64 remains active and best.
>
> Scored PTX resolution: authorized v75 job `9ae81acb80274f40` timed out after
> preparation and produced no correctness records or score.  Its SASS contains
> a blocking `USETMAXREG.TRY_ALLOC.CTAPOOL` retry loop but no register release;
> the consumer warpgroups could not satisfy the 120-register request and the
> producer waited at the following CTA barrier.  v75 is rejected.
>
> Local follow-up: v76 removes the same NCW16/WGPAIR local frame without dynamic
> register allocation.  It shadows the epilogue-only `g1`/`g2`/`nw` pointers
> once per CTA in 24 bytes of shared memory and reloads them only in the fused
> epilogue.  Exact CUDA 12.4 SM90a compilation reports 96 registers, zero stack,
> zero spills, no WGMMA serialization, and unchanged HGMMA/HMUL2/warpgroup
> instruction counts.  The ordinary v64 WGPAIR configuration remains the tuner
> control.  Authorized scored job `0b36057067044648` passed 5/5 at 7,178.23 us /
> 1.873512x, 27.09 us slower than v64.  EPISHADOW did win its paired M=256
> tuner comparison, 298 versus 301 us, but the gain was too small to improve the
> aggregate reliably.  `submission.py` remains byte-identical to v64.
>
> Scored follow-up: v77 replaces exact SiLU only in opt-in tuner variants with a
> reciprocal-free saturated odd polynomial.  A standalone CUDA benchmark was
> 47% faster than exact SiLU.  Five reduced-dimension end-to-end NVFP4 MoE
> simulations measured 0.96-0.99% total error on normal inputs and 0.72-0.73%
> on skewed inputs, with worst rows below 1.16%.  Every focused gk/CPB2/WGPAIR
> SM90a build preserves its control's registers, stack/spills, tensor-core count,
> and async barriers.  M=256 stacks the polynomial with v76 EPISHADOW; M=128
> uses polynomial-only because EPISHADOW did not change CPB2's spills.
> Authorized job `c6413be663324360` passed 5/5 at 7,137.57 us / 1.884184x,
> improving v64 by 13.57 us (0.190%).  POLYSILU won M=32 by 160 versus 161 us
> and was selected on a 36-us tie at M=1; exact controls remained selected for
> the other four buckets.  v77 is the new scored best and active `submission.py`.

## Current result

`submission.py` is the validated v77 packed-NVFP4 implementation. Scored H200
job `c6413be663324360` passed all five correctness cases and measured
**1.884184x / 7,137.57 us** (95% CI 1.855922-1.912447x).  It improves the prior
v64 best by 13.57 us / 0.190%.  Corrected v74 has been measured and rejected;
v75 timed out and is rejected; scored v76 was 0.38% slower.  The public leader
was 2.6888x / 5,001.7 us when last checked, so this implementation is useful
but not yet leaderboard class.

| scored shape | SGLang | v77 | speedup | reference floor |
|---|---:|---:|---:|---:|
| M=1 decode | 76.51 us | 29.29 us | 2.612x | 11.4 us |
| M=16 decode | 188.02 us | 111.87 us | 1.681x | 52.5 us |
| M=32 decode | 204.78 us | 151.95 us | 1.348x | 70.8 us |
| M=8 verify | 143.46 us | 79.14 us | 1.813x | 37.3 us |
| M=128 verify | 454.99 us | 236.51 us | 1.924x | 101.3 us |
| M=256 verify | 619.66 us | 285.72 us | 2.169x | 108.9 us |

The result is materially stronger than the first Marlin/Humming hybrid
(`f4c16fcd8b534956`, 1.250x / 10,757.9 us) and the first recovered fused kernel
(`2d7b9c00a2614b79`, 1.824x / 7,373.7 us).

## v51 NCW24 result

`submission_v51.py` preserves the first next-kernel experiment while
`submission.py` is restored to the validated v50 control. The candidate adds an
`NCW=24, RPW=1, TPX=64` build: six consumer warpgroups cover three 128-row
blocks per persistent work unit. Since 64 W13 blocks and 32 W2 blocks are not
divisible by three, unit counts are ceiling-divided (22 and 11), each ring
header carries `valid_blocks`, and the producer's TMA transaction count and
weight-copy loop include only those blocks. Inactive tail warpgroups continue
to consume/release every mbarrier slot and contribute to completion counters,
but skip LDS, dequantization, WGMMA, and epilogue stores.

The two NCW24 slots use 176,128 bytes; the full M=257 dynamic allocation is
about 186,120 bytes, below Hopper's 227-KiB per-block opt-in limit. The launch
uses 800 threads, leaving a compile-time ceiling of 81 32-bit registers per
thread from Hopper's 64-K register file.

Scored H200 job `0462e0cce19241f1` passed 5/5, but the autotuner decisively
rejected NCW24: **304 us versus 256 us** for NCW16 at M=128 and **365 us versus
304 us** at M=256. The job selected the unchanged NCW16 path and scored 1.844x /
7,293.2 us, statistically indistinguishable from v50's 1.840x / 7,309.5 us.
NCW24 is therefore retained only as negative evidence, not as the active
kernel. Extra consumer warpgroups are not the missing structural gain.

## v52 result: two persistent CTAs per SM

`submission_v52.py` tests latency hiding across CTAs instead of adding more
warpgroups to one CTA. It adds `CPB=2` support to the fused-gather WGMMA kernel,
launches `2 * SM_count` persistent blocks, and refuses to launch unless
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` confirms that both blocks are
concurrently resident. This guard is mandatory because the kernel uses
grid-wide monotone counters.

Two configurations are exposed to the existing cold-L2 graph tuner:

- `NCW=8, RPW=1, TPX=64, NSL=2, CPB=2`: 110,344 bytes per block at M=257,
  220,688 bytes per SM, with two independent producer/ring pipelines and four
  total consumer warpgroups per SM.
- `NCW=8, RPW=2, TPX=32, NSL=2, CPB=2`: 114,440 bytes per block and 228,880
  bytes per SM. It preserves two-block activation reuse but narrows the token
  pass to make the second CTA resident.

Both are below Hopper's 228-KiB shared-memory capacity per SM by construction;
the runtime occupancy check also covers compiler register allocation. Scored
job `464df2779a444197` passed 5/5. At M=128, the `RPW=2, TPX=32, CPB=2` variant
won at **242 us versus 255 us** for NCW16 and was selected. At M=256 it measured
306 us versus 303 us, so the tuner correctly retained NCW16. The final scored
times were 238.3 us and 294.5 us for M=128/256. `submission.py` therefore stays
on v52; `submission_v50.py` remains the reproducible control.

## Constraints that determine the architecture

- H200 is SM90 Hopper. It has BF16/FP16/FP8 Tensor Cores but no native FP4
  matrix instruction. NVFP4 must remain packed in HBM and be decoded to BF16
  fragments inside the kernel.
- Weights must remain losslessly NVFP4 and activations must remain BF16. An
  offline BF16/FP8 weight expansion or a W4A8 shortcut is not legal here.
- The EP8 rank owns 32 of 256 experts. Routing is top-8 and skewed; the busiest
  rank has few routes at decode sizes and most/all experts active at M=128/256.
- Timings use CUDA Graph replay with cold L2. The kernel must optimize HBM
  traffic and device-side scheduling rather than depend on cache warmth.
- Post-expert RMSNorm makes W2 an inseparable part of the route computation;
  normalization and weighted reduction should remain fused with its epilogue.

Primary implementation references:

- Challenge: https://openkernelchallenge.dev/problems/muse_moe_w4a16
- Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/
- Hopper grouped GEMM (persistent TMA + WGMMA):
  https://github.com/NVIDIA/cutlass/blob/main/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu
- CUTLASS grouped scheduler notes:
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md
- SGLang Marlin MoE implementation:
  https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/moe_wna16_marlin.py

## Implemented v50 design

### Lossless prepare-time packing

Each 16-row x 512-K NVFP4 tile is stored in MMA/WGMMA fragment order as packed
E2M1 codes plus E4M3 block scales. This is a lossless layout transform, not a
precision conversion. A second WGMMA-oriented layout groups eight row tiles and
64-K chunks. Prepare also builds all candidate CUDA extensions and tunes only
the six scored M buckets under cold-L2 graph replay; total preparation remained
inside the 300-second limit (about 148 seconds before tuning in the last run).

### Small/medium persistent `mma.sync` family (`moe_k`)

One CTA is resident per SM. TMA streams 2,304-byte packed pieces through
per-warp rings; warps decode E2M1/E4M3 with integer permutes plus BF16x2
multiplication and feed `m16n8k16` BF16 MMA. Routing, W13, SwiGLU, W2, norm
statistics, normalization, and route reduction are one launch.

The measured shape choices are:

| M | selected shape |
|---:|---|
| 1 | `SW=4, KS2=1, MAXM=2, PPS=2` |
| 8 | `SW=4, KS2=1, MAXM=8` |
| 16 | `SW=2, KS2=2` |
| 32 | `SW=2, KS1=2, KS2=4, MAXM=32` |

`SW=2` is a real v49 improvement: it doubles independent tile groups and
halves the reduction fan-in. A third M=8 ring stage and wider per-slot M=8
loads did not win, confirming that producer latency is not the limiting factor.

### Large persistent TMA/WGMMA family (`moe_x`)

For M=128/256, a one-CTA-per-SM persistent kernel uses one producer warp and
four consumer warpgroups (`NCW=16`). Each work unit covers two 128-row blocks,
uses a two/three-slot TMA ring, decodes packed NVFP4 into register A fragments,
and issues BF16 register/shared `m64nNk16` WGMMA. It processes all routes of an
expert per weight pass, then fuses W2 gain, sum-of-squares, RMSNorm, and weighted
top-8 combination.

v50 removes the standalone `xgather` launch. The persistent CTA already builds
stable per-expert route ranks for scheduling, so CTAs cooperatively write the
same WGMMA-swizzled activation buffer from those ranks and publish it through a
one-time grid barrier. This removed duplicated routing work and improved the
large-shape check without changing arithmetic.

The last tuner selected `NCW=16` for both M=128 (254 us synthetic median) and
M=256 (303 us synthetic median). A nominal 512-row unit (`NCW=16, RPW=2`) was
slower: larger per-warp accumulator/live state outweighed activation reuse.

## What the research ruled out

- Extra TMA stages: consumer wait-full is small; consumers, not producers, are
  the bottleneck.
- Moving dequantization to 3-4 producer warps: an earlier H200 experiment was
  correct but regressed because a few transform warps replaced work previously
  distributed across eight consumer warps.
- Preconverted BF16 scales or expanded records: extra bytes/register pressure
  cancel the removed integer instructions and approach an illegal precision
  expansion.
- Launch-level groups of experts: repeated persistent tails cost more than any
  L2 reuse of the intermediate.
- More K split on the ring kernel: M=32/128/256 tuner results consistently
  penalized split factors 2/4/8.
- Pointwise-only work: the current path is a single main kernel (plus no gather
  on v50), so launch fusion cannot explain the remaining ~2,290 us score gap.

The measured consumer stage spends substantial time in both exact NVFP4 decode
and WGMMA issue/retirement. Minor launch and ring changes cannot turn 1.84x into
2.6x.

## Historical NCW24 plan (completed and rejected by v51)

The next bounded implementation should be a **24-consumer-warp narrow-N
persistent WGMMA kernel**, initially only for M=128/256:

1. Use `NCW=24, RPW=1, TPX=64`: six consumer warpgroups compute three 128-row
   blocks per work unit, 50% more consumer warpgroups than v50.
2. Keep `WCH=4`. With `UBL=3`, one slot is about 88 KiB and two ping-pong slots
   are about 176 KiB. Route metadata and the M=256 list keep total dynamic
   shared memory near 184 KiB, inside H200's opt-in limit.
3. Launch 800 threads (24 consumer warps plus one producer warp). One resident
   CTA gives an 81-register/thread budget. This is viable for the observed
   narrow expert token counts (`N<=16` normally, `TPX<=64`) but must be checked
   in ptxas for spills.
4. Make row-block counts ceiling-divided: W13 has 22 units and W2 has 11.
   Add `valid_blocks` to each ring header. On the final unit, inactive
   warpgroups still arrive at the empty-slot barrier but skip LDS, WGMMA, and
   epilogue stores. TMA transaction bytes include only valid weight blocks, so
   the last expert never reads across its allocation.
5. Preserve the v50 fused gather and exact register-source dequant path. Do not
   add decode-to-shared transformation or another grid-wide intermediate.
6. Accept this family only if paired cold-L2 medians improve both M=128 and
   M=256 by at least 5%; otherwise the extra warps are already issue-saturated.

For M=16/32, the next small experiment is `SW=1` with the current K splits. It
tests whether fully independent warp tiles beat the proven `SW=2` balance. It
is a secondary 3-5% opportunity, not the path to the leaderboard.

If `NCW=24` fails to provide a structural gain, the remaining credible route is
an expert-scoped cooperative-cluster pipeline that makes W13 output chunks
available to W2 without a grid-wide phase boundary. That design is much larger:
eight CTAs own disjoint W2 row ranges for one expert, cooperate on W13 chunks,
and retain per-route W2 accumulators while streaming K. It should not be built
until the simpler six-warpgroup consumer test establishes that additional
consumer parallelism is valuable.

## Validation policy

- Keep `submission_v48.py`, `submission_v49.py`, and `submission_v50.py` as
  reproducible controls; active `submission.py` is v50.
- Run AST/source checks locally, then official off-board checks for correctness,
  held-out M=257, CUDA Graph replay, and H200 resource behavior.
- Compare paired cold-L2 medians, not isolated profiler samples. Do not spend a
  scored submission unless a candidate clears run variance and all tests.
- No files outside `muse_moe_w4a16` are modified by this work.
