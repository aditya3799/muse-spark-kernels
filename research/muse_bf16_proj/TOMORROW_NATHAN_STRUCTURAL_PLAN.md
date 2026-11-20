# Muse BF16 Projection — Next-Session Handoff

Date saved: 2026-10-28 (Asia/Calcutta)

## Resume here: v54 router TMA-multicast research

The next implementation target is a new **v54 router architecture**, based on
the protected v50 best (`submission_qkv32_o16_m16trace_v50.py`). Do not resume
the wide-Pooler work: v53 was a decisive negative result and all experimental
Pooler routes fell back to the v50 incumbents.

### Why router is the next target

The v50 router has repeatable estimated headroom in every scored workload. The
first v54 prototype should target the three shapes with the best combination of
headroom and structural reuse:

| Shape | v50 corrected time | Estimated floor | Gap |
|---|---:|---:|---:|
| M32 | 4.821 us | 1.864 us | 2.958 us |
| M128 | 5.969 us | 2.212 us | 3.758 us |
| M256 | 7.928 us | 2.676 us | 5.252 us |

The existing wide router is at
`submission_qkv32_o16_m16trace_v50.py:1156-1266`, and its launch selection is
at lines 1507-1537. It uses N64 output tiles and clusters only along split-K.
Consequently, each output group independently loads the same activation tile.

### Proposed v54 architecture

Implement a two-dimensional, eight-CTA cluster route:

- Actual output tile: **N32**.
- Use `wgmma.m64n64k16.f32.bf16.bf16`.
- Pack each FP32 weight's BF16 high and low components into the two N32 halves
  of the WGMMA N64 operand, then fold the two accumulator halves exactly.
- Cluster geometry: **K-split 4 x N-peer 2 = 8 CTAs**.
- Multicast each activation TMA tile to the two N-peer CTAs with
  `cp.async.bulk.tensor...multicast::cluster`.
- Give each N-peer its own weight tile.
- Reduce DSMEM partials only among the four split-K CTAs belonging to the same
  N32 output tile.
- Start with M32, M128, and M256. Preserve all v50 paths as controls and
  automatic fallbacks.

Expected grid behavior:

- M32: retain roughly the same CTA count while halving duplicated activation
  traffic between N peers.
- M128: retain 64 CTAs, change from N64/split8 to N32/split4/N-peer2, and cut
  split-K reduction width in half.
- M256: use 128 CTAs instead of the current 64, improving H200 utilization
  while retaining split-K 4 and sharing activation loads between N peers.

This is materially different from v34. v34 only doubled split-K and used
nonportable 16-CTA clusters. Its adjusted timing regressed M128 by about 51%
and M256 by about 36%, so do not repeat split16/split8 alone. See
`profiles/submission_routerwide_v34.7d569819f5554ac5.log:61-62`.

### Expected result and confidence

If v54 reduces M32/M128/M256 router time by 15-30%, the aggregate projection is:

- approximately **3164-3142 us**;
- approximately **1.527-1.537x**;
- approximately **22-44 us** better than v50's 3186.412227 us.

Pre-benchmark confidence is about **65%** that the architecture will beat the
corresponding v50 routes. This is enough to implement and check, but it is below
the agreed 75-80% threshold for spending a final scored submission.

### Required v54 validation and promotion gates

1. Copy v50 to a new candidate; never edit the protected v50 file.
2. Compile every route for SM90a and reject any route with spills or a local
   stack frame.
3. Run exhaustive one-hot coefficient validation for each enabled shape.
4. Require finite output and FP64-relative random/spike error below `2e-5`.
5. Run repeated paired cold-L2 timing against the exact v50 incumbent.
6. Enable a candidate only when its overhead-adjusted ratio is below about
   **0.94**. Otherwise fall back automatically.
7. Use an unscored H200 check first. Do not make a final scored submission
   until correctness passes, the projected aggregate beats 3186.412227 us, and
   confidence reaches at least 75-80%.

### Research references

- NVIDIA Hopper Tuning Guide: TMA, warp specialization, clusters, and DSMEM:
  <https://docs.nvidia.com/cuda/hopper-tuning-guide/>
- NVIDIA PTX ISA: `cp.async.bulk.tensor` cluster multicast:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- NVIDIA CUTLASS Hopper warp-specialized GEMM example:
  <https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu>

## Final v53 scored result

Job `0946457ed69249c5` passed 9/9 with no flags at **3218.850116 us /
1.5007843874x**.  It did not replace v50's 3186.412227 us best.  The M128
persistent controls measured 26.08--26.14 us versus 23.616 us incumbent and
were disabled.  The valid M256 routes measured 42.72--65.95 us versus about
28.8 us incumbent and were disabled.  The experimental N128/split4 M128 route
failed the exhaustive one-hot check and was also disabled.  Current leaderboard
position after the run: **137 / 1248**, best still **1.5161x**.  Daily quota is
now 10/10 scored and 20/20 checks.

## 2026-10-28 local v53 wide-Pooler sweep (preferred next check)

`submission_pooler_persistent_sweep_v53.py` supersedes v52 as the preferred
unscored-check candidate.  It retains both v52 configurations as controls and
adds five spill-free alternatives: a 32-thread M128 producer, N128/split4 for
M128, and M256 variants with three stages, no split/reduction, or split4.
Prepare exhaustively validates each launchable route, measures all survivors
against the incumbent, and activates only the fastest route when its paired
cold-L2 ratio is below **0.95**.  This is stricter than v52's 0.97 gate.

Local SM90a compilation passed for all seven routes.  Ptxas reports 90--114
registers, zero stack, and zero spills.  Two exploratory N32 routes were
removed after compilation exposed 648/1216-byte stack frames and spills.  A
route-selection unit test passed.  On the local RTX 3050, where Hopper routes
are necessarily disabled, v50/v52/v53 outputs were bit-exact; v53-to-v50
paired median ratios were 1.0042 at M128 and 1.0108 at M256, within local
fallback noise.  This benchmark does not predict H200 kernel speed.

Use v53, not v52, for the next unscored H200 check.  Do not score it unless
all validations pass and an enabled route is comfortably below the 0.95 gate.

## 2026-10-28 local v52 persistent wide-Pooler candidate (not submitted)

`submission_pooler_persistent_v52.py` is the next candidate.  It preserves
the authoritative v50 routes and adds guarded Pooler N=4096 routes only for
M=128 and M=256.  Unlike the older WGB experiments, it launches exactly 64
N64 output tiles times split-K 2 = 128 CTAs (one H200 wave).  The M256 route
uses two consumer warpgroups so each staged weight tile feeds both M128 slabs.

Local SM90a compilation passed.  Ptxas reports 114 registers for M128 and 102
for M256, with zero spills and zero stack in both kernels.  The embedded H200
validation checks exhaustive coefficients, FP64 random/spike error, launch
occupancy, and local-memory attributes.  A repeated paired cold-L2 gate keeps
a route only when it is at least 3% faster than the incumbent (`ratio < 0.97`).

This is not yet 75--80% confidence because no H200 runtime is available
locally.  After the daily quota reset, spend one unscored check on v52 and
inspect its validation/gate lines and per-shape medians.  Do not use the final
scored attempt unless the check is fully correct and projects an aggregate
below v50's 3186.412227 us.  Candidate SHA-256:
`7CE983097A249F698E5D8E7BD13101EFCEAA14C0DFB1463B523E9423D4A36654`.

## 2026-10-28 unpooler-32 and O16 follow-up

The authoritative scored best is now
`submission_qkv32_o16_m16trace_v50.py` at **3186.412227 us /
1.5160624729x** (job `40e982512af34a70`, rank 129 / 1235 at submission).
It passed 9/9 with no flags.  SHA-256:
`D0D1124EC0B2775DFD3C48ABCF446C0B2DCB642FF3F0ECD3192FCC97D68574E1`.
Keep the earlier v33 file untouched as the simpler fallback.

### Unpooler-32 structural sweep: v33 geometry confirmed optimal

`submission_u32_sweep_v49.py` tested PF 1/2/3/4, Q 1/2/3, launch-bound
cap 2, WK 2/3/4/6/8, and split-2 variants.  All launchable variants passed
all 4096 one-hot columns plus FP64 random/outlier validation.  Cap-2 and WK8
spilled 112 local bytes and were rejected.  Every other alternative was slower
than `(BN=32,R=64,WK=4,PF=2,Q=2,C12,CAP=1,split=1)`; the closest were Q1
at ratio 1.0033 and Q3 at 1.0098.  Split-2 WK4 was 1.3317x slower.  The gate
kept the baseline.  Scored job `df25da797f464cd3` passed 9/9 at
3240.765787 us; do not continue configuration-only work in this family.

### QKV32 plus exact O16 route

v50 combines the already proven QKV32 `m32x_q` specialization with Nathan's
exact MIT-licensed `m16trace_early` O16 source.  Both CUDA and C++ source match
Nathan byte-for-byte and compile locally for SM90a.  The O16 path reproduces
all 1024 coefficients exactly and passes FP64 random/spike validation.

In v50, QKV32 measured 16.064 us versus 16.800 us (ratio 0.9562) and was
enabled.  Its official corrected median was **12.08 us**.  O16 tied its
incumbent at 9.728 us in the cold-L2 gate and was disabled; the aggregate became
the new best above.

`submission_qkv32_o16_m16trace_force_v51.py` allowed the exact O16 route on a
near-tie because Nathan's saved profile used it.  Job `0e93a62d54514233`
passed 9/9 and profiled the expected kernel at **5.1 us**, but its corrected
O16 median was 5.57 us and its noisy aggregate was 3236.337680 us / 1.492675x.
Do not promote v51.  At save time the quota is **1 scored / 0 checks remaining**;
do not spend the last run on a duplicate or variance-only rerun.

## 2026-10-28 QKV32 M32/m32x follow-up

The scored production best remains
`submission_qkv_o32_u16_u32_r32_v33.py` at **3187.835967 us /
1.5153853742x** (job `617f2d50e27040c4`). Its SHA-256 is still
`D89BD4826822A014EC8E67CC5987AE133E55136567A543C9433EB0BDB8375936`.

### Generic M32: exact-valid but slower, rejected

`submission_qkv32_m32_v46.py` integrated the dormant generic M32 QKV32
entry `(FMT=1, MT=4, RED=4, NTMAX=256)` from Nathan's source. The embedded
CUDA was byte-identical, compiled locally for SM90a, and passed all 8192
one-hot columns plus FP64 random/outlier validation. Scored job
`50a20ebab80a4453` passed 9/9 at **3203.285851 us / 1.508076x**. The paired
H200 gate measured 17.440 us for M32 versus 16.736 us for final GVC
(ratio 1.0405), so it correctly fell back. This is not the kernel responsible
for Nathan's historical 12.02-us QKV32 result.

### Actual m32x_q specialization: real QKV32 win, aggregate run not promoted

Nathan's official profile showed that QKV32 used the later
`m32x_q_gemv<FMT=1,MT=4,RED=5,NTMAX=128,CP=0>` specialization, not generic
M32. `submission_qkv32_m32x_v47.py` integrated that exact profiled CUDA and
its decode-5 lossless C12 layout. The source matched byte-for-byte, local
SM90a compilation passed, all 8192 coefficients were reproduced exactly, and
random tests covered m=9/16/17/24/32.

Scored job `37e7d7408f884bbd` passed 9/9. The gate measured 16.160 us versus
16.960 us (ratio 0.9528) and enabled m32x. The scored QKV32 median improved
from v33's **12.587 us to 12.04 us**, and the profile showed the expected
11.5-us m32x kernel. However, unrelated shapes in that worker run were broadly
slower and the aggregate was **3246.429063 us / 1.488035x**. Treat m32x as a
credible per-shape structural improvement, but do not replace the faster
scored v33 entry with this noisy aggregate result.

### m32x_q structural sweep: baseline specialization confirmed optimal

`submission_qkv32_m32x_sweep_v48.py` tested baseline CP0, CP-async depths
1/2/3, RED4, RED6, and NTMAX256 in one extension. Scored job
`3ed841f913a2401c` passed 9/9 at **3229.411184 us / 1.495876x**. Six variants
passed exhaustive one-hot and FP64 validation; NTMAX256 failed the launch
attribute gate. Paired H200 prepare timings were:

| Variant | Candidate | Incumbent | Ratio |
|---|---:|---:|---:|
| RED5, NTMAX128, CP0 | **16.448 us** | 16.800 us | **0.9790** |
| RED5, NTMAX128, CP1 | 16.704 us | 16.992 us | 0.9943 |
| RED5, NTMAX128, CP2 | 17.696 us | 16.928 us | 1.0454 |
| RED5, NTMAX128, CP3 | 18.720 us | 16.928 us | 1.1061 |
| RED4, NTMAX128, CP0 | 17.632 us | 17.344 us | 1.0396 |
| RED6, NTMAX128, CP0 | 16.832 us | 17.184 us | 0.9906 |

The original Nathan specialization remains the family optimum. Do not spend
more attempts on CP depth, reduction mode, or launch-bound-only m32x changes.
At save time the quota is **4 scored / 0 checks remaining**.

## 2026-10-28 small-QKV and O128 follow-up

No candidate in this follow-up replaced v33. The authoritative scored best is
still `submission_qkv_o32_u16_u32_r32_v33.py` at **3187.835967 us / 1.5153853742x**.

### QKV occupancy-cap variants: correct, rejected

`submission_qkv16_sweep_v42.py` expanded the small-QKV C12 GEMV schedules.
Official check `cb78b084119748c8` passed 9/9 at 3242.971 us. The tuner selected
V21 (fast decode + pushed split-K partials + two resident blocks/SM), but QKV16
measured 9.491 us versus v33's 9.396 us.

`submission_qkv_small_cap2_v43.py` carried V21 to QKV m1/m8/m16/m32. Local
SM90a compilation passed and official check `5c10fe3d261a45f7` passed 9/9 at
3252.2 us. QKV32 improved only from 12.587 to 12.568 us while QKV1/8/16 and
the verify cases regressed. Do not continue V21 occupancy-cap propagation.

### O128 direct WGMMA: correct, structurally exhausted, rejected

`submission_o128_v44.py` extended the already integrated O32 direct WGMMA
kernel with Nathan's O128 configuration `(128,128,4,3,1,1)`. It includes
exhaustive one-hot validation, FP64 random/outlier validation, and a paired
gate against the final tuned cuBLAS/Lt incumbent. Local SM90a compilation
passed. Because the daily check quota was exhausted, scored job
`8320cf71b6d64d93` was used; it passed 9/9 at **3211.5 us / 1.504x**. The gate
saw 10.912 us versus 11.328 us and enabled the route, but the official O128
corrected median was 7.486 us versus v33's 7.324 us. The raw GPU-event medians
were effectively tied (11.504 versus 11.536 us).

`submission_o128_sweep_v45.py` then tested twelve validated O128 variants over
BK 64/128/256, legal ring depths, pipeline modes, and L2 policy. Scored job
`bc3a15f0557e475b` passed 9/9 at 3232.4 us. Every variant passed exhaustive
validation. The original `(128,128,4,3,1,1)` was fastest in the cold ranking at
10.880 us; the paired gate narrowly chose the no-producer `(128,128,4,1,1,1)`
variant at 10.784 versus 11.040 us. Its official event median was 11.488 us,
only 0.016 us better than v44 and 0.048 us better than v33, while its corrected
median was worse. This direct O128 kernel family has no material remaining
headroom; do not spend more runs on BK/stage/pipeline/L2-only variants.

## 2026-10-28 wide-shape structural experiments

No new route was accepted.  `submission_qkv_o32_u16_u32_r32_v33.py` remains
the scored production candidate at **3187.835967 us / 1.5153853742x**.

### QKV256 dual-M weight reuse: correct but rejected

`submission_qkv_o32_u16_u32_r32_v37.py` added an opt-in Hopper kernel with
two 128-thread WGMMA consumer groups and one producer group per CTA.  Each
four-stage pipeline load shared one 64-row weight tile across two independent
128-token activation slabs.  The QKV128 and QKV256 routes both passed exhaustive
all-8192-column one-hot validation plus FP64 random/outlier validation.

Official check `096cd43a414f464e` passed 9/9, but QKV256 regressed sharply:

| Shape | v33 corrected median | v37 corrected median | Delta |
|---|---:|---:|---:|
| QKV128 | 13.444 us | 14.903 us | +1.459 us |
| QKV256 | 17.137 us | 28.872 us | +11.735 us |

The QKV256 time is approximately two QKV128 passes.  This shows that the route
is WGMMA issue/compute limited rather than weight-TMA-bandwidth limited: sharing
the weight transaction does not offset executing two accumulator streams inside
one CTA.  Even perfect distribution from the present 108 CTAs across all 132
H200 SMs cannot recover the roughly 41% reduction required to beat v33.  Do not
score v37 or spend another check on split-K-only tuning of this dual-consumer
design.  A future QKV256 attempt needs a different compute mapping, not just
shared weight staging.

### QKV256 split/store/pipeline sweep: split 3 confirmed, still rejected

`submission_qkv256_split_sweep_v38.py` preserved the original QKV128 extension
exactly and compiled the dual-M experiments into a separate extension.  One
prepare-time sweep covered split-K 2 through 6, RED4 direct stores versus RED5
shared/stmatrix stores, a 32-thread versus 128-thread producer tail, and WGMMA
wait depths 1 through 3.  A 0.90 incumbent-ratio gate prevented activation of a
noisy or marginal candidate.

Official check `4567e297cc104ca6` passed 9/9.  No QKV256 experiment was
activated, so the measured verify-32 QKV route remained the v33 fallback at
17.3 us.  Key cold-L2 prepare timings were:

| Variant | Time | Finding |
|---|---:|---|
| split 2, RED5, wait 2 | 41.760 us | insufficient parallelism |
| split 3, RED5, wait 2 | 32.352 us | best split at baseline pipeline |
| split 4, RED5, wait 2 | 44.736 us | reduction/scheduling overhead dominates |
| split 5, RED5, wait 2 | 40.928 us | still too much reduction overhead |
| split 6, RED5, wait 2 | 50.080 us | worst split |
| split 3, RED4, wait 2 | 32.512 us | direct and stmatrix stores effectively tied |
| split 3, RED5, wait 1 | **26.208 us** | best dual-M variant |
| split 3, RED5, wait 3 | 45.600 us | excess WGMMA backlog is very costly |
| split 3, RED5, wait 2, 128-thread producer | 32.864 us | producer-tail trimming helps only slightly |
| same-run torch.mm incumbent | **23.136 us** | dual-M best is 1.1328x slower |

The sweep establishes split 3 and one outstanding WGMMA group as the local
optimum for this mapping, but the remaining gap is structural.  Do not spend
more checks on split/store/stage-only variations of dual-M.  Local compile
support for this diagnostic path is available as `--q256-sweep`.

### QKV256 independent token tiles: rejected

`submission_qkv256_independent_tiles_v39.py` removed dual-M entirely and ran
one consumer warpgroup per CTA.  It swept TM128 split-K 1 through 6, TM64 S1/S2,
stage counts 8 through 12, wait depths 1/2, and producer versus consumer-driven
TMA refill.  Official check `8eb76e9d80e7486a` passed 9/9; the route gate kept
the incumbent.

| Mapping | Best time |
|---|---:|
| TM128, S1, ST8, producer WG | **31.104 us** |
| TM128, S3, ST8, wait 1 | 31.488 us |
| TM64, S1, ST8 | 46.656 us |
| TM64, S2, ST10, wait 1 | 47.840 us |
| TM64, S1, consumer-driven refill | 60.960 us |
| same-run torch.mm incumbent | **23.616 us** |

Independent TM128 is limited by only 72 CTAs at S1; adding split-K reduction
does not help.  TM64 reaches 144 CTAs without reduction but reloads the weights
across four M slabs and is substantially slower.  Removing the producer
warpgroup also loses badly because it destroys TMA/WGMMA overlap.

### QKV256 independent output tiles: rejected

`submission_qkv256_output_tiles_v40.py` changed to normal-orientation WGMMA and
tiled output rows with NR32/16/8.  It included the theoretically attractive
NR16 x TM256 mapping: 144 independent CTAs, no split-K, one full-M slab, and 32
accumulators per consumer thread.  Official check `ff9ba8903ae24876` passed 9/9;
the route gate again kept the incumbent.

| Mapping | Best time |
|---|---:|
| NR32 x TM128, 144 CTAs | 46.944 us |
| NR32 x TM256, 72 CTAs | **42.144 us** |
| NR16 x TM256, 144 CTAs | 70.592 us |
| NR8 x TM256, 288 CTAs | 96.288 us |
| same-run torch.mm incumbent | **23.200 us** |

Narrowing NR multiplies activation TMA traffic across 72--288 output groups and
uses the slower normal WGMMA orientation.  More CTA parallelism cannot recover
that cost.  Together v37--v40 rule out dual-M sharing, independent token slabs,
and independent narrow-output tiling in this kernel family.  Further QKV256
work needs a new algorithm outside this descriptor-kernel mapping; switch
optimization effort to another scored shape instead.

### Pooler32 dormant CPS/WGMMA route: valid, gated out

`submission_pooler32_v41.py` exposed the exact Pooler32 entry that was present
in the generalized CPSSM tables but never prepared or dispatched. The route
uses lossless C12 weights, `(WR=1, WK=8, KU=16, S=1)`, a depth-1 pipeline, and
the existing exhaustive one-hot plus FP64 random/outlier validation. A paired
cold-L2 gate compares it with the actual post-tuning incumbent before enabling
the dispatch.

Local SM90a compilation passed for both the C12 kernel and its raw-BF16 twin.
Official check `a814b4986c97408d` passed 9/9. The route validated with a maximum
of eight escapes per warp, but was slower and correctly disabled:

| Route | Prepare-time paired median | Ratio |
|---|---:|---:|
| Pooler32 CPSSM candidate | 21.472 us | 1.0468 |
| v33 incumbent | 20.512 us | 1.0000 |

The official profile therefore retained the fallback at 16.170 us versus the
scored v33 profile's 16.125 us; the aggregate 3211.234 us is ordinary run noise,
not a new result. The saved Nathan v18 profile is also slower at 16.459 us, so
transplanting its later Pooler32 bundle is not credible. Do not score v41 or
continue Pooler32 work in this kernel family. `compile_extension_local.py`
now accepts `--p32` to compile this exact SM90a instance.

### Router128/256 doubled split: rejected

`submission_routerwide_v34.py` changed only the wide router dispatch:

- M128: `rhlaunch<8,false>` -> `rhlaunch<16,false>`
- M256: `rhlaunch<4,true>` -> `rhlaunch<8,true>`

Official check `7d569819f5554ac5` passed 9/9, but the prepare-time paired gate
rejected both routes:

| Shape | v33 candidate | doubled split | Result |
|---|---:|---:|---|
| Router128 | 10.432 us | 14.944 us | 1.51 adjusted ratio, reject |
| Router256 | 12.128 us | 15.936 us | 1.365 adjusted ratio, reject |

The extra CTAs shorten each CTA's K stream and make the cluster reduction
dominant.  Do not revisit this exact split-only change.

### Pooler128/256 cluster split-K: correct but much slower

`submission_poolerwide_diag_v35a.py` exercised the existing `wgb` TMA/WGMMA
kernel with BF16 output and 128-CTA geometries.  Check `2452911a66eb4463`
passed 9/9.  Every candidate validated, but all lost badly:

| Shape | cuBLAS incumbent | best `wgb` candidate |
|---|---:|---:|
| Pooler128 | 20.22 us | 38.27 us |
| Pooler256 | 30.24 us | 55.74 us |

The DSMEM cluster reduction is not viable for these Pooler shapes.

### Pooler workspace/counter reduction: unsafe for production

`submission_pooler_workspace_v36.py` generalized router `wgr` to a single
BF16 weight stream and BF16 epilogue.  Local SM90a compilation passed and the
official fallback path remained 9/9 correct.

- Check `fdf583fe31f74ddb` showed the 128-CTA candidates returning `-5` from
  the all-CTAs-resident safety guard at their target M.
- `submission_pooler_workspace64_v36b.py` and v36c reduced the grids to 64
  CTAs.  Checks `873d0187d0f74dc4` and `98fe3b13a9d54ce3` timed out inside
  preparation before wide candidate timing, consistent with a counter-wait
  residency/scheduling deadlock.
- v36c batched the exhaustive one-hot check at target M (still checking all
  8192 columns), so the second timeout was not caused by the old 8-row
  validation batching.

Do not score v34-v36c.  A future Pooler kernel must avoid both cluster-wide
DSMEM reduction and an in-kernel global counter wait.  The credible design is
a two-kernel partial/reduction path with PDL, or a no-split persistent kernel
that reuses weights across multiple M tiles.

### Tooling repair

`compile_extension_local.py --rh` had stale bindings for removed `rho32`,
`rhu32`, and `rhq128` functions.  The helper now binds only the functions
present in `RHCU`: `rhrun`, `rhrun32`, and `rhtail`.  Local SM90a router
compilation succeeds after this repair.

## 2026-10-28 resume result — authoritative current state

The structural-parity sequence is complete. Start future work from
`submission_qkv_o32_u16_u32_r32_v33.py`, not from the older v27/v28 state below.

- New scored best: **1.5153853742x / 3187.835967 us**
- 95% CI: **[1.4926546x, 1.5381162x]**
- Scored job: `617f2d50e27040c4`
- Leaderboard snapshot after submission: **rank 126 / 1219**, displayed speedup 1.5154x
- Passed: **9/9**, no flags or rejections
- v33 SHA-256: `D89BD4826822A014EC8E67CC5987AE133E55136567A543C9433EB0BDB8375936`
- Scored profile: `profiles/submission_qkv_o32_u16_u32_r32_v33.617f2d50e27040c4.json`
- Unscored v33 check: `6b5e17d9025149ca` (1.4917063x; same code hash, noisier run)

Validated fixed routes now integrated:

| Route | Scored corrected median | Validation |
|---|---:|---|
| QKV-128 descriptor TMA/WGMMA | 13.444 us | all 8192 one-hot columns + FP64 random/outlier |
| Direct O-32 | 5.811 us | all 1024 one-hot columns + FP64 random/outlier |
| Lossless C12 unpooler-16 | 13.244 us | lossless pack/unpack, exhaustive one-hot, random/outlier |
| Lossless C12 unpooler-32 | 15.401 us | all 4096 one-hot columns + FP64 random/outlier |
| Transposed router-32 `(32,16,4,1,2,1)` | 5.024 us | all 8192 split coefficients + FP64 random/outlier |

Milestone files and checks:

- `submission_qkv128_v29.py` — check `44fa1bfd90dc4008`
- `submission_qkv_o32_v30.py` — check `ebe89cfd73bf435c`
- `submission_qkv_o32_u16_v31.py` — check `714842b60c6445c8`, 1.4955192x
- `submission_qkv_o32_u16_u32_v32.py` — check `585c420f20e84632`; U32 improved by 0.852 us despite aggregate noise
- `submission_qkv_o32_u16_u32_r32_v33.py` — check `6b5e17d9025149ca`, scored job `617f2d50e27040c4`

Local compile support in `compile_extension_local.py` now includes `--q128`, `--o32`, `--u16`,
`--u32`, and `--r32`; use `--sm90a-compile` through `compile_extension_local.ps1`.

Recommended next work: preserve v33 as the scored fallback and use new candidate files for one-variable
tuning. The most credible remaining shape gaps are O-128, O-256, O-16, pooler-32, and QKV-16.
Run an unscored check after each isolated route and only spend another scored attempt when corrected
per-shape medians improve without a correctness or nearby-shape regression.

## Current state

- Best submission: `submission_integrated_v27.py`
- Best job: `203d55885baf4fed`
- Best speedup: **1.4842574538x**
- Best aggregate time: **3254.691 us**
- Best 95% CI: **[1.46199x, 1.50652x]**
- v27 SHA-256: `3FB0F20BB410EAEF43EB511E568DDD3621734468074022770925FA84E11C8B5C`
- Latest experimental submission: `submission_final_v28.py`
- Latest job: `06bae18d016c45bf`
- v28 passed all 9 correctness tests, but scored **1.4591682014x / 3310.653 us**.
- All 10 scored submissions and 20 checks are currently used. Limits reset at 00:00 UTC.

The leaderboard should retain v27 because OKC keeps the best scored result.

## What v28 proved

### Successful result

The transposed router-32 design was valid and faster:

- v27 router-32: **5.874 us**
- v28 router-32: **5.572 us**
- Nathan router-32: **5.184 us**

The `m64n32`, sixteen-way split-K direction is correct. It needs pipeline refinement, not another architectural restart.

### Failed result

All three v28 register-A dense kernels were rejected safely during preparation:

```text
register-A dense rejected: o coefficient mismatch at 0
register-A dense rejected: unpooler coefficient mismatch at 0
register-A dense rejected: qkv_gate coefficient mismatch at 0
```

The problem is the weight fragment layout. Our `_pack` function produces the ordering used by the existing `mma.sync` kernels. Register-A WGMMA needs Nathan's distinct lane/register ordering.

The failed kernels never reached scored dispatch. Correctness remained intact because they fell back to the v27 routes.

## Exact remaining gaps versus Nathan

The most useful corrected per-shape timings are:

| Shape | Nathan | v27 | Recoverable gap |
|---|---:|---:|---:|
| QKV-128 | 14.212 us | 15.428 us | **1.217 us** |
| Unpooler-16 | 13.414 us | 14.422 us | **1.008 us** |
| Unpooler-32 | 15.932 us | 16.934 us | **1.001 us** |
| Router-32 | 5.184 us | 5.874 us | **0.690 us** |
| O-32 | 5.847 us | 6.497 us | **0.649 us** |
| O-128 | 6.798 us | 7.409 us | **0.610 us** |
| O-256 | 8.657 us | 9.197 us | **0.540 us** |
| O-16 | 5.281 us | 5.802 us | **0.521 us** |

Nathan's complete submission scored approximately **1.5095x**. Matching only the main QKV/unpool/router routes should be sufficient to cross roughly 1.50x. Fine-tuning after structural parity gives a plausible target of **1.515–1.525x**.

## How our design currently differs from Nathan's

| Area | Our v27/v28 framework | Nathan's framework |
|---|---|---|
| Overall organization | Compact generic kernels, cuBLASLt, and runtime gates | Many kernels specialized for exact shapes |
| Small dense weights | Full BF16 with `mma.sync` packing | Separate register-A WGMMA packing |
| Unpooler | Full BF16 | Lossless C12 compression with escape restoration |
| QKV-128 | v28 attempted register-A plus three-way DSMEM reduction | Descriptor-to-descriptor TMA/WGMMA producer-consumer kernel |
| Router-32 | Simplified transposed `m64n32` TMA pipeline | Same broad geometry with a more refined pipeline and ownership scheme |
| Configuration choice | Prepare-time benchmarking can vary between runs | Mostly fixed H200 configurations validated offline |
| Safety | Strong generic fallback and validation gates | Fixed per-shape production routes |

The next implementation should preserve our cleaner dispatch and fallbacks while reproducing the successful data layouts and microarchitectures independently.

## Priority 1 — QKV-128

Do not continue the v28 register-A QKV experiment as the primary path. Reproduce Nathan's descriptor-based TMA/WGMMA design.

Nathan's selected configuration in `submission_nathen_v18.py` is:

```python
(64, 36, 128, 3, 8, 539, 5)
```

Source location: `submission_nathen_v18.py:4617`.

Interpreted structure:

- 64 output rows per group.
- 36 output groups for 2304 rows.
- 128 tokens per CTA group.
- Three-way split-K.
- Eight pipeline stages.
- Warp-specialized producer/consumer execution.
- TMA descriptors for weights and activations.
- WGMMA accumulators with one clustered reduction.

Start with structural parity, validate it, and then sweep:

- Pipeline stages 6–10.
- Split-K 2, 3 and 4.
- WGMMA wait depths 0, 1 and 2.
- Producer warpgroup placement.
- Weight `evict_first` and activation `evict_last` L2 policies.
- TMA issue distance and ring-buffer reuse point.

## Priority 2 — Correct register-A WGMMA packing

Nathan's default BF16 unit mapping is at `submission_nathen_v18.py:1066`:

```python
bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
v = bits.view(
    n // 16, 2, 8, k // 64, 4, 4, 2, 2
).permute(
    0, 3, 2, 4, 5, 6, 1, 7
)
units = v.reshape(n // 16, k // 64, 32, 32)
```

The step-major byte arrangement is:

```python
words = units.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4).contiguous()
```

This layout must replace the v28 `_rpack64` assumption for register-A WGMMA.

Validation requirements before timing:

1. Pack/unpack must preserve every BF16 bit.
2. Run every K column as one-hot input.
3. Require bit-exact output equal to `weight.T` for one-hot rows.
4. Run random and outlier inputs against the FP64 reference.
5. Only benchmark after all four checks pass.

## Priority 3 — Lossless C12 unpooler kernels

Nathan's key advantage for the memory-bound unpooler is lossless C12 compression, not just a different launch geometry.

The format stores:

- The low eight bits of each BF16 value directly.
- Exponent delta and sign in a compact nibble/bit table.
- Exact escape entries for values outside the selected exponent window.

This reduces a 16x64 BF16 fragment from 2048 bytes to 1536 bytes, a **25% reduction in weight traffic**, while reconstructing the original BF16 patterns exactly.

Relevant implementation begins at `submission_nathen_v18.py:1093`.

Integrate independently:

1. Unpooler-16 using Nathan's `m32x_round3_decoder9` / `d9_cap2` idea.
2. Unpooler-32 using the final `decodediet` route.
3. Maintain an uncompressed BF16 twin as fallback when an escape cap cannot represent a weight tile.

Structural innovation to test after parity:

- Select the exponent window per output group or per K tile instead of globally.
- Measure escape caps 1, 2, 4 and 6.
- Vectorize decode with `lop3`/`prmt` where possible.
- Decode the next fragment while the current WGMMA group is executing.
- Store escape metadata in the order each warp consumes it.

## Priority 4 — Direct O-32

Nathan's final V17 route identifies O-32 as `owner_variant='direct'` at `submission_nathen_v18.py:6525`.

Rebuild the v28 direct kernel using the corrected WGMMA fragment mapping. Then tune:

- One versus two activation stages.
- One versus two K partitions.
- Weight prefetch distance.
- WGMMA group depth.
- Direct output conversion versus shared-memory epilogue.

Expected recoverable gap: approximately **0.649 us**.

## Priority 5 — Finish router-32

Keep the v28 transposed architecture and match Nathan's selected configuration:

```python
(32, 16, 4, 1, 2, 1)
```

Relevant source location: `submission_nathen_v18.py:905`.

Investigate the remaining approximately 0.39 us gap between v28 and Nathan by comparing:

- When the next TMA stage is issued.
- Whether one WGMMA group remains outstanding.
- Named barrier use instead of full block synchronization.
- DSMEM destination ownership and reduction order.
- Register lifetime around hi/lo accumulation.
- Four-stage versus deeper rings.

## Secondary opportunities

Only pursue these after the first five routes are stable:

- O-128: Nathan's dedicated O-layer path, approximately 0.61 us gap.
- O-256: final custom configuration `(256, 64, 5, 7, 1, 1, 1)`, approximately 0.54 us gap.
- O-16: Nathan's `m16trace_early` route, approximately 0.52 us gap.
- Router-1/16: v28 already moved close to Nathan; lower priority.

## Structural innovations beyond Nathan

These are the most credible ways to exceed Nathan rather than merely match him.

### Tile-local adaptive C12

Choose the exponent window separately for each output/K tile. This can reduce escapes and decode work, particularly for unpooler weights. Packing remains lossless.

### Deterministic H200 route selection

Avoid selecting close candidates from noisy prepare-time measurements. Use checks to establish fixed winners and activate them by shape/device. Retain runtime gating only when a candidate wins by at least 3–4%.

### Persistent QKV scheduling

Nathan's QKV geometry launches 108 CTAs on an H200 with 132 SMs. Explore a persistent work queue or a mixed work assignment that uses more of the 132 SMs without creating a costly partial second wave. This is higher risk and should follow a correct reproduction of Nathan's 108-CTA kernel.

### Overlapped lossless decode

Use separate producer work to load and decode C12 fragments into registers while consumer warps execute WGMMA on the previous fragment. This may move decompression latency under tensor-core work.

### Shape-specific epilogues

Avoid generic shared-memory reductions when a shape permits direct BF16 stores or warp-local reduction. O-32 is the first candidate.

## Next-session execution order

1. Copy `submission_integrated_v27.py` to a new candidate. Do not start from v28.
2. Port only the QKV-128 descriptor kernel.
3. Compile locally for SM90a.
4. Run exhaustive one-hot and numerical validation on H200 through an OKC check.
5. Inspect the profile before adding another route.
6. Add corrected WGMMA packing and direct O-32.
7. Add C12 unpooler-16.
8. Add C12 unpooler-32.
9. Refine router-32.
10. Once structural parity is reached, run controlled one-variable tuning sweeps.

Use separate candidate files for each milestone so a regression can be isolated. Keep v27 unchanged as the permanent known-good fallback.

## Expected progression

- Structural QKV-128 integration: recover up to roughly 1.2 us on that shape.
- QKV-128 plus unpooler-16 and unpooler-32: projected aggregate near 1.50x.
- Add complete router-32 and direct O-32: projected approximately 1.504–1.512x.
- Successful structural innovations: plausible final range **1.515–1.525x**.

These projections are targets, not guarantees. Every route should be accepted only after exact correctness and graph-replay profile evidence.

## Important files

- `submission_integrated_v27.py` — known-good base.
- `submission_final_v28.py` — experiment showing the router improvement and incorrect register-A packing.
- `submission_nathen_v18.py` — MIT-licensed source for studying the successful structures.
- `NATHEN_FERNANDES_MIT_LICENSE.txt` — license notice.
- `profiles/submission_integrated_v27.203d55885baf4fed.json` — best profile.
- `profiles/submission_final_v28.06bae18d016c45bf.json` — latest profile and rejection evidence.
- `profiles/submission_nathen_v18.e3182690ad2446eb.json` — Nathan timing reference.
