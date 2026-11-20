# Decode-layer component integration research

Date: 2026-11-02

Scope: `muse_bf16_proj`, `muse_attention_decode`, and `muse_moe_w4a16` in the parent directory

Active decoder baseline: S052, job `6828c406e66940ed`, `2.213593835x`, `10218.812 us`, 13/13

Protected decoder SHA-256: `63AF668FBB4C49033BE0DB7113472518E639EE428C4B9D151CF02A67DA39C7C2`

This is an integration study, not a source promotion. The active `submission.py` was not changed.

## Executive conclusion

The component repositories do contain useful work that is not yet in S052, but a wholesale component-source replacement is not the right move. The decoder already contains most of the mature component mechanisms, and previous full-backend swaps lost because isolated kernel wins did not preserve PDL overlap, fused launches, and whole-chain scheduling.

The best remaining integration opportunities are:

1. **BF16 `m=1` exact-C12 GEMV routes** from the protected v105 projection entry. S052 has exact-C12 SK2, but it does not contain v105's dedicated `cgemv_kernel`/`rgemv_kernel` family. The current decode-1 dense rows leave the clearest low-token dense gap.
2. **BF16 RS128 for QKV at `T=128`** from v150. This is a new, repeated H200 win that post-dates the consolidated BF16 notes. Port QKV first. Treat pooler as a second experiment against the decoder's fused pooler+router plan, and do not assume that the standalone unpooler win transfers.
3. **Runtime route-skew-aware MoE scheduling.** MoE remains 38-63% of S052's profiled layer time. The same token count changes by tens of microseconds across routed-expert distributions, and fixed layer/T policies have already failed to generalize. The next large gain should choose/schedule work from the GPU-visible per-expert route counts, not only from total `T`.
4. **Selective BF16 v105 small-M retunes** and the missing MoE MID scale flags are cheap secondary checks, but their proven local gains are sub-microsecond to about 0.8%; they are not the main speedup.
5. **Attention changes are currently small or unvalidated.** A continuous q8 flat pipeline is mildly better than turning it off, but the measured whole-layer envelope is tiny. The repaired five-stage q8 candidate has no completed result and should not be imported as a winner.

## Why component timings cannot be copied directly

The decoder is a chained CUDA-graph workload, not the sum of five independent boards. In S052:

- projections can overlap through PDL;
- pooler and router can share one launch;
- the residual boundaries overlap or expose waits in neighboring profiler rows;
- component profiler rows may include time waiting on an earlier kernel;
- the MoE route distribution changes between official seeds even at the same `T`;
- full and sliding cases have different score weights (`16` versus `46`).

Therefore every number below is used either as structural evidence or as a first-order envelope. Promotion still requires an exact whole-decoder check.

## Active S052 bottlenecks

| Shape family | Attention | MoE | Main implication |
|---|---:|---:|---|
| decode-1 sliding/full | 4.8 / 10.9 us | 29.9 / 34.7 us | MoE and dense launch floor dominate |
| decode-16 sliding/full | 7.6 / 89.6 us | 58.5 / 71.1 us | MoE dominates sliding; full attention is also material |
| decode-32 sliding/full | 9.1 / 172.2 us | 89.2 / 80.2 us | route distribution changes MoE behavior at fixed `T` |
| verify-1 sliding/full (`T=8`) | 7.1 / 14.4 us | 57.0 / 73.1 us | small MoE and dense launch floor dominate |
| verify-16 sliding/full (`T=128`) | 11.8 / 102.6 us | 148.8 / 158.5 us | strongest BF16 RS128 target plus large MoE target |
| verify-32 sliding/full (`T=256`) | 15.0 / 187.1 us | 205.5 / 179.9 us | MoE variance is larger than many proposed micro-wins |

The important signal is the route-sensitive MoE spread. For example, S052 reports `205.5 us` versus `179.9 us` at the same `T=256`, and `148.8 us` versus `158.5 us` at the same `T=128`. S054's layer-aware M128 dispatch then regressed from `2.213594x` to `2.204859x`, proving that one fixed seed/layer label is not a robust scheduling key.

## BF16 projection findings

### What S052 already has

S052 is not using an old generic GEMM baseline. It already contains:

- lossless 12-bit BF16 packing with exact escape correction;
- SK2 swapped-operand tensor-core kernels for `T <= 32`;
- a one-launch fused pooler+router path;
- cuBLAS/cuBLASLt and custom-kernel tournaments;
- fixed whole-chain Lt picks at `T=128/256` where isolated tuning made the wrong choice;
- BF16 projection v131 L2 hints in the embedded `v86y4` lineage;
- QKV/O/pooler/unpooler/router shape-specific dispatch.

Previous decoder evidence matters:

- E030's wholesale standalone-v55 backend replacement passed but regressed 1.70% and increased preparation from about 56 to 188 seconds.
- S033b's exact standalone dense-v87 composition passed and improved its then-current control by only 0.63%, inside run variation and below the later S040/S052 line.
- forcing separate BFP pooler and router at large `T` lost roughly 10-12 us in end-to-end composition because it bypassed the fused Lt plan.

The correct approach is to add individual routes behind the existing decoder composition, not replace the embedded BF16 subsystem.

### Candidate BF16-1: v105 `m=1` C12 GEMV family

This is the clearest missing mature mechanism.

The protected BF16 v105 source contains dedicated row-major exact-C12 GEMV kernels for `m=1`:

| Projection | Standalone v105 kernel profile | S052 decode-1 row/envelope | Observed gap |
|---|---:|---:|---:|
| QKV | about 7.3 us | about 9.7-10.2 us | about 2.4-2.9 us |
| O | about 4.1 us | about 4.7-4.8 us | about 0.6-0.7 us |
| Pooler | about 13.2 us | part of 18.2 us fused pooler+router | must compare as a chain |
| Unpooler | about 2.7 us | about 6.3-6.5 us | about 3.6-3.8 us |
| Router | about 2.5 us | fused or `_gle_route` about 2.5-2.7 us | no independent gap |

The row mapping is inferred from dimensions and kernel order. Some S052 rows include PDL wait time, so these are not additive proof. They are strong enough to justify an isolated port.

Recommended integration:

- copy the v105 `cgemv_kernel`, C12 row-major packer, escape tables, and launcher as a separate guarded route;
- prepare the extra packed representation once from weights only;
- first enable only QKV and unpooler at `T=1`;
- retain the current fused pooler+router path as the control;
- then test a second variant that uses cgemv pooler plus the existing exact router, because two launches may erase the apparent 2-3 us projection saving;
- preserve the current PDL boundary and time the entire decode-1 chain, not isolated calls.

First-order score envelope from S052 profiler totals:

- if QKV+unpooler really remove `6 us` from both decode-1 cases: about `2.2421x` (`+1.29%`);
- if all dense changes remove `8.5 us`: about `2.2548x` (`+1.86%`).

These are optimistic envelopes, not expected check results. A useful promotion gate is at least `3 us` end-to-end on both decode-1 cases with no regression elsewhere.

### Candidate BF16-2: v150 faithful RS128, QKV first

`submission_rs128_faithful_v150.py` is newer than the consolidated BF16 architecture note. It preserves v105 and adds an Apache-2.0 RS128 producer/consumer kernel ported from `chenwainuo/okc` commit `3d418e1`. Preserve that provenance and license if it is copied.

Source SHA-256: `E405D6464EF5D64FDC896E300DD0F5AC9CD772D4E65F4EFF7FF2338A2D0F7258`.

Two H200 jobs passed 9/9 and enabled all three intended `m=128` routes:

| Job | Score | Speedup | Enabled route-tuner result |
|---|---:|---:|---|
| `ad856299d0994a65` | 3047.079 us | 1.585387x | QKV 0.9005x, pooler 0.8435x, unpooler 0.9281x core ratio |
| `323a268f980745ff` | 3081.224 us | 1.567819x | QKV 0.9048x, pooler 0.8250x, unpooler 0.8972x core ratio |

The executed profile rows were approximately:

- QKV: `11.9-12.3 us`, versus the older v105 profile around `13.2 us` and S052 around `16.4-16.6 us`;
- pooler: `16.7-16.8 us`, versus older v105 around `20.8 us`;
- unpooler: `17.2-18.0 us`, versus older v105 around `19.2 us`.

Port order for the decoder:

1. **QKV-only RS128.** This is the cleanest comparison against S052's `16.4-16.6 us` QKV Lt kernel. A `4.4 us` whole-chain saving on both `T=128` cases projects to about `2.2198x` (`+0.28%`).
2. **QKV plus alternate pooler/router plan.** S052's fused `pr` Lt row is about `24.8 us`; RS128 pooler plus the current wide router is plausibly competitive, but the extra launch and PDL interaction must be measured. If total saving reaches `7.4 us`, the first-order envelope is about `2.2241x` (`+0.48%`).
3. **Do not enable RS128 unpooler by default.** S052's two equal large projection calls total about `29.4 us`, or roughly `14.7 us` each. If one is the unpooler, that is already faster than v150's `17.2-18.0 us`. Keep it only as a prepare-time/whole-chain candidate.

Resource and integration cautions:

- v150 reports 168 registers, zero local bytes, and 144,520 bytes dynamic shared memory;
- it needs its own exact packing/escape representation;
- it increases build and preparation work;
- the original route tuner subtracts a replay floor and requires a 3% per-route win plus a 0.9% projected global gate;
- the decoder must retain its current fixed Lt control, because the component incumbent is different.

### Candidate BF16-3: selected v105 SK2 retunes

S052's embedded SK2 table predates five small route retunes present in v105:

| Route | S052 pick | v105 pick | Component comment |
|---|---|---|---|
| O, `T=16` | `o8h_r88` | `o16t_r68` | about 0.15 us local |
| pooler, `T=16` | older `pooler` | `pooler16t_r192` | about 0.43 us local |
| unpooler, `T=16` | `unpooler16h_r100` | `unpooler16t_r72` | about 0.60 us local |
| QKV, `T=8` | `qkv8_r176` | `qkv8t_r144` | about 0.12 us local |
| QKV, `T=16` | `qkv16_r176` | `qkv16t_r200` | about 0.08 us local |

These are easy source-level ports, but their isolated total is only around 1-1.5 us and pooler-16 participates in the decoder's fused pooler+router kernel. Treat them as one guarded low-risk candidate after the two BF16 experiments above, not as a rank-moving design.

### BF16 directions not to repeat

- wholesale current-component replacement;
- generic mid-M TMA/WGMMA migration (measured 1.36-2.40x slower than SK2);
- wide compressed unpooler variants;
- more generic cuBLASLt plan sweeps without a new mechanism;
- pooler DSMEM/multicast/workspace reductions;
- dual-C12 router and wider custom router paths;
- forced near-tie routes that bypass whole-layer composition.

## Attention findings

### What S052 already has

S052 includes the fused component architecture: QK normalization, RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head norm, and gate. Its embedded source is a stripped production fork, not a byte-identical copy of the current standalone file.

Relevant compile-state difference:

- standalone scored lineage: continuous pipeline enabled for q1 and q8 flat paths;
- S052: `OKC_CONT=0`, with decoder-specific `QDIRECT=0`, `PK2=0`, and `W0=1` choices.

The standalone profile is often 1-8 us lower on full cases than the S052 attention row, but this is cross-run evidence from different compile flags and integration context. It is not a safe full-source-swap claim.

### Candidate ATT-1: isolated continuous flat-pipeline build

The clean first check is one decoder candidate that changes only continuous mode while preserving S052's API, W0, QDIRECT, PK2, PDL, and all non-flat dispatch.

Direct standalone evidence is modest:

- the experiment that disabled continuous mode for q8 regressed the full q8 profiler by about `0.660 us` at B16 and `1.093 us` at B32;
- q1 results were mixed and small;
- applying only those two recovered deltas to S052 projects about `2.2140x`, just `+0.018%`.

This is a cheap correctness/performance check, but it should not consume a scarce scored attempt unless the whole-decoder check shows a larger interaction.

### Candidate ATT-2: selective standalone lineage transplant

A larger but riskier experiment is to transplant only the current flat full-context attention specialization into S052 while retaining decoder launchers and fallbacks. The difference between standalone and decoder profile rows gives an optimistic cross-run envelope of about `+0.63%` whole-layer (`~2.2275x`) if all observed attention deltas transferred. That number is deliberately an upper bound; source similarity is only about 0.85 and there are hundreds of differing blocks.

Required gates:

- exact FP8 KV/cache and norm/gate parity on all 13 decoder cases;
- no change to sliding paths in the first candidate;
- same dynamic shared-memory ceiling and no resource regression;
- matched-seed full-layer profile, not a comparison of independent component jobs.

### Pending, not validated: repaired q8 five-stage ring

The repaired `submission_q8_st5_vt1.py` preserves the proven two-CTA topology and expands the flat q8 K/V ring from four to five stages within the same 0x1c000 dynamic shared-memory budget. Its final audited source had no completed job result: the submit command timed out without a job ID or payload during an API outage.

Do not call this a component winner or import it into the decoder yet. First recover a standalone H200 verdict. The original pre-repair five-stage candidate was rejected, so only the final repaired hash is relevant.

### Attention directions not to repeat

- q_len-specialized CONT1/CONT8 hybrid that turns q8 continuous mode off;
- page-1 producer specialization (regressed q8 B16/B32 by about 1.2/3.1 us);
- dynamic chunk work stealing that caused 59-66% q8 regressions;
- denominator-free q8 mode;
- one-CTA/two-producer F2, which failed synchronization correctness;
- a blind full-file replacement that also changes W0/QDIRECT/PK2 and launch glue.

## MoE findings

### What S052 already has

S052 already contains more than the old component-board winner:

- tiny, tiny2, MID, B4, and large-B4 compile-time builds;
- lossless schedule-specific NVFP4 repacking and the tuned fused GEMM1/SwiGLU/GEMM2/post-norm kernel;
- KPERM3-style stage-wide layout;
- fused/optimized routing and activation gather choices;
- compact norm partials for `M=32/128/256`;
- FINPF on the tiny path (added later in the decoder even though the base define tuple initially lacks it);
- the stronger S052 large-B4 finalizer.

Recent component investigations strongly narrow the search:

- removing WGMMA was neutral and raw-dequant ablations implied only a 1.2-1.5% arithmetic upper bound;
- x-gather readiness was only about 3-6 us while producer empty-ring waits were much larger;
- whole-unit cyclic scheduling lost 7.6-9.5%;
- split-fixup alternatives lost every paired seed;
- local KPERM3 v139 was statically worse in S062 (same 96 registers but more stack, spills, and IMADs);
- split finalizers, full-row finalizers, packed route IDs, extra rings, and MID FT32 have already failed in the decoder.

The remaining problem is scheduling balance across the actual expert-load histogram.

### Candidate MOE-1: GPU route-skew-aware hybrid scheduling

This is the highest-upside research direction.

The route prologue already computes per-local-expert counts on GPU. Use those counts to choose work granularity or execution family for each expert rather than selecting one kernel solely from total `T`.

Two viable implementation shapes are:

1. **One persistent kernel with two device work queues.** Build light/skinny and heavy/dense expert jobs from the route counts, then let CTAs claim appropriately sized units. Preserve the current fine-grained cost-weighted fractional partition inside each queue.
2. **Two fixed graph launches with GPU-side guards and disjoint expert ownership.** One launch handles experts below a count threshold and one handles heavy experts. Both launches are always captured, so no input-dependent host branch is introduced. This is simpler but pays an extra launch/barrier and needs exact shared-scratch ownership.

The threshold should be based on modeled work, not just count. Useful features already available in the prologue are:

- local route count per expert;
- number of full 16-row/256-row units;
- split-edge count;
- phase-A versus phase-B predicted cost;
- maximum and variance of expert counts.

Why this is more credible than another static schedule:

- standalone history shows CPB2-like geometry can win at M128 while NCW16-like geometry wins at M256;
- S052 shows that even the same `T` changes materially with routing;
- S054's fixed sliding/full policy did not generalize to another official seed;
- v153 proved that coarser whole-unit scheduling reduces spills but loses badly through imbalance.

The design must retain fine-grained fractional ownership for split tails. The new mechanism is the route-aware partition into scheduling families, not cyclic whole-unit assignment.

Performance gate:

- target at least 5% MoE reduction across multiple routing seeds before integration;
- reject if any high-weight sliding case regresses by more than 1%;
- a 5% reduction across all S052 MoE rows projects to roughly `2.269-2.271x`;
- a 10% reduction projects to roughly `2.331x`;
- a 20% reduction only on the large `T=128/256` MoE rows projects to about `2.303x`.

This is the only component-derived direction with credible several-percent whole-layer upside.

### Candidate MOE-2: MID scale broadcast flags

The current standalone MID build enables:

- `OKC_SKSC2=1`;
- `OKC_FTSC2=1`.

Its source comment reports about `+0.8%` at batch 16. S052's MID tuple has FINPF but not these two flags. This is a small, simple, isolated `T=16` candidate. The likely saving is around half a microsecond on the MoE row and is not sufficient by itself for a scored promotion.

FINPF is not a missing decoder port: the active file later appends `-DOKC_FINPF=1` to the tiny build, and tiny2/MID already contain it.

### MoE directions not to repeat

- v139 local 4x4 K permutation in place of S052's stage-wide layout;
- joined weight/activation completion and dual WGMMA accumulator banks;
- FT32 at M32;
- cyclic whole-unit scheduling;
- second split-fixup buffers;
- more TMA rings, readiness polling, or activation-gather work as the primary lever;
- full-row or split external finalizers;
- packed route IDs/direct global route reloads;
- multicast/codebook lookup/dequant microcode rewrites;
- layer-name-based large-B4 dispatch from one routed seed.

## First-order score envelopes

These projections use S052 profiler totals and official shape weights. They answer "is this direction large enough?"; they do not predict the next official score.

| Hypothetical realized change | Projected speedup | Relative change |
|---|---:|---:|
| attention q8 continuous recovery only (`0.66/1.093 us`) | 2.2140x | +0.018% |
| RS128 QKV, `-4.4 us` on both T128 cases | 2.2198x | +0.281% |
| RS128 QKV plus competitive pool/router, `-7.4 us` | 2.2241x | +0.476% |
| all cross-profile attention deltas transfer (optimistic ceiling) | 2.2275x | +0.629% |
| v105 m1 QKV+unpooler, `-6 us` on decode-1 | 2.2421x | +1.289% |
| full m1 dense envelope, `-8.5 us` on decode-1 | 2.2548x | +1.863% |
| m1 full envelope plus RS128 QKV | 2.2612x | +2.149% |
| all MoE rows `-5%` | about 2.270x | about +2.5% |
| all MoE rows `-10%` | about 2.331x | about +5.3% |
| m1 full + RS128 QKV + all MoE `-5%` | 2.3189x | +4.758% |

The table explains the recommended effort split: attention flags and MoE micro-flags are useful polish; m1 GEMV can produce a real near-term step; routing-aware MoE is needed for a large jump.

## Ranked integration plan

### Experiment 1: BF16 m1 C12 GEMV graft

- Create a new candidate from exact S052; do not edit active `submission.py`.
- Add only the v105 cgemv source, packer, and launch bridge.
- Candidate A: QKV+unpooler only.
- Candidate B: add O.
- Candidate C: compare fused pooler+router against cgemv pooler + existing router.
- Use prepare-time validation for pack recovery and exact one-hot probes.
- Profile both decode-1 layer types and verify that every other kernel/source hash is unchanged.
- Promote only if the end-to-end saving is repeatable, not merely visible in isolated GPU rows.

### Experiment 2: BF16 v150 RS128 QKV128 graft

- Preserve v150 attribution and exact packer.
- Build only the QKV specialization first to control compile/preparation cost.
- Compare against S052's fixed QKV128 Lt plan in the actual projection->attention chain.
- If it wins, add a separate pooler experiment while retaining the fused `pr` plan as control.
- Do not enable unpooler unless the decoder-level tuner proves it faster.

### Experiment 3: route-skew MoE diagnostic

- Instrument or offline-model the existing GPU route counts for several official-style seeds.
- Predict current cost-weighted fractional partition tails per expert.
- Define one heavy/light threshold policy and audit exact ownership exhaustively.
- First compare scheduling only; keep math, packing, finalizer, and K layout byte-identical.
- Require multi-seed paired evidence before a whole-decoder port.

### Experiment 4: low-cost component deltas

- BF16 v105 SK2 picks for QKV8/QKV16/O16/pooler16/unpooler16.
- MoE MID `SKSC2+FTSC2` flags.
- Attention continuous flat pipeline with all decoder-specific flags preserved.

These can be grouped only after individual static/resource checks. Do not combine unrelated unmeasured deltas into one scored candidate.

### Experiment 5: attention five-stage candidate only after standalone verdict

- Recover or rerun the final repaired q8-five-stage standalone candidate.
- If and only if it wins both q8 B16 and B32 full cases, port the exact flat specialization into an S052-derived candidate.

## Promotion checklist

- Active S052 hash remains unchanged before and after candidate generation.
- Candidate is a separate file with one mechanism per check.
- Python syntax and exact SM90a compilation pass.
- Weight transforms read weights only during `prepare()`; no activation-derived tuning state.
- No host synchronization or input-dependent host dispatch enters graph replay.
- Exact route ownership and scratch non-aliasing are audited.
- All 13 correctness cases pass.
- Profile uses matched-seed/paired evidence where routing affects MoE.
- Whole-layer improvement clears noise and does not trade away high-weight sliding cases.
- Only then consider copying the candidate to `submission.py`.

## Integration results (2026-11-02)

The two highest-priority BF16 mechanisms were implemented as isolated,
self-contained S052-derived candidates and then composed.  All generated
sources are hash-pinned to the protected decoder/component inputs.

### S073: exact-C12 GEMV at T=1

- Candidate: `submission_s073_cgemv_m1.py`.
- Final check: `117bb37666e34562`, 13/13 correctness, no flags.
- Both routes validated and enabled: QKV relative error `1.62e-4`, unpooler
  exact in the validation sample.
- Profile: QKV `8.8 us` and sliced unpooler `3.5 us`, versus approximately
  `10.4 us` and `6.2-6.4 us` for S052's corresponding kernels.
- Whole-layer check: `10262.18 us`; useful kernel-level gain, but not enough
  aggregate evidence by itself to promote.

### S074: faithful RS128 QKV

- Candidate: `submission_s074_rs128_qkv.py`.
- Final live-dispatch check: `e32d6f7b463b4d9f`, 13/13 correctness, no flags.
- Exact unpack/one-hot checks passed; FP64 validation relative error was
  `3.83e-3`; resources were 168 registers, zero local bytes, and 144520 shared
  bytes.
- After covering the later S052 cuBLASLt wrapper, verify-16 profiles contained
  `rq<128,...>` at `11.7-11.9 us`, replacing the roughly `17 us` QKV kernel.
- Whole-layer check: `10278.18 us`; again, the target kernel won but the noisy
  aggregate did not beat the protected incumbent.

### S075: composed S073 + S074

- Candidate: `submission_s075_cgemv_rs128.py`; repeat artifact differs only by
  a trailing comment.
- Checks `abf3d1b56f004dc7` and `631a4f0106ec448a` both passed 13/13 with all
  three routes enabled and no flags.  Prepare time was 157.3 and 165.4 seconds,
  below the 300-second limit.
- Scores were `10318.89 us` and `10285.04 us` (mean `10301.96 us`).
- Same-day untouched-S052 control `a70938130b9e441a` passed 13/13 at
  `10241.84 us`; the older S052 result was `10218.81 us`.
- Decision: do not promote.  The component kernels are real local wins, but
  two combined whole-layer samples missed the control.  `submission.py` was
  restored to protected S052 hash
  `63AF668FBB4C49033BE0DB7113472518E639EE428C4B9D151CF02A67DA39C7C2`.

This result narrows the next useful work: the standalone kernels do not need
more tuning.  Any future attempt should address their interaction with the
full layer (especially PDL/launch scheduling and MoE variance) and must beat
S052 end-to-end before promotion.

### S076: complete RS128 pooler/unpooler transfer

- Candidate: `submission_s076_rs128_pool_unpool.py`, generated by
  `scripts/build_s076_rs128_pool_unpool.py` from hash-pinned S075.
- Check `944117f7364f43e8` passed all 13 correctness cases with no flags.
  Prepare took 163.5 seconds.  Exact reconstruction and one-hot checks passed;
  pooler/unpooler FP64 validation errors were `3.170e-3` and `2.445e-3`.
- Both new routes activated.  The verify-16 profile contained three live
  `rq<128,...>` calls (QKV, pooler, unpooler) totalling `49.4 us`, plus the
  existing exact-FP32 router at `9.7 us`.
- Despite the faster projection kernels, the whole-layer result was only
  `2.190488x / 10326.60 us`, versus the same-day S052 control's
  `2.208616x / 10241.84 us`.  Do not promote; active `submission.py` remains
  exact S052.

Post-check audit corrected an earlier profile interpretation.  A profiler row
with `calls=2` already reports the aggregate time for both calls; multiplying
that row again manufactured a nonexistent approximately `29 us` overlap.
S052 also has `PDL_MAX_T = 32`, so PDL is disabled on verify-16 (`T=128`).  The
same-day S052 control's rows sum exactly to `250.8 us`; there is no missing GPU
busy interval to recover at this shape.

The direct transfer did produce a small real local win.  On sliding verify-16,
S052's dense rows total `70.3 us`, while S076's three RS128 calls, separate
exact router, and O projection total `67.9 us` (a `2.4 us` reduction).  On the
full case those totals are `70.8 us` and `66.1 us` (a `4.7 us` reduction).
However, the official paired timing moved from `248.03` to `249.39 us` on the
high-weight sliding case and from `334.05` to `329.99 us` on full.  Applying
the official 46:16 weights gives `270.23` versus `270.19 us`: only a `0.04 us`
net verify-16 improvement, far inside run variance.  Unchanged T=16/T=32
cases then supplied most of the worse whole-problem score.

So the integration did not fail correctness or dispatch.  Its useful gain was
too localized, and splitting the old pooler/router path into RS128 pooler plus
a separate approximately `9.7 us` exact router consumed much of the component
advantage.  The next structural target remains pooler/router/route fusion.
PDL at T=128 is still worth one controlled experiment, but it is a new regime,
not restoration of overlap that S052 previously had.

### S077: experimental T=128 PDL chain

- Candidate: `submission_s077_rs128_pdl.py`, generated by
  `scripts/build_s077_rs128_pdl.py` from hash-pinned S076.
- The RS128 kernel issues two immutable weight stages before
  `griddepcontrol.wait`, executes the wait on every thread before reading X,
  calls `griddepcontrol.launch_dependents` before its output epilogue, and is
  launched with programmatic stream serialization.
- Prepare-time validation keeps PDL off.  Runtime RS128 calls enable it, and
  `PDL_MAX_T` is raised from 32 to 128 so the surrounding producer/consumer
  kernels participate consistently.
- Python/source invariants pass.  The exact selected RS128 device template and
  a minimal CUDA PSS launcher both compile for `sm_90a` with CUDA 12.4.
- With the check quota exhausted, the user authorized scored job
  `93ba69588c9244b8`.  It passed all 13 cases with no flags; all three RS128
  routes validated and activated.  Score was `10252.76 us / 2.206264x`, CI
  `[2.173170, 2.239358]`, versus same-day S052 control `10241.84 us /
  2.208616x`.  Do not promote the aggregate result.
- The target mechanism itself worked.  Relative to S076, official verify-16
  timing improved from `249.39` to `242.32 us` sliding and from `329.99` to
  `326.59 us` full.  With 46:16 weights, that is `270.19` to `264.07 us`, a
  `6.12 us` (`2.27%`) T=128 gain.  The profile now reports a PDL/stream-overlap
  interval (`256.3 us` summed kernel time versus `253.1 us` GPU busy).
- The new T=128 path was not the source of the whole-problem miss.  The first
  remaining hypothesis was interaction with the inherited S073 T=1 package;
  the next controlled candidate therefore removed that appendix while keeping
  S077's PDL chain unchanged.

### S078: controlled removal of S073

- Candidate: `submission_s078_rs128_pdl_no_cgemv.py`, generated by
  `scripts/build_s078_rs128_pdl_no_cgemv.py` from hash-pinned S077.  The builder
  physically removes the complete S073 appendix; every other S077 byte is
  preserved apart from the identifying footer.
- No unscored checks were available after the quota reset, so scored job
  `675f204641404941` was used.  It passed all 13 cases with no flags and all
  three RS128 plans validated.  The result was `10338.24 us / 2.188022x`, CI
  `[2.155201, 2.220842]`; reject and keep S052 active.
- The controlled timings disprove the claim that S073 caused S077's aggregate
  miss.  Weighted decode-1 was effectively unchanged (`70.58` to `70.51 us`).
  The unchanged T=128 mechanism also remained healthy: verify-16 moved from
  `264.07` to `263.39 us` under the official 46:16 weights.
- The approximately `85.48 us` aggregate-score regression therefore cannot be
  assigned to the deleted route.  It is dominated by cross-run normalization
  and unchanged-load variance.  S073 remains a local kernel win but is neutral
  at whole-layer resolution.
- The next structural isolation is QKV-only RS128/PDL at T=128, restoring
  S052's fused pooler/router and original unpooler.  This tests whether the
  separate router and the additional RS128 calls consume the useful overlap.

### S079: QKV-only RS128/PDL

- Candidate: `submission_s079_qkv_rs128_pdl.py`, generated by
  `scripts/build_s079_qkv_rs128_pdl.py` from hash-pinned S077.  It removes both
  the S073 and S076 appendices, retaining the QKV RS128/PDL implementation and
  restoring S052's pooler/router and unpooler routes.
- Scored job `00f19df8d24b460e` passed all 13 cases with no flags.  QKV exact
  reconstruction, one-hot validation, and FP64 validation passed.  The result
  was `10274.59 us / 2.201577x`, CI `[2.168553, 2.234600]`; reject.
- Verify-16 was `242.70 us` sliding and `332.67 us` full, or `265.92 us` with
  the official 46:16 weights.  This is `1.85 us` slower than S077's `264.07 us`
  but still `4.31 us` faster than the same-day S052 control's `270.23 us`.
- Therefore the RS128 pooler/unpooler calls add a small useful gain inside the
  T=128 PDL chain; the separate router did not erase it.  Neither appendix
  removal produces a better aggregate, so no candidate is promoted.  Further
  scored attempts should wait for the higher-impact pooler/router/route fusion
  rather than spending quota on more packaging combinations.

### S080: guarded combined T=128 pooler/router RS128 projection

- Candidate: `submission_s080_rs128_pr_fused.py`, generated by
  `scripts/build_s080_rs128_pr_fused.py` from hash-pinned S077.  Candidate
  SHA-256 is
  `9201d106ab2a6be10b0ee9fbade8b3d78fef0495910543809fce2f680585be93`.
- It reuses the decoder's existing contiguous BF16 `state["pr"]` matrix:
  4096 pooler rows followed by the router's 256 BF16-high and 256 BF16-low
  rows.  One KS=4 RS128/PDL launch therefore covers all 4608 outputs at T=128,
  replacing S077's 4096-row RS128 pooler launch plus separate router launch.
- The proven S077 CUDA payload is byte-identical (decoded CUDA SHA-256
  `40d0e06012c434fe7ac70d1516715c429bf31fb8d40a2481a5b867a5a7c1bd15`).
  The only pre-existing-source changes are three combined-route dtype guards,
  allowing the route kernels to consume either FP32 or BF16 combined output.
- Precision is guarded in `prepare()`.  The path requires bit-exact packed
  weight reconstruction, the existing one-hot/counter/FP64 RS128 validation,
  router relative error at most `6e-3`, and exact ordered top-8 agreement for
  every row of three deterministic 128-token probes.  Any failure leaves the
  complete S077 pooler plus exact-FP32 router path active.
- Local static validation passed: Python syntax, one runtime dispatch, all
  three route guards, and byte-identical copied CUDA.  H200-only packing,
  numerical, and route checks necessarily run on the judge.
- Scored job `b63134e6813a4519` was submitted with content hash
  `4048757d0d157f27645c970bba4c7a69ef55c905a628cb3477636c4fc413a2d6`.
  At the 2026-11-03 checkpoint it was still `running` on cluster `nyc3b`, with
  no result yet.  Do not submit another candidate or promote S080 until this
  job finishes and its log proves whether `s080_pr_enabled` was true.

### Resume checkpoint (2026-11-03)

- Protected active `submission.py` remains exact S052, SHA-256
  `63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.
- Decoder quota after starting S080: one scored run left, zero checks left for
  the current UTC day.  A new UTC day should reset the quota, but query it
  before use.
- First command on resume:
  `../bin/okc status b63134e6813a4519 --json` from the repository in WSL.
- If S080 passes, inspect
  `profiles/submission_s080_rs128_pr_fused.b63134e6813a4519.{log,txt,json}`.
  Confirm the `s080 combined pr ... enabled True` line, compare verify-16
  sliding/full and the RS128/router rows against S077, then decide promotion.
- If the guard reports `enabled False`, the score is a fallback S077 rerun and
  should not be interpreted as evidence about the fused launch.  Use the probe
  diagnostics to design the next precision-preserving version.

## Source evidence used

- Decoder: `submission.py`, `EXPERIMENTS.md`, S052 profile/job `6828c406e66940ed`.
- BF16: `FAILED_OPTIMIZATIONS_AND_CURRENT_ARCHITECTURE.md`, protected v105, v147-v150 profiles/logs, and v150 generator/source.
- Attention: `RESUME_NOTES_2026-11-01.md`, scored incumbent profile `ab5f2ee782e445c3`, and the repaired q8-five-stage notes.
- MoE: `OPTIMIZATION_CHECKPOINT.md`, v101 profile `db4b3ef0899740a7`, v139-v156 diagnostics, and decoder S053-S072 results.
