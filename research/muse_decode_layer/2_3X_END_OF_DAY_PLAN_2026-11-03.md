# Muse decode layer: evidence-based plan to 2.3x

Date: 2026-11-03, 23:20 IST

## Executive decision

Protect the exact S081 incumbent in `submission.py`:

- scored job: `725c19cbeb9b4d48`
- result: **2.2611966426x / 10.003685 ms**, 13/13, no flags
- SHA-256: `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`

The 2.3x threshold is **9.834913 ms**, so the exact remaining gap is
**168.772 score-us**, or **1.6871%** of current latency. A single known
component change cannot close it. The only evidence-supported route is a
portfolio:

1. retain the exact-C12 M=1 QKV/unpooler CGEMV win;
2. remove small-token route and final-boundary launch cost through producer
   epilogues using last-arrival finalizers, with no grid-wide spin or barrier;
3. carry the measured M=16 MoE sliding win only as part of that larger bundle;
4. validate BF16 v177 as optional margin, not as the main mechanism;
5. score only after two behavior-equivalent checks predict at least **2.315x**,
   with a preferred center at or below **9.745 ms** (about 2.321x).

The minimum modeled bundle can cross 2.3x, but only the full bundle has enough
margin for the observed roughly 0.6% run-to-run noise. This is a conclusive
execution plan, not a guarantee: the decisive unknown is whether the two
small-token finalizers save at least about 2.2 us per affected load without
slowing their producer kernels.

Because the current time is already 23:20 IST, literal midnight is not a
credible implementation-and-validation deadline. The operational deadline
used below is the current quota day ending at 05:30 IST.

## Exact target math

The benchmark constant implied by the incumbent is 22,620.3 score-us.

| State | Score latency (us) | Speedup | Remaining gap to 2.3x |
|---|---:|---:|---:|
| Protected S081 | 10,003.685 | 2.261197x | 168.772 us |
| Exact 2.3x threshold | 9,834.913 | 2.300000x | 0 |
| Safe submission center | 9,745 or lower | 2.32122x or higher | 89.9 us margin |

The first-order score leverage of one raw microsecond saved in one shape is:

| Shape | Sliding leverage | Full leverage | Both branches |
|---|---:|---:|---:|
| decode 1 | 18.14 | 6.31 | 24.45 score-us |
| decode 16 | 8.41 | 2.93 | 11.34 score-us |
| decode 32 | 6.93 | 2.41 | 9.34 score-us |
| verify 1 | 11.20 | 3.89 | 15.09 score-us |
| verify 16 | 4.66 | 1.62 | 6.28 score-us |
| verify 32 | 3.53 | 1.23 | 4.76 score-us |

This makes decode-1 sliding the dominant target: one microsecond there is
worth more than five microseconds in verify-32 full.

## What is already bankable

### Exact-C12 M=1 CGEMV

S090 is numerically exact and has reproduced the intended local mechanism:
QKV and unpooler fall to about 9.0 us and 3.4 us. The direct gain is about
2.4 us in both decode-1 branches, worth roughly **59.6 score-us**. Its scored
aggregate was only 2.254826x because unrelated route/MoE timings moved against
it; this is not evidence that the target kernels failed. Do not score C12 by
itself again. Include it only in the final structural bundle.

Modeled alone, C12 moves 10,003.685 -> about **9,944.117 us**, or
**2.27474x**. It closes only 35% of the required gap.

### M=16 MoE dispatch

S092 reduced decode-16 sliding MoE from about 73.0 to 68.3 us while leaving
the full branch effectively flat (63.3 versus 63.4 us). That local 4.7 us
signal is worth about **39.5 score-us**, but it did not survive whole-layer
variance and must not be treated as an independent win. Carry it only after
the structural candidate is already close to target, and remove it if a
same-seed comparison does not reproduce the touched row.

### BF16 v177

The corrected v177 K128 GQX source now passes static CUDA 12.4 SM90a checks:
zero spills, no C7515 dependency barrier between its eight HGMMAs, and all
launch shapes fit H200 shared memory. It still needs an H200 component check.
Expected decoder value is only about **15--30 score-us**; it is margin, not a
path to 2.3x.

## The primary structural bet

The remaining launch-floor kernels are route (roughly 1.8--3.7 us) and the
final residual/norm boundary (roughly 1.8--2.2 us at the smallest loads and
larger at T=16/32). The plan is to eliminate their separate graph nodes for
T in `{1, 8, 16, 32}` without changing attention or MoE topology.

### A. Pooler/router last-arrival route finalizer

Extend the existing fused pooler/router `sk2f` completion machinery. Every
producer CTA performs its normal stores, then release-increments a generation-
tagged completion counter. Only the last arriver performs an acquire fence and
runs the route work; all other CTAs exit. There must be no polling, grid-wide
barrier, cluster-wide barrier, or assumption about CTA scheduling.

Implementation requirements:

- reuse the existing release/acquire pattern around `rcnt`, not a new spinning
  protocol;
- use double-buffered or generation-tagged counters so CUDA graph replay cannot
  observe a stale completion;
- preserve exact ordered top-8 behavior, local-ID mapping, normalization, and
  BF16/FP32 rounding points;
- first support T=1, then T=8, and extend to 16/32 only after the last-arriver
  workload remains bounded;
- emit a diagnostic marker and counter so the remote log proves that the fused
  path actually executed.

The main risk is that current routing is parallel across multiple programs,
whereas a last CTA may have insufficient parallelism. Kill this branch if the
producer grows by more than 0.4 us or if route-equivalent work takes as long as
the separate route launch. Do not try to rescue it with a grid barrier.

### B. Unpooler last-arrival boundary-2 finalizer

Extend the small-M unpooler epilogue so each CTA stores its ordinary output and
any norm partials it already owns, then release-increments a per-generation
completion counter. The last arriver acquires the completed row data and runs
the exact boundary-2 residual/norm computation. The separate
`_sl_cluster_finish`/boundary-2 node is omitted only on this guarded path.

Implementation requirements:

- preserve the reference reduction order and the incumbent FP32/BF16 rounding
  points; accuracy is checked on every supported T;
- pre-load only dependency-free residual/gate vectors before the PDL wait;
- do not hold additional vectors across the unpooler main loop if this reduces
  occupancy;
- use a fresh output/scratch generation for replay safety;
- start with T=1 and T=8; add T=16/32 only if register count, spills, and the
  producer duration are unchanged within measurement noise.

The realistic saving from either fusion is the removed graph/launch floor, not
the full displayed duration of the old kernel. The combined route plus boundary
target is **at least 2.2 us per affected small load**. Below 1.5 us, the branch
does not support a robust 2.3x result and should be stopped.

### Why this is preferable to a persistent megakernel

Local experiments measured cross-SM synchronization around 1.4 us when idle
and about 6.3 us under a 4.3-TB/s stream. S015's 16-CTA fused router was
2.79--3.03x slower than the separate chain. PDL already exists in the
incumbent and improved the chain only around 0.9%. A last-arrival finalizer
adds one release/acquire handoff and no waiting CTAs; a persistent megakernel
or global barrier repeats mechanisms already disproved here.

## Secondary structural fallback

If route finalization cannot clear its gate, extend the existing GLE deferred
normalization path from T=1 to T=8. This requires an M=16 fused pooler/router
configuration because GLE projects `2*T` rows. It can remove verify-1's first
boundary launch while keeping the same linear decomposition already validated
at T=1.

Gate this fallback independently:

- exact parity for T=8 routing and output;
- no escape-table overflow;
- at least 1.0 us per verify-1 branch after accounting for the doubled-row
  pooler/router work;
- no regression in the O projection or fused pooler/router producers.

Do not make this the first implementation: it is a larger code-generation
change than the T=1 last-arrival prototypes and its M=16 fused projection is
not currently available.

## Quantified portfolio

The estimates below use the exact incumbent score model. “Small finalizers”
means a combined 2.5 us at decode 1, 2.2 us at decode 16/32, and 2.1 us at
verify 1, across both sliding and full branches.

| Bundle | Modeled score us | Modeled speedup | Decision |
|---|---:|---:|---|
| Incumbent | 10,003.685 | 2.26120x | protect |
| C12 only | 9,944.117 | 2.27474x | insufficient |
| C12 + M16 sliding | 9,904.411 | 2.28386x | insufficient |
| C12 + small finalizers | 9,803.570 | 2.30735x | crosses, but fragile |
| C12 + small finalizers + M16 | about 9,764 | about 2.317x | viable |
| Prior row + 15--30 us BF16 margin | about 9,734--9,749 | about 2.320--2.324x | preferred score candidate |

The target for implementation is therefore not merely 2.300x. It is a
two-check center of at least **2.315x**, preferably **2.32x**, because the S090
replications differed by roughly 0.6% at whole-layer level.

## Experiment order and quota allocation

Current decoder budget is 3/10 scored and 14/20 checks used, leaving seven
scores and six checks before reset. Do not use a scored run during development.

### Static/local gates before any remote check

For every candidate:

1. preserve the incumbent file and build a separate candidate;
2. syntax/import and upload-size checks;
3. exact source-marker assertion for every intended dispatch;
4. CUDA 12.4 SM90a compilation, register/spill/SASS inspection;
5. replay-safe counter audit and a proof that no CTA spins or waits for an
   unscheduled CTA;
6. local algebraic parity for the route and both boundary reductions.

### Six-check decision tree

1. **F1:** T=1 boundary-2 finalizer alone. Pass only if 13/13, no flags,
   target producer does not regress, and decode-1 improves at least 0.8 us in
   both branches.
2. **F2:** T=1 route finalizer alone, or T=8 boundary finalizer if the route
   static gate fails. Pass only with direct touched-kernel evidence; aggregate
   speedup alone is not evidence.
3. **F3:** extend the winning finalizer(s) to T=8/16/32. Require at least
   90 score-us of modeled direct saving before continuing.
4. **Bundle:** add exact-C12 and, only if its touched row reproduces, M16 MoE.
   Require check speedup at least 2.315x and all intended dispatch markers.
5. **Replication:** behavior-identical rebuild with different non-semantic
   source content. Require 13/13 and at least 2.30x again; the two profiles must
   agree on the targeted kernels within 0.5 us or 5%, whichever is larger.
6. **Margin/fallback:** add validated BF16 v177, or test GLE T=8 if the bundle
   remains between 2.29x and 2.315x. Otherwise keep this check in reserve for a
   correctness fix.

### Scored submission rule

Spend one scored run only when all of the following are true:

- two checks pass 13/13 with no flags;
- both checks are at least 2.30x and their mean/center is at least 2.315x;
- the score model from touched kernels predicts 9.745 ms or better, or there is
  equally strong repeated whole-layer evidence;
- route IDs, weights, local IDs, graph replay, and every intended dispatch are
  explicitly verified;
- `submission.py` is promoted with a hash guard from the protected S081 hash.

If the first score is below 2.3x but the targeted profile reproduces, use at
most one content-distinct scored replication. If the targeted kernels do not
reproduce, stop; do not spend the remaining quota on noise.

## Time-box through 05:30 IST

| Window | Work | Exit condition |
|---|---|---|
| 23:20--00:15 | implement and compile T=1 boundary finalizer | exact local parity, no spills, no spin |
| 00:15--01:00 | first remote check and profile inspection | >=0.8 us direct saving both d1 branches |
| 01:00--02:00 | route-finalizer feasibility or T=8 boundary extension | one structural path clears direct gate |
| 02:00--03:00 | extend to T=16/32; assemble C12/M16 bundle | modeled direct saving >=140 score-us |
| 03:00--04:00 | two whole-layer checks | both correct; center >=2.315x |
| 04:00--05:00 | BF16 margin or GLE T=8 fallback; final replication | predicted <=9.745 ms |
| 05:00--05:30 | hash-guarded promotion and one score | score only if every gate is green |

The schedule is intentionally front-loaded with kill decisions. If neither
finalizer shows a direct win by 02:00, a reliable 2.3x result in this quota day
is no longer supported by evidence; preserve the incumbent and carry the work
past reset instead of submitting a weak candidate.

## Explicitly closed paths

Do not spend time today on:

- attention L2::64, q_len=1 occ3 integration, q_len=8 F2 topology, or WST B16;
- RS128 pooler variants that fail routing replay;
- M=32 standalone MoE transplant, packed route IDs, joined WGS completion,
  FT32, large-boundary preload, or cluster multicast;
- a 16-CTA fused router, persistent layer megakernel, or any cross-SM spinning
  barrier;
- a blind CUTLASS Top-K transplant: NVIDIA's Hopper example is optimized only
  for K=2/4, warns of serious cost for generic K>4, and requires the N tile to
  cover the complete N reduction. The current K=8 router is split across 32
  tiles.

## Research sources

- Public incumbent and experiment lessons:
  <https://github.com/keell0renz/muse-open-kernel-challenge>
- Other current public decoder implementation:
  <https://github.com/vjeux/okc-2026>
- NVIDIA CUDA Programmatic Dependent Launch guide:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>
- NVIDIA CUTLASS Hopper collective/EVT example:
  <https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu>
- NVIDIA CUTLASS Hopper GEMM + Top-K + softmax example:
  <https://github.com/NVIDIA/cutlass/blob/main/examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu>

## Bottom line

The banked changes alone reach only about 2.284x. Reaching 2.3x today requires
one genuinely structural result: eliminate at least about 2.2 us from the
small-load route/boundary chain without slowing the producers. C12 plus that
mechanism crosses the threshold; M16 and BF16 supply the margin needed to make
a scored run rational. Every other recent branch is either mathematically too
small or already experimentally closed.
