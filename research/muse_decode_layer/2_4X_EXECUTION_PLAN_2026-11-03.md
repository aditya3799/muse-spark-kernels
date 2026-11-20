# Muse decode layer: execution plan to 2.4x

Date: 2026-11-03

## Decision

Keep the exact 2.2611966426x incumbent (`submission.py`, SHA-256
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`)
protected. Do not spend another decoder attempt on a sub-percent transplant.

The 2.4x target requires a further 1.06138x latency improvement. The exact
incumbent score is 10003.685 us, so the target is about 9424.2 us. The gap is
about 579.5 weighted microseconds.

The next decoder submission must combine:

1. a standalone MoE win of at least 12% at the affected token buckets, with a
   15% target; and
2. the already measured attention/projection bundle, provided its in-layer
   confirmation remains positive.

Anything smaller cannot plausibly reach 2.4x and should remain a component
experiment.

## Exact incumbent load table

| Load | Sliding us | Full us | Weighted load us (`46*S + 16*F`) |
|---|---:|---:|---:|
| decode 1 | 66.277 | 73.656 | 4227.26 |
| decode 16 | 128.729 | 199.577 | 9114.78 |
| decode 32 | 144.790 | 275.864 | 11074.16 |
| verify 1 | 108.271 | 116.842 | 6849.96 |
| verify 16 | 242.844 | 330.065 | 16451.84 |
| verify 32 | 306.514 | 476.722 | 21727.21 |

The problem score is the geometric mean of the six load speedups. Within a
load, sliding has weight 46 and full has weight 16. This makes a full-attention
micro-optimization less valuable than the same saving in sliding or MoE.

## Work completed in this pass

### Rejected after H200 measurement

- Recovered WST q_len=8 full-attention kernel: correct, but 1452.9 us / 2.592x
  standalone versus the exact attention incumbent at 1444.3 us / 2.6076x.
  It saved only about 2.8 us at full B16 and was neutral at B32.
- Recovered ST small-token MoE kernel: correct, but 6956.4 us / 1.933x.
  Its M1 time was 41.4 us versus roughly 25 us for the incumbent.
- BF16 four-warpgroup RS256: correct fallback, but every candidate geometry
  lost its prepare-time paired gate.
- Attention L2::64 policy: 13/13 correct, but decoder score fell to 2.226x.
- SC6 4W/2X MoE, cluster multicast, three-CTA attention, boundary preload,
  fused router, and standalone boundary reuse were already measured losses.

### Measured positive components, held for a combined candidate

1. **Full q_len=1 attention occupancy.** Three resident CTAs per SM with all
   128 producer threads improved the standalone attention score from 2.6076x
   to 2.6546x. The repeatable touched gains were about 2.55 us at full B16,
   4.65 us at full B32, 2.65 us at verify-full B16, and 4.87 us at
   verify-full B32.
2. **RS128 pooler at T=128.** The protected projection tournament measured
   about 24.38 -> 20.99 us for pooler M128. The decoder already embeds the
   RS128 implementation, but `RQ_USE` omits `('pr', 128)`.
3. **MoE production hybrids.** Standalone v186 measured 2.3893x and confirms
   the M128 production build, while v187 combines the independently paired
   M32 B4 and M128/M256 MFIN+SC6 choices. The current decoder already contains
   the B4/MFIN/SC6 mechanisms, so this is confirmation rather than a new large
   transplant.

Applying the attention and pooler savings optimistically to the incumbent
predicts only about 2.2726x. This bundle is worth preserving, but not worth a
decoder attempt on its own.

## Public-source audit

- The incumbent is the current public keell0renz decoder at commit
  `890809cee421500b00126d084e81b338ef9181ab`; there is one public branch.
- The current vjeux decoder remains commit
  `f35a41ee1c13e316dabbf772830d47b23b453393`; its newest fork has no decoder
  changes.
- The newly updated AutoclickerI decoder scores 1.8892x. Its standalone stage
  timings are all slower than the incumbent. Its rigorous excluded-work oracle
  puts deleting route plus both residual boundaries at 5.13%, and its legal
  route-fusion implementations regressed.
- operatorstack and lee101 decoder/component kernels are slower than the
  incumbent and have already been used as donors in the AutoclickerI audit.
- Current SGLang and vLLM Muse MegaMoE work does not provide a drop-in Hopper
  W4A16 kernel. The published MegaMoE work is primarily Blackwell or
  communication-fused. Hopper supports BF16 and FP8 WGMMA, but not native
  NVFP4 tensor-core MMA.

Primary public references:

- https://github.com/keell0renz/muse-open-kernel-challenge
- https://github.com/vjeux/okc-2026
- https://github.com/AutoclickerI/muse_kernel
- https://github.com/operatorstack/open-kernel-challenge-solutions
- https://github.com/lee101/openkernels
- https://github.com/sgl-project/sglang/issues/44678
- https://github.com/vllm-project/vllm/issues/64632
- https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wgmma_programming.html

## The remaining score requirement: MoE-sized, but no supported 12% mechanism yet

The active profiler attributes approximately:

| Token count | MoE GPU time |
|---:|---:|
| 1 | 22-24 us |
| 16 | 63-73 us |
| 32 | 75-85 us |
| 8 | about 57 us |
| 128 | 151-156 us |
| 256 | 198-207 us |

A robust 15% MoE reduction across the hot buckets would be large enough to
cover the remaining target when combined with the banked component bundle.
Route and boundary work alone is not. The research in this pass did not find a
remaining implementation whose evidence supports that saving.

## Go/no-go results for the two remaining MoE ideas

### Route-count-specialized light/heavy schedules: no-go before implementation

A deterministic 128-seed model reproduced the official routing construction
and measured the local-expert work distribution. The decisive large-token
buckets are overwhelmingly heavy:

| M | Median local routes | Median nonempty experts | route share count <= 2 | route share count >= 8 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 100.00% | 0.00% |
| 8 | 13 | 8 | 68.00% | 5.05% |
| 16 | 25 | 12 | 44.77% | 21.65% |
| 32 | 48 | 16 | 26.86% | 32.97% |
| 128 | 188.5 | 23 | 4.76% | 73.64% |
| 256 | 373 | 25 | 1.46% | 89.61% |

At M128/M256 only 4.76%/1.46% of routed work is in count-at-most-two experts;
73.64%/89.61% is already count-at-least-eight. A light-body queue therefore
cannot produce the required 12-15% large-bucket saving even with zero queue
overhead. Close this architecture without spending remote quota. The CPU model
is not used as a timing claim; it is only a structural rejection of the work
partition.

### Separately tuned W13/W2 phases: no-go as an implementation architecture

The active WGS kernel already has a finer dependency than a phase-wide barrier:

- the weight and activation producers stream both phases through the same
  resident rings;
- each phase-A epilogue writes the phase-B layout directly into `xB`;
- `doneAs` publishes readiness per phase-B stage;
- phase B waits only for the two phase-A units that produced its matching 256
  intermediate columns; and
- MFIN publishes phase-B expert completion directly to the finalizer, avoiding
  the old pre-final grid barrier.

Splitting W13 and W2 into launches would add grid completion, a launch, and a
second routing/range/prologue pass while removing this per-stage overlap.
Historical experiment E034 already tested an even finer per-(expert,
token-pass) W13-to-W2 readiness protocol. It passed 13/13 but was 1.41% slower
overall; extra releases, acquires, counters, and cleanup outweighed the overlap.
S070 separately split the finalizer and likewise regressed the aggregate.

Do not implement a two-launch phase split. A future phase experiment is allowed
only if it preserves the current single-launch per-stage producer/consumer
handoff and presents a static reason for at least a 12% compute reduction (for
example, a materially narrower exact WGMMA shape with no new spill/stack). No
such body was found in the public or local candidates audited here.

## Continuation results: the banked bundle did not survive integration

- Guarded q_len=8 F2 job `b0caf5d92cc34e00` still produced CUDA 719. Guard3
  job `a07879fc61f24181` repaired the empty-barrier arrival mismatch and removed
  the launch failure, but the first q_len=8 full case had relative L2 error
  0.702 against a 0.15 bound. Close F2: it has neither correctness nor timing
  evidence.
- M256 two-slab RS128 job `8943f1b52d4b413d` passed 9/9, but every candidate
  lost its paired gate: qkv +6.7%, O +34.6%, and pooler +10.3%. Its dispatch
  correctly stayed disabled.
- RS128 O/m128 half-split job `31f7a75e4dec4b35` passed 9/9, but the candidate
  was 12.32 us versus the 11.04 us incumbent and correctly stayed disabled.
- Combined decoder job `caef5b4dc2274dff` passed the 13 correctness cases but
  failed the timing recheck on verify-16 routing after enabling RS128 pooler.
  The pooler path is unsafe in the decoder replay/overlap context despite its
  standalone projection tolerance.
- q_len=1-only decoder isolation `9434c003de174c1f` passed 13/13 at 2.254x /
  10.04 ms, below the 2.2611966426x incumbent. Its targeted B16 full attention
  changed only about 90.8 -> 90.5 us and B32 regressed to 174.3 us. The
  standalone occupancy win does not transfer to the whole layer.
- MoE v188 is an AST-identical replication of v187. Its 2.3704x versus
  v187's 2.3839x is direct evidence of run variance, not a new mechanism.
- Exact standalone-v178 M=16 decoder dispatch (S092) passed 13/13 in check
  `c947bd1c889b4303` but scored only 2.238111x / 10.106873 ms versus the
  protected 2.261197x / 10.003685 ms incumbent.  Its sliding M=16 MoE row
  improved 73.0 -> 68.3 us while the full row was flat at 63.3 -> 63.4 us;
  the isolated gain is too small for promotion and does not warrant a replica.
- User-authorized scored S090 job `43aec79e497c41c9` passed 13/13 with no
  flags at 2.254826x / 10.031951 ms. Its exact-C12 M=1 CGEMV kernels enabled
  and saved about 2.4 us locally, but the aggregate remained below the
  protected 2.261197x incumbent and below 2.3x. Close S090; do not repeat it.
- BF16 v177's initial K128 GQX revision serialized every HGMMA, but the current
  hash `aaa46a08611f70c2e8a574b136b27b2bdfed397a03927a3fea05c17438affd3a`
  emits all eight operations through one inline-PTX operand list.  Exact
  `sm_90a` compilation now has zero spills and no C7515; SASS has no dependency
  barrier between the eight HGMMAs.  The revised candidate clears the static
  gate but still needs an unscored standalone BF16 check after quota reset.
  Do not spend decoder quota before that result.

Boundary fusion still has an excluded-work ceiling of about 5.13% even when
route plus both residual boundaries are deleted. It remains an engineering
direction, but it cannot reach 2.4x by itself and prior legal implementations
regressed.

The 2.4x target still requires a score-sized mechanism, but smaller verified
gains are now useful intermediate results.  Re-open the decoder for either a
new component with a measured score-sized projection, or a narrowly scoped
candidate backed by an already validated component implementation.  A small
candidate must be replicated before promotion so route variance is not
mistaken for progress.

## Decoder integration order

Create a new candidate file; never edit the incumbent in place.

1. Port only a newly measured, score-sized component mechanically.
2. Do not include q_len=1 occ3, RS128 pooler, M256 two-slab RS128, O/m128 HALF,
   or q_len=8 F2; each now has decisive negative integration/component evidence.
3. Run syntax/static checks and exact SM90a resource inspection.
4. Use an unscored decoder check first. Require 13/13 and inspect the targeted
   kernel/profile delta rather than accepting a favorable aggregate seed.
5. For a small gain, run a content-distinct behavior-identical replication.
   Promote only when the two-run evidence is favorable and the touched kernels
   move in the intended direction.  Keep a non-reproducible point estimate as
   a saved candidate, not as `submission.py`.

## Stop conditions

- Do not chase favorable route seeds or retry unchanged files for variance.
- Do not resubmit the rejected micro bundle or its q_len=1-only isolation.
- Do not revisit cluster multicast, 4W/2X, ST, boundary preload, L2::64, or the
  fused-router recipe without a new mechanism and new standalone evidence.
- Do not approximate or change weight precision outside the task rules.
- The route-count hybrid and two-launch W13/W2 split are closed by the evidence
  above. Do not revive them with threshold or scheduling micro-sweeps.
- The current evidence does not support 2.4x from the remaining measured
  mechanisms. Preserve the 2.2612x winner until a genuinely new component
  clears the quantitative promotion gate.
