# Next large optimization: 2D clustered MoE operand multicast

## Decision

The next architecture worth implementing is a four-CTA Hopper cluster for the
large-token `moe_x` path (`T=128/256`).  Treat the cluster as a 2 x 2 grid:

- one axis owns different expert output-row groups;
- the other axis owns adjacent token passes of the same expert;
- activations are multicast along the row axis;
- packed NVFP4 weights are multicast along the token-pass axis.

This is a structural dataflow change, not another instruction or launch tune.
It is also different from the rejected W13-to-W2 counter pipeline: W13 remains
globally before W2, so no CTA can occupy the machine while waiting for work on
which it depends.

Do not modify the S010 winner until a standalone component candidate clears the
gates below.  S010 remains the exact source with SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.

## Why this is the best remaining large target

The S010 profile puts `moe_x` at:

| Shape | Whole shape | `moe_x` | Profile share |
|---|---:|---:|---:|
| verify-16 sliding | 318.2 us | 212.7 us | 64.5% |
| verify-16 full | 492.6 us | 211.6 us | 41.7% |
| verify-32 sliding | 402.6 us | 266.5 us | 63.1% |
| verify-32 full | 755.0 us | 247.1 us | 32.1% |

Those four cases are large enough that a real 15-25% `moe_x` reduction would
improve the verify-16 load by about 9.6-17.1%, the verify-32 load by about
8.6-15.3%, and the six-load geometric score by about 3.0-5.1%.  That is safely
larger than run noise and materially larger than the recent sub-percent ideas.

The current two large configurations already expose the operand-reuse problem:

| Tokens | Current configuration | Activation bytes/slot | Packed-weight bytes/slot |
|---|---|---:|---:|
| 128 | `NCW8, RPW2, TPX32, CPB2` | 16,384 | 36,864 |
| 256 | `NCW16, RPW1, TPX64, CPB1` | 32,768 | 36,864 |

Within one CTA, `RPW2`/`UBL2` already reuses an activation across two 128-row
blocks.  Increasing reuse inside the CTA again was tested and lost because it
increased live accumulators.  The cluster instead adds reuse across CTAs while
leaving every CTA's accumulator and register footprint unchanged.

## Why the previous large experiments do not close this direction

- E033/TPX128 reused a weight pass by doubling the token accumulator footprint
  inside each CTA.  It was 1.53% slower overall and as much as 7.2% slower in a
  large MoE profile.  The proposed cluster keeps TPX32/64 and gives the second
  token pass to another CTA.
- The standalone v61 checkpoint proved that a two-CTA Hopper cluster launch and
  initial cluster synchronization are effectively free (302 us versus 303 us
  for ordinary NCW16).
- v62-v63 changed W13-to-W2 readiness and introduced dependency waiting.  This
  design does not interleave W2 and therefore cannot reproduce that scheduler
  inversion.
- E034 pass-scoped counters and E035 paired WGMMA changed control overhead but
  not operand traffic.  Their integrated results were mixed and inside noise.
- A four-query/eight-query custom attention path is still interesting, but the
  prior 64-query WGMMA prototype has an unresolved layout bug and the correct
  alternatives were more than 3x slower than FA3.  Even a 2x full-attention
  kernel affects only the weight-16 full branch.  `moe_x` is the broader and
  better validated large target.
- The BF16-v55 wholesale projection port and the available prefill ports were
  already tested and regressed.  They do not expose a comparable remaining
  structural reuse opportunity.

## Hopper mechanism

Hopper thread-block clusters co-schedule their CTAs within one GPC.  SM90 TMA
supports `cp.async.bulk...shared::cluster.global...multicast::cluster`, which
transfers one global tile to the same shared-memory destination in multiple CTAs
and completes against cluster transaction barriers.  NVIDIA CUTLASS uses the
same 2D pattern for GEMM: multicast A across CTAs that share A and B across CTAs
that share B.

The active kernel already uses TMA transaction barriers and a cluster-addressed
global-to-shared instruction, so the change is localized to scheduling, launch,
barrier initialization, and the producer's copy operations.  Consumer dequant,
WGMMA, epilogues, routing, and numerical order remain unchanged.

Primary references:

- CUDA Programming Guide, asynchronous copies and cluster multicast:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
- PTX ISA, `cp.async.bulk` and `cp.async.bulk.tensor` multicast forms:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- NVIDIA CUTLASS Hopper WGMMA/TMA tutorial and cluster launch:
  https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
- NVIDIA CUTLASS SM90 multicast primitives:
  https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp

## Cluster work mapping

Use `CLUSTER=4` and give each CTA a `(row_rank, pass_rank)` coordinate.

### Paired-pass mode

For two adjacent passes of one `(phase, expert)`:

```text
               pass p                 pass p+1
row group r    CTA (r, p)             CTA (r, p+1)
row group r+1  CTA (r+1, p)           CTA (r+1, p+1)
```

- The two CTAs in a column use the same activation tile, so issue one activation
  TMA multicast to their two-bit mask.
- The two CTAs in a row use the same packed-weight tile, so issue one weight TMA
  multicast to their two-bit mask.
- Each CTA waits on its existing local full barrier and executes the unchanged
  dequant/WGMMA consumer loop.

For a fully populated 2 x 2 group, TMA source transactions for both operands
fall by 50%: four activation loads become two and four weight loads become two.
This does not promise a 50% kernel gain because consumer dequant and WGMMA are
unchanged, but it is large enough to target a 15-25% end-to-end `moe_x` gain.

### Odd-pass mode

Never leave `pass_rank=1` idle.  When an expert has one unpaired pass, reinterpret
all four ranks as row ranks for that pass:

- multicast its activation tile to all four CTAs;
- assign four consecutive row groups, one to each CTA;
- load each CTA's distinct weight tile normally.

W13 has 32 current row units and W2 has 16, both divisible by four, so this mode
has no row tail in either active large configuration.  Experts with only one
pass still get four-way activation reuse and full cluster utilization.

### Singleton packing fallback

If cluster scheduling or the odd-pass mode proves too expensive, the fallback
is a two-CTA token-pass cluster.  Pair adjacent passes when available; pack two
unrelated singleton units into the same cluster otherwise and disable multicast
for that pair.  Do not accept a design that idles half of the cluster for every
single-pass expert.

## Scheduler and synchronization plan

1. Build cluster-level segments from the same device-resident expert counts.
   Keep all W13 cluster groups before all W2 cluster groups.
2. In paired mode, decode a cluster-group index into one expert, two passes, and
   two row groups.  In odd mode, decode it into one pass and four row groups.
3. Have cluster rank zero acquire the next cluster-group index.  Broadcast it
   through distributed shared memory after one cluster synchronization at the
   work-unit boundary, not once per K tile.
4. Every CTA initializes the same-offset transaction barrier for each ring slot.
   The issuing rank uses a multicast mask; each destination CTA counts only the
   bytes that arrive in its own shared-memory slot.
5. Keep the current two/three-stage ring.  Do not reduce `WCH` or add a barrier
   per K16 instruction; the earlier WCH2 result showed that extra slot turnover
   loses.
6. Retain the expert-wide W13 completion counters.  W2 begins only after its
   expert's W13 is complete, exactly as in S010.
7. Preserve the existing fused gather choice: `FGATHER` at T=128 and the S010
   T=256 path unchanged except for cluster scheduling and TMA copies.

## Quantitative gates

This is a high-upside architecture, but multicast alone is not assumed to win.
The diagnostics show producers can wait for consumers, and repeated tiles may
hit L2.  Use hard gates so a legal but non-critical-path improvement is rejected
before it consumes decoder attempts.

### Phase 0: mechanics-only control

- Port v61's cluster launch to a new standalone candidate.
- Test `CLUSTER=2`, then `CLUSTER=4`, with unchanged work assignment.
- Require 5/5 correctness and no more than 2 us regression at M=128 and M=256.
- Confirm no spills and unchanged per-CTA dynamic shared memory/register count.

### Phase 1: weight-pass multicast only

- Implement the two-pass scheduler and packed-weight multicast for a two-CTA
  cluster first.
- Report paired-pass coverage, bytes multicast, singleton coverage, producer
  wait, consumer wait, and cold-L2 graph medians.
- Require at least 8% at both M=128 and M=256.  This stage directly tests the
  mechanism that TPX128 could not isolate from accumulator pressure.

### Phase 2: full 2D cluster

- Add the row axis and activation multicast, then the four-way odd-pass mode.
- Require at least 12% at both M=128 and M=256, with a target of 15-25%.
- Require improvement on multiple fresh routing seeds, not only the tuner seed.
- Keep a paired `CLUSTER=2` fallback if cluster-4 placement hurts one bucket.

### Phase 3: decoder integration

- Integrate only the independently winning M=128/M=256 configurations.
- Run syntax/resource checks, then `okc check`, never a scored run first.
- Promotion requires 13/13, no nearby-shape cliff at T=264, at least 8% lower
  `moe_x` in all four verify profiles, and at least 2% lower aggregate decoder
  latency.  The expected successful band is 3-5% score improvement.
- Spend a scored attempt only after the unscored profile shows the mechanism,
  not merely a favorable headline within the confidence interval.

## Main risks

- **Insufficient multi-pass coverage.**  If hot experts rarely exceed TPX32/64,
  weight multicast cannot pay.  Measure coverage before building the 2D path.
- **L2 already absorbs the second read.**  Multicast still removes TMA and
  L2-to-SM traffic, but the benefit may be below the Phase-1 gate.
- **Cluster scheduling overhead.**  A cluster barrier per work group is
  acceptable to test; a barrier per K tile is not.
- **GPC placement/occupancy.**  Use `cudaOccupancyMaxPotentialClusterSize` and
  `cudaOccupancyMaxActiveClusters`; keep the existing persistent-residency
  checks, now at cluster granularity.
- **Barrier byte accounting.**  Every destination barrier must expect exactly
  its local activation plus weight bytes, including short token tails.
- **Load imbalance.**  Odd passes must switch to row-parallel mode or be packed;
  they must not strand half the cluster.

## Recommendation

Proceed with Phase 0 and Phase 1 in the standalone MoE directory.  Do not begin
with the full four-CTA implementation and do not touch S010 yet.  The two-CTA
weight-pass multicast is the decisive experiment: it isolates cross-CTA weight
reuse from the accumulator-pressure failure in E033.  If it clears 8%, extend
to the 2D cluster; if it does not, reject the architecture rather than trying
small multicast-mask or cache-policy variations.
