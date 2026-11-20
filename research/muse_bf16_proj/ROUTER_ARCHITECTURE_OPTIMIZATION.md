# Router Architecture and Optimization

This document describes the protected v50 router in `submission_qkv32_o16_m16trace_v50.py`, the optimizations currently active, and the architectural directions closed by the v63 and v64 H200 experiments. Detailed measurements are in `ROUTER_OPTIMIZATION_SEQUENCE.md`.

## Problem contract

The router computes a single projection:

```text
X:       BF16  [m, 8192]
W:       FP32  [256, 8192]
output:  FP32  [m, 256]

output = X @ W.T
```

The required output is the raw FP32 router logits. Sigmoid, softmax, top-k selection, and expert-weight normalization are not part of this contest operator.

## Numerical representation

For m=1, the `rr<1,1>` path reads the original FP32 weights and performs FP32 FMA accumulation directly.

For the other specialized shapes, each FP32 weight is represented by two BF16 values prepared once:

```python
hi = W.to(torch.bfloat16)
lo = (W - hi.float()).to(torch.bfloat16)
```

The kernels accumulate both terms in FP32:

```text
logit ~= FP32(X * hi) + FP32(X * lo)
```

This preserves the contest-approved FP32-equivalent router path while allowing BF16 MMA/WGMMA instructions. The low component is not an independent approximation: it is the BF16-rounded residual left after subtracting the high component from the original FP32 weight.

## Prepared weight layouts

- `router_wide`: `[N/64, K/64, hi_or_lo, N64, K64]`; used by m=128/256 so one TMA region places the high and low N64 tiles next to each other.
- `router_mid`: `[N/16, K/64, hi_or_lo, N16, K64]`; used by the m=32 specialized path with a 32-row TMA box containing high and low N16 tiles.
- `router_tail`: register/MMA-native ordering for m=8/16; high and low fragments are adjacent and already arranged in `mma.sync.m16n8k16` lane order.
- `router_raw`: the original FP32 `[256,8192]` matrix retained for the m=1 direct-FP32 route and validation.

Packing is performed during `prepare()`, outside the timed kernel invocation. All active routes are validated against the original matrix before they are enabled.

## Shape-specialized dispatch

| m | Active kernel | Grid and ownership | Compute and reduction |
|---:|---|---|---|
| 1 | `rr<1,1>` | 256 CTAs; one output row per 512-thread CTA | Each thread loads four FP32 `float4` vectors, accumulates in registers, then performs a 16-warp shared-memory reduction. |
| 8 | `rhtailk<1>` | 256 CTAs arranged as sixteen 16-CTA clusters; each cluster owns N16 | Sixteen ranks split the 128 K64 chunks; eight warps per CTA use `mma.sync`; final vectors are reduced through DSMEM. |
| 16 | `rhtailk<2>` | Same 256-CTA/cluster-16 topology as m=8, processing two M8 groups | Register-resident high/low MMA fragments and activation vectors; DSMEM split-K reduction. |
| 32 | `rr_kernel<32,16,4,1,2,1>` | Shape-specialized M32/N16 route | Four-stage TMA pipeline feeds register/MMA accumulation of the high and low components; the route is correctness- and latency-gated. |
| 128 | `rhwide<8,false>` | M64 x N64 tiles, 2 M tiles x 4 N tiles x split-K 8 = 64 CTAs; cluster size 8; 128 threads/CTA | Eight-stage TMA ring; separate N64 WGMMA operations for high and low weights; distributed DSMEM reduction. |
| 256 | `rhwide<4,true>` | M64 x N64 tiles, 4 M tiles x 4 N tiles x split-K 4 = 64 CTAs; cluster size 4; 256 threads/CTA | Producer/consumer specialization; high and low N64 tiles are treated as one N128 WGMMA result and folded in FP32; DSMEM reduction. |

The two wide grids deliberately contain exactly 64 CTAs, approximately one wave on H200. This keeps every CTA's K stream long enough to amortize launch, TMA, cluster, and WGMMA startup costs.

## Wide-kernel data flow

```text
global packed weights + BF16 activations
             |
             | TMA, 128-byte swizzle, L2 64-byte promotion
             v
eight-slot shared-memory ring
             |
             | mbarrier producer/consumer hand-off
             v
WGMMA FP32 accumulators
             |
             | st.async.shared::cluster DSMEM exchange
             v
cluster-local split-K sum
             |
             v
raw FP32 [m,256] output
```

Each ring slot contains one high-weight tile, one low-weight tile, and one M64 activation tile. TMA completion is tracked by per-slot `mbarrier` objects. The WGMMA consumer waits only for the required slot, commits asynchronous WGMMA groups, and overlaps later TMA refills with current computation.

The split-K result never uses a global workspace in the active wide kernels. Each cluster rank sends `float4` partials directly to the owning rank with asynchronous DSMEM stores tied to a remote `mbarrier`; the owner sums the rank fragments and writes final FP32 logits. Reduction and output storage therefore remain in the same kernel and are graph-safe without a global atomic counter.

## Small-m memory behavior

- m=1 issues 32 KiB of weights per CTA using 16-byte `ld.global.nc.L1::no_allocate` vectors.
- m=8/16 issue 32 KiB of packed high/low weights per CTA with the same 16-byte non-allocating load form; activation reads are also vectorized.
- m=32 uses a four-stage TMA pipeline rather than a shallow load/compute loop.

The small routes therefore already meet the requested 24 KiB minimum bytes-in-flight target. Their remaining cost is primarily fixed launch, synchronization, and reduction latency rather than a lack of load width.

## Active optimizations

| Optimization | Current implementation |
|---|---|
| Shape specialization | Separate kernels and launch geometry for m=1, 8, 16, 32, 128, and 256. |
| FP32-equivalent BF16 compute | FP32 weights split into BF16 high plus BF16 residual-low components, both accumulated in FP32. |
| Direct FP32 m=1 route | Avoids high/low duplication and uses each original FP32 weight once. |
| Register residency | Small-m weight and activation fragments stay in registers through MMA and only final partials enter shared memory/DSMEM. |
| Vectorized cache policy | 16-byte `ld.global.nc.L1::no_allocate` weight loads avoid polluting L1. |
| Staged asynchronous movement | m=32 uses four TMA stages; m=128/256 use eight-stage TMA rings. |
| WGMMA specialization | m=128 performs two N64 high/low operations; m=256 performs one N128 operation and folds its halves. |
| One-wave wide launch | Both wide shapes use 64 CTAs to balance H200 occupancy against per-CTA useful work. |
| DSMEM split-K | Cluster-local asynchronous stores and barriers replace global workspace traffic and a second reduction kernel. |
| Fused completion | Accumulation, split-K reduction, and final FP32 storage occur in one router kernel. |
| Runtime safety gate | Exact coefficient recovery, random FP64-reference validation, and paired timing decide whether a specialized route is enabled. |

## Rejected experiments

### v63: weight multicast

v63 multicast an identical weight tile across M64 peers in 16-CTA clusters while retaining 64 total CTAs. It was correct, but m=128 became 38% slower and m=256 became 24% slower. This shows that eliminating repeated logical weight requests does not repay the larger-cluster synchronization and scheduling cost; L2 already services the repeated tiles effectively.

### v64: additional CTA parallelism

v64 replaced the wide topology with 96-CTA cluster-3 and 128-CTA cluster-4 variants using only DSMEM reduction. Every variant recovered packed coefficients exactly and passed FP64-reference checks, but all were slower:

| m | CTA count | Incumbent | Candidate | Regression |
|---:|---:|---:|---:|---:|
| 128 | 96 | 10.528 us | 17.280 us | +64.1% |
| 128 | 128 | 10.560 us | 14.528 us | +37.6% |
| 256 | 96 | 12.064 us | 17.504 us | +45.1% |
| 256 | 128 | 12.096 us | 15.200 us | +25.7% |

The 96-CTA routes used a centralized rank-zero DSMEM reduction because the output `float4` vectors cannot be divided evenly across three ranks. The 128-CTA routes used a balanced distributed four-rank reduction. The latter were less damaging, but still conclusively slower than the 64-CTA one-wave kernels.

An exact 132-CTA topology is not compatible with the current M64 tile and legal WGMMA N widths under the requested cluster-size limit of four. The legal N16/N32 decompositions each produce 32 output tiles, making 96 and 128 the exact cluster-3/4 grid sizes.

## Architectural conclusion

The router is latency, scheduling, and synchronization limited rather than weight-bandwidth limited:

- Logical bandwidth is already roughly 2.0-4.2 TB/s depending on shape, implying substantial L2 reuse at wide shapes.
- Weight multicast reduced requested traffic but lost badly to cluster overhead.
- Increasing CTA count exposed more parallel work but lost to shorter K streams and additional DSMEM/scheduling overhead.
- The successful topology is the smallest cluster and CTA count that still fills one H200 wave while giving each CTA a long, efficiently pipelined K stream.

Future work should therefore target instruction issue, barrier critical paths, WGMMA/TMA overlap, and launch latency inside the existing 64-CTA topology. Large clusters, weight multicast, and additional split-K parallelism should not be retried without new profiler evidence.

## Constraints on other proposed directions

- Programmatic dependent launch requires an owned dependent kernel. This contest submission owns only the projection kernel, so there is no legal consumer in which to place `cudaGridDependencySynchronize`.
- Extra fusion is unavailable: the split-K reduction is already fused, while sigmoid/top-k/normalization would change the required operator output.
- A persisting-L2 window conflicts with cold-L2 replay and the prohibition on cross-call cache-residency strategies.
- Storing the low residual as FP8 violates the allowed router representation; only the BF16 high/low split is permitted.
