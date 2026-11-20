# Next Optimization Plan: Route-Skew-Aware Hybrid MoE

## Status and constraints

- Current winner: S010 (`submission.py`), 13/13 correct, 13.365432974 ms, 1.692447977x.
- Preserve S010 byte-for-byte until a candidate clears every correctness and performance gate.
- Whole-decoder quota remaining when this plan was written: 8 scored submissions and 1 unscored check.
- Standalone MoE quota remaining when this plan was written: 2 scored submissions and 0 unscored checks until the daily reset at 00:00 UTC.
- Do not spend the final whole-decoder check on an exploratory kernel.

## Why MoE is the correct target

MoE is the largest remaining controllable cost in the decoder. Depending on the shape, it contributes roughly 32-65% of total layer time. The dominant kernels are `moe_k(P)` for small and medium token counts and `moe_x(P)` for the T=128 and T=256 verify cases.

Representative S010 timings:

| Shape | Whole layer | Dominant MoE work |
|---|---:|---:|
| decode, T=1 | 86-96 us | `moe_k`: about 28 us |
| decode, T=16 | 149-244 us | `moe_k`: about 78-79 us |
| decode, T=32 | 193-373 us | `moe_k`: about 102-111 us |
| verify, T=16 | 318-493 us | `moe_x`: about 212 us |
| verify, T=32 | 403-755 us | `moe_x`: about 247-267 us |

A real 20% reduction in the large-token MoE path could save approximately 42-53 us, equivalent to roughly 5-11% of end-to-end layer time. Router, normalization, and glue micro-fusions are too small to provide a comparable gain.

## Core hypothesis

The current T=256 MoE path uses one scheduling strategy for every expert even though routed-token counts are skewed. A single configuration is unlikely to be optimal for both heavily loaded and lightly loaded experts:

- Heavy experts have enough work to amortize a CPB2 persistent WGMMA schedule and fused activation gather.
- Light experts benefit from the current NCW16 path, which has lower fixed scheduling and synchronization overhead.

The proposed candidate will classify experts by their local routed-token count and dispatch them through two fixed persistent work queues:

1. **Heavy queue:** proven CPB2/fused-gather implementation.
2. **Light queue:** current NCW16 implementation.

This changes only physical scheduling. It must preserve routing, stable route ordering, numerical behavior, expert ownership, and the exact logical computation.

## Phase 1: Establish evidence before implementation

Do not begin by guessing a threshold. First determine whether the workload and existing kernels support the hypothesis.

1. Audit the current embedded MoE kernel and the standalone v52/v57 variants.
2. Record the exact differences between CPB2 and NCW16:
   - CTA and warp-group structure
   - tile sizes
   - register and shared-memory use
   - W13-to-W2 handoff
   - activation gather behavior
   - work counter and queue overhead
3. Confirm whether CPB2 was previously tested globally at T=256, and identify why it lost.
4. Extract or instrument per-GPU, per-expert routed-token counts for the benchmark shapes.
5. Build histograms for sliding and full-attention cases and quantify:
   - median and maximum load
   - number of empty/light/heavy experts
   - fraction of routed work owned by the heavy tail
   - stability of the distribution across repetitions
6. Estimate the break-even routed-token count from measured kernel behavior, occupancy, and fixed overhead rather than selecting it arbitrarily.

### Phase 1 go/no-go gate

Proceed only if the measurements show both:

- a meaningful mixture of light and heavy experts within the same T=256 workload; and
- a credible region where CPB2 is faster for heavy experts while NCW16 remains faster for light experts.

If either condition is absent, stop this direction without spending a remote submission.

## Phase 2: Scheduler and kernel design

### Expert classification

- Classify using the route count already available after routing/localization.
- Keep the operation deterministic and GPU-resident.
- Avoid an extra global launch if classification can be fused into existing route-list construction.
- Start with a compile-time threshold for clean code generation and low dispatch overhead.
- Evaluate at most a small, evidence-driven threshold set; do not perform a broad blind sweep.

### Work queues

- Build fixed-capacity heavy and light expert queues in existing scratch storage where possible.
- Use atomic append only during queue construction, outside the WGMMA hot loop.
- Store expert ID plus the minimum metadata needed by each worker.
- Sort or bucket heavy work by descending route count only if the existing stable per-expert ordering remains intact and the cost is demonstrably amortized.
- Do not introduce cluster multicast, cross-SM barriers, or route reshaping.

### Heavy path

- Reuse the already proven CPB2/fused-gather machinery rather than rewriting the math path.
- Preserve packed NVFP4 weight access and the existing scale interpretation.
- Keep the producer/consumer WGMMA pipeline and validate that the selected expert load is large enough to hide its fixed cost.
- Avoid increasing shared-memory use to a level that reduces occupancy enough to erase the gain.

### Light path

- Retain the current NCW16 implementation with the smallest possible changes.
- Do not force light experts through the CPB2 setup cost.
- Ensure empty experts consume no persistent worker slot beyond queue construction.

### Synchronization

- The two paths may execute as separate kernels initially to establish the attainable speedup safely.
- Fuse dispatch only after the split version proves that the scheduling policy wins.
- Prefer independent persistent counters for heavy and light queues.
- No grid-wide or cluster-wide synchronization.

## Phase 3: Correctness validation

Validate the standalone MoE component before touching the decoder winner.

1. Compile for SM90a and inspect:
   - registers per kernel
   - static and dynamic shared memory
   - local-memory use and spills
   - occupancy limits
2. Run local structural and syntax checks.
3. Compare output against the reference for all supported token counts, EP configurations, and adversarial routing distributions.
4. Add focused cases for:
   - all routes concentrated on one expert
   - uniform expert load
   - many empty experts
   - counts immediately below, at, and above the threshold
   - maximum-capacity experts
5. Confirm stable routing and that no route is dropped, duplicated, or assigned to the wrong local expert.
6. Use an unscored standalone MoE check only after the daily quota resets and the candidate passes local/static gates.

## Phase 4: Performance validation

Profile components, not just the aggregate runtime.

Measure:

- route-list and queue-construction time
- heavy-path W13, activation, W2, and reduction time
- light-path time
- total MoE time
- launch and synchronization overhead
- achieved occupancy and evidence of register/shared-memory pressure

Compare against S010's current embedded backend with repeated runs. Treat changes within ordinary run-to-run noise as no improvement.

### Performance gate

The candidate must deliver all of the following before decoder integration:

- at least 15% reduction in total MoE time at T=128;
- at least 15% reduction in total MoE time at T=256;
- no material regression in smaller token-count MoE paths;
- no correctness flags or unexplained numerical drift;
- an improvement large enough to remain convincing after queue and launch overhead.

The 15% gate is intentionally strict because only one whole-decoder unscored check remains. A mixed or shape-specific result is not sufficient.

## Phase 5: Decoder integration

Only after the standalone gate clears:

1. Create a new decoder candidate file; do not edit `submission.py` in place.
2. Port the validated scheduler and kernels mechanically.
3. Keep the S010 attention, BF16 projection, normalization, routing semantics, and all unrelated code unchanged.
4. Re-run static resource inspection after integration because surrounding compilation can change register allocation.
5. Use the final whole-decoder unscored check.
6. Promote the candidate to `submission.py` only if it is 13/13 correct and produces a clear aggregate speedup across both sliding and full-attention cases.
7. Spend a scored decoder submission only on that validated winner.

## Failure criteria and fallback

Reject the direction without further quota if:

- expert loads do not show useful within-shape skew;
- the break-even regions overlap poorly or are unstable;
- queue construction consumes most of the projected gain;
- CPB2 resource use causes low occupancy at T=256;
- either T=128 or T=256 improves by less than 15%;
- small-token paths regress materially;
- correctness depends on route reordering or approximation.

If rejected, retain S010 and use the evidence to choose the next architecture-level MoE target. Do not fall back to router, norm, or glue micro-optimizations, because their maximum end-to-end contribution is too small relative to the remaining MoE cost.

## Immediate next action

Perform the Phase 1 audit and route-count characterization. No remote submission should be spent until the daily standalone MoE check quota resets and the evidence supports the hybrid split.
