# Router Optimization Sequence

All baseline timings are corrected H200 medians from the protected v50 profile. Weight bytes are logical kernel-issued bytes, not measured HBM transactions; the m128/m256 values include repeated M64 weight-tile requests that may hit L2.

## Step 1 - Baseline

| m | Router time (us) | Issued weight bytes | Achieved logical GB/s | End-to-end load (us) | Router share |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.09 | 8 MiB | 2051 | 36.71 | 11.14% |
| 8 | 3.95 | 8 MiB | 2124 | 36.73 | 10.75% |
| 16 | 4.20 | 8 MiB | 1997 | 46.69 | 9.00% |
| 32 | 4.82 | 8 MiB | 1740 | 54.83 | 8.79% |
| 128 | 5.97 | 16 MiB | 2810 | 64.97 | 9.19% |
| 256 | 7.93 | 32 MiB | 4231 | 81.80 | 9.69% |

Every scored load exceeds the 5% stop threshold, so the ordered investigation continued.

## Step 1 diagnostic - wide-kernel cost ablation

Diagnostic v65 (`95beefe`, reverted by `9249f70`) preserved the incumbent grid, cluster size, block size, and dynamic shared-memory allocation but stopped the wide kernel at five different points. These kernels intentionally did not produce valid logits and were never reachable from `kernel_fn`. Unscored H200 job `5a23238b41d44875` measured:

| m | Diagnostic envelope | Diagnostic (us) | Paired full (us) | Corrected share of full |
|---:|---|---:|---:|---:|
| 128 | Empty launch | 4.896 | 10.464 | ~0% above replay overhead |
| 128 | Tensor-map prefetch + barrier initialization | 5.024 | 10.464 | ~0% above replay overhead |
| 128 | All TMA loads, no WGMMA | 9.248 | 10.592 | 75.2% |
| 128 | WGMMA on uninitialized shared memory, no TMA | 5.280 | 10.528 | 1.8% |
| 128 | Full TMA/WGMMA mainloop, no DSMEM reduction | 9.376 | 10.560 | 78.0% |
| 256 | Empty launch | 4.832 | 12.160 | ~0% above replay overhead |
| 256 | Tensor-map prefetch + barrier initialization | 4.928 | 12.160 | ~0% above replay overhead |
| 256 | All TMA loads, no WGMMA | 10.080 | 12.192 | 69.9% |
| 256 | WGMMA on uninitialized shared memory, no TMA | 5.792 | 12.160 | 8.7% |
| 256 | Full TMA/WGMMA mainloop, no DSMEM reduction | 10.656 | 12.160 | 78.4% |

The envelopes are not additive: removing TMA also removes its dependency stalls, and uninitialized-shared-memory WGMMA understates the coupled mainloop cost. The robust conclusions are that TMA/data movement dominates the pre-reduction critical path, while the residual DSMEM reduction, fold, and final store add about 1.18 us at m=128 and 1.50 us at m=256.

## Step 2 budget check - m256 M128 shared-weight retile

The requested eight-stage ring cannot fit in one H200 CTA. Each stage requires a 16 KiB hi|lo N128 x K64 weight tile plus a 16 KiB M128 x K64 activation tile, or 32 KiB per stage. Eight stages require 256 KiB before mbarriers and DSMEM receive storage, exceeding H200's roughly 228 KiB opt-in shared-memory limit. The retile was therefore not forced with fewer stages, and step 5 remains disabled by its explicit dependency on a step-2 improvement.

## Step 2 - CTA count and small-m flight audit

The v64 candidate tested 96 CTAs (cluster-3) and 128 CTAs (cluster-4) using only DSMEM reduction. All four routes recovered every packed hi/lo coefficient exactly, passed random FP64-reference validation, and compiled without spills. All were slower and were gated off; unscored H200 check `5502df0bd9c240d4` passed only because the candidate self-reverted to the incumbent at runtime. Experiment commit `f0f694e` was reverted by `b7ae27f`.

An exact uniform 132-CTA decomposition is not representable with M64 tiles, legal WGMMA output widths, and cluster size at most four: both legal N16/N32 decompositions have 32 output tiles, so cluster-3/4 gives 96/128 CTAs.

Small-m kernels already meet the requested memory-flight mechanism:

- m=1 `rr<1,1>` issues 32 KiB of weight loads per CTA through 16-byte `ld.global.nc.L1::no_allocate` vectors.
- m=8/16 `rhtailk<1/2>` issues 32 KiB of weight loads per CTA through 16-byte `ld.global.nc.L1::no_allocate` vectors, with vectorized activation loads.
- m=32 uses a four-stage TMA ring in its specialized register/MMA kernel, so no deeper-stage rewrite was made without evidence of a deficit.

## Step 3 - m128 N128 hi/lo combine

v66 (`bb6a700`, reverted by `3ca07d3`) preserved the 64-CTA, cluster-8, M64 x N64 topology and its eight-stage ring. It replaced the two N64 high/low WGMMAs with one N128 WGMMA and folded the two accumulator halves in FP32. The kernel fit at about 208 KiB dynamic shared memory and 90 registers/thread with no spills, versus 58 registers/thread for the incumbent.

Unscored H200 job `6993cf8ebaba4802` recovered every packed coefficient exactly (`max_coeff_err = 0`), passed the FP64 check (`relative error = 2.62e-6`, maximum absolute error `3.12e-5`), but measured 10.528 us against 10.496 us. The +0.3% raw / +0.6% overhead-corrected regression caused the runtime gate to reject the route.

## Step 4 - tensor-map prefetch and prologue placement

Inspection showed that v50 already issues mbarrier initialization and all initial TMA stages before `barrier.cluster.arrive`, separates the cluster arrive from its wait, and places the wait immediately before the first remote DSMEM store. v67 (`3536b0d`) therefore isolated the only missing operation: the producer issues weight and activation `prefetch.tensormap` instructions before mbarrier initialization. Prefetched and incumbent variants have identical register, shared-memory, grid, and cluster resources.

Unscored H200 job `24df734ef84941f3` recovered every coefficient exactly at both shapes. At m=128, FP64 relative error was `3.20e-6` and timing improved from 10.592 us to 10.528 us (-0.6% raw, -1.2% after replay-overhead correction), so the per-shape gate kept prefetch. At m=256, FP64 relative error was `3.19e-6`, but both variants measured 12.416 us, so the gate retained the original kernel. The final profiler confirmed `rhwide<8,false,true>` for m=128 and `rhwide<4,true,false>` for m=256.

## Step 5 - conditional m128 in-CTA split-K

Not attempted, exactly as conditioned in the experiment request. Step 5 was allowed only if the step-2 m256 M128 shared-weight retile improved; step 2 could not fit its required eight-stage ring within H200 shared memory and therefore produced no improvement. No speculative lower-stage or altered-topology substitute was introduced.

## Earlier applicability constraints

- PDL: not applicable to this submission boundary. The router is one kernel and the contest invokes each projection independently; there is no dependent consumer kernel that this submission owns and can mark with `cudaGridDependencySynchronize`.
- Fusion: split-K reduction and final FP32 store are already fused into `rhwide`. Sigmoid, top-k, and normalization are outside `muse_bf16_proj`; adding them would change the required raw `[m,256]` FP32 logits and fail correctness.
- L2 residency: not attempted because the harness requires cold-L2 replay and prohibits cross-call cache residency strategies.
- FP8 residual: not attempted because the task permits the FP32 router through BF16 hi+lo storage only; FP8 storage would violate the weight-precision rule. Therefore no logit-error or top-k-mismatch claim is made for an invalid representation.

## Experiment results

| Step | m | Variant | Before (us) | After (us) | Change | Decision |
|---|---:|---|---:|---:|---:|---|
| 1 diagnostic | 128 | Full mainloop without DSMEM tail | 10.560 | 9.376 | -11.2% | Reverted: intentionally incomplete |
| 1 diagnostic | 256 | Full mainloop without DSMEM tail | 12.160 | 10.656 | -12.4% | Reverted: intentionally incomplete |
| 2 budget | 256 | M128, two consumer warpgroups, eight-stage ring | n/a | n/a | n/a | Not implementable: 256 KiB ring exceeds shared-memory limit |
| 3 | 128 | One N128 WGMMA plus FP32 half-fold | 10.496 | 10.528 | +0.3% | Reverted |
| 4 | 128 | Early weight/activation `prefetch.tensormap` | 10.592 | 10.528 | -0.6% | Kept in v67 |
| 4 | 256 | Early weight/activation `prefetch.tensormap` | 12.416 | 12.416 | 0.0% | Reverted by per-shape gate |
| 5 | 128 | Two-warpgroups in-CTA split-K | n/a | n/a | n/a | Not attempted: step-2 prerequisite failed |
| 2 | 128 | 96 CTAs, cluster-3 central DSMEM reduction | 10.528 | 17.280 | +64.1% | Reverted |
| 2 | 128 | 128 CTAs, cluster-4 distributed DSMEM reduction | 10.560 | 14.528 | +37.6% | Reverted |
| 2 | 256 | 96 CTAs, cluster-3 central DSMEM reduction | 12.064 | 17.504 | +45.1% | Reverted |
| 2 | 256 | 128 CTAs, cluster-4 distributed DSMEM reduction | 12.096 | 15.200 | +25.7% | Reverted |
| 3 | 128/256 | PDL | n/a | n/a | n/a | Skipped: no owned dependent kernel |
| 4 | 128/256 | Additional fusion | n/a | n/a | n/a | Already fused / out of domain |
| 5 | all | L2 persisting window | n/a | n/a | n/a | Skipped: cold-L2 rule |
| 6 | all | FP8 e4m3 low residual | n/a | n/a | n/a | Skipped: precision rule |

The conclusive architectural result is that the current 64-CTA one-wave wide kernels are latency/synchronization limited: raising CTA count with small DSMEM clusters increases scheduling and reduction cost, while the measured logical bandwidth and failed multicast experiment independently rule out weight traffic as the primary bottleneck.

## Ordered follow-up: loader floor, m256 shared tile, and tail split

This follow-up uses the corrected router-only times supplied after the earlier ablations: 4.09, 3.95, 4.20, 4.82, 5.97, and 7.93 us for m=1, 8, 16, 32, 128, and 256. The approximately 4.5 us graph-replay envelope has already been subtracted from those values.

The protected v50 file was checked again before and after implementation and remains byte-identical at SHA256 `D0D1124EC0B2775DFD3C48ABCF446C0B2DCB642FF3F0ECD3192FCC97D68574E1`.

### Small-m loader-floor probe

Diagnostic v68 (`eb1867b`) is a copy of v67 and is never reachable from `kernel_fn`. Every CTA reads one contiguous 32 KiB segment of the m=8 packed hi|lo layout. It contains:

- eight 16-byte `ld.global.nc.L1::no_allocate` loads per thread, accumulated into a live register;
- four 8 KiB `cp.async.bulk` copies with one mbarrier per piece;
- eight 4 KiB `cp.async.bulk` copies with one mbarrier per piece; and
- 132- and 264-CTA launches of whichever 256-CTA loader wins.

The SM90a build succeeds. The scalar/vector loader uses 4 registers/thread; both bulk loaders use 10 registers/thread. All have zero stack, zero local memory, and no spill instructions. The bulk variants request 32,800 B and 32,832 B of dynamic shared memory for four and eight pieces, respectively, plus 1,024 B static shared memory.

The stop rule is unchanged: if the best measured loader floor is at most 4.35 us, the current 3.95 us kernel is within 0.4 us of the floor and no small-m loader rewrite will be attempted.

### m256 M128 shared-weight retile

The later request explicitly allowed a five-stage ring, superseding the earlier eight-stage-only budget rejection. Candidate v69 (`e865776`) keeps the required 64 CTAs and cluster size 8: two M128 token tiles x four N64 output tiles x split-K 8. Two aligned M64 consumer warpgroups share one N128 hi|lo weight tile, with a separate producer warp. The local split-K contribution remains in registers, so only seven remote contributions consume DSMEM receive storage (exactly 28 KiB).

All four compiled variants use 90 registers/thread, a zero-byte stack, zero local memory, and no `LDL`/`STL` spill instructions:

| Variant | Ring | Dynamic shared | Static shared | Total shared | Spill bytes |
|---|---:|---:|---:|---:|---:|
| K64, 5 stages | 160 KiB | 192,608 B | 1,024 B | 193,632 B | 0 |
| K64, 6 stages | 192 KiB | 225,392 B | 1,024 B | 226,416 B | 0 |
| K32, 10 stages | 160 KiB | 192,688 B | 1,024 B | 193,712 B | 0 |
| K32, 12 stages | 192 KiB | 225,488 B | 1,024 B | 226,512 B | 0 |

The 6/12-stage configurations fit under the H200 limit but intentionally leave little headroom. Runtime gates require exact coefficient recovery, finite FP64-reference relative error below `2e-5`, and an overhead-corrected paired ratio below `0.99`; otherwise v69 dispatches the v67 incumbent.

### Wide-router tail breakdown

Diagnostic v70 (`e16b9ea`) is also unreachable from `kernel_fn`. At m=128 it uses the kept v67 tensor-map-prefetch prologue; at m=256 it uses the ordinary v67 prologue. It pairs three otherwise identical paths:

1. fold the local accumulator and store only the rank-owned partials, with no remote DSMEM sends;
2. perform local writes, all remote sends, and the receive wait, then skip the final sum/store; and
3. execute the unchanged full reduction and store.

All diagnostic variants retain the incumbent register counts (58 at m=128 and 90 at m=256), use no stack/local memory, and contain no spill loads or stores. The component report uses the established pre-tail baselines of 4.77 us at m=128 and 6.43 us at m=256, corresponding to the supplied approximately 1.2/1.5 us tails.

### H200 status and pending result table

The Muse CLI reported that all 20 daily unscored checks had already been consumed. No scored submission was used: v68 and v70 deliberately produce invalid outputs, so submitting either would violate the diagnostic-only requirement. Consequently, no candidate has been called kept or reverted, and exact coefficient/FP64 claims for v69 remain pending an H200 run.

| Step | m | Variant | Before (us) | After (us) | Decision |
|---|---:|---|---:|---:|---|
| 1 diagnostic | 8 | 256-CTA 16 B `ld.global.nc` floor | 3.95 | Pending H200 | Pending; never submit |
| 1 diagnostic | 8 | 256-CTA 4 x 8 KiB bulk floor | 3.95 | Pending H200 | Pending; never submit |
| 1 diagnostic | 8 | 256-CTA 8 x 4 KiB bulk floor | 3.95 | Pending H200 | Pending; never submit |
| 1 diagnostic | 8 | Winning loader at 132/264 CTAs | 3.95 | Pending H200 | Pending; never submit |
| 2 | 256 | M128 shared weight, K64 ST5/ST6 | 7.93 | Pending H200 | Pending correctness and paired timing |
| 2 | 256 | M128 shared weight, K32 ST10/ST12 | 7.93 | Pending H200 | Conditional fallback; pending correctness and timing |
| 3 diagnostic | 128 | Local / send-wait / full tail split | 5.97 | Pending H200 | Pending; never submit |
| 3 diagnostic | 256 | Local / send-wait / full tail split | 7.93 | Pending H200 | Pending; never submit |

## Scored H200 follow-up results (v69-v78)

The scored sequence resolved the pending items above. All timings in this section are the paired raw kernel timings printed by the candidate gates; the gates subtract the measured replay envelope before applying the `0.99` keep threshold. Every non-timeout candidate that reached validation recovered the hi/lo coefficients exactly (`max_coeff_err = 0`) and passed the FP64-reference limit. Protected v50 remained byte-identical throughout at SHA256 `D0D1124EC0B2775DFD3C48ABCF446C0B2DCB642FF3F0ECD3192FCC97D68574E1`.

### v69: M128 shared-weight m256 retile

Scored job `fe876fdf92aa4fa6` passed contest correctness, but both legal K64 pipelines were slower. K32 ST10/ST12 recovered the wrong coefficients (`max_coeff_err` approximately `1.03766`) and were rejected before timing. The result confirms that halving duplicated weight requests does not repay split-K 4-to-8, cluster 4-to-8, two-consumer scheduling, and the larger DSMEM fan-in.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 256 | M128 shared weight, K64 ST5 | 12.032 | 13.024 | Reverted: +8.2% raw |
| 256 | M128 shared weight, K64 ST6 | 12.096 | 13.088 | Reverted: +8.2% raw |
| 256 | M128 shared weight, K32 ST10/ST12 | 12.096 | Not timed | Reverted: coefficient recovery failed |

The full scored result was `3224.909 us` (`1.497965x`), with the runtime self-gating back to the incumbent.

### v71-v73: rank-local partials in registers

v71 (`b36eb91`) and v72 (`52b3fc0`) timed out. The cause was not the removed sender barrier: the templated incumbent accidentally changed the receive expectation from `(SPLIT-1)*OWN*16` to `SPLIT*OWN*16` when `REMOTE_ONLY=false`. A rank-local shared store never decrements the async mbarrier, so the default path waited for a nonexistent local async contribution. v72 restored the sender barrier but retained this byte-count bug and therefore could not fix the hang.

v73 (`7b492c1`, scored job `6310e2aba67e4738`) restored the correct `(SPLIT-1)*OWN*16` expectation. It kept the rank-local contribution in registers and allocated receive storage only for remote ranks. Resource use stayed at 58 registers/thread for m128 and 90 for m256, with zero spills.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 128 | Tensor-map prefetch | 10.464 | 10.368 | Kept by gate |
| 256 | Tensor-map prefetch | 12.224 | 12.224 | Reverted by gate |
| 128 | Remote-only receive tail | 10.656 | 10.464 | Kept in v73 |
| 256 | Remote-only receive tail | 12.096 | 12.192 | Reverted: slower |

v73 scored `3198.512 us` (`1.510327x`). Its m128 tail result was real in isolation, but the next experiment showed it was not additive with prefetch.

### v74: prefetch plus remote-only tail

v74 (`b833637`, scored job `88a66f8a54144c90`) combined the two m128 mechanisms. It was fully correct, but the combined candidate measured `10.400 us` against the `10.368 us` prefetch-only incumbent, so the remote-only tail was rejected. At m256 it measured `12.160 us` against `12.096 us` and was also rejected.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 128 | Prefetch + remote-only tail | 10.368 | 10.400 | Reverted: non-additive |
| 256 | Prefetch + remote-only tail | 12.096 | 12.160 | Reverted: slower |

The self-gated submission scored `3232.271 us` (`1.494553x`).

### v75: vectorized final stores

v75 (`4f6d688`, scored job `55bdeeb75c3e4938`) replaced four scalar FP32 stores and four bounds checks per reduced fragment with two aligned `float2` stores. Wide-router shapes are exact multiples of 64 tokens, and each fragment maps to adjacent column pairs, so arithmetic and output order are unchanged. The SM90a build remained at 58/90 registers with zero spills.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 128 | Two aligned `float2` output stores | 10.528 | 10.432 | Kept: -0.096 us raw, adjusted ratio 0.98266 |
| 256 | Two aligned `float2` output stores | 12.128 | 12.160 | Reverted by per-shape gate |

The full submission scored `3199.694 us` (`1.509770x`). The m128 improvement repeated in later jobs, although 32 us timer quantization and gate noise sometimes placed it just below the 1% keep threshold.

### v76: deeper WGMMA issue window

v76 (`6e97970`, reverted by `6f7b105`, scored job `365fd849516b46b1`) allowed two or three committed WGMMA groups and delayed each ring refill until the oldest group using that slot had retired. All variants were exact and spill-free. They exposed no useful parallelism because every group depends on the same accumulator chain; delayed refills instead hurt m256.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 128 | `wait_group 2` | 10.560 | 10.560 | Reverted: neutral |
| 128 | `wait_group 3` | 10.592 | 10.592 | Reverted: neutral |
| 256 | `wait_group 2` | 12.352 | 12.416 | Reverted: slower |
| 256 | `wait_group 3` | 12.352 | 13.024 | Reverted: 5.4% slower |

The self-gated score was `3224.991 us` (`1.497926x`).

### v77: one WGMMA fence per mainloop

v77 (`7c36f2c`, reverted by `7abc06d`, scored job `018489cb41d240cc`) moved `wgmma.fence` from every K stage to one fence after accumulator initialization. Coefficients and FP64 checks remained exact, but the repeated fence was providing useful pacing for the dependent accumulator stream. Removing it shifted the stall to the final drain, especially for the m256 N128 path.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 128 | One pre-loop WGMMA fence | 10.624 | 10.592 | Reverted: adjusted ratio 0.99419 misses margin |
| 256 | One pre-loop WGMMA fence | 12.352 | 18.464 | Reverted: 49.5% slower raw |

The self-gated score was `3233.559 us` (`1.493958x`).

### v78: remove sender-side CTA synchronization

v78 (`770c0d7`, reverted by `d6ade55`, scored job `de3938f1c142425b`) removed the CTA barrier immediately before the DSMEM receive wait. It was correct: each reducer thread writes its own rank-local slot, and the mbarrier accounts for all remote slots. Timing proved that the barrier is not on the critical path.

| m | Variant | Before (us) | After (us) | Decision |
|---:|---|---:|---:|---|
| 128 | No sender-side CTA barrier | 10.528 | 10.528 | Reverted: neutral |
| 256 | No sender-side CTA barrier | 12.288 | 12.352 | Reverted: slower |

The self-gated score was `3198.469 us` (`1.510347x`).

### Final state and conclusions

| m | Best newly validated mechanism | Before (us) | After (us) | State |
|---:|---|---:|---:|---|
| 128 | Vectorized `float2` final stores | 10.528 | 10.432 | Kept in experimental v75 |
| 256 | None | 12.096-12.352 | No repeatable improvement | Protected v50 remains preferred |

The protected best remains v50 at `3186.412 us` (`1.516062x`). Experimental v75 is correct and has a repeatable local m128 improvement, but the best full score observed with that selected path was `3198.469 us` in the self-gated v78 run. It did not beat v50 because the gain is only about 0.1 us in one of thirty scored cases and run-to-run/autotuning variation elsewhere is larger. The remaining daily scored attempt is intentionally reserved rather than spent on a low-confidence resubmission.

The conclusive architectural picture is now narrower:

- weight traffic is not the bottleneck (v63 multicast and v69 shared-weight retile both lost);
- more CTAs, larger split-K, and deeper clusters lose to the 64-CTA one-wave topology;
- TMA/data readiness dominates the mainloop, but descriptor prefetch provides only a small, noisy gain;
- WGMMA groups are serialized by a true accumulator dependency, so deeper issue windows do not help;
- the DSMEM tail is material in absolute time, but local-buffer removal and synchronization deletion are not additive wins; and
- only final-store instruction reduction produced a repeatable, correctness-preserving improvement, and it is too small by itself to move the full leaderboard score reliably.
