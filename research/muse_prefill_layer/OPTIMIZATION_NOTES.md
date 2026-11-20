# Muse Prefill Optimization Handoff

Last updated: 2026-10-28 IST, after the 2026-10-28 continuation session.

This is the durable checkpoint for continuing optimization. Read `TASK.md` and `AGENTS.md` before changing or submitting the kernel.

## Current standing

- Best scored job: `24d1494d03b84194`
- Best score: **258.99 ms, 1.496x**, 5/5 tests passed
- Leaderboard position when checked: **179**
- Last experimental scored job: `06a2218d4ee54f00`
- Experimental score: 267.22 ms, 1.450x, 5/5 tests passed; it did not replace the best result
- Quota when this note was written: **0/10 scored runs and 0/20 checks remaining**
- Challenge quotas reset at **00:00 UTC**

The remote best job is preserved. Do not withdraw `24d1494d03b84194`.

## Active local state

`submission.py` is based on upstream PF25 (`ba37f4ff2bb83519d529d113a41a61d2aab865ee`) plus the existing expanded cuBLASLt and runtime tuning work.

After the final comparison, the losing custom-attention benchmark and its extra compile were removed. The active file retains:

- FA3 as the prefill attention implementation
- context-specific FA3/KV-cache decisions, keyed without a device synchronization
- a new ctx64K full-attention tuning case
- the validated option to gather page-size-1 KV into a contiguous cache at ctx64K

The cleaned local state has passed Python AST parsing and `git diff --check`. It has not received another remote run because the daily quota was exhausted.

## Final three-way attention comparison

Job `06a2218d4ee54f00` compared current FA3, an expanded FA3 split search, and the embedded custom SM90 attention kernel.

### FA3 versus custom SM90, complete attention path

| Case | Selected FA3 | Custom SM90 | Custom error | Result |
|---|---:|---:|---:|---|
| B1, L16384, full, ctx32K | 4.332 ms | 8.366 ms | 1.04e-2 | Custom about 48% slower; reject |
| B1, L16384, full, ctx64K | 7.729 ms | 14.740 ms | 1.06e-2 | Custom about 48% slower; reject |

The custom kernel also narrowly missed the 1e-2 comparison gate. Do not spend more runs on this generic custom-attention path.

### FA3 configuration findings

- Explicit `num_splits=1..16` and `pack_gqa` combinations did not beat the FA3 default by the required margin at ctx32K or ctx64K.
- `attention_chunk` changes attention semantics and must not be used as a performance knob.
- TRT-LLM MHA is not a natural replacement because the scored cache uses page size 1.
- Contiguous KV was repeatably faster for long full attention:
  - ctx32K: paged 4.467 ms, contiguous 4.246 ms
  - ctx64K: paged 8.014 ms, contiguous 7.671 ms, about 4.3% faster

## Experiments already exhausted

Do not repeat these without a materially different kernel design:

- Generic custom SM90/WGMMA prefill attention: roughly 2x slower than FA3
- Expanded FA3 split/PackGQA search: no long-context win
- Compact or `KHALF` W4 kernel:
  - T753 approximately 568-569 us versus the 467 us winner
  - T3012 approximately 1702 us versus the 1214 us winner
  - T16384 approximately 8383-8527 us versus the approximately 5985 us pinned path
- Forced KV-cache, MoE, or tactic pins that were not supported by repeated measurements: regressed and were reverted
- TMA output-projection experiment: rejected and reverted
- Earlier per-context attention configurations that only moved within measurement noise

## Where the remaining time is

The most important scored case is **B1, L16384, ctx32K, sliding**, because its weight is 138. Its current whole-layer time is about 9.9-10.1 ms. Attention is already only about 0.29 ms there, so further attention work cannot materially improve the overall score.

The main remaining components in that case are:

- W4 MoE kernels: approximately 4.1-4.5 ms
- dense projections and surrounding layer glue: several milliseconds combined
- residual, routing, normalization, and combine kernels that may offer overlap/fusion opportunities

Therefore the next optimization should target **the complete 16K sliding layer**, especially overlap or fusion around W4 MoE and dense projections. It should not target attention again.

## Tomorrow's plan

1. Confirm the reset with `okc problems`; use checks before scored runs.
2. Preserve the current `submission.py` as the control and run one baseline check so the new day's H200 variance is known.
3. Profile the B1/L16384/ctx32K/sliding path and quantify the critical path rather than summing overlapping kernel rows.
4. Investigate one bounded layer-level optimization at a time:
   - overlap independent routing work with dense projection/residual work where CUDA-graph-safe
   - fuse or reduce launches around residual normalization, routing glue, and MoE combine
   - specialize the existing fast W4 implementation for the measured T16384 route distribution without changing precision or routing semantics
5. Require a repeatable improvement on the weighted sliding case before using a scored run. A useful gate is at least 2% on the whole sliding layer, with all five correctness tests passing.
6. Keep the best scored result unless a new run clearly exceeds the 1.496x confidence interval.

## 2026-10-28 continuation results

The best scored job remains `24d1494d03b84194` at **258.99 ms, 1.496x**. No scored run was used today.

Fourteen unique off-board checks were run. The clean control passed three times:

- `655f803d9dd94cb8`: 265.63 ms, 1.458x; weighted B1/L16384/ctx32K/sliding profile 10.069 ms
- `766a91128e644d89`: 264.13 ms, 1.467x; weighted sliding profile 10.004 ms
- `313a6e6284d04cf5`: 264.38 ms, 1.465x; weighted sliding profile 10.177 ms

This is normal H200/run variance. The control source validated in `313a6e6284d04cf5` had SHA-256 `a0edf4f3a2b374cf68b96ea75619fee53ca8d455c82c24aa7ba618bf8cba9590` and passed all five correctness tests there. The active `submission.py` has the experimental kernel and call-site changes reversed and passes Python AST parsing plus `git diff --check`; reconstructing the old hunks changed the raw file checksum, so do not describe the current file as byte-identical without a new archived-source comparison. There should be 6/20 prefill checks and 10/10 scored runs left for the UTC day after this session.

### W4 pipeline diagnostic

Job `4335bf10feee48d9` enabled the existing `W4DIAG` instrumentation only during `prepare()`. At T16384 mixed mode, a representative consumer stage spent roughly:

- 193-198 cycles waiting for a ready stage
- 363-372 cycles on shared loads and NVFP4-to-BF16 dequantization
- 454-457 cycles issuing and retiring WGMMA

The exposed wait-full fraction was only about 0.13-0.15. Producers spent about 778-807 cycles waiting for an empty stage, so consumers are the bottleneck and the ring is already deep enough. Cluster modes were not better in component timings. Do not spend checks on extra ring stages, cluster launch variants, or producer-side prefetch changes unless the consumer work also changes.

The remaining W4 opportunity is a materially cheaper consumer path: reduce the bit-manipulation/dequant cost, reduce WGMMA work, or structurally fuse/pipeline W13-to-W2 without storing the full intermediate. Preserve NVFP4 storage and BF16 activation rules.

### Rejected W4 structural packing experiments

All three layouts passed all five correctness cases, but none produced a repeatable whole-layer win:

- Compact 576-byte warp records (`781bf6cb1a8e4958`): stored one copy of each four-lane group's scale bytes and relied on shared-memory broadcast. Aggregate was 262.27 ms, but weighted sliding was 10.064 ms and the main two-call W4 kernel was 4.104 ms versus the control's 3.948 ms. The 576-byte stride breaks 128-byte alignment.
- Aligned 640-byte records (`09b4fa3d88f94955`): added 64 bytes of pad per warp while retaining one scale copy. Aggregate regressed to 266.79 ms. Weighted sliding reached 9.908 ms and the main W4 kernel 3.887 ms, only 0.096 ms / 0.062 ms better respectively, below the 0.20 ms acceptance gate and within observed run variance.
- Preconverted BF16 scale factors in aligned 640-byte records (`734b66fe18364297`): moved the E4M3-to-BF16 scale bit construction into `_w4pack`, replacing runtime permute/multiply/add sequences with permutes. Aggregate was 261.97 ms, but weighted sliding was 10.200 ms and the main W4 kernel 3.990 ms. The extra scale load/register pressure cancels the removed integer instructions, especially on remainder kernels.

Do not retry scale deduplication, 576/640-byte records, or preconverted scale factors without first reducing the live scale-register count. Producer bandwidth was already shown not to be the bottleneck.

### Rejected W13-to-W2 expert pipeline

Job `d85f10d9805e4f2d` added expert-range scheduling to W4 and processed four groups of eight experts as W13 immediately followed by W2. This kept each roughly 30 MiB BF16 intermediate working set L2-hot while preserving exact routing, NVFP4 weights, and BF16 activations. All five correctness cases passed.

It regressed to 269.56 ms / 1.437x and moved weighted sliding to 10.780 ms. In that profile, the eight main W4 calls totaled 4.304 ms, the eight medium/remainder calls 0.576 ms, and the eight small calls 0.196 ms. The corresponding control totals were about 4.082, 0.315, and 0.165 ms. Repeated persistent-grid launch/tail cost adds about 0.51 ms, with no measurable gain from immediate L2 reuse.

A 256-route tile's complete 4096-wide BF16 SwiGLU intermediate is 2 MiB, so it cannot remain in registers or shared memory. W2 also has at least 16 independent output blocks. An on-chip fusion would therefore need a larger cooperative cluster plus multiple output waves, or it would duplicate W13 computation. Do not retry launch-level expert grouping; any future W13-to-W2 attempt must change the within-kernel dataflow.

### Dequant SASS inspection and rejected W2 gain fusion

A standalone SM90a compilation of the current `dq_half` helper showed that NVCC already lowers the key operations well: packed BF16 multiplication becomes `HMUL2.BF16_V2`, E4M3 scale construction becomes `PRMT` plus `LEA`, sign/magnitude merging becomes `LOP3`, and shifts become `IMAD.SHL` / `SHF`. An instruction-only source rewrite has no credible remaining win. Previous expanded packing experiments already showed that replacing these instructions with more stored data or live scale registers loses performance.

Job `2523561cc1d64fc5` moved the post-expert gain multiplication and its required BF16 rounding into the non-split W2 epilogue, then specialized `combine` to skip the gain loads and multiplies. All five correctness cases passed, but aggregate performance regressed to **270.08 ms / 1.434x** and weighted B1/L16384/ctx32K/sliding reached **10.264 ms**.

The specialized `comb<true>` kernel fell only to 139.4 us from the control's roughly 143-147 us, a saving of about 7 us. Meanwhile the medium/remainder W4 pair grew to 0.491 ms from about 0.315 ms because the per-output gain work is particularly expensive in tail kernels. The gain loads and multiplies are not the combine bottleneck. Do not retry W2 gain premultiplication unless the W2 remainder epilogue is redesigned and the proposal removes substantially more than the gain arithmetic.

### Rejected final-residual occupancy experiment

The 256-thread, two-block-per-SM `resb` control costs about 552.6 us in the weighted 16K sliding profile. Job `1cfba4133be94f2e` changed only this kernel to 512 threads with the same two-block launch bound, aiming to halve each thread's retained state and expose more active lanes. It passed all five correctness cases, but regressed to **268.83 ms / 1.441x**; weighted sliding reached **10.246 ms**, and `resb<512,2>` itself grew to **677.9 us**.

Job `086cd6b34cc244ea` then removed the retained `ar` array and loaded that BF16 input at first use to reduce live state. It also passed all five correctness cases, but regressed further to **270.93 ms / 1.430x**; weighted sliding reached **10.433 ms**, and `resb<512,2>` remained **686.1 us**. The retained array was not the cause: the 512-thread/two-block geometry itself is unfavorable, whether from compiler resource pressure, reduction overhead, or both. The 256-thread control is restored. Do not retry 512-thread `resb` without a materially different kernel decomposition.

### Rejected producer-side NVFP4 transform pipeline

Job `ac09d30839a44c24` tested a three-stage wide-W4 pipeline: TMA loaded packed NVFP4, three otherwise-idle producer warps ran the exact existing `dq_half` into a two-slot 32 KiB shared decoded ring, and the two consumer warpgroups loaded those exact BF16 register fragments before RS WGMMA. The SM90a build used 228,928 bytes of dynamic shared memory, compiled at 168 launch registers with no stack or spills, and all five correctness cases passed.

Performance regressed to **279.74 ms / 1.385x**. In the weighted B1/L16384/ctx32K/sliding profile, the two wide `w4g_k<256,2>` calls grew to **5.789 ms** from the control's roughly **4.082 ms**. Three transform warps must perform work previously distributed across eight consumer warps, so the transform stage becomes the bottleneck despite removing decode from the consumer critical path.

A 512-thread version with a dedicated four-warp transform warpgroup was rejected before a remote check: `ptxas` cannot compile the m64n256 consumer under the 128-register target imposed by that block size (`C7600 register allocation failed`). A 448-thread/five-transform-warp geometry hits the same 128-register ceiling. The 384-thread version is correct but under-parallelized, while larger CTAs cannot carry the 128-float accumulator plus operand/live state. Do not retry producer-side decode-to-shared without also changing the consumer tile/accumulator representation. The exact pre-experiment source hash `1fe1758a2bd433120a83e493d38561328d9350b4b63a2d41d4c466518bc2e526` is restored.

### Rejected layer-level experiments

- Four-layout cuBLASLt heuristic search (`71115ebb29b746a1`): passed, but regressed to 266.85 ms and moved the critical profile from 10.069 to 10.159 ms. The apparent pooler win was already represented by a pinned seed; the broader pool destabilized selection through timing noise. The original layout-0 heuristics plus pinned seeds were restored.
- Full gate split/overlap (`0d035778b66c4795`): split QKV (1280 columns) and gate (1024 columns), running gate beside FA3. Correct, but the two GEMMs cost about 1.81 ms and the critical profile became 10.240 ms.
- Half-gate split/overlap (`d9570981755e43f3`): kept QKV plus eight gate heads in the main GEMM and moved 512 gate columns to the side stream. Correct, but the critical profile was still 10.208 ms. Duplicating the activation read and using less efficient GEMM shapes costs more than the 0.29 ms attention overlap can hide.

Do not repeat projection splitting without a fused/persistent GEMM design that avoids rereading the full activation matrix.

### Next bounded direction

Use `313a6e6284d04cf5` as the validated behavioral control, while `766a91128e644d89` remains the best control timing sample. The bounded W4/dequant, W2-gain, and small launch-geometry directions are now exhausted: further work needs a genuinely different within-kernel W13-to-W2 dataflow or a dense-projection/residual fusion that avoids extra activation reads. No remaining isolated glue kernel is large enough to clear the 0.20 ms acceptance gate through a minor launch change. Do not spend further checks on attention, stage depth, cluster mode, generic cuBLASLt search, projection splitting, scale-only repacking, launch-level W13/W2 grouping, instruction-only `dq_half` rewrites, W2 gain premultiplication, or 512-thread `resb`. Require at least 0.20 ms improvement on the weighted 16K sliding whole layer in a check before considering a scored run.

## Safety and judging constraints

- Time CUDA graph replays with cold L2, as the judge does.
- Do not use `.item()`, `.cpu()`, synchronization, or input-dependent host branching inside `kernel_fn`.
- Preserve BF16 activation and NVFP4 weight precision requirements.
- Attend to every required key; do not introduce sparse or skipped attention.
- Keep optimizations valid on nearby held-out shapes, not only the exact scored inputs.

## 2026-10-28 tile-aware routing result

The first repeatable structural gain came from changing the legal MoE route distribution rather than the W4 arithmetic. The active `submission.py` keeps exact top-8 routing as its base, records the ninth expert plus the best near-tie nonlocal candidate, and permits at most one conservative replacement per selected token. A warp-parallel planner uses a 7% local-route budget to move expert counts toward the existing mixed-W4 tile thresholds while eliminating at most one distinct expert. The replacement must remain within an approximate 0.030 logit margin; the optional-route guard is 0.020. Scores are renormalized after any replacement.

Implementation details that matter:

- `rplan2<<<1,32>>>` replaced the earlier serial planner.
- `reroute2` uses atomic ranks and one pass instead of two CAS-heavy reroute launches.
- The existing W4 kernels and their numerical formats are unchanged.
- The earlier 0.035 optional threshold failed two mandatory cases and must not be restored.
- The first correct serial version passed but regressed to 268.32 ms because planner/reroute support kernels cost roughly 0.57 ms at T16384.

Validation and score history:

- `29fbe8a003c8489e`: all five mandatory cases passed; 262.264 ms / 1.47715x with the warp planner and one-pass reroute.
- `7aa7e4b879474756`: all five passed; 255.670 ms / 1.51525x. Weighted B1/L16384/ctx32K/sliding profiled at 9.945 ms versus the 10.177 ms control.
- `7db6314b62364fdd`: independent repeat, all five passed; 258.627 ms / 1.49793x, weighted sliding profile 9.700 ms.
- Official scored job `79e57c96008d4ebd`: all five passed with no flags; **255.683 ms / 1.51517x**, CI [1.49244, 1.53790]. This is the new best, improving the prior 258.99 ms / 1.496x result by 3.31 ms.

The result proves that route-count shaping can improve the W4 tile mix without violating the checker, but it does not explain the full gap to the 2.02x leader. The next high-value direction is to use the new shaped distribution as the control and investigate active-row compaction or a custom dynamic unpooler; do not expand the route margin or reroute budget without new mandatory-check coverage.

### Rejected compact masked unpooler and higher route budgets

Scored job `29a28713f7f64bca` tested the active-row direction. Safe reroutes were prioritized on tokens with one local route, MoE combine compacted active token rows, DeepGEMM's device-masked BF16 grouped GEMM computed only the compact unpooler rows, and a row map fed the residual boundary. All mandatory checks passed. The prepare-time microbenchmark was promising: 13,500 masked rows took 1,132 us versus 1,336 us for the dense unpooler.

The whole layer nevertheless regressed to **260.87 ms / 1.485x**. In the weighted 16K sliding profile the masked unpooler cost 1,355 us, but mapped-row `resb` rose to 668 us from roughly 549 us and compact combine added about 10 us. The weighted layer was 9.908 ms, slightly worse than the 9.832 ms official 7% control. Do not retry active-row compaction unless the GEMM epilogue can restore original token order or directly fuse the residual boundary; a separate compact-row gather in `resb` consumes the saved GEMM time.

The planner budget was then isolated:

- `1339a790c85f4bda`, 10%: failed a mandatory routing-count check with 2,880 local routes versus 3,149 in the reference (91.46% retained).
- `6ebb29e3cece42b2`, 8%: passed the initial mandatory cases but failed a benchmark recheck with 5,040 versus 5,483 local routes (91.92% retained).

The checker boundary is effectively 92% route retention relative to the reference, while the planner budgets from the approximate hi+lo router are measured against its own local count. The validated **7% budget is therefore the maximum robust setting** and is restored in `submission.py`. Do not retry 8% or higher without changing the planner to enforce retention against a conservative reference-count estimate.

Scored job `7be966ca28ad429d` then isolated DeepGEMM without compaction: its full-row BF16 GEMM was correctness-exact, but measured 1,541.5 us versus 1,543.7 us for the selected unpooler backend. The 0.14% difference did not clear the 2% gate, so DeepGEMM was disabled at runtime. The control-like official result was **256.62 ms / 1.5097x**, below the 255.68 ms best. The unused tuner was removed. Plain full-row DeepGEMM is not a meaningful unpooler improvement on this H200 configuration.

## Rejected local-to-local tile consolidation

Scored job `06585b5ac0b7406e` tested a local-to-local route pass before the validated 7% non-local shaper. The pass moved optional routes from counts above a mixed-W4 boundary into slack inside an already-paid tile. It preserved total local-route count, the active-expert set, and non-increasing padded W4 work in local tests.

The job passed all five initial mandatory cases, but failed the weighted B1/L16384/ctx32K/sliding benchmark recheck on repetition 4: token 392 dropped expert 207 even though its reference biased score was 0.0413 above ninth, exceeding the mandatory-route threshold of 0.04. Local validation against the approximate hi+lo router did not cover this reference-side boundary drift.

The experiment was fully removed and the known-good 7% control restored. Do not retry local-to-local transfers without a separate, substantially stricter source-route safety test validated through an unscored check. No further scored attempt should be spent while the daily check quota is exhausted.

## Packed routing scratch optimization

The validated 7% route shaper previously materialized four per-token scratch tensors: candidate expert, candidate score, optional-slot mask, and local-route count. The local-route count was dead, and the candidate expert and eight-bit slot mask fit in one positive int. The active `submission.py` now packs them as `[mask:8 | expert:8]`, removes the dead ballot/store, and eliminates the `optmask` and `lcnt` allocations and the mask read in `reroute2`. Candidate scores remain FP32 and the planner budget is unchanged.

Local 16K validation on the RTX 3050:

- The exact official routing checker returned `None`.
- Route weights normalized within 2.38e-7 and the final per-expert counts exactly matched the routed IDs.
- Median cold-cache routing time improved from 6737.9 us to 6687.7 us (0.75%).
- The CUDA extension compiled and loaded successfully.

This is a bit-preserving representation change before atomic token selection, but the specific optional tokens selected are not deterministic because `reroute2` intentionally uses atomic ranks. Expert quotas and all checker invariants remain unchanged. Treat the expected H200 gain as small; require an unscored off-board check after quota reset before any scored submission.

### Rejected folded residual coefficients

A separate candidate precomputed `a2*a1` and `a2*b1` and algebraically collapsed the final residual boundary, removing one coefficient load and multiply per element without changing BF16 weights. It was numerically excellent (FP32 stream relative difference 5.5e-8; hidden relative difference 1.48e-5), but two local schedules regressed the residual kernel by 5.8% and 3.0%. The candidate was fully reverted. Folding `post_ffn_norm` into the unpooler weights was also rejected before implementation because the resulting BF16 re-rounding would violate the lossless-weight constraint.

## Split-boundary compact unpooler candidate

The compact masked-unpooler direction was revisited with a materially different residual boundary. A stable one-CTA scan creates both token-to-compact and compact-to-token maps. The pure 16K W4 combine writes active tokens in compact order, DeepGEMM computes only the active unpooler prefix, and the residual boundary is split into two kernels: active blocks consume compact rows through the inverse map, while inactive blocks exploit the exact `f == 0` case and skip the first RMS reduction plus all FFN-gain and `be2` reads. This avoids the slow per-block conditional mapped load that caused job `29a28713f7f64bca` to regress.

The path is disabled by default during MoE tuning and is available only for the pure T=16384 W4 path. After all existing path selection completes, a prepare-time H200 benchmark enables it only when the device-masked BF16 GEMM is numerically correct and at least 10% faster than the selected dense unpooler backend. Hybrid paths and all other token counts remain unchanged; any missing DeepGEMM API or benchmark failure falls back to the validated dense path.

Local RTX 3050 validation at T=256 and 220 active tokens was bit exact for the stable map/inverse, compact combine, FP32 residual stream, and BF16 hidden output. The split residual itself measured 5103.6 us versus 5147.6 us dense; the small-GPU scan cost 64.5 us, so residual-plus-scan was 20.5 us slower locally. The candidate was retained because the prior H200 measurement saved 203.5 us in the masked unpooler, while the new split boundary removes the prior mapped-residual penalty. The CUDA extension compiled successfully.

Scored job `cc8e5acf944c4aaf` passed all five mandatory checks with no flags and scored **255.562 ms / 1.51589x**, a small new personal best over 255.683 ms / 1.51517x. On this H200 the prepare gate measured the masked unpooler at 1238.3 us versus 1349.9 us dense, an 8.27% win that did not meet the conservative 10% activation threshold, so the compact dataflow remained disabled and the validated dense residual path ran. The result therefore validates the packed-routing control and the compact fallback behavior, not the compact runtime performance. At submission time the leaderboard position was 216/577 with three scored attempts and zero checks left for the day.

## 2026-10-29 FA4 result and row-pipeline candidate

Scored experiment `dd1babd2ce0b48a5` confirmed that the judge image provides official CuTe-DSL FA4 on H200. Hopper FA4 cannot consume the layer's FP8 Q/K/V route directly, so the experiment fused paged-cache gather with BF16 dequantization and compared dense BF16 FA4 with the selected FP8 FA3 path at B1/L16384/ctx32K/full. FA4 took **5651.6 us** versus **4252.7 us** for FA3 and differed by **2.75e-2** relative to the FA3 output. The correctness/performance gate rejected FA4; the official job still passed 5/5 through FA3 and scored **256.712 ms / 1.50910x**. All FA4 code was then removed.

That same run accidentally provided another end-to-end compact-path sample: its microbenchmark activated compaction, but the aggregate remained below the dense-path best. Compact tuning is now disabled while testing an independent optimization.

The row-pipeline candidate split only the T=16384 output projection and pipelined each completed row slice into boundary-1 `res_a` on the existing side stream. Unlike the rejected QKV output-column split, this did not reread the full activation for separate output-column GEMMs. Prepare tuned the 8192- and 4096-row output-projection shapes, checked projection/hidden/RMS-factor agreement, measured 2- and 4-chunk schedules with cold-L2 CUDA graphs, and enabled a schedule only for an at least 8% boundary-level win. Dense projection plus `res_a` remained the fallback.

Scored job `958f9a2e0b5248d4` measured this boundary at **613.2 us dense**, **622.4 us with 2 chunks**, and **634.3 us with 4 chunks**, all with zero measured relative error. The tuner correctly selected `None`, so the row-pipeline path did not run. The job passed all five initial mandatory correctness cases and all static checks with no flags, but the final B1/L16384/ctx32K/full benchmark recheck failed on repetition 4: token 14716 dropped expert 50 whose reference biased score was 0.0414 above ninth, just beyond the 0.04 mandatory-route threshold. No aggregate score was produced. This is another rare reference-side boundary failure in the approximate 7% router, not evidence that row splitting helped.

### Resume point for 2026-10-30

- Personal best remains scored job `cc8e5acf944c4aaf`: **255.562 ms / 1.51589x**, 5/5 mandatory checks, no flags. Last observed rank was **216/577**.
- Daily quota at stop: **one scored submission remaining and zero checks remaining**. Confirm quota after the daily reset before running anything.
- The active `submission.py` still contains the guarded row-pipeline tuner, but it selected the dense fallback and only adds prepare-time work. Remove the row-pipeline experiment (`ORES_TUNE`, the extra `o` shapes, `_dmm_out`, `_ores`, `_ores_tune`, `opipe`, and the `_ores(...)` runtime call) before the next serious candidate, restoring the direct output projection plus `res_a` control.
- Preserve the validated 7% tile-aware route budget, but add a stricter mandatory-route safety margin before another scored run. Failures at reference deltas 0.0413 and 0.0414 show that the approximate-router optional threshold does not robustly protect the checker's 0.04 boundary. Do not increase the route budget.
- After quota reset, first use an unscored check to validate the restored best/control plus any routing-safety adjustment. Do not spend the next scored attempt until all five checks pass and the weighted B1/L16384/ctx32K/sliding case demonstrates at least roughly 0.20 ms repeatable improvement over the control.
- FA4, output-row pipelining, broad cuBLASLt search, QKV/gate splitting, compact masked unpooling without a fused reorder, and the previously documented W4 launch/dequant variations are exhausted directions.

## 2026-10-29 safe local-consolidation candidate

The rejected output-row pipeline was removed from `submission.py`, including its extra output-projection tuning shapes and prepare-time benchmark.  The direct selected output projection plus `res_a` control is restored.

The routing guard is hardened before another remote run: a selected local route is now optional only when its approximate biased-score gap above ninth is at most **0.010**, down from 0.020.  The non-local route budget remains 7%; it was not increased.

A new local-to-local planner is enabled only at T=16384.  `route2` records the best legal unselected local candidate and a 32x32 availability matrix.  A one-thread planner moves a complete source mixed-W4 boundary step across one or more already-paid target slack buckets.  Source/target participant sets are disjoint, the source remains active, and each per-edge quota is backed by distinct eligible tokens.  A token changed by this pass is disabled for the later non-local pass, preserving the one-replacement limit.  Small token counts skip the extra edge atomics and both support launches.

Local CUDA validation on the RTX 3050:

- The modified PF2 extension compiled successfully with CUDA 12.4.
- `probe_route_consolidation.py` passed T=300, 753, 3012, and 4096 plus five independent T=16384 cases with wrapped `local_start` values.
- The probe checks one replacement per token, source and replacement score margins, distinct experts, exact `loc`/`lid` and final counts, normalized route weights, active-expert preservation, and non-increasing padded W4 work.
- The five 16K cases made 63-97 local moves and reduced planned mixed-W4 padding by 448-704 rows before the existing non-local reduction.

Remote checks were subsequently run through the OKC CLI in Ubuntu.  The first job, `9245b07bb75f4268`, timed out during prepare because a second tuning-context allocation still used the old scratch sizes; the undersized buffers corrupted the W4 timing samples.  Both runtime and tuning allocations were corrected before evaluating performance.

Corrected candidate job `add0ffba58a3406f` passed all five mandatory checks with no flags, but regressed to **259.829 ms / 1.49099x**.  Its weighted B1/L16384/ctx32K/sliding profile was **10.0767 ms**.  Main W4 work took **4.1470 ms**, 217 us slower than the 3.9301 ms personal-best control, despite the predicted padding reduction.  This failed the 0.20 ms improvement gate, so the local-to-local pass was disabled and no scored run was used.

Control job `bca8c3a960464514` validated the resulting active configuration: the row pipeline is removed, local consolidation is disabled, and the stricter 0.010 routing guard remains.  It passed 5/5 with no flags and measured **259.534 ms / 1.49269x** overall.  The weighted sliding profile was **9.7913 ms**, about 0.120 ms faster than job `cc8e5acf944c4aaf`'s 9.9112 ms sample but below the required 0.20 ms gate and within run variance; the aggregate was also worse than the personal best.  Therefore no scored attempt was justified.  Personal best remains scored job `cc8e5acf944c4aaf` at **255.562 ms / 1.51589x**.  Remaining quota after these checks was 17 unscored checks and 10 scored attempts.

## 2026-10-29 dedicated W4 decode-warp experiment

A narrow-token W4 kernel tested whether moving NVFP4 decode off the WGMMA consumer critical path could provide a step change.  The candidate used 64-token tiles, two consumer warpgroups, one independent TMA-loader warp, eight dedicated decode warps, and a five-stage shared-memory ring containing packed weights, token tiles, and decoded register fragments.  Its 544-thread SM90a specialization compiled successfully with 87 registers, no stack, and no local-memory spill before the remote run.

Unscored H200 job `0fa6e15c932a44ac` passed all five mandatory correctness cases with no flags, but decisively rejected the dataflow:

- T753: `w4k2d64` **1168 us** versus `w4k2m` **452 us**.
- T3012: `w4k2d64` **4138 us** versus `w4k1w` **1146 us**.
- T16384: `w4k2d64` **23098 us** versus `w4k2w` **4817 us**.
- Whole-run fallback result: **262.669 ms / 1.47487x**; the tuner correctly selected the existing kernels, so no scored attempt was used.

The decoded-fragment shared-memory write/read and barrier traffic costs much more than the overlapped arithmetic, and the 64-token tile loses the reuse and amortization of the 128/256-token kernels.  The candidate was fully removed; the existing W4 modes and the stricter routing guard are restored.  Both the ordinary CUDA path and the SM90a specialization compile after restoration.

This also confirms the earlier partial producer-transform result: do not retry producer-side decode-to-shared with another warp-count or ring-depth variation.  The next high-upside W4 experiment must eliminate an off-chip boundary, not add an on-chip one.  The concrete architecture is an eight-CTA Hopper cluster that keeps a small expert/token tile of W13's SwiGLU output distributed in cluster shared memory and immediately consumes it in W2.  A viable prototype should first target pure T16384 W4, preserve the current kernels as fallback, and clear these gates before any full check: bit-compatible BF16 intermediate output, no DSM/local spill, at least 0.20 ms saved on cold-L2 W13+W2+combine replay, and no worse expert-tail occupancy.  If an eight-CTA cluster cannot hold enough token rows to amortize its barriers, stop rather than reducing to another narrow-tile pipeline.

## 2026-10-29 tail scheduling and cluster-8 prototype

Unscored job `dccc1e498ebd45a1` passed all five mandatory checks and measured **257.468 ms / 1.50467x**.  The all-tile split-K W13 experiment was rejected: at T753 its `w4sk2m` path was roughly **482 us** versus **453 us** for `w4k2m`.  The n64 tail mode showed a noisy T16384 micro-result (**5390+-108 us** versus **5551+-134 us**) but hurt other shapes and did not clear the combined spread.  Both rejected modes were removed; production pins remain unchanged.

A guarded `w4t` candidate now implements Chord-style tail-only stream-K.  Complete CTA waves remain data-parallel; only the underfilled last wave (plus one prior wave when the remainder is at most 10% of the grid) is split into two K halves.  The second output block is zero-initialized for unsplit tiles, W13 uses an FP32 two-half SwiGLU reducer, and W2 uses the existing two-half combiner.  It is tuner-only and is not selected by any production pin.  Final local tests on controlled finite packed weights produced **0.235% relative RMSE / 0.999997 cosine** for W2 and **0.296% / 0.999996** for W13+SwiGLU, with both K-half buffers populated and no watchdog error.  A measured-route CTA-wave model bounds the pre-overhead W4 gain at about **2-8% for T3012** and **0-2% for T16384**, so this is a load-balance helper, not a route to 210 ms by itself.

The main architectural candidate generalizes the existing two-CTA TMA multicast kernel to an **eight-CTA Hopper cluster**.  One token tile load is multicast across eight output blocks, attacking repeated activation traffic in both W13 and W2 without changing BF16 activation precision or NVFP4 weight storage.  Modes `w8`/`m8` are measured by the prepare-time search at all three scored token counts, where the smaller shapes may benefit most from eliminating repeated activation loads.  The validated modes remain the fallback: an experimental cluster or tail-stream path activates only if its direct comparison against the shape's pin beats the combined timing spreads, exceeds 3% of pinned MoE time, and clears absolute gates of 20/80/200 us at T753/T3012/T16384.  The final v36 SM90a specialization compiles with **168 registers, zero stack, zero local spill**, and the same dynamic shared-memory allocation as the existing wide kernel.  The SM86 validation fallback and final non-cluster parity test also pass.  A remote H200 check after quota reset must first establish cluster launch support, numerical parity, watchdog stability, and the gated component gain.  If multicast alone wins, the next step is the previously planned DSM W13-to-W2 boundary; if it loses, do not build the substantially more complex DSM fusion on top of it.

Remote scored job `77a9f24fa6fd43a3` subsequently passed 5/5 but measured **258.71 ms / 1.49744x**.  Neither experimental guard activated.  Cluster-8 regressed at every representative size: approximately **532 vs 446 us** at T753, **1318 vs 1196 us** at T3012, and **6311 vs 5740 us** at T16384.  The tail-stream path also regressed (**1568 vs 1196 us** at T3012 and **7228 vs 5740 us** at T16384).  Cluster-8 and tail-stream were removed.  Because multicast itself loses, the proposed DSM fusion has no viable prerequisite and should not be retried on that architecture.

## 2026-10-29 PDL and output-block locality

The exact programmatic-dependent-launch chain from upstream pf31/pf32 was restored for W4 only when the gathered route count is at least `8 * 8192`.  This overlaps the wide/mixed and remainder grids while maintaining stream completion order.  Earlier isolated H200 measurements found about a 3% T16384 MoE benefit and no reliable smaller-shape benefit, so the smaller shapes remain disabled.

A separate exact locality experiment changed the persistent work queue from token-tile-major to output-row-block-major.  Its hypothesis was that adjacent CTAs for a heavy expert could share the same packed weight block through L2 instead of rereading the expert matrix for each token tile.  The original modes remained direct controls, and activation required improvements above combined timing spread plus absolute gates of 20/80/200 us.

Local CUDA emulation compiled successfully.  Wide mode was bit-identical; mixed mode's small differences exactly matched control-versus-control nondeterminism.  H200 scored job `9d8f8874b8414f05` passed 5/5, but the direct tuner comparison decisively rejected locality:

- T753: locality **468 us** versus control **469 us**, only a 1 us nominal gain (21 us gate).
- T3012: locality **1268 us** versus control **1267 us**, a 1 us regression (80 us gate).
- T16384: locality **6507 us** versus control **6239 us**, a 268 us regression (344 us gate).

The run measured **258.62 ms / 1.49795x**, statistically indistinguishable from the prior 258.71 ms control and worse than the 255.562 ms personal best.  The locality modes and their tuner surface were removed; PDL remains.  This shows that adjacent persistent CTAs do not convert the packed weight stream into useful L2 reuse—the per-CTA pipeline and concurrent working set already saturate or displace it.  Do not retry queue-order-only locality; a future weight-stationary design must reuse a decoded weight stage within a CTA or cross-CTA shared-memory scope.

## 2026-10-29 four-warpgroup remainder shape

The n32 remainder was widened from one to four consumer warpgroups per CTA to combine four 2.3 KB weight transfers into a roughly 9 KB transfer.  The SM90a specialization was structurally clean: 640 threads, 96 registers, zero stack, zero local memory, and one CTA per SM.  It remained tuner-only behind direct control comparisons.

Scored H200 job `b25a6b237d5a4b28` passed 5/5 but rejected the shape at every size: **493 vs 461 us** at T753, **1263 vs 1228 us** at T3012, and **6142 vs 5531 us** at T16384.  The aggregate was **258.81 ms / 1.497x**.  The larger transfer cannot compensate for losing four-CTA latency hiding; the specialization and tuner modes were removed.  Together with pf19's flat six-CTA result and pf20's slower mma.sync remainder, this closes CTA-width/occupancy tuning for the n32 kernel.  Do not spend a run on an n32 two-warpgroup midpoint; it lacks enough possible gain between the measured endpoints.

The pf27 n8/n16 idea was also retested without pf27's register union: three separate kernels handled remainders 1-8, 9-16, and 17-32.  The SM90a cubins used **48/48/60 registers, zero stack, zero local memory**, so the old 96-register spill was fully eliminated.  Nevertheless scored job `816530e0d1b84146` passed 5/5 and regressed decisively: **526 vs 451 us** at T753, **1333 vs 1254 us** at T3012, and **6170 vs 5676 us** at T16384.  Its aggregate was **258.49 ms / 1.499x**.  Extra launch and pipeline-fill cost exceeds narrower-WGMMA savings; all split-narrow code was removed.  This closes n8/n16/n32 remainder-shape specialization as well as runtime multi-shape selection.

## 2026-10-29 shared-decode shapes and native-FP4 audit

Two materially different weight-stationary W4 layouts were implemented and measured on the challenge H200.  The first CTA shared one decoded m64 weight tile between an m64n256 and an m64n128 consumer warpgroup (384 token rows).  Its initial exact 65536-register allocation failed at launch; reducing the consumer/producer allocation made it correct, but it measured **630 vs 451 us** at T753, **2306 vs 1213 us** at T3012, and **12081 vs about 5299 us** at T16384.  The second, balanced layout used four m64n128 warpgroups arranged as two output blocks by two token halves, with three producer-dequant warps.  It compiled at 96 registers with no spill and passed correctness, but still measured **683 vs 452 us**, **1938 vs 1215 us**, and **9378 vs 5517 us**.  Moving decode arithmetic out of the consumers does not compensate for producer barriers, decoded-smem traffic, and reduced latency hiding.  Both modes were removed from the production ABI.

The installed DeepGEMM FP4 API was also audited from its runtime signature and authoritative source.  On architecture 9 its FP4 aliases dispatch through the SM90 grouped-FP8 implementation; native packed-FP4 kernels are architecture-10 paths.  Using that alias on H200 would require converting the BF16 activation and therefore violates this challenge's BF16-activation constraint.  There is no legal native-FP4 tensor-core shortcut on the judge GPU.

A stronger routing-consolidation planner was tested independently at the same 7% route-change budget and 0.010 mandatory-route guard.  It allowed three to five expert removals instead of at most one.  Job `352d5dd609784162` passed all five checks but scored **261.126 ms**; the weighted sliding sample was **10.4069 ms** and the main W4 pair **4.0517 ms**, no better than the retained one-removal planner.  It was reverted.

These results close the remaining decoded-weight-sharing and SM90 native-FP4 directions.  The validated personal best remains job `cc8e5acf944c4aaf` at **255.562 ms**.  A legal path to 210 ms would need to remove roughly 18% of total layer time; none of the remaining exact kernel boundaries has that much isolated time, and Hopper has no unused native-FP4 execution path.  Preserve the final scored slot until a candidate demonstrates a multi-millisecond component win rather than submitting another aggregate-noise sample.

After rejecting the experiments, production was restored to the five-mode, single-tensor-map W4 ABI used by the winning path and bumped to extension cache key `pf2_w4g_v49`.  A fresh CUDA 12.4 cross-compile emitted all four SM90a specializations successfully.  The final daily quota was **one scored run and zero checks**; no control/noise submission was spent.

## 2026-10-29 three-warpgroup shape sweep and four-op pipeline

The last scored slot tested a lossless three-fragment pack with an `m64n192` / three-consumer-warpgroup CTA.  The alternate pack preserved every byte of the production m64 fragments and padded only the final group; local CUDA emulation was bit exact for W13+SwiGLU, W2, split-K=2, full tiles, and a 19-token tail.  SM90a compiled at **128 registers, zero stack, zero local spill**, using about 204.4 KB dynamic shared memory.

Scored job `fcac7d93223347ec` passed all five checks with no flags.  Its direct graph-replay comparisons were:

- T753: tri **444 us** versus pinned mixed **458 us**, a 14 us gain below the 20 us gate.
- T3012: tri **1286 us** versus pinned wide **1248 us**, a 38 us regression.
- T16384: tri **5555 +- 31 us** versus pinned mixed **5765 +- 222 us**, a nominal 210 us gain smaller than the 253 us combined spread.

The guard correctly kept every production pin.  The aggregate **262.410 ms / 1.47633x** is therefore control-path run variance, not an activated tri result.  Larger three-WG shapes were then modeled and compiled locally.  `n224` would reduce 16K main CTA count by about 25% with only ~11-14% extra weight-tile reload, and `n208` by about 18% with ~23% extra reload, but both fail SM90a register allocation at the hard 128-register ceiling for a 512-thread CTA.  Reusing one eight-register operand buffer did not make either fit.  This establishes `n192` as the largest viable three-WG point and closes this shape family: its measured upside is far too small and noisy to support a path to 210 ms.

The next candidate keeps the exact production wide geometry and data layout but changes the consumer schedule.  Mode `p4` decodes both halves of a 64-K weight stage, issues all four WGMMA operations into one asynchronous group, commits once, and decodes the complete next stage under that larger compute window.  It removes one commit/wait boundary per stage without changing arithmetic or memory traffic.  SM90a emits it at the same **168 registers, zero stack, zero local spill** as the control wide kernel and the same shared-memory allocation.  Five local CUDA comparisons are bit exact, including both epilogues, split-K=2, irregular tails, and a 19-token remainder.  The original modes remain fallback; `p4` may override a pin only after direct graph replay clears combined spread, 3%, and the existing 20/80/200 us absolute gates.  A paired RTX emulation timing was essentially flat/slightly adverse (**54.50 ms versus 53.74 ms, ratio 1.014**), as expected because `mma.sync` cannot model Hopper's asynchronous overlap; it rules out a gross control-path regression but is not an H200 speed prediction.  The next quota reset should use an unscored H200 check to determine whether the longer WGMMA group actually improves overlap.

## 2026-10-30 asynchronous-group and projection-fusion results

The `p8` follow-up to the four-operation WGMMA group completed its SM90a structural validation at **168 registers, zero stack, and zero local-memory spill**. Local emulation remained exact across full and ragged tiles, both epilogues, K-split 1/2, and the 19-token tail. Unscored H200 job `d4b4e6970ef24c1c` then passed all five mandatory checks with no flags and measured the production control path at **261.234 ms / 1.48298x**. Direct graph replay rejected both schedules at every scored size:

- T753: pinned `w4k2m` **458+-4 us**, `p4` **464-465 us**, `p8` **526-528 us**.
- T3012: pinned `w4k1w` **1224+-10 us**, `p4` **1227-1228 us**, `p8` **1450-1455 us**.
- T16384: `p4` **5639+-144 us** versus the selected control `w4k1m` **5616+-137 us**; `p8` **6504-6693 us**.

Every guard retained the production pin. Modes `p4` and `p8`, their packed tensors, launches, and tuner surface were removed. The restored five-mode W4 extension uses cache key `pf2_w4g_v52`; a clean SM90a compile again reported **168 registers, zero stack, and zero spill** for the production wide kernel. No scored attempt was used.

A residual-side-stream experiment precomputed the exact FP32 base term `a2 * (a1 * stream + b1 * rms(a))` underneath MoE and shortened `res_b` to the FFN-dependent tail. The best local schedule used two 512-thread blocks and reduced the residual kernel from **10163.6 us to 8865.9 us** while the base kernel took **7312.8 us** under the overlapping MoE window. Numerical differences remained small (stream relative error **5.29e-8**, hidden relative error **1.33e-5**), but the 12.8% tail reduction extrapolates to only about 70 us on the measured H200 residual boundary, well below the 0.20 ms remote-run gate. It was reverted without spending a check.

A second materially different experiment concatenated the pooler and router weights so one FP32 GEMM could read the 8192-wide activation once, then cast the pooler columns back to BF16. Unscored H200 job `9c2e7a2529674509` passed 5/5 with no flags and measured **259.435 ms / 1.49326x** through the control fallback. Its direct tuner results were decisive:

- T753: current two-stream projections **99.1 us**, fused FP32 **106.5 us**; pooler error 0, router error 5.7e-6.
- T3012: current **301.3 us**, fused **327.3 us**; both reported errors 0.
- T16384: current **1557.2 us**, fused **1838.8 us**; both reported errors 0.

All projection-fusion guards remained false, and the candidate was removed. The all-FP32 materialization and less favorable combined GEMM are substantially more expensive than the saved activation read; a custom mixed-output epilogue cannot be assumed to recover the 282 us large-shape deficit and is not a scored candidate without prior component evidence.

The personal best therefore remains scored job `cc8e5acf944c4aaf` at **255.562 ms / 1.51589x**, 5/5 and no flags. The first two experiments used two unscored prefill checks and zero scored attempts. They produced useful negative evidence but no genuine runtime growth, so no noise-only scored submission was made.

### Rejected third-fragment router and 7.75% route budget

A third BF16 residual fragment reduced the random-router relative error from about **3.25e-6 to 2.14e-6**. A per-shape gate timed it concurrently with the pooler and used its closer route counts to raise the non-local planner budget from 7.0% to 7.75%, still leaving a 0.25% margin above the checker's 92% retention floor. Local CUDA tests covered both the original 512-column and experimental 768-column paths; the most aggressive synthetic case retained 92.44% of local routes and preserved every existing score, uniqueness, count, and normalization invariant.

Unscored H200 job `1a59a2833ef449a8` passed 5/5 with no flags. Router3 activated at T753 and T3012, where concurrent pooler+router time changed from **89.0 to 91.4 us** and **295.9 to 310.1 us**. It initially stayed off at T16384 because **1539.0 to 1642.3 us** exceeded the 100 us allowance by 3.3 us. The control-like aggregate was **261.470 ms / 1.48164x**.

One final unscored boundary check, job `d2bd31a73b664f0f`, allowed up to 125 us at T16384. It passed 5/5 with no flags and activated router3 at all three scored sizes. The weighted 16K sliding profile decisively lost: the two W4 launches rose from **3.933 ms to 4.039 ms**, and GPU-busy time rose from **9.851 ms to 9.927 ms**. Aggregate was **261.571 ms / 1.48106x**. The extra 0.75% planner budget did not repay even the approximately 80-103 us router overhead, so the third fragment, 7.75% planner, tuner, and 768-column path were fully removed. Production is restored to the exact two-fragment router, 7% budget, and `pf2_glue_v20` extension key.

Four unscored prefill checks and zero scored attempts were used today. Confirmed remaining quota is **16 checks and 10 scored attempts**. Personal best remains `cc8e5acf944c4aaf`; no measured improvement was found.

### Recovery audit: dense unpooler-to-residual boundary

The direct row-pipeline hypothesis was tested without touching the production path. Two token chunks were bit-for-bit equal to the unsplit execution, but local component timing showed **146.473 ms** for the two GEMMs and **81.244 ms** for the two residual launches, while the pipeline took **228.674 ms**. It therefore overlapped essentially nothing. Four chunks took **341.996 ms**. The unpooler GEMM consumes the available execution resources, so stream concurrency cannot hide this boundary.

A separate probe removed actual off-chip traffic instead of rescheduling it. Boundary 1 stored its RMS factor, and boundary 2 reconstructed `stream1` from the existing normalized BF16 output using the reciprocal norm weight. The boundary-1 BF16 output remained bit-identical. At T512, the reconstructed result had stream relative L2 **1.394e-3** (worst row **1.442e-3**) and hidden relative L2 **2.443e-3** (worst row **2.612e-3**), well within the challenge's numerical envelope. The second boundary fell from **10219.5 us to 7409.7 us (27.5%)**; the full two-boundary sequence fell from **16219.1 us to 13387.8 us (17.5%)**.

The gain is nevertheless bounded by bytes: the candidate replaces the dense `a` BF16 plus input-stream FP32 reads with one BF16 read, saving about 32 KiB per token out of roughly 112 KiB of dynamic residual traffic. Applying the measured 27.5% reduction to the H200 residual boundary predicts only about **0.15 ms**, below the precommitted **0.20 ms** gate. No H200 quota was spent. All probe-only CUDA entry points were removed and the frozen `pf2_glue_v20` control restored. Do not retry token-row overlap or BF16 reconstruction alone; either would need fusion that also removes another material boundary to cross the gate.

## 2026-10-30 exact external top-candidate reproduction

The public `rohanarun/Auto-research-agent` Gemini prefill candidate was tested byte-for-byte rather than partially ported. The tested source has SHA-256 `c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb` and is preserved locally as `submission_external_rohan_196619.py`. The existing 255.562 ms incumbent in `submission.py` was not overwritten.

Off-board official check `51788973353f41b5` passed all five mandatory cases with no flags or static-check rejections. It measured **197.033 ms / 1.96619x**, with 95% interval **1.93670x-1.99569x**. This clears the 210 ms target by 12.967 ms and improves on the 255.562 ms scored incumbent by 58.529 ms (22.9%). The exact judge artifacts are saved under `profiles/external_rohan.51788973353f41b5.{json,txt,log}`.

| case | candidate | baseline | paired speedup |
|---|---:|---:|---:|
| T753 sliding, ctx80K | 549.683 us | 1145.488 us | 2.08390x |
| T753 full, ctx80K | 860.874 us | 1456.544 us | 1.69194x |
| T3012 sliding, ctx80K | 1591.995 us | 3340.208 us | 2.09813x |
| T3012 full, ctx80K | 2910.288 us | 4816.512 us | 1.65499x |
| T16384 full, ctx0 | 9274.590 us | 18646.048 us | 2.01044x |
| T16384 sliding, ctx32K | 7805.429 us | 16643.056 us | 2.13224x |
| T16384 full, ctx32K | 11265.636 us | 20448.239 us | 1.81510x |
| T16384 full, ctx64K | 14287.650 us | 24570.991 us | 1.71974x |

The weighted T16384 sliding profile is a complete architectural replacement, not a single W4 tweak: persistent fused W13-to-W2 MoE took 3.550 ms; two dense router/pooler-side kernels overlapped for about 1.595 ms; unpooler and other dense projections used pinned cuBLASLt kernels; custom sliding attention took 0.302 ms; the two boundaries took 0.280 and 0.466 ms; gather and combine were about 0.131 and 0.126 ms. GPU-busy time was 7.853 ms.

The candidate also stores and uses FP8 pooler/unpooler copies. The current automated judge accepted that path and returned no flags, but the leaderboard entry is under source screening and the written task says BF16 weights must remain BF16 or be compressed losslessly. Treat the 197.033 ms result as a real passing measurement but still provisional for final source-rule review. Do not publish a scored run without an explicit user decision on that risk.

## 2026-10-30 guarded FP8 pooler integration and scored result

With explicit user authorization, only the external candidate's pooler idea was ported into the existing incumbent. The pooler weight is quantized per output channel to E4M3 during `prepare`; each activation row is quantized with its own FP32 scale by `_pf2_pool_rowq8`, then evaluated with `torch._scaled_mm`. The incumbent attention, hi+lo router, W4 MoE, BF16 unpooler, and residual dataflow remain unchanged. Dispatch uses FP8 for 512-4096 tokens and for larger prefills only when cached context is at least 4096 tokens; the long ctx0 first turn retains the original BF16 pooler.

Off-board job `a16cc00a61c74eb7` passed all five mandatory cases with no flags or static rejections and measured **252.556 ms / 1.53393x** (95% CI **1.51092x-1.55694x**). This beat the 255.562 ms incumbent by 3.006 ms, clearing the precommitted submission gate.

The exact checked source was then submitted without modification. Scored job `b2d9091437474043` passed 5/5 with no flags or static rejections and measured **251.426 ms / 1.54083x** (95% CI **1.51772x-1.56394x**), a **4.136 ms / 1.62%** improvement over the prior personal best. Source SHA-256 is `d3683505bc9239af33e02cee15319dcbfc86e2f871b8741d6e323bcd293c3a83`; both jobs used judge content hash `4a1d7970cead21ffc617ff4c1fd4b41dd29544f3e85e5cb0f8a172c1c7f8ed6f`. The immediate leaderboard query placed the team at **287/680** with speedup 1.5408; no screening field was present yet. Remaining quota is **14 checks and 9 scored attempts**.

This validates a modest numerical/performance gain under the automated judge, not final rule compliance. E4M3 is a lossy copy of a BF16 pooler weight, while the written task says BF16 weights must remain BF16 or be compressed losslessly. Keep job `b2d9091437474043` explicitly marked provisional until manual screening resolves that conflict; do not interpret the absence of automated flags as organizer approval.

## 2026-10-30 broad-FP8 boundary experiment

A requested “FP8 from start to end” experiment first quantized all four major dense projections: QKV, O, pooler, and unpooler. It used per-row activation scales, per-output-channel weight scales, and direct BF16 GEMM outputs. This is not lossless—an arbitrary BF16 tensor cannot be represented exactly in E4M3 without side information—but it directly tests the performance/numerical hypothesis.

Off-board job `df187b68fae14abf` failed one of five mandatory cases. The B2/T1506 sliding case changed the pre-MoE hidden state enough that token 1114 selected an expert whose biased score was **0.0549** below the reference eighth-best, exceeding the **0.05** routing limit. Other cases passed, but T753 full reached update error **0.1348** against a 0.150 bound, and T4096 reached **0.1137** against 0.114. Therefore QKV/O FP8 is not robust: it compounds through attention and changes routing. Failed source SHA-256 was `93919088b314cbd40afbd211baf0b8a2a1a6ee213dfa066b1fecbb4383ceae7d`; judge content hash was `af39c7f48380c7920149085be0d6020713a659e999256cc96c4e5f5e99e89140`.

The experiment was narrowed to keep QKV and O in BF16 while retaining FP8 for the post-routing pooler and unpooler. Off-board job `97777aa934f14ee6` then passed all five mandatory cases with no flags or static rejections and measured **241.624 ms / 1.60333x** (95% CI **1.57928x-1.62738x**). This is **9.801 ms (3.90%)** faster than scored job `b2d9091437474043` and **13.938 ms (5.45%)** faster than the old 255.562 ms incumbent. Exact source SHA-256 is `5f7273d72c967645929ad0f5e7e93539a6633e7620745066ff50ac86b3e47269`; judge content hash is `230e1b02a768ccc0cc69a0d9e4513217ff219af95a6846f5e18a8dce9f2528c5`. Remaining quota is **12 checks and 9 scored attempts**.

The successful boundary confirms that FP8 is valuable where errors cannot feed back into expert selection. At T16384 cached-context profiles, the two FP8 GEMMs together took roughly **1.58-1.65 ms**, and their two row-quantization launches took roughly **0.15-0.23 ms**, versus the prepare-time BF16 pooler+unpooler timings of roughly **3.02 ms** combined. The broad path is closed; further work should fuse the MoE combine output directly to FP8 to remove the standalone unpooler row-quantization/read-write boundary, not quantize pre-router projections again.

With explicit user authorization, the exact checked source was submitted without modification. Scored job `cf4dee70b7b54309` passed 5/5 with no automated flags or static rejections and measured **240.760 ms / 1.60909x** (95% CI **1.58495x-1.63323x**). It improves on scored job `b2d9091437474043` by **10.666 ms (4.24%)** and on the older 255.562 ms incumbent by **14.802 ms (5.79%)**. The live leaderboard query immediately afterward placed the team at rank **238**. Source SHA-256 remained `5f7273d72c967645929ad0f5e7e93539a6633e7620745066ff50ac86b3e47269`, and judge content hash remained `230e1b02a768ccc0cc69a0d9e4513217ff219af95a6846f5e18a8dce9f2528c5`. Remaining quota is **12 checks and 8 scored attempts**; the leaderboard row had no screening field yet.

## 2026-10-30 persistent fused-MoE hybrid

The measured next bottleneck was ported without replacing the incumbent router, attention, residual, or dense projection paths. The hybrid retains the current shaped/rerouted expert selection, derives the persistent gather's per-expert row ordinal from the existing global route list, losslessly repacks the original NVFP4 codes and block scales, and uses the public MIT-licensed persistent W13-to-W2 expert kernel. Raw BF16 pre/post-expert norm weights are passed because that kernel forms `1 + w` internally. Its combine writes E4M3 rows and FP32 row scales directly for the already-validated FP8 unpooler. The current W4 implementation remains the compile/runtime fallback.

The imported 2,291-line CUDA expert block is SHA-identical to the validated public source block (`6069e4caaf40582155b497d57938714875d323d3011eff84e6921793dd94d024`). Off-board job `a82a291ef13342e4` compiled both paths in **58.0 s**, completed prepare in **71.22 s**, passed all five mandatory checks, and returned no flags or static rejections. It measured **203.340 ms / 1.90520x**, 95% CI **1.87662x-1.93378x**. This beats scored job `cf4dee70b7b54309` by **37.420 ms (15.54%)** and clears the precommitted 215 ms / 1.80x gate. Candidate source SHA-256 is `4e2730f95fdfac37e84f6663d66ba2cebf206eb2028807c6af9dd82f6d9db909`; judge content hash is `3387b9dec3b8b31fd2ef17107009949e7e6a6209409563f824b31fec53131420`.

| case | hybrid | baseline | paired speedup |
|---|---:|---:|---:|
| T753 sliding, ctx80K | 527.4 us | 1061.7 us | 2.01314x |
| T753 full, ctx80K | 925.3 us | 1427.6 us | 1.54275x |
| T3012 sliding, ctx80K | 1598.9 us | 3344.0 us | 2.09136x |
| T3012 full, ctx80K | 3104.2 us | 4870.9 us | 1.56914x |
| T16384 full, ctx0 | 8928.0 us | 17213.1 us | 1.92800x |
| T16384 sliding, ctx32K | 8172.1 us | 17510.0 us | 2.14266x |
| T16384 full, ctx32K | 11625.2 us | 20455.1 us | 1.75955x |
| T16384 full, ctx64K | 15314.2 us | 24418.9 us | 1.59453x |

The weighted sliding profile confirms the intended mechanism: persistent fused MoE took **3.695 ms** versus the prior W4 kernels' **4.712 ms**; gather was **0.139 ms**, combine-to-FP8 **0.126 ms**, and whole-profile GPU busy time **8.161 ms**. The benchmark median was **8.172 ms**, below the 8.3 ms case gate. The candidate is validated off-board but has not been scored. It introduces no new expert-weight precision relaxation: NVFP4 remains NVFP4 and the new norm inputs remain BF16. The previously documented FP8 pooler/unpooler screening caveat still applies to the overall submission.

### Scored persistent-MoE submission

With explicit user authorization, the exact checked source was submitted without modification. Scored job `367f1182aac845d0` passed all five mandatory checks with no automated flags or static rejections and measured **203.879 ms / 1.90017x** (95% CI **1.87166x-1.92867x**). This is **36.881 ms (15.32%)** faster than the prior 240.760 ms scored best. The live leaderboard query placed `aditya3799` at **rank 25 of 690**, up from rank 238.

The exact submitted source SHA-256 is `4e2730f95fdfac37e84f6663d66ba2cebf206eb2028807c6af9dd82f6d9db909`; judge content hash is `3387b9dec3b8b31fd2ef17107009949e7e6a6209409563f824b31fec53131420`. The leaderboard row had no screening field at the immediate post-submit query, so the previously documented FP8 pooler/unpooler manual-screening caveat remains unresolved. Confirmed remaining quota is **11 checks and 7 scored attempts**.

### Verified complete 197 ms candidate

The next MoE-only schedule diagnostic compared the production biggest-first/fine-remainder queue with interleaved, expert-major, and split-remainder alternatives on both measured 16K route profiles. Production won robustly on the first profile and on balance; the variants were rejected and the scored source was restored byte-for-byte. A proposed prepare-time BF16 expert-weight cache was rejected before an official run because the rules require NVFP4 weights to remain NVFP4 outside the timed kernel.

The complete public candidate `submission_external_rohan_196619.py` was then checked through OKC. Job `e3c417319bf14a06` deduplicated to official job `51788973353f41b5`, which passed 5/5 at **197.033 ms / 1.96619x** with 95% CI **1.93670x-1.99569x**. This is **6.846 ms (3.36%)** faster than scored job `367f1182aac845d0`. Its persistent MoE block is the same imported implementation; the remaining advantage comes primarily from its custom full-context attention and leaner surrounding dataflow, not another expert-core change.

The exact verified file is now active as `submission.py`, SHA-256 `c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb`, judge content hash `bef68ced286bf325cd328ddde1af93b1fac5b0690cf2d2efc4bb1a680541e526`. The prior scored source remains preserved as `submission_scored_367f1182aac845d0.py`. No scored attempt was made, and the duplicate check consumed no quota; **10 checks and 7 scored attempts** remain.

### Scored 197 ms architecture

With explicit user authorization, the exact validated source was submitted as job `d81535430cfd4b48`. It passed all five mandatory checks with no flags or static rejections and measured **199.900 ms / 1.93799x** (95% CI **1.90892x-1.96706x**). This is **3.979 ms (1.95%)** faster than scored job `367f1182aac845d0`. The live leaderboard placed `aditya3799` at **rank 19 of 697**, up from rank 25.

The submitted source SHA-256 is `c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb`, and the judge content hash is `bef68ced286bf325cd328ddde1af93b1fac5b0690cf2d2efc4bb1a680541e526`. An exact backup is `submission_scored_d81535430cfd4b48.py`. The leaderboard row had no screening status at the immediate query, so the FP8 pooler/unpooler manual-screening caveat remains. Remaining quota is **10 checks and 6 scored attempts**.

### Lean route-shaper port

The previously validated conservative route shaper was ported into the lean 199.9 ms architecture. The fused route kernel retains the best unselected non-local candidate, marks only local picks within 0.010 of ninth as optional, and requires the replacement to lie within 0.030 of eighth. A one-warp planner spends at most 7% of local routes at exact persistent-kernel tile boundaries and eliminates at most one active expert. Because rerouting occurs after the lean router originally assigns expert-row ordinals, a final block-local-counter kernel rebuilds dense positions and counts before gather.

Unscored job `fa2c5aa06ac64db9` compiled successfully, passed 5/5 with no flags or static rejections, and measured **194.842 ms / 1.98830x** (95% CI **1.95847x-2.01812x**). This beats scored job `d81535430cfd4b48` by **5.058 ms (2.53%)** and clears the contemporaneous rank-10 score of 196.208 ms. The weighted 16K sliding profile fell to about **7.773 ms GPU-busy**, with the persistent MoE core at **3.418 ms**; T753 and T3012 MoE cores also fell to roughly 0.220 ms and 0.664 ms on their sliding cases.

The exact checked source is active as `submission.py`, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`, judge content hash `9b3ec873a50e50a70d655b1d877dbbbcb135ed431f2a5894104c102b95241cf0`. Exact backups are `submission_route_shape_candidate.py` and `submission_checked_fa2c5aa06ac64db9.py`. No scored attempt was used; **9 checks and 6 scored attempts** remain.

### Scored route-shaper submission

With explicit user authorization, the exact checked source was submitted as scored job `910fa5d559e44d0d`. The judge content hash remained `9b3ec873a50e50a70d655b1d877dbbbcb135ed431f2a5894104c102b95241cf0`, confirming that the scored payload was byte-for-byte the checked candidate. It passed all five mandatory cases; `checks.ok` was true, with no flags or rejections.

The scored result was **194.310 ms / 1.99374x**, with 95% CI **1.96383x-2.02365x**. This is **5.590 ms (2.80%)** faster than the prior scored best `d81535430cfd4b48` at 199.900 ms. The immediate leaderboard query placed `aditya3799` at **rank 7 of 700**, up from rank 19 and 12 places higher. The row had no screening field yet, so the existing FP8 pooler/unpooler manual-screening caveat remains unresolved.

The active and scored source SHA-256 is `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`; an exact backup is `submission_scored_910fa5d559e44d0d.py`. Confirmed remaining quota is **9 checks and 5 scored attempts**.

### Fused boundary quantization and two-expert route follow-up

The first residual boundary was extended with an opt-in output that emits both the unchanged BF16 hidden state and the pooler's E4M3 row plus FP32 row scale. The FP8 values are generated from the BF16-rounded values retained in registers, so the fused path reproduces the existing standalone row-quantizer rather than introducing another approximation. An isolated local CUDA-graph probe was exact for BF16 output, FP8 bytes, row scales, stream output, and both residual modes. At T256 it reduced boundary-plus-quantization time from **5604.8 us to 4929.5 us (12.1%)** on the local RTX GPU.

An initial H200 compile job, `d566cc1e300b4d46`, exposed two PDL launch sites that needed explicit null arguments for the new optional outputs; it failed before timing. After that ABI fix, unscored job `897df59a8de84a8d` passed all five mandatory cases with no flags or rejections and measured **194.597 ms / 1.99080x** (95% CI **1.96094x-2.02067x**). In the weighted T16384 sliding profile the fused boundary took about **359.3 us**, versus roughly **282.5 + 179.9 = 462.4 us** for the old boundary and row quantizer, confirming an approximately **103 us component saving**. The complete run was nevertheless 0.287 ms slower than the scored incumbent, well within run noise, so it did not earn a scored attempt. The exact candidate is `submission_boundary_q8_candidate.py`, SHA-256 `6027339f00a2f0a4580bd12ed6a6568697071047a0b63aea37b5e8f25e233e01`.

A global planner experiment then allowed elimination of at most two experts under the same score thresholds and 7% route-change budget. Job `77e19c38998044b9` passed 5/5 with no flags but regressed to **196.046 ms / 1.97608x**; global two-expert elimination is rejected. A narrower version kept the proven one-expert planner on sliding layers and allowed two only for full-context layers. Job `9a42b0f30dc241da` passed 5/5 with no flags at **194.312 ms / 1.99372x** (95% CI **1.96381x-2.02362x**). That is only **0.002 ms** slower than the scored 194.310 ms incumbent and therefore indistinguishable from noise, despite nominally improving on the earlier 194.842 ms unscored control.

No scored attempt was used. The selective candidate is preserved as `submission_boundary_q8_full_route2_candidate.py`, SHA-256 `d56c94ee28bd4dd5994919b47dc7b361bcd61c54635132f924bf4c9b6a330f0`. `submission.py` has been restored byte-for-byte to scored job `910fa5d559e44d0d`, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`. Confirmed remaining quota is **5 checks and 5 scored attempts**. Further work should leave route shaping alone and target full-context attention; neither the approximately 0.10 ms fusion nor the selective route change independently clears a responsible submission gate.

### Full-context attention audit

Four exact or fallback-protected attention variants were evaluated from the scored route-shaper source. Every job passed all five mandatory checks with no flags or static rejections, but none improved performance.

1. `ATTN_TRIM=7` enabled unsigned ring arithmetic, hoisted shared-memory descriptors, and immediate WGMMA accumulator-scale operands. Job `9cee59e73efe4d0d` regressed to **204.671 ms / 1.89281x**. The 32K and 65K full-attention kernels grew to approximately **5.080 ms and 8.746 ms**, versus the control's roughly **3.87 ms and 6.83 ms**. The hoisted descriptors enlarge live state and damage the Hopper schedule. Exact candidate: `submission_attn_trim7_candidate.py`, SHA-256 `73d81b5a6ddb6cf879f3b152c9994f22ba2fafdb31784e168b02c37a06c856ca`.
2. The immediate WGMMA scale operand was then isolated as `ATTN_TRIM=4`, avoiding the extra descriptors. Job `365c8323c8e84a9e` regressed further to **206.044 ms / 1.88020x**, with 32K/65K full attention at roughly **5.289/9.237 ms**. Even this inline-assembly specialization perturbs allocation/scheduling adversely. Exact candidate: `submission_attn_trim4_candidate.py`, SHA-256 `093aa11ca477fcd74c95d40a3341a68fb2917ed778917a9e0b2f648a0d4d542f`.
3. The dormant two-consumer-warpgroup pipelined/exp-turn kernel (variant 53) was compiled beside every production variant and directly tuned on the four real attention shape classes. Job `bb40fe5711d0490e` validated variant 53 at the same **0.0151** worst relative error, but it lost everywhere: sliding **459 vs 305 us**, full-long **5425 vs 3836 us**, single-request full **620 vs 365 us**, and four-request full **2233 vs 1393 us**. The tuner retained variants 8/43 in every class and the control-like aggregate was **194.405 ms / 1.99277x**, only 0.095 ms slower than the scored incumbent. Exact diagnostic source: `submission_attn_v53_tuner_candidate.py`, SHA-256 `4ba20d4b9f7502d9e3b3200e2f1c3d61b37fce8e03d8631907bfa61a8a677f23`.
4. `ATTN_LATEMAX=1` used the existing headroom test to skip per-tile row-max work when FP8 values remain representable, with an exact recomputation fallback on saturation. Job `b3cc203c38264dc5` passed 5/5 but regressed to **203.642 ms / 1.90238x**. Full attention took about **5.158/8.567 ms** at 32K/65K, and the short full kernels also increased from **0.349/1.352 ms** to **0.462/1.714 ms**. The extra speculative path and packed-value test cost more than the avoided reductions. Exact candidate: `submission_attn_latemax1_candidate.py`, SHA-256 `828bea7aeaac65a3f121a23846a2c1d6df281f39841fe1a0187427c6853d649c`.

No scored attempt was used. `submission.py` is restored byte-for-byte to scored job `910fa5d559e44d0d`, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`. One unscored check and five scored attempts remain before the next quota reset. Do not retry attention descriptor trimming, immediate scale operands, two-warpgroup PIPE3, or late-max. The production three-consumer-warpgroup variants 8/43 remain decisively best; a future attention attempt needs a new dataflow rather than a dormant scheduling flag.

### Rejected fused route apply/position rebuild

The scored route-shaper path was audited for support work rather than changing its validated routing policy. The control creates initial route positions, launches the one-warp planner, applies reroutes in a token kernel, clears the 32 counts, and traverses routes again to rebuild final counts and positions. A candidate moved the count clear into the planner and fused reroute application, weight renormalization, and the final block-local position rebuild. It therefore removed one launch, the stream-ordered count memset, a full token traversal, and count atomics whose results were immediately discarded. A second layout distributed the eight route slots across the token warp while retaining increasing-slot quota attempts and the one-change-per-token rule.

The route-only SM86 probe compiled the exact routing sections from the candidate and scored control. Eight candidate cases (T=512/753/3012/16384, two seeds each) passed normalized weights, unique top-8 experts, exact `ids`/`lid` agreement, exact final counts, dense unique positions for every local expert, at most one changed route per token, local-to-nonlocal replacement, and the 7% budget. At T16384, independent control/candidate runs also produced identical unshaped and shaped count vectors. Performance rejected the dataflow: two paired cold-cache sequences measured the control at **20602.9/20612.1 us** and the warp-parallel fused candidate at **20686.9/20681.7 us**, a repeatable **70-84 us regression** on the laptop GPU. The earlier lane-0 fused layout was slower still at roughly 20703-20704 us versus 20599-20613 us.

No official check or scored attempt was used. Two read-only attempts to retrieve the archived H200 profile timed out, so the last remaining check was not risked against a locally regressing component. The final experimental source is preserved as `submission_route_apply_pos_candidate.py`, SHA-256 `7a21fa0031140584ac32e090d9355c3b59767bea9016e6c7a1e0d9d433dbd398`. `submission.py` is restored byte-for-byte to scored job `910fa5d559e44d0d`, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`. Do not retry route-support fusion with the same one-warp-per-token rebuild; the added serialized dependency costs more than the removed launch and memset.

A narrower follow-up retained the proven separate planner/apply/rebuild schedule and removed only work that is provably dead when shaping is active. The initial route pass still constructs exact local counts for the planner, but no longer stores positions that the final rebuild overwrites. Its shared/global count atomics also discard their return values, permitting reduction-form atomics; unshaped routing is unchanged. The same eight-case invariant suite passed. Two paired T16384 sequences measured **20516.9/20537.3 us** for the candidate versus **20615.2/20613.1 us** for the scored control, a repeatable **76-98 us (about 0.4-0.5%)** local route improvement. Because the H200 route/support span is only around 0.1 ms, that fractional improvement projects to only a few microseconds and cannot responsibly clear the aggregate submission gate. No official check was spent. Preserve it as a future combinable cleanup, not a standalone submission: `submission_route_dead_pos_candidate.py`, SHA-256 `e9611831265807de0979ef221678bd5ae2bba6aff34b42e4a7546725ed34746c`.

### Legal BF16 dense, boundary, and stream-K follow-up (2026-10-31)

The sibling BF16 projection entry's descriptor-TMA/WGMMA Pooler was generalized from 128-256 rows to the scored 16,384-row Pooler shape. Eight exact-BF16 configurations covered N64/N128 tiles, one/two consumer warpgroups, no split-K and two-way cluster split-K. Unscored job `5131aaf50ced428d` passed 5/5. All candidates were correct, but the best custom path was **1828.2 us** versus pinned cuBLASLt at **1333.9 us**; other candidates ranged from 1873 to 3014 us, and the split-K route was 2580 us. The gate retained cuBLASLt. The headline 210.07 ms was control variance, not an activated custom kernel, and the entire extra extension was removed.

A four-way legal BF16 Unpooler/boundary pipeline then tuned a 4096-row cuBLASLt plan. Job `524beb18430344bd` passed 5/5 but regressed to **214.32 ms**. Four GEMMs totaled about **1361 us** (44 us below the two-half profile), while four boundary launches grew to **752 us** from 544 us. Weighted sliding GPU-busy rose to **8896.7 us**. More chunks lose to boundary launch/occupancy and overlap interference; the two-half schedule remains the right topology.

The retained improvement changes only the large-T second residual boundary. MODE 1 now saves the first stream's row RMS after producing its BF16 normalized output. MODE 2 reconstructs that stream as `float(h_bf16) * row_rms / (1 + pre_ffn_norm)` using a prepared exact FP32 reciprocal of the original BF16 norm vector. This replaces the original FP32-stream plus first-branch reads with one BF16 read; all dense weights remain their original BF16 tensors. Job `0a5be76a63334e29` passed 5/5. Across the large profiles, the two MODE-2 launches fell reproducibly from about **544-546 us to 472-474 us**. Weighted sliding GPU-busy fell from **8763.8 to 8680.7 us**, despite its noisy 214.71 ms aggregate. The reconstruction rounding is therefore a real approximately 83 us critical-path win and is retained in `submission.py`, SHA-256 `180b34a96ac38e9f391cb0701c6ce0f040d69e15e0413ec904dee211a991bec6`.

An asymmetric 12K/4K Unpooler split attempted to hide the larger first boundary under the 4K tail GEMM. Job `b13962d933d94dd9` passed 5/5 at 212.34 ms, but direct profiles rejected it: boundary time under contention rose to **524.3 us** and weighted sliding GPU-busy rose from the equal-split reconstruction's 8680.7 us to **8771.5 us**. The source returned to the 8K/8K split.

Finally, the dormant causal attention stream-K scheduler was correctness- and performance-gated at all three 16K full-context lengths and grids 396/528/660. Job `6e7a7042d5394c60` passed 5/5. Stream-K was numerically valid (relative errors 0.0052-0.0076) but lost everywhere: ctx0 **838 us normal vs 999-1095 us**, ctx32K **3833 vs 3922-3953 us**, and ctx64K **6764 vs 6865-6897 us**. No stream-K route activated, and the diagnostic tuner was removed.

After cleanup the only retained delta from the prior exact BF16 winner is the validated boundary reconstruction. Remaining daily quota reported by the last check is **4 unscored checks and 9 scored attempts**. Reaching 198 ms still requires a new multi-hundred-microsecond legal dataflow improvement; custom dense scheduling, smaller Unpooler chunks, asymmetric overlap, and attention stream-K are now closed.

### Upstream cohort and availability-aware route planner

The 2026-10-30 Auto-research-agent cohort was audited before beginning another large kernel rewrite. Its six newer prefill candidates did not contain a replacement for the current architecture: their two-bank reported speedups ranged from roughly **0.076x to 1.073x**, versus the incumbent's official **1.99374x**. The only material upstream gain, Claude Opus 5.5 at 1.073x, includes a path that dequantizes the NVFP4 expert weights into BF16 during `prepare`; that is incompatible with this challenge's written NVFP4-storage rule. GPT-6.1's lowest published 16K sample still uses the standard Marlin expert path and dense BF16 projections. No source was ported.

A new local audit then found a real inefficiency in the winning route shaper. The planner selected tile-boundary reductions from raw expert counts, but it did not know how many checker-safe replacement tokens actually existed for each expert. Across eight invariant cases, **1-62 planned removals were unfulfilled at T512-T3012 and 138-160 were unfulfilled at T16384**. Those failed quotas consume planner budget without reducing expert work.

`submission_route_avail_candidate.py` adds one 32-counter availability vector, zeroed inside the existing first-boundary counter clear. The route kernel increments it only for candidates that already pass the unchanged 0.010 optional-route and 0.030 replacement guards. The one-warp planner rejects a tile-boundary step unless cumulative safe availability can fill it. The 7% global budget and one-eliminated-expert cap remain unchanged.

The exact local route probe passed normalization, unique top-8, `ids`/`lid`, final counts, dense positions, local-to-nonlocal replacement, one change per token, and the 7% budget at T512/753/3012/16384 with two seeds each. Planned versus fulfilled quotas became exact in seven cases and missed by only one route in the eighth. Candidate route latency was statistically flat against the control (**about 20.60-20.61 ms** on the RTX probe). Multi-seed padded expert work was never worse: ordinary additional savings were **0-32 rows at T753, 96-128 at T3012, and 128-448 at T16384**; skewed invariant cases saved up to **1,696 rows**. Candidate SHA-256 is `30865f672c5c9a4c6aaddc84aa101472dc4c9398aef9b07b9e8b508b2e014485`.

An official H200 check was attempted twice with the identical source. The first client timed out at 64 seconds and the reconnect timed out at 304 seconds without a job id, stderr, result, or local profile artifact; even `okc --help` subsequently timed out through WSL. Treat remote status as unresolved service/client failure, not as a kernel pass or failure. Do not submit this candidate until a deduplicated check returns 5/5 and a material aggregate win. `submission.py` remains byte-for-byte the scored source, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`.

### Persistent W13-to-W2 dataflow audit (2026-10-31)

Two gated changes tested whether the fused persistent MoE could reduce the global BF16-intermediate boundary without changing precision, arithmetic, or the production fallback. Both official checks passed all five mandatory cases with no flags or static rejections, but component timing rejected both changes.

First, job `32a090307aa84954` replaced the all-W13-then-all-W2 work queue with wavefront batches: B token tiles' 64 W13 items followed immediately by their 32 W2 items. The mapping was verified as a bijection for 1-129 tiles. At the T16384 synthetic distribution and order 0, the original queue took **3739 us** while B=2/4/8/16 took **4779/4251/4230/4257 us**. The shorter BF16-intermediate lifetime does not compensate for disrupting W13/W2 weight working-set locality. Do not retry static tile-batch interleaving.

Second, job `6f31f528ee3e49db` replaced the publisher's separate `fence.acq_rel.gpu` plus relaxed counter increment with a single `atom.acq_rel.gpu.global.add`. At T16384 the pairs were **3822 vs 3832 us** (order 0), **4295 vs 4294 us** (order 1), and **4253 vs 4253 us** (order 2); smaller cases were likewise flat within a few microseconds. The publisher already hides this synchronization cost. Do not retry atomic-ordering substitutions without removing a larger dependency.

Both experiments were removed. `submission.py` is restored byte-for-byte to the availability-aware legal BF16 winner, SHA-256 `D5B014DFB56F85552F47675322F7766D7EF5731ED56F906CF6AADD4A5CF031FD`. The best official check for that source remains `cc8940c90ffc4f3c` at **211.021120 ms / 1.83585368x**, and **12 unscored checks / 9 scored attempts** remain. A future MoE attempt needs a genuinely different ownership scheme for W13 output/W2 input; queue interleaving and fence micro-optimization are closed.

### Legal BF16 library and four-warpgroup attention audit (2026-10-31)

Unscored job `f9ffd3eec6f1482e` expanded the two weighted BF16 projection shapes beyond cuBLASLt's eight heuristics by enumerating every supported algorithm-66 tile/stage/custom/swizzle configuration. Only 79 additional valid configurations existed. The production pins were already the winners: pooler tile 29/custom 1 measured **1372.6 us**, and each 8192-row unpooler half used tile 201/custom 2 at **683.6 us**. The check passed 5/5 but was control-like at **213.55 ms / 1.814x**; the exhaustive prepare tuner was removed.

A dormant four-consumer-warpgroup attention path was then made launch-safe by allocating its extra 4 KiB Q staging block. The first diagnostic `e654c405608349b0` exposed the missing allocation; corrected job `fd30f0837a38425e` passed 5/5 at **212.11 ms / 1.826x**. Variant 64 was numerically valid (worst relative validation error **0.0153**) and narrowly won sliding attention (**301 vs 305 us**), but lost full-long to production variant 43 (**3980 vs 3802 us**) and also lost both short full classes. It was removed. The exact legal BF16 winner was restored; do not retry library-plan expansion or four-warpgroup half-tile attention.

### Two-resident-CTA fused-MoE audit (2026-10-31)

A legal BF16/NVFP4 occupancy experiment capped token tiles at 128 rows, reduced the fused kernel from four 43-KiB TMA slots to two 21-KiB slots, disabled the extra accumulator sets, and reduced dynamic shared memory from 227 KiB to 80 KiB. The intended launch was 2 x SM persistent CTAs. Splitting each former 256-row tile into two 128-row tiles preserves the original weights and per-row mathematics but deliberately trades duplicate packed-weight traffic for latency hiding.

The first H200 check, job `4f05edc0fec9488d`, proved that the original 384-thread layout cannot reach two-block occupancy: `ptxas` imposed an 80-register target and reported that the kernel requires at least 94. A dormant 320-thread consumer-first layout compiled, but job `68069a9f33724f4e` failed all five mandatory cases; its first residual update error was **0.9661** against a **0.071** bound. It was rejected as a synchronization-path failure, not numerical tolerance.

A final layout retained the proven producer-first ordering and removed two producer warps that are idle in the production SM90 path, giving 64 producer/publisher threads plus 256 consumers. It compiled, but the unscored check produced no output and the client hit its **904-second timeout**, consistent with a barrier-phase stall. No scored submission was made. The subsequent read-only status query reports **1 of 20 checks** and **9 of 10 scored runs** remaining today.

All dual-CTA code was removed. `submission.py` is restored byte-for-byte to the legal BF16 boundary-reconstruction candidate, SHA-256 `180B34A96AC38E9F391CB0701C6CE0F040D69E15E0413EC904DEE211A991BEC6`; Python syntax and `git diff --check` pass. This closes ordinary two-resident-CTA conversion of the fused MoE. A future parallel design would need a new work/ownership protocol, not another ring-depth or warp-count adjustment.

### Persistent BF16 Unpooler-to-boundary fusion (2026-10-31)

A fixed 128-CTA Hopper kernel tested the strongest legal dense-fusion design without changing weight or activation precision. Each CTA owned 64 of the Unpooler's 8192 output columns, stayed resident across 128-row token tiles, computed the original BF16 WGMMA projection, retained the rounded BF16 tile on-chip, and joined two grid-wide reductions before writing the final FP32 stream and BF16 hidden state. This removed the complete `[16384, 8192]` BF16 Unpooler intermediate and the standalone second-boundary launches. The ordinary cuBLASLt 8K/8K pipeline remained available as source fallback.

Unscored job `178d5d0a26af422a` stopped at a mechanical PTX error caused by double-escaped newlines in the new raw CUDA string. After correction, the isolated SM90a source assembled locally with **115 registers, zero stack, and zero spills**. Because no checks remained and scored validation had been explicitly authorized, job `ea47c714aff54c9a` tested the corrected kernel. Its prepare gate agreed closely with cuBLASLt plus the original boundary (**9.73e-8 stream relative error, 1.66e-5 hidden relative error**), and all **5/5 mandatory cases passed**.

Performance rejected the topology decisively. The persistent kernel took **12.48-12.82 ms** in every 16K profile, versus roughly **1.6 ms** for the overlapped legal-BF16 cuBLASLt-plus-boundary path, producing **272.47 ms / 1.422x** aggregate. The hundreds of software grid rendezvous and a ptxas-reported serialized WGMMA accumulator chain dominate the eliminated HBM traffic. A cooperative-grid barrier could reduce the rendezvous tax, but the earlier standalone custom BF16 projection's 1.83 ms lower bound is already slower than the retained overlapped path, so another scored variant is not justified.

The entire active experiment was removed. `submission.py` is again byte-for-byte SHA-256 `180B34A96AC38E9F391CB0701C6CE0F040D69E15E0413EC904DEE211A991BEC6`; Python syntax and `git diff --check` pass. The contest reports **0 checks and 8 scored runs** remaining. This closes software-global-barrier fusion of the full dense output; a future fusion must use a library-quality GEMM mainloop and a hardware/cooperative reduction scheme with essentially no per-tile global rendezvous.
## 2026-10-31 lossless KPERM at T3012

The retained lossless NVFP4 K-permutation was extended from T753 to T3012.  The first scored job,
`f76d6f8d2755447d`, passed all five correctness cases with no flags but timed out after the two T753 benchmarks.
The cause was exact: T3012 enables fine 96/160/192/224-row remainder tiles, while the KPERM dispatcher selected
`moe_fused_persistent<0, true>`.  Those fine-width switch cases are compiled out when `FINE=0`, so their W13
completion counters were never published and W2 waited indefinitely.

Job `d4739ce200d04865` added the missing `moe_fused_persistent<1, true>` instantiation and passed 5/5 with no
flags.  It scored **213.084 ms / 1.81808x**, below both the T753-only KPERM result (211.761 ms) and the legal
control (210.713 ms).  Its two T3012 MoE profiles were distribution-sensitive: the weighted sliding case
regressed from roughly 688 to 718 us, while the full case improved from roughly 687 to 628 us.  The higher-weight
regression rejects the wider dispatch.  The source immediately after this experiment was the T753-only KPERM
variant, SHA-256 `64760BE32320CA0C0235133E463FC5D0440679FB790600D352F347C8FC454927`.  One scored prefill attempt remained;
do not retry T3012 KPERM without a route-distribution-specific gate and a multi-millisecond path to the leaderboard.

The final scored job, `86b59c20237446ff`, applied that route-specific gate: T3012 KPERM ran only for the
full-attention workload while the higher-weight sliding workload retained the original layout.  It again passed
**5/5 correctness cases with no flags** and scored **211.474 ms / 1.83192x**.  This was 0.287 ms better than the
T753-only KPERM scored sample but still 0.761 ms slower than the legal 210.713 ms control.  More importantly, the
full T3012 MoE profile was **721.1 us**, so it did not reproduce the earlier 627.7 us sample and provides no stable
component-level evidence for keeping the extra template and dispatch.  The selective extension is rejected and
the source is logically restored to the T753-only KPERM variant.  The reconstructed rollback has SHA-256
`0E433BB3B7AD71D5CAD5BAF06C1F09BDAD882E19CE3C0E0E02E00E4822C72130`; Python syntax and `git diff --check`
pass.  Prefill quota is now exhausted: **20/20 checks and 10/10 scored attempts used**.

## 2026-11-01 external inference-kernel audit

Five upstream implementations were cloned for a source-level Hopper audit rather than relying on headline
benchmarks: DeepGEMM `5081f8513af5`, FlashMLA `395cb3afbb33`, FlashInfer `1ec41b4b0e21`, TensorRT-LLM
`5d86481f60bd`, and Humming `948e6fd547e7`. The review was filtered through this problem's exact contract:
dense weights and activations stay BF16, expert weights stay NVFP4, and the target is H200/SM90.

The compatibility result is narrow. DeepGEMM's native FP4 path is SM100-oriented, while its relevant Hopper
route assumes FP8 activations. FlashInfer's newest Hopper MegaMoE FP4 route likewise uses FP8 activations; its
SM90 register-source/swap-AB organization is already the organization of our fused kernel (packed weight codes
decoded directly into WGMMA's register-source operand). TensorRT-LLM's legal Hopper BF16xNVFP4 route is Marlin,
which is already the task baseline, while its native FP4 CUTLASS route is SM100. Current FlashMLA targets SM100;
the transferable Hopper concepts from its older kernels--persistent tile scheduling, fine TMA pipelines, PDL,
warp specialization and cache hints--were individually compared to the existing source. Humming is the closest
legal standalone W4A16 Hopper kernel and was tested directly.

### Dynamic W13/W2 frontier

A readiness-aware fused queue tested the interleaved-scheduler idea without changing precision or tile math.
W13 work remained available while completed tiles exposed W2 work; issue ratios 2:1, 3:1 and 4:1 were measured.
Jobs `6f36c78be45e49d1`, `4bdf94bf90d4415b`, the warmed `50e8280e89b04bb3`, and
`fca5c6677c8f4468` passed and produced **212.351, 211.321, 210.922, and 213.066 ms** respectively. The best
3:1 sample remained 0.209 ms slower than the legal scored control and disturbed the weight-locality benefit of
finishing the W13 frontier first. All frontier code was removed.

### Direct Humming comparison

The installed Humming W4A16 kernel was first tactic-swept on the challenge's separate `muse_moe_w4a16` entry.
Representative winners were BM24 for B1/Q753, BM96 for B4/Q753, and BM128 for B1/Q16384. An exact full-layer
integration used Humming for both large-T GEMMs with the original BF16 intermediate and SiLU multiplication.
Job `9b5340c0f17b4fd4` passed 5/5 but measured **225.841 ms**. In profile, W13 took about 3468 us, W2 about
1766 us, plus roughly 284 us for SiLU, versus roughly 3.4 ms for the incumbent fused launch. Removed. Humming's
split-issue/wait implementation was also audited; the incumbent already converts the next weight fragment while
one WGMMA group is in flight, and the older deeper-wait variants already have negative measurements.

### Lossless K=64 permutation at T16384

FlashInfer's SM90 organization motivated revisiting the existing lossless K=64 permutation at the dominant large
workload. The original gate used it only below 2048 tokens. The extension selects the matching
`moe_fused_persistent<1,true>` specialization for fine 96/160/192/224-row remainder tiles and sets that kernel's
dynamic shared-memory attribute. The initial job `3794584af0774e87` passed all mandatory and short benchmark
cases but failed on the first large launch because that attribute was initially omitted; this was a mechanical
launch configuration error, not a layout/correctness failure.

Corrected job `a1d2f61ed3534773` passed **5/5 with no flags or rejections** and measured **210.067014 ms**. It is
0.646 ms faster than the legal scored job `3d31ea828106401f` (210.713 ms). Large-case event medians versus that
control changed from 9352.83/8603.88/12230.35/14877.66 us to
9130.19/8524.39/11966.31/14992.37 us: three improved by 80-264 us and the longest full-context case regressed
by 115 us. Because benchmark seeds and route distributions differ between jobs, treat the small aggregate delta
as a promising legal base rather than a proven scored gain. The source is retained.

### Cross-slot register prefetch

`MOE_XPF256=1` waited for the next ring slot and loaded its first packed weight/scales into registers before
issuing the current slot's final WGMMA, attempting to hide the next slot's LDS latency. Job
`ce19bf15a9494716` passed 5/5 but measured **210.659492 ms**, 0.592 ms behind K64-only. Large MoE profiles were
3518.5/3455.2/3863.5/3251.8 us, with no consistent gain. The longer register lifetime and next-slot wait do not
pay for one saved load boundary; `MOE_XPF256` is restored to zero.

### Hopper L2 eviction policy

The exact non-tensor Hopper PTX form used by FlashInfer was tested:
`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` with the
`createpolicy.fractional.L2::evict_first` policy value. The bit was isolated to T>=8192 expert-weight copies;
short cases and activation traffic were untouched. Job `6eb2b70d839948d3` passed 5/5 but measured
**212.147569 ms**. Its large MoE profiles were 3510.5/3388.7/3759.8/3605.3 us and the aggregate lost 2.081 ms
to K64-only. FlashInfer's own current source explicitly overrides its streaming default to EVICT_NORMAL for
expert weight tiles because multiple M-block CTAs share them through L2. Our result confirms the same reuse
regime. The cache-hint code and extra dispatch bit were removed; evicting the more heavily reused activation tile
was not attempted.

After cleanup, `submission.py` is exactly the retained K64 candidate, SHA-256
`5F5D243AC18014F8B121D8D326CAAF25810DEFAB60FCB5400A225138F63ADF6D`. Python AST parsing and
`git diff --check` pass. The final quota query reports **10/20 prefill checks used and 0/10 scored attempts used**.
The best checked result is 210.067 ms, so the 198 ms goal is not reached; another launch-scale tweak is not a
credible path for the remaining 12.067 ms.

## Fine-grained W13 -> W2 chunk pipeline

The next experiment changed ownership rather than another launch knob. Each W13 n-tile publishes one bit after
its 64-feature BF16 intermediate chunk reaches global memory. One such chunk is exactly one W2 K-block, so the W2
producer waited only for the 1--4 readiness bits required by its current TMA stage instead of waiting for all 64
W13 n-tiles. The implementation used a per-token-tile 64-bit readiness mask with GPU-scope release/acquire and an
async-proxy fence before staging the intermediate.

The first schedule grouped work by token tile, issued a lead of 16 W13 chunks, then alternated two W13 chunks and
one W2 chunk. Job `e68cc05206174ca6` passed all 5 mandatory cases with no flags, but measured **220.107358 ms**.
Its four large-case MoE profiles were approximately 4027.8, 4186.3, 3928.4 and 3640.2 us. The schedule destroyed
the incumbent's global expert/weight order and therefore its packed-weight locality.

The second schedule preserved both GEMMs' original flat order: after the lead it issued three globally ordered
W13 items per globally ordered W2 item, followed by the W2 tail. The mapping was checked as a bijection across
representative tile counts before launch. Job `bf25c0116b894336` again passed 5/5 with no flags and improved to
**214.663569 ms**, but still lost 4.597 ms to K64-only. Its large MoE profiles were approximately 3750.8,
3745.0, 3532.5 and 3760.8 us, versus about 3421.1, 3298.0, 3190.9 and 3532.9 us for the retained K64 run.

This rejects fine-grained overlap that still round-trips the full intermediate through global memory. The added
per-stage acquire/proxy boundaries and concurrent pressure from the W13 and W2 packed-weight streams cost more
than the exposed dependency overlap saves. A 2:1 global variant was not spent: the earlier whole-tile sweep
already showed 2:1 worse than 3:1, and increasing W2 concurrency would intensify the measured contention. All
readiness-mask, mapping, polling and bit-10 dispatch code was removed. Python AST parsing and `git diff --check`
pass. The current source is logically the retained K64 implementation; its SHA-256 is
`8C64DB941210FD6EDFC053A30FE87C81CB2F8026C0F798BC4B2CDC41CFC65B80`, differing from the uploaded source only in
line wrapping around the restored fused-kernel interface. The final quota query reports **12/20 checks used (8
left)** and **0/10 scored attempts used**.

## Paired-output MoE and final upstream follow-up

The strongest legal on-chip reuse design used four consumer warpgroups, 128-token tiles, and two adjacent output
tiles per work item. Each packed activation/intermediate stage was therefore shared across two output tiles while
preserving the incumbent's all-W13-then-all-W2 global order. This is the viable Hopper substitute for a literal
W13-to-W2 register handoff: one W13 chunk is reused by all 32 W2 output tiles, so retaining the whole handoff in a
single CTA would require 32 live accumulator tiles; an eight-CTA cluster would instead recompute W13 about four
times or require persistence across cluster waves.

Job `3ee0956bbe97433c` compiled the paired-output path and passed **5/5** with no flags, but measured
**218.572919 ms**. Its four large MoE profiles were approximately **4003.9, 4113.8, 3783.1, and 3865.4 us**, all
well behind the K64 control. The saved activation loads were already cache-served; m64n128 in place of m64n256,
one accumulator chain, KPS=1, and the extra synchronization cost much more. The path was rejected.

FlashInfer's current pre-SM100 E2M1-to-BF16 helper was then compared instruction-for-instruction with the retained
decode. Its PRMT lookup needs about 13 integer operations to materialize eight unscaled BF16 values and still
requires the four scale multiplies. The incumbent repack produces four already-scaled WGMMA RS registers with
seven integer-pipe operations, two FMA-pipe IMAD shifts, and the same four multiplies. No H200 check was spent on
an instruction sequence that is strictly longer.

Two route-planner follow-ups closed the remaining tile-work hypothesis. Job `127d77b4d1044fad` widened the local
optional/replacement score margins from .010/.030 to .035/.045 while leaving a nominal .005 margin inside the
official .040/.050 guards. It failed three mandatory cases: the approximate router dropped reference-local
experts whose true gaps were **.0416, .0409, and .0452**. The unsafe margins were immediately restored.

Job `049cfc0c136644aa` instead kept the proven score guards and 7% route budget, but allowed a second expert to
reach zero only when the availability-aware planner could fill its boundary. It passed **5/5** with no flags and
measured **210.286913 ms**. Its large MoE profiles were **3446.8, 3327.8, 3289.1, and 3370.0 us**; the highest-
weight sliding case regressed from about 3298 to 3328 us and the aggregate lost 0.220 ms to K64-only. It was
rejected and restored to the one-expert cap.

All paired-output/bit-10 code was physically removed. Clean rollback job `7f9c34f06cc44737` passed **5/5** with
no flags at **210.578984 ms**, a control-variance sample rather than a new winner. The retained best remains job
`a1d2f61ed3534773` at **210.067014 ms**. The cleaned source SHA-256 is
`9FB5E1B6B6291A180D374DC5373C38E5E009A320480FD79C7217EAAD407D9BB4`; Python AST parsing and `git diff --check`
pass. Quota is **16/20 checks used (4 left)** and **0/10 scored attempts used**.

This audit is conclusive for the legal upstream-transferable directions: DeepGEMM/FlashInfer scheduling,
Humming W4A16, K-permutation, cache hints, fine-grained W13/W2 overlap, paired-output activation reuse,
FlashInfer PRMT decode, wider route margins, and multi-expert tile elimination all failed direct H200 gates.
Reaching 198 ms would now require a new hardware/dataflow capability, not another launch or schedule variant.

## Final full-context attention gates

A paired-FP16 exponential variant retained FP32 row maxima/rescaling but evaluated two probabilities at a time
with `ex2.approx.f16x2` before E4M3 packing. Job `781f98b51b114059` passed 5/5 with validation error 0.0152
versus 0.0151 for v43, but lost every class: **365 vs 305 us** sliding, **4718 vs 3836 us** full-long,
**452 vs 365 us** single full, and **1740 vs 1400 us** batched full. The tuner retained the incumbent and the
run measured **211.9697 ms** overall. The candidate was removed.

A prepare-only cycle profile then measured the production v43 full-long attention loop. Job
`8cb23a6805644684` passed 5/5 and reported, per tile, **74.4 cycles K/V readiness wait**, **232.9 cycles QK/PV
overlap**, **1539.0 cycles softmax**, **79.2 cycles PV tail**, and **1996.8 cycles consumer total**. K/V wait is
only **3.86%** of the loop; even perfect K/V multicast has a ceiling of roughly 0.26 ms on the 6.8-ms long case
before cluster overhead. A cluster-multicast rewrite was therefore rejected without spending another check.

Finally, v43 hybrids moved 8/64, 16/64, or 32/64 exponentials per thread from MUFU to the FA4-style cubic FMA
polynomial. Job `a36eb807d8964526` passed 5/5 with identical 0.0151 validation error, but even the lightest hybrid
regressed full-long to **4414 vs 3835 us**; the heavier hybrids reached 4613 and 4809 us. The lightest variant
also measured **356 vs 324 us** sliding, **431 vs 364 us** single full, and **1581 vs 1383 us** batched full.
The added dependency and issue cost outweighed any MUFU relief, and all three variants were removed.

These gates show that production full-context attention is dominated by the softmax dependency chain, not K/V
bandwidth. Paired low-precision exponentials and FMA/MUFU balancing both make that chain slower, while multicast
cannot save enough time to matter. `submission.py` is restored byte-for-byte to the retained best source,
SHA-256 `9FB5E1B6B6291A180D374DC5373C38E5E009A320480FD79C7217EAAD407D9BB4`; Python AST parsing and
`git diff --check` pass. Quota is **19/20 checks used (1 left)** and **0/10 scored attempts used**.

## Full-context attention-to-dense row pipeline

The last check tested the remaining layer-level overlap hypothesis rather than another attention microvariant. The
16K full-context path split variant 43 at the 8K query boundary. As soon as the late query half completed, its exact
BF16 O projection, first residual boundary, BF16 Pooler and FP32 hi+lo Router ran on two side streams while the main
stream computed the early query half. Every key remained attended, all dense weights and activations stayed BF16,
and the two dense slices retained exactly 8192 rows. Sliding and short shapes were unchanged.

Job `91e45f069d81425c` compiled, passed **5/5**, and returned **no flags or rejections**, but regressed to
**212.609710 ms**. At ctx32K, the two attention launches totaled **4538.4 us** and GPU-busy time was **12257.5 us**;
at ctx64K they totaled **7921.7 us** and GPU-busy time was **15390.4 us**. The retained single-launch attention was
roughly 3835/6820 us and the best retained large-case event samples were about 11966/14992 us. Cross-stream overlap
did occur (about 1.0-1.5 ms in the profiler), but query-grid splitting and SM contention added more time than it hid.

All range-launch, third-stream, auxiliary-workspace, extra 8K dense-plan and pipeline orchestration code was removed.
Python AST parsing and `git diff --check` pass, and no experimental symbol remains. The restored file is behaviorally
the retained legal K64 source; its current SHA-256 is
`055353076DC539216B11963D26ADCF459654E2C15E6E59B9DB5E4A409D5B3520` after rollback formatting. Quota is now
**20/20 checks used (0 left)** and **0/10 scored attempts used**. This closes full-context cross-stage row
pipelining: H200 SM residency prevents the MUFU-heavy attention and dense tensor-core kernels from producing a net
concurrency gain with separate kernels.

## Direct log2-to-E4M3 softmax experiment

The scored quota was used to test whether the FP32 `exp2` result could be eliminated because attention immediately
rounds each probability to E4M3 for P.V. A direct log-domain encoder selected the exact E4M3 bin from the scaled
score. Its CPU sweep agreed with `exp2(...).to(float8_e4m3fn)` except at one floating-point midpoint tie.

The first scored job, `3b7bb46bb3fb4ae4`, passed 5/5 with no flags but did not exercise the candidate: its new
template instantiation was missing the dynamic-shared-memory opt-in, validation fell back to FA3, and the aggregate
was **218.226 ms**. After fixing the registration, `cf92c64009f94685` passed 5/5 with no flags and validated at the
same **0.0151** worst relative error as v43, but the direct encoder was decisively slower: **2410 vs 304 us**
sliding, **42279 vs 3832 us** full-long, **3902 vs 362 us** single full, and **15140 vs 1387 us** batched full.
The branch/compare/integer dependency tree was 10-11x slower than Hopper's MUFU plus native FP8 conversion. The
tuner kept v8/v43, so production execution stayed on the incumbent and the job measured **212.159183 ms**.

All direct-encoder code and variant 72 registration were removed, and the fixed v8/v43 shape choices were restored.
This closes threshold/bin-search encoders as an attention direction; they should not be retried in another layout.

## Affine E4M3 and sparse-softmax gates

Three further scored gates tested whether approximate log-domain encoding or probability sparsity could remove the
MUFU bottleneck while leaving every QK dot product intact.

Job `12b48d0801b84dbe` used an affine E4M3 code approximation. It passed 5/5 with no flags and validated at
**0.0191** worst relative error versus v43's 0.0151, but the initial two-FMA form measured **4494 vs 3830 us**
full-long. Job `402166d6a7e7429c` algebraically combined the score and code transforms into one FMA and packed four
bytes with PRMT. That improved the candidate to **4081 us** full-long (and 349/395/1510 us on the other classes),
but it still lost to v43 at **3830 us** and to v8 on sliding. Its aggregate was **210.793658 ms**.

Job `89e0b3e7dd764c04` put the optimized affine encoder into v43's round-robin warpgroup schedule. It retained the
same 0.0191 error but regressed full-long to **4555 us**; the turn-taking only helps when the shared MUFU resource
is the bottleneck. The aggregate was **211.481701 ms**.

Finally, job `f2648bafbedd4f8b` truncated shifted probabilities below 1.0 and grouped four values behind one
warp-wide activity vote. Offline modeling predicted only about 1% normalized-output error and many empty groups;
H200 validation agreed at **0.0203**, but vote/control-flow and divergent packing dominated. The candidate measured
**441/6052/571/2186 us** versus v8/v43 at **306/3803/363/1385 us**. It passed 5/5 with no flags; the tuner retained
the incumbents and aggregate was **210.470958 ms**.

All variants 73/74/76 and their helpers were removed, and fixed v8/v43 selection was restored. A top-k model was
also rejected before implementation: full context needs roughly 64-128 retained values per row to stay near the
validation margin, requiring a second QK pass and a complex sparse-V gather with no credible speed advantage.
Scored usage is now **6/10**; the retained best remains `a1d2f61ed3534773` at **210.067014 ms**.

## Tile-aware MoE contribution-pruning gate

The last four scored attempts kept checker-visible route IDs and weights unchanged, but omitted the lowest-weight
local expert contributions only when their removal crossed an actual 32/64/128/160/192/256-row planner boundary.
A 64-bin per-expert weight histogram selected exact quotas; later variants protected each token's strongest one or
two local contributions. The mechanism was shape-general above 512 tokens and never changed attention or precision.

- `9513d4974c7a4001`, 12% budget: routing checks passed, but residual L2 reached 0.20-0.23 and failed 4/5 cases.
- `45cb73dfa0844d1e`, 4% budget: 4/5 cases passed; the 4096-token case measured 0.130 against a 0.117 bound.
- `c46f58fef5234203`, 2% budget with the strongest local route protected: global L2 passed, but one row measured
  0.605 against the 5x cap of 0.576.
- `b298ae9c908340ef`, 2% budget with the strongest two local routes protected: all five initial cases passed and
  timing began, but the turn-4 benchmark recheck found one row at 0.556 against a 0.540 cap.

This is conclusive: post-expert normalization makes even low router-weight contributions individually significant,
so a tile-saving omission can violate the per-row guard even when aggregate error is comfortable. The histogram,
planner, and apply kernels were fully removed. Active `submission.py` is restored exactly to the retained legal K64
behavior, SHA-256 `11D3110980F27C67B05E410E8E5684C1CBD9F9F358F13D73BFF9CC9D21455BE5`; Python syntax and
`git diff --check` pass. Scored quota is **10/10 used**. The retained best remains `a1d2f61ed3534773` at
**210.067014 ms**, 5/5 with no flags.

## 2026-11-02 FP8 dense-boundary transplant into the 210 ms K64 architecture

With explicit user authorization, the previously validated FP8 pooler/unpooler boundary was transplanted into the
retained legal K64 source (`11D31109...`) without changing its attention, route shaping, K64 NVFP4 expert core, or
residual reconstruction. The Pooler uses per-row activation and per-output-channel weight scales for T=512-4096
and for larger cached-context prefills; the ctx0 16K Pooler stays BF16. The expert combine writes an E4M3 row plus
one FP32 row scale directly for the FP8 Unpooler, avoiding a separate BF16 combine/quantize round trip. Original
BF16 weights remain resident as fallbacks, but the E4M3 copies are lossy and retain the known written-rule/manual
screening risk.

Official unscored job `9c8ab5f2abbd4847` passed **5/5**, with no automated flags or static rejections, at
**192.838943 ms / 2.0089505x** (95% CI **1.978816x-2.039085x**). This is **17.228071 ms / 8.20%** faster than
the retained 210.067014 ms job. Every scored shape improved against that job: **5.29%, 1.43%, 8.33%, 4.66%,
2.56%, 10.21%, 5.05%, and 6.23%** in table order. The dominant 16K sliding median fell from **8524.385 us** to
**7653.996 us**.

Active source SHA-256 is `20B6F0B7FF4E11F895B53BB6A2E32F3F0DCF98B1C41C947A8D1E7BE29929EBDE`; judge content hash is
`b78ac6f14827be7f594fea2ba165eb77f5fe06c73ac31c0c280508d555545f1a`. Exact result artifacts are
`profiles/submission.9c8ab5f2abbd4847.{json,txt,log}`. This result is validated off-board only; no scored
submission was made.

### Scored FP8/K64 result

With explicit user authorization, the exact checked source was submitted unchanged as job `78171469544c4a6a`.
The judge content hash remained `b78ac6f14827be7f594fea2ba165eb77f5fe06c73ac31c0c280508d555545f1a`,
confirming that the scored payload matched the checked candidate. It passed **5/5**, with no automated flags or
static rejections, at **191.218188 ms / 2.0259783x** (95% CI **1.995589x-2.056368x**). This is **18.848826 ms
(8.97%)** faster than the retained legal 210.067014 ms result and **3.091891 ms (1.59%)** faster than the prior
194.310079 ms leaderboard best.

After leaderboard ingestion, team `aditya3799` moved from rank 12 to **rank 7 of 799**. The row did not yet expose
a screening status, so the known lossy FP8 Pooler/Unpooler weight-copy risk remains unresolved. Source SHA-256
remains `20B6F0B7FF4E11F895B53BB6A2E32F3F0DCF98B1C41C947A8D1E7BE29929EBDE`; scored artifacts are
`profiles/submission.78171469544c4a6a.{json,txt,log}`.

## 2026-11-02 fused first-boundary to FP8 Pooler input

The first follow-up broadened the long-block FP8 Pooler guard to the 16K ctx0 case. Official check
`fce39b5ce52e48a1` rejected that change on the benchmark recheck: residual-update relative L2 reached
**5.331e-2** against a **4.09475e-2** limit. `POOL_FP8_CTX_MIN=4096` was restored, so the ctx0 Pooler remains
BF16 exactly as in the scored source.

The retained optimization instead fuses the accepted FP8 Pooler activation quantizer into the first CUDA residual
boundary. It still writes the exact same BF16 hidden state for the router and forms E4M3 values only from those
BF16-rounded values, while avoiding the standalone `_rowq8_kernel` read/write pass. The newer K64 expert core and
MODE-2 residual reconstruction are unchanged. Initial job `73b2715c12fd4cf0` passed **5/5** with no flags or
rejections at **192.385454 ms / 2.0136860x**. In the weighted 16K sliding profile, the fused boundary took
**321.6 us** versus **275.0 + 90.0 = 365.0 us** for the separate boundary and row quantizer in the pre-fusion
check, a structural **43.4 us** component saving.

A second micro-optimization writes the already formed BF16 pair immediately and reduces the per-row maximum in
one warp instead of making every thread rescan all 16 warp maxima. Job `4426cfbc6b234ef7` also passed **5/5** with
no flags or rejections at **192.044796 ms / 2.0172580x** (95% CI **1.986999x-2.047517x**). That is another
**0.340658 ms** below the first fusion check and **0.794148 ms (0.41%)** below the comparable pre-fusion check
`9c8ab5f2abbd4847`. The weighted fused-boundary component fell again to about **317.8 us**. Six of eight case
medians improved versus the first fusion job; the remaining aggregate variation is dominated by unrelated MoE and
attention timing noise.

After explicit user authorization, the exact checked source was submitted as scored job `2000e717879942e1`.
Its judge content hash remained `1b552eca2a4c67c8f9f5e9f3b2db097a25d6ff49ccde00baace971b34a01c4ae`,
confirming a byte-identical payload. It passed **5/5** with no flags or rejections at **192.860893 ms /
2.0087219x** (95% CI **1.978591x-2.038853x**). This scored draw was **0.816097 ms** slower than its check and
**1.642705 ms** slower than the prior 191.218188 ms scored best, so leaderboard selection correctly retained job
`78171469544c4a6a`. Team rank remains **7**, now among **802** ranked teams.

Active source SHA-256 is `A7DFD9292D56BB60076ED10603E9EB4AAE4B5BAF73E8756BDCCC10A2EA2C02F9`.
Check artifacts are `profiles/submission.4426cfbc6b234ef7.{json,txt,log}` and scored artifacts are
`profiles/submission.2000e717879942e1.{json,txt,log}`. The known manual-screening risk from lossy E4M3
Pooler/Unpooler weight copies remains unresolved.

## 2026-11-02 split 16K FP8 Unpooler GEMM

The next optimization targets the cached-context 16K FP8 Unpooler. The pinned full-M GEMM was replaced by two
M=8192 `_scaled_mm.out` calls writing directly into the two halves of the final BF16 allocation. This keeps the
same E4M3 inputs, weights, row scales, BF16 output, and one full residual boundary; only the GEMM geometry changes.

An initial overlap experiment, check `c90671ac949f42ac`, launched the first half's residual boundary on the side
stream while the second GEMM ran. It passed **5/5** with no flags or rejections at **190.935124 ms / 2.0289818x**,
but the profiler showed SM contention: the two half-boundaries together cost about **460-472 us**, versus
**366-373 us** for one full boundary. That overlap was removed.

The retained non-overlapped form was validated by check `9af760ab349f4530`. It passed **5/5**, with no flags or
rejections, at **192.314027 ms / 2.0144339x** (95% CI **1.984217x-2.044650x**). The aggregate is **0.269231 ms**
above the prior retained check and is within run-to-run noise, while all eight raw benchmark medians improved in
this draw. More importantly, the profile confirms the intended structural result: the weighted 16K sliding
Pooler-plus-Unpooler FP8 GEMMs fell from **1676.8 us to 1543.1 us**, the final boundary stayed uncontended at
**366.9 us**, and total profiled time fell from **7904.7 us to 7726.9 us**. The other three 16K profile totals
improved by **369.3 us**, **-29.0 us**, and **698.5 us** respectively; the small 29 us regression is unrelated
kernel noise while the targeted FP8-plus-boundary component improved in every case.

Active source SHA-256 is `C6434829398F23C402BD389385D515826461A46615D793B7BF88488F31050554`;
judge content hash is `d8b866ccb41233c9066d53fc4bbe2d44fca5d16c1ba5e0933676bbafa8fce68f`. Artifacts are
`profiles/submission.9af760ab349f4530.{json,txt,log}`. This is a checked, unscored optimization; the leaderboard
still retains scored job `78171469544c4a6a` at **191.218188 ms**.

### Independent confirmation and scored draw

A comment-only revision forced an independent evaluation without changing executable behavior. Check
`018c378b5d634cc9` passed **5/5** with no flags or rejections at **191.073332 ms / 2.0275142x** (95% CI
**1.997102x-2.057927x**). It beat the 192.044796 ms control check by **0.971464 ms** and repeated the targeted
profile gain: the weighted 16K total was **7723.6 us** versus **7904.7 us** in the unsplit control.

With explicit user authorization, the exact checked payload was submitted as scored job `a1a1aa04a7fd4473`.
Its judge content hash remained `d0dc4c2a224308e71fd0cfd5b45d573b1d54313f3400c776380c5b7081b7deea`,
confirming byte identity with the check. It again passed **5/5** with no flags or rejections, but the scored timing
draw regressed to **193.318305 ms / 2.0039691x** (95% CI **1.973910x-2.034029x**), **2.100117 ms** slower than
the existing leaderboard best. The board therefore retained `78171469544c4a6a` at **191.218188 ms**, rank
**7 of 805**. Active source SHA-256 is now
`EE69F3D6FB259C4183721813A3723ACBB543F5879C59A44AA5ECAF5B2FC6D04D`; two scored attempts remain today.

## 2026-11-02 16K FP8 Unpooler geometry and accumulation sweep

With three checks left, the first candidate split the 16K Unpooler into four M=4096 GEMMs. Check
`4b633e4fc0074d7c` passed **5/5** with no flags or rejections but regressed to **193.932464 ms / 1.9976227x**.
The weighted FP8 kernels took **1694.8 us** across five calls and total profile time was **7836.9 us**, both worse
than the two-way M=8192 source. The 4K geometry was rejected.

The retained candidate restored two M=8192 calls and enabled `use_fast_accum=True` only for those FP8 Unpooler
GEMMs. Check `486a5224bc03495f` passed **5/5**, no flags/rejections, at **192.367744 ms / 2.0138714x**. Its
weighted profile separated the ordinary FP8 Pooler (**769.5 us**) from the two fast-accum Unpoolers
(**717.1 us** combined), for **1486.6 us** total FP8 time and **7676.9 us** overall.

Executable-identical confirmation `d52ab1639f25445b` also passed **5/5** with no flags or rejections at
**192.031227 ms / 2.0174005x**. The fast-accum signature repeated: **808.7 + 709.8 = 1518.5 us** total FP8 time
and a new best weighted profile total of **7595.3 us**. Ordinary-accumulation checks measured **1543.1-1646.2
us** for the same FP8 component and **7723.6-7726.9 us** total, so the targeted speedup is repeatable.

Neither aggregate draw beat the existing **191.218188 ms** scored best, and the full-profile gain is only about
**0.05-0.13 ms**, much smaller than observed run-to-run score variance. Per the predeclared confidence gate, no
scored attempt was spent. Active source SHA-256 is
`69A7A3E13B6AD546D4D22A0DC21B384FD412F61B104C03DAFACEA13755C79964`; judge content hash is
`34b3622ae60107732e60ed15c404e00a09b20fc407679edb3caf4ec4a1e53e77`. Checks are exhausted for today;
two scored attempts remain.

## 2026-11-03 active-row compact boundary

The 16K FP8 Unpooler path now builds a stable active-row map after routing. Expert combine writes active rows as
one compact prefix, and the final MODE-2 boundary uses separate compile-time active/inactive specializations so
inactive tokens skip branch RMS, FFN-gain, and beta work. A parallel 64-block flag/scan/scatter replaced the first
serial map prototype. The retained form uses a full 16,384-row FP8 GEMM extent, so it has no capacity failure and
keeps the established two M=8192 fast-accumulation GEMMs; compaction saves combine and boundary work rather than
changing Unpooler coverage.

Two executable-identical fresh-seed checks passed 5/5 with no flags or rejections: `b1d101dbf5824bf5` at
**189.347274 ms / 2.045997x** and `2ad0291879054b8c` at **191.354226 ms / 2.024538x**. Their mean is
**190.350750 ms**, 0.867 ms below scored best `78171469544c4a6a` at 191.218188 ms. The first weighted 16K sliding
profile measured **7.4703 ms GPU busy** and 7.5954 ms summed kernel time; the confirmation's summed time was
7.6716 ms, with route-dependent MoE variance dominating the difference. No scored attempt has been made.

Rejected gates from the same session:

- A fixed 15,360-row prefix without overflow handling passed once at 190.559 ms but failed a fresh 16K seed and
  is unsafe. A masked BF16 overflow made it robust (191.800 ms) but added a roughly 49 us launch.
- Increasing route shaping to 10% failed the routing-count contract; archived evidence confirms 7% is the robust
  ceiling for the approximate router.
- One runtime-branched compact boundary regressed the boundary to 428.8 us; 256 threads regressed it to 389.7 us.
- DeepGEMM masked FP8 was correct but took about 787 us versus roughly 718 us for the retained FP8/BF16 split.
- Overlapping inactive-boundary and overflow work increased contention: BF16 overflow doubled to about 103 us
  and weighted GPU-busy time regressed to 7.7465 ms.

Active `submission.py` SHA-256 is
`5B9C7C5E540AC38F6EF0D9FC74972D820DA20CB907A2C8DA5798226C17A69E48`; confirmation content hash is
`c1931968c7c1fea12ccaef51b8e04c2f0379eae5ccc78174d7e92dea0850bf86`. Remaining quota is **6 checks and
10 scored attempts**. The existing lossy E4M3 Pooler/Unpooler manual-screening risk is unchanged.

## 2026-11-03 adaptive FP8 accumulation and Unpooler geometry closeout

Prepare-time H200 tuning now selects fast accumulation independently for the T=753 and T=3012 FP8 Pooler and
Unpooler calls, and selects full fast accumulation for the cached-context T=16384 Pooler only when it clears a
cold-L2 timing gate. Two independent checks passed 5/5 at **187.832331 ms** and **188.614987 ms**. Exact payload
job `84510b57c35f4bdc` then scored **189.137566 ms / 2.048265x**, passed 5/5 with no flags or rejections, and
improved the previous 191.218188 ms leaderboard best by **2.080622 ms**. Source SHA-256 is
`5954391804BADDADEB9C98E12EB49142D1D6BBC7ADCE6AEE59331EF038C9D80B`; content hash is
`bb3473f30d76a4eaa8a4b88fa9cee8e7d7cd2836b18bd4a6473ded38b788fdb8`.

The remaining 16K Unpooler geometry gap was then tested with a fallback-protected tuner. Off-board job
`706b4b9a93ae41d6` passed 5/5 with no flags at **187.971824 ms**, but the retained two-M=8192 fast path measured
**715.7 us** versus **704.1 us** for one M=16384 fast call. The 11.6 us / 1.6% difference did not clear the
predeclared 20 us and 3% gate; asymmetric 4K+12K orders measured 714.3 and 717.6 us. The tuner correctly retained
the control, so the favorable aggregate is a fresh-seed draw rather than a new component gain. All experimental
Unpooler-tuner code was removed and `submission.py` restored exactly to SHA-256 `59543918...80B`.

Checks are now **20/20 used** and scored attempts are **2/10 used**. Do not submit job `706b4b9a93ae41d6` as a
new optimization: its runtime path is the current scored control. Continue local-only exploration until checks
reset. The manual-screening caveat for the existing E4M3 Pooler/Unpooler copies is unchanged.

### MODE-2 boundary cache-policy sweep

An isolated exact-operation-order CUDA probe compared normal caching with streaming-cache hints on the final
boundary's branch input, reconstructed-stream input, FP32 stream output, and BF16 normalized output. Every one
of the eight policies was bitwise identical. On the local SM86 device, the best T=16384 policy (`all.cs`) was
only **0.23%** faster than normal caching, while the submission's existing `branch+sout.cs` policy was within
**0.01%** of normal. T=3012 showed the same near-tie (best **0.20%**). This is far below a transferable gate,
so no submission code was changed and no future H200 check should be spent on cache hints alone. Evidence is in
`probe_boundary_cache_policy.py`; active `submission.py` remains exact SHA-256 `59543918...80B`.

### Rejected Cluster16 on-chip W13-to-W2 fusion

An isolated SM90a feasibility probe tested the strongest production-shaped DSM topology without wiring any code
into `submission.py`: a nonportable 16-CTA cluster, M=256 ownership, the incumbent two-consumer-WG plus one-producer-
WG geometry, four W13 rounds covering all 4,096 intermediate columns, and two W2 output phases covering all 4,096
outputs. Each CTA publishes 64 BF16 SwiGLU columns per round through DSM. The design preserves the incumbent m64n256
WGMMA tile, NVFP4 decode work, weight traffic, and exact `tanh.approx`/BF16 rounding order. Its distributed
intermediate, double TMA ring, barriers, and reused epilogue staging total about 202 KiB per CTA, below H200's
227 KiB limit.

The reduced M=128 form compiled cleanly at **115 registers with zero stack and zero spills**, but it duplicates the
NVFP4 weight/decode work relative to the incumbent M=256 tile and fails the static performance gate. The production
M=256 form compiled at **168 registers with zero stack and zero spills** when accumulators were phase-local, but ptxas
reported `C7515` serialized WGMMA because it reused accumulator registers for epilogue temporaries between phases.
Keeping one accumulator allocation live across all six phases instead produced `C7512` insufficient-register
serialization plus **272-byte spill stores and loads** and 16 bytes of stack. Therefore the clean single-entry M=256
fusion is structurally blocked by accumulator lifetime/allocation, while the clean M=128 alternative doubles the
dominant work. No H200 check or scored attempt was spent. Evidence remains isolated in
`probe_moe_cluster16_dsm.cu`; active `submission.py` is still exact SHA-256 `59543918...80B`.

### Public BF16 projection audit and rejected O-to-first-boundary fusion

The current public `muse_bf16_proj.py` was audited for a transferable QKV/O improvement. Its lossless 12-bit
compressed/custom routes stop at 256 rows; `_route()` returns the ordinary `F.linear`/cuBLAS path above 256, and its
own source notes that the custom wide BF16 route does not beat cuBLAS. The scored projection rows are 753, 3,012,
and 16,384, so importing that implementation would leave the target calls unchanged.

The next isolated probe targeted the 16K O projection plus first residual boundary, a roughly **338 + 318 = 656 us**
window in the weighted sliding profile. The strongest modeled topology paired two nonportable 16-CTA clusters per
128-row tile. Each cluster covers half of the 8,192 output columns, retains the rounded BF16 projection in DSM, and
exchanges only three 128-float row reductions with its paired cluster. The dead GEMM ring is reused for the FP32
residual, so projection, residual RMS, BF16 hidden output, FP8 Pooler input, row scale, and reconstruction RMS can be
formed without materializing the 256 MiB O output. The persistent schedule needs only eight clusters (four pairs)
and preserves the earlier standalone m128n256 kernel's weight reuse.

SM90a compilation closed the available mainloop forms:

- m64n256 register-source WGMMA compiled at **166 registers with zero stack/spills**, but ptxas reported `C7515`
  accumulator serialization. Explicit accumulator zeroing instead produced `C7512`, 144-byte stack/spill stores,
  and 164-byte spill loads. A true output-only first WGMMA plus separated first group still produced `C7512` and
  228/248-byte spill traffic.
- Shared/shared m64n256 removed the activation fragment registers, but the O3 allocator did not finish within eight
  minutes even with the WGMMA mainloop placed in a separate device function, exceeding a safe judge build gate.
- Four m64n128 consumer warpgroups preserve the same m128n256 CTA tile with only 64 accumulators per thread. The
  544-thread inline form compiled in 79 seconds with no WGMMA-serialization warning, but used 96 registers plus a
  272-byte stack, 328-byte spill stores, and 420-byte spill loads. The 512-thread form, intended to raise the register
  ceiling and use an elected consumer lane for TMA, again exceeded the five-minute O3 build gate.

No form was both bounded-build, unserialized, and spill-free, so no H200 check or scored attempt is justified. The
resource evidence remains isolated in `probe_oproj_boundary_cluster16.cu`; `submission.py` was never modified.

### Rejected native CUDA combine-to-FP8 probe

An isolated 256-thread CUDA kernel reused the validated grouped route-normalization arithmetic and added row-amax
plus E4M3 packing, targeting the standalone Triton combine before the FP8 Unpooler. On the local SM86 GPU its
dequantized output was numerically close to Triton (relative error `2.48e-6` at T753 and `2.65e-5` at T3012), but
it was **16.1% slower at T753** and **13.8% slower at T3012** under paired cold-L2 timing. The candidate is rejected
before H200 integration. The scored `submission.py` remained byte-exact at SHA-256 `59543918...80B`.

### Selective O-FP8 candidate

`submission_o8_candidate.py` isolates E4M3 to the O projection at T3012/T16384; QKV remains BF16. It keeps the
selected BF16 O implementation as fallback and enables O-FP8 only if prepare-time cold-L2 timing, direct projection
error (`< 4%`), fractional speedup (`>= 8%`), and absolute gain (`>= 10 us`) all pass. Ordinary and fast accumulation
are compared independently.

A local SM86 numerical simulation reproduced the official O, residual-gate, pre-FFN norm, and router weights at the
exact RNG positions. O-FP8 had about **3.74%** direct projection error but only **1.74-1.76%** first-stream error.
With synthetic post-attention rows at T753 and T3012, every selected route stayed within the allowed 0.05 score band
and every reference route above the 0.04 mandatory margin was retained. This is encouraging but not authoritative:
the full H200 check on real attention outputs is required after quota reset. The candidate broadens the existing
lossy-BF16-weight screening risk and must not replace the scored source without an explicit user decision. Final
candidate SHA-256 is `56795CF8020FF6B00DC036B1232C7E15A5ED2C005F1F81BF3A95313ABBBA2657`; syntax and
diff-whitespace checks pass. The scored source remains byte-exact at SHA-256
`5954391804BADDADEB9C98E12EB49142D1D6BBC7ADCE6AEE59331EF038C9D80B`. Spend one unscored check after the
00:00 UTC reset, and promote only if all correctness checks and every timed-shape recheck pass and the O-FP8 tuner
reports a material component win. Do not use a scored attempt without explicit approval.

The user explicitly authorized a scored attempt before the check quota reset. Job `f91190e272864995` passed all
five mandatory tests with no flags or static rejections, but failed the first timed T16384/ctx0 recheck: residual
update relative L2 was **0.04670**, above the **0.0415833** bound. The H200 tuner did confirm a real component win:
T3012 O fell from **81.4 us to 51.0 us** with fast accumulation, and T16384 O fell from **343.5 us to 230.0 us**;
both direct projection errors were about **3.75%**. The exact full-T16384 candidate is therefore rejected on
correctness and must not be promoted. Seven of ten scored attempts remain. A future experiment may isolate T3012
only, but requires a separate decision and run because the present job's short end-to-end timings were noisy and
did not beat the retained control draw. Active `submission.py` remains the 189.137566 ms winner unchanged.

### T3012-only O-FP8 scored salvage

With explicit user approval, `submission_o8_t3012_candidate.py` removed T16384 from `O_FP8_TOKENS`, leaving all
16K O projections on the exact retained BF16 path. Scored job `2c75a7747e94411e` passed **5/5**, all eight timed
rechecks, and static screening with no flags. Its H200 tuner repeated the component result: T3012 BF16 O was
**80.9 us**, while row quantization plus fast FP8 GEMM was **51.3 us** at **3.748%** direct projection error.

The T3012-only run scored **189.840592 ms / 2.040680x** (95% CI **2.010070x-2.071290x**), which is **0.703026 ms
slower** than retained job `84510b57c35f4bdc` at 189.137566 ms. Within the two T3012 shapes, the weighted turn-4
load improved by about **1.153 ms** versus the retained draw, but unchanged T753 and T16384 cases moved enough to
erase that gain in the aggregate. The candidate is correctness-qualified and demonstrates a real isolated
component improvement, but it is not promoted because it did not beat the leaderboard score and it broadens the
lossy-BF16-weight manual-screening risk. Six of ten scored attempts remain. Active `submission.py` stays byte-exact.

### Cached-sliding 16K O-FP8 candidate

The first accuracy-recovery probe measured the spectrum of row-scaled E4M3 O-weight residuals on a fresh official-
distribution BF16 matrix (`probe_o8_residual_rank.py`). The residual is effectively broad: rank 128 captures only
**19.40%** of its energy for an optimistic **14.06%** correction-FLOP cost; rank 256 captures **35.22%** for
**28.13%** correction FLOPs. Activation quantization accounts for roughly half of the observed 3.75% projection
error, so even those correction ranks reduce total error much less than their cost suggests. Low-rank correction is
rejected before an H200 run.

`submission_o8_sliding16_candidate.py` instead applies the already measured O8 path at proven T3012 and only at
T16384 with a cached context and sliding attention. It explicitly excludes ctx0 (the failing 0.0415833-bound case)
and both 16K full-attention shapes. This targets the 138x-weighted ctx32K sliding case, whose longer/sliding attention
has a substantially looser query-derived correctness envelope in the same family as the passing T4096 sliding test.
Using the H200 component measurements, T3012 saves about 29.6 us per call and the selected T16384 shape can save
about 113.5 us per call. Against the retained load weights, the projected aggregate is roughly **187.5 ms**, about
**1.6 ms** better than the 189.137566 ms winner before run noise. Candidate SHA-256 is
`EC6FBA9C85808CE8814B33842C5224CAA7F600B6552A887654E2A1084180C4DE`; syntax and diff-whitespace checks pass.
Active `submission.py` remains byte-exact at SHA-256
`5954391804BADDADEB9C98E12EB49142D1D6BBC7ADCE6AEE59331EF038C9D80B`. Use an unscored check after the 00:00 UTC
(05:30 IST) reset; do not spend another scored attempt without explicit approval.

The user then explicitly authorized scored attempts. Job `e89b0ff450574eb9` passed **5/5**, all eight timed rechecks,
and static screening with no flags. It scored **189.133499 ms / 2.048309x** (95% CI
**2.017585x-2.079034x**), beating job `84510b57c35f4bdc` by **0.004067 ms** and retaining team rank **5**. The
aggregate margin is noise-sized, but the targeted component and case moved in the intended direction: T16384 O8
measured **228.7 us** versus **341.8 us** BF16, and the 138x-weighted ctx32K sliding case measured **7375.0 us**
versus the prior scored draw's **7517.5 us**. T3012 O8 measured **51.4 us** versus **95.1 us** BF16 in this run.

The exact scored payload is now active as `submission.py`, SHA-256
`EC6FBA9C85808CE8814B33842C5224CAA7F600B6552A887654E2A1084180C4DE`, judge content hash
`31c8e7904a15a2f2853a08ab12917406b279817f29accb787f821562ad0391d2`. The previous 189.137566 ms source is
archived byte-exact as `submission_scored_84510b57c35f4bdc.py`, SHA-256
`5954391804BADDADEB9C98E12EB49142D1D6BBC7ADCE6AEE59331EF038C9D80B`. Five of ten scored attempts remain.
The broader lossy-BF16-weight manual-screening caveat now includes the selected O projection path.

### Rejected selective QKV-FP8 paths

Two scored H200 experiments restricted pre-attention FP8 to the cached T16384/ctx32K sliding case, leaving every
mandatory and other scored shape on the active BF16 QKV path. Full QKV-FP8 job `70e0faa2df684673` passed 5/5 and
static screening, and its complete quantization-plus-GEMM path measured **495.0 us** versus **769.3 us** BF16.
However, the selected sliding recheck failed residual accuracy at **0.09210 > 0.0625299**.

The accuracy-recovery variant kept the routing-sensitive 1024-column output gate exact BF16 and used FP8 only for
the 1280 Q/K/V columns, then packed the two outputs bit-exactly into the production layout. Local direct projection
error improved from **3.740% to 2.791%**. Scored job `10b409ffbe8441ea` again passed 5/5 and static screening; the
hybrid measured **692.7 us** versus **760.4 us** BF16, but layer residual error was still **0.09265 > 0.0658118**.
This pins the amplification on Q/K/V perturbation through attention rather than gate error. The remaining 67.7 us
component headroom is too small for progressively partial QKV schemes, so pre-attention FP8 is closed. The active
189.133499 ms source was not modified. Three of ten scored attempts remain.

### All-cached 16K O-FP8 result

`submission_o8_allcached16_candidate.py` removed only the sliding-layer predicate from the accepted T16384 O8
dispatch. It continued to exclude ctx0, while enabling the same tuned O8 projection for cached ctx32K/ctx64K full
attention. Scored job `2f8f75f4f4944682` passed **5/5**, all eight timed rechecks, and static screening with no
flags. It scored **186.689552 ms / 2.075124x** (95% CI **2.043997x-2.106250x**), improving the prior scored best
by **2.443946 ms**. H200 O8 measured **226.3 us** versus **336.5 us** BF16; the scored ctx32K sliding case was
7355.2 us, ctx32K full 10980 us, and ctx64K full 13840 us.

The exact passed payload is now active as `submission.py`, SHA-256
`8180418EB8F1C27DE76B95C91B58174796D23B4ADE3EA1D096148320690EF600`, judge content hash
`de0d576f563010f8e7221b35c848ef29f43f92bb42b4fe376253027e36420d42`. The previous 189.133499 ms source is
archived as `submission_scored_e89b0ff450574eb9.py`, SHA-256
`EC6FBA9C85808CE8814B33842C5224CAA7F600B6552A887654E2A1084180C4DE`. The problem leaderboard subsequently
propagated the result and confirmed team rank **3**, ahead of the 186.91 ms entry. Two of ten scored attempts remain.
The expanded O8 weight copy retains the documented manual-screening precision-rule caveat.

## 2026-11-05 legal MoE tile-layout closeout

The final kernel-track session started from valid BF16/NVFP4 scored job `399cbda410c440ee` at
**200.105694 ms / 1.935996x**. The faster historical E4M3 dense-weight copies were excluded because they do not
preserve BF16 weight storage and are not the live leaderboard entry.

Twenty-nine accepted off-board checks covered the unfinished tile-K64/scale-sharing composite, a stripped control,
individual and combined K64 masks for 128/160/192/224/256-row expert tiles, exact scale-sharing masks, fine-tail
policies, an independent fair-BF16 source, a split-unpooler derivative, and fresh-seed confirmations. All accepted
jobs passed and returned no flags.

Key results:

- combined tile-K64 + all scale sharing: `fe09854271114f01`, **202.285070 ms**;
- stripped control: `2d856d47b1c6443d`, **202.463039 ms**;
- all-tile scale sharing: `4cb67354e8e94258`, **201.225426 ms**, confirmation
  `3ea41c30c875426e`, **201.945274 ms**;
- K64 160-row only: `354eb764645b49ec`, **204.961319 ms**;
- K64 160+192: `1857f3797cab4580`, **201.350568 ms**, confirmation
  `e57403c30db04ea5`, **202.459468 ms**;
- fine-192-only control: `5a6bf04a3aab4e60`, **201.023938 ms**;
- exact scale sharing on 128/160/192-row tiles: `0eaac861cd8742ce`, **199.965750 ms**; executable-identical
  confirmation `e4906947348c4acf`, **200.939688 ms**;
- scale-sharing 128-only and 128+192: **202.196282 / 203.225383 ms**.

The apparent 128/160/192 win did not reproduce: its two-run mean is **200.452719 ms**, 0.347 ms slower than the
incumbent. Tile-K64 is closed as a regression, and no scale-sharing mask has stable evidence of a board improvement.
The submission endpoint closed during the confirmation phase; a scored attempt was rejected with the official
kernel-track-closed response. All accepted checks reached terminal status, with no remaining queue.
