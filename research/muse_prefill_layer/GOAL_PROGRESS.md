# Muse Prefill Optimization Goal - Current Resume Checkpoint

Last updated: 2026-11-01 IST, after the fine-grained W13->W2 pipeline experiment.

This section supersedes every older checkpoint below. Keep the older sections as experiment history only.

## Outcome and standing

- Best rule-safe **scored** result is still job `3d31ea828106401f`, **210.713 ms**.
- Best rule-safe **checked/off-board** candidate is now job `a1d2f61ed3534773`, **210.067 ms**, 5/5 mandatory
  checks, no flags or rejections. It is 0.646 ms faster than the scored control, but only about 0.31%; that is too
  close to run/route variance to spend a scored attempt yet and remains 12.067 ms above the 198 ms goal.
- Active `submission.py` is logically restored to the checked K64 candidate. Its current SHA-256 is
  `8C64DB941210FD6EDFC053A30FE87C81CB2F8026C0F798BC4B2CDC41CFC65B80`; the difference from the uploaded
  `5F5D...` source is line wrapping around the temporarily extended fused-kernel interface, not executable logic.
- The only retained change from the prior legal source is a lossless K=64 permutation of the original NVFP4
  weights, with the matching activation-fragment order, for T>=8192 as well as the existing T<2048 path.
  Dense weights/activations remain BF16 and expert storage remains NVFP4.
- Current prefill quota is **12/20 checks used (8 left)** and **0/10 scored attempts used (10 left)**.

## External repository audit

The relevant upstream kernels were checked out under `D:\OpenKernelChallenge\inference_research`:

- DeepGEMM `5081f8513af5`
- FlashMLA `395cb3afbb33`
- FlashInfer `1ec41b4b0e21`
- TensorRT-LLM `5d86481f60bd`
- Humming `948e6fd547e7`

The source-level compatibility audit matters more than repository headline numbers. DeepGEMM and FlashInfer's
newest native FP4 paths either target SM100 or use FP8 activations on Hopper, which violates this task's BF16
activation contract. TensorRT-LLM's legal Hopper BF16xNVFP4 route resolves to Marlin, already represented by the
challenge baseline. FlashMLA main now targets SM100; its older Hopper techniques (persistent scheduling, fine TMA
pipelines, PDL and warp specialization) are already present in this source. Humming provides the closest legal
Hopper W4A16 comparison and was tested directly.

## Measured upstream-derived experiments

1. A DeepGEMM/FlashInfer-style dynamic W13/W2 frontier was tested at 2:1, 3:1 and 4:1 issue ratios. Jobs
   `6f36c78be45e49d1`, `4bdf94bf90d4415b`, `50e8280e89b04bb3` and `fca5c6677c8f4468` measured
   **212.351 / 211.321 / 210.922 / 213.066 ms**. Even the warmed 3:1 sample lost to the scored control; the
   original all-ready-first queue was restored.
2. The exact Humming `muse_moe_w4a16` kernel was tuned and integrated for T>=8192. It passed 5/5, but job
   `9b5340c0f17b4fd4` measured **225.841 ms**; the separate W13, SiLU and W2 launches were much slower than the
   fused persistent kernel. Removed.
3. FlashInfer's K=64 layout idea transferred cleanly. The first launch (`3794584af0774e87`) exposed one missing
   dynamic-shared-memory attribute on the new `<FINE=1,KPERM=true>` specialization. After the mechanical fix,
   job `a1d2f61ed3534773` passed 5/5 at **210.067 ms** and is retained.
4. Cross-slot register prefetch (`MOE_XPF256=1`) passed 5/5 in job `ce19bf15a9494716` but measured
   **210.659 ms**, 0.592 ms behind K64-only. Removed.
5. FlashInfer-style `EVICT_FIRST` was applied only to T>=8192 expert-weight bulk copies. Job
   `6eb2b70d839948d3` passed 5/5 but measured **212.148 ms**. This agrees with FlashInfer's own comment that
   expert weights spanning several M-blocks should keep normal L2 priority for cross-CTA reuse. Removed.
6. A larger producer-consumer redesign published each W13 64-feature chunk separately and let W2 wait only for
   the 1--4 chunks needed by its current K-stage. The per-tile 2:1 schedule (`e68cc05206174ca6`) passed 5/5 at
   **220.107 ms**. Preserving each GEMM's global flat order with a 3:1 stream (`bf25c0116b894336`) also passed
   5/5 but reached only **214.664 ms**. The latter improved the first design by 5.444 ms, yet remained 4.597 ms
   behind K64-only. The readiness mask, polling, schedule mapping and dispatch flag were removed.

## Exact next step

- Do not score the 210.067 ms sample solely on its aggregate delta; it is a useful legal base, not evidence of a
  stable 12 ms path to 198 ms.
- Do not repeat ordinary frontier ratios, Humming's separated full kernel, cross-slot register prefetch, or
  evict-first on GEMM operands. The current kernel already uses the upstream RS/swap-AB structure (packed weights
  decoded into the register-source operand), persistent work stealing, fine TMA staging, and WGMMA issue/wait
  overlap.
- The measured chunk pipeline shows that merely overlapping W13/W2 while retaining the global BF16 intermediate
  is insufficient: per-stage acquire/proxy fences and mixed weight-stream pressure outweigh the overlap. A future
  W13->activation->W2 design must actually eliminate or localize intermediate traffic (or remove another
  multi-millisecond whole-layer boundary), while preserving BF16 activations and NVFP4 expert storage. Keep the
  K64 candidate as the source of truth for that work.

---

## Archived 2026-10-31 checkpoint

Last updated: 2026-10-31 IST, after exhausting the prior prefill quota.

The following is preserved as the prior checkpoint; the 2026-11-01 section above is authoritative.

## Outcome and standing

- Best rule-safe scored result: job `3d31ea828106401f`, **210.713 ms**.  It keeps dense weights/activations BF16
  and expert weights NVFP4; it does not use the lossy FP8 dense-weight copies that triggered screening concern.
- Current top-10 cutoff at the final query: approximately **192.496 ms**.  The legal gap is therefore about
  **18.217 ms**.  A 198 ms result would no longer reach top 10.
- Faster historical job `910fa5d559e44d0d`: **194.310 ms / 1.99374x**, but it uses guarded FP8 pooler/unpooler
  weight copies and its leaderboard screening is under review/flagged.  Preserve the job and exact backup
  `submission_scored_910fa5d559e44d0d.py`; do not withdraw it without explicit user authorization.
- Current daily quota is exhausted: **20/20 checks and 10/10 scored attempts used**.  Query `okc problems --json`
  after the next reset before starting new remote work.

## Active source

- `submission.py` retains the rule-safe architecture plus the lossless NVFP4 K-permutation only for the small
  T753 expert workload (`KPERM_MAX_T = 2048`).
- Current SHA-256: `0E433BB3B7AD71D5CAD5BAF06C1F09BDAD882E19CE3C0E0E02E00E4822C72130`.
- This is a logically reconstructed rollback of the proven T753-only source, not a byte-identical copy of its
  earlier `64760B...` archive.  Python AST parsing and `git diff --check` pass; the current file is the source of
  truth for resuming.
- The KPERM is a pure permutation of the original NVFP4 expert weights and matching BF16 activations.  It does
  not reduce precision.  Dense projection weights remain their original BF16 tensors.

## Final KPERM experiment sequence

1. T753-only KPERM, job `e2ddd06fa31e4423`: **211.761 ms / 1.82944x**, 5/5, no flags.  The T753 MoE kernels
   measured about 235-239 us and showed the expected component gain.
2. Broad T3012 KPERM, job `f76d6f8d2755447d`: 5/5 and no flags, but benchmark timeout.  Root cause was dispatching
   the `FINE=0` KPERM template for fine 96/160/192/224-row tiles, so W13 completion counters were never published.
3. Corrected broad T3012 KPERM, job `d4739ce200d04865`: **213.084 ms / 1.81808x**, 5/5, no flags.  The weighted
   sliding T3012 MoE regressed from roughly 688 to 718 us, so the extension was rejected.
4. Full-attention-only T3012 KPERM, final job `86b59c20237446ff`: **211.474 ms / 1.83192x**, 5/5, no flags.  Its
   full T3012 MoE measured 721.1 us and failed to reproduce the earlier 627.7 us sample.  Reject this selective
   extension too; it is noise/distribution-sensitive and not retained.

## Exact next step

- Resume only after quota reset.  First re-query the problem limits, leaderboard, and screening state.
- Do not spend more runs on T3012 KPERM, ordinary multi-CTA/ring-depth variants, static W13/W2 queue interleaving,
  fence substitutions, generic custom BF16 projections, smaller/asymmetric Unpooler chunks, or dormant attention
  scheduling flags.  Each has direct negative evidence in `OPTIMIZATION_NOTES.md`.
- Closing an 18 ms top-10 gap requires a genuinely new legal architecture, not another launch-scale tweak.  The
  highest-value target remains the weight-138 16K sliding case: redesign ownership/dataflow around the NVFP4 MoE
  and the BF16 dense pipeline so a multi-millisecond intermediate or synchronization boundary is removed while
  retaining the original precisions.
- Read `TASK.md`, `AGENTS.md`, this top checkpoint, and the end of `OPTIMIZATION_NOTES.md` before editing or
  submitting.  Never treat automated 5/5 acceptance as permission to use lower-precision weight copies.

---

# Historical Checkpoints

Last updated: 2026-10-30 IST

## Active goal

The public external candidate crossed 210 ms off-board, and its FP8 pooler idea has now produced a smaller but real scored improvement when isolated inside our incumbent. Continue separating automated numerical acceptance from final source-rule screening: the written rules' BF16-weight requirement makes any lossy FP8 weight copy provisional.

## Verified best and quota

- Personal-best scored job: `367f1182aac845d0`.
- Result: **203.879 ms / 1.90017x** (95% CI **1.87166x-1.92867x**), 5/5 mandatory checks, no flags or static rejections.
- This is **36.881 ms (15.32%)** faster than the prior 240.760 ms scored incumbent.
- Exact submitted `submission.py` SHA-256: `4e2730f95fdfac37e84f6663d66ba2cebf206eb2028807c6af9dd82f6d9db909`; judge content hash: `3387b9dec3b8b31fd2ef17107009949e7e6a6209409563f824b31fec53131420`.
- At the post-submit query the job was rank **25 of 690** with speedup 1.9002; its leaderboard row did not yet contain a screening status.
- Its validating off-board job was `97777aa934f14ee6`: **241.624 ms / 1.60333x**, 5/5 with no flags or rejections.
- Exact external-candidate check: `51788973353f41b5`.
- Result: **197.033 ms / 1.96619x** (95% CI **1.93670x-1.99569x**), 5/5 mandatory checks, no flags.
- This is **43.727 ms (18.2%)** faster than the current scored incumbent and **12.967 ms below** the 210 ms target.
- Candidate source: `submission_external_rohan_196619.py`, SHA-256 `c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb`.
- `submission.py` is now the scored hybrid: the incumbent attention, router, W4 MoE, and residual paths plus guarded E4M3 pooler and unpooler projections.
- Confirmed quota after scored job `367f1182aac845d0`: **11 unscored checks and 7 scored attempts** remain.

## Completed continuation of yesterday's plan

1. `p8` finished SM90a compilation at **168 registers, zero stack, zero spill**; its local parity suite was exact.
2. Unscored H200 job `d4b4e6970ef24c1c` compared the production pin, `p4`, and `p8`. Both experimental schedules lost at all three token counts. The run passed 5/5 and measured 261.234 ms through the unchanged control path.
3. `p4` and `p8` were fully removed. The restored five-mode W4 extension is keyed `pf2_w4g_v52`; the production wide kernel recompiles at 168 registers with no stack or spill.
4. A split residual boundary saved 12.8% of the local tail kernel but projected to only about 70 us on H200, below the 0.20 ms gate. It was reverted without a remote run.
5. A fused pooler/router projection was tested in unscored H200 job `9c2e7a2529674509`. It passed 5/5, but was slower at T753/T3012/T16384: 106.5/327.3/1838.8 us versus 99.1/301.3/1557.2 us for the current two-stream path. It was removed.
6. A third BF16 router residual enabled a guarded 7.75% route budget. Jobs `1a59a2833ef449a8` and `d2bd31a73b664f0f` both passed 5/5 with no flags. With the candidate active at 16K, weighted W4 time rose from 3.933 ms to 4.039 ms and GPU-busy time from 9.851 ms to 9.927 ms. The entire candidate was removed and the exact two-fragment/7% control restored.

Full timings and architectural conclusions are in `OPTIMIZATION_NOTES.md`.

## Recovery audit after the no-growth review

The dense unpooler-to-residual boundary was isolated under a strict no-quota gate.

1. A two-chunk token pipeline was bit-exact, but local timing was **228.7 ms** against **146.5 ms GEMM + 81.2 ms residual**. There was effectively no overlap because the dense GEMM occupied the GPU. Four chunks regressed to about **342.0 ms**.
2. A materially different residual kernel reconstructed `stream1` from the already-produced normalized BF16 tensor, eliminating the dense BF16 `a` and FP32 input-stream reads. It preserved the first boundary output bit-for-bit. Against the current fused residual, stream relative error was **0.139%** and hidden relative error **0.244%**.
3. The reconstruction kernel reduced local residual time from **10.220 ms to 7.410 ms (27.5%)**, and the full two-boundary probe from **16.219 ms to 13.388 ms (17.5%)**. On the measured H200 residual cost this projects to only about **0.15 ms**, below the agreed **0.20 ms** remote-run gate.
4. No H200 check or scored attempt was spent. The probe-only CUDA ABI was removed and `submission.py` restored to `pf2_glue_v20`.

## Current source state

- `submission.py` now contains the validated off-board persistent-MoE hybrid from job `a82a291ef13342e4`: **203.340 ms / 1.90520x**, 5/5, no flags or static rejections. Source SHA-256 is `4e2730f95fdfac37e84f6663d66ba2cebf206eb2028807c6af9dd82f6d9db909`.
- The current shaped router, attention, residual path, QKV/O BF16 projections, and FP32 stream remain unchanged. The new core losslessly repacks NVFP4 weights, keeps BF16 expert activations/norms, and retains the prior W4 path as fallback.
- Guarded FP8 pooler and FP8 unpooler remain enabled; their previously recorded manual precision-screening caveat is unchanged.
- The Python source parses and `git diff --check` passes.
- The protected production pins and the stricter routing safety guard remain unchanged.
- Probe files are retained as evidence so rejected experiments are not accidentally repeated.
- `probe_unpool_res_pipeline.py` retains the exact pipeline evidence; the reconstruction probe was removed with its probe-only ABI after its result was recorded above.

## Confidence and next decision

- Confidence that the persistent-MoE hybrid is numerically and operationally accepted by the current automated judge: **high** after job `a82a291ef13342e4` passed 5/5 and cleared the strict performance gate.
- Confidence that the lossy FP8 pooler survives final manual precision screening: **low until the organizers explicitly accept it**.
- The isolated pooler improves our architecture by only 1.62%; reaching 210 ms still requires porting other architectural gains or a materially different legal dataflow.

Instruction grouping, CTA/remainder geometry, decoded-weight sharing, cluster multicast, tail split-K, routing consolidation, larger approximate-route budgets, compact unpooling without a fused reorder, FA4, output-row pipelining, residual splitting, and naive pooler/router fusion have all failed their direct gates. Do not reopen them without a materially different dataflow.

## Exact resume point

1. Preserve scored job `367f1182aac845d0` and source hash `4e2730f9...` as the on-board best: **203.879 ms / 1.90017x**, rank **25 of 690** at the immediate post-submit query.
2. Its exact off-board validation is job `a82a291ef13342e4` at **203.340 ms / 1.90520x**. Confirmed remaining quota is **11 checks and 7 scored attempts**.
3. The persistent expert core adds no new precision-rule issue, but the existing lossy FP8 pooler/unpooler copies retain the written-rule/manual-screening risk. Automated acceptance is not final organizer approval, and the new leaderboard row had no screening field yet.
4. For further optimization, preserve the submitted source and target long-context full attention in an experimental copy; the 65K full case still spends about 7.65 ms there.

## 2026-10-30 verified 197 ms candidate

- `submission.py` now contains the exact complete candidate validated by official job `51788973353f41b5`: **197.033 ms / 1.96619x**, 5/5 passed. Source SHA-256 is `c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb`.
- This improves on scored job `367f1182aac845d0` by **6.846 ms (3.36%)**. The scored incumbent is preserved exactly in `submission_scored_367f1182aac845d0.py`.
- The production MoE schedule remains unchanged after alternative queue orders and remainder handling lost on the two real 16K route profiles. Caching BF16-dequantized NVFP4 weights was rejected as rule-incompatible before any official run.
- Job `e3c417319bf14a06` deduplicated to the validated job and consumed no quota. Remaining quota is **10 checks and 7 scored attempts**. No scored submission has been made for the 197 ms source.

## Scored result `d81535430cfd4b48`

- The exact active source passed 5/5 with no flags or static rejections at **199.900 ms / 1.93799x**, 95% CI **1.90892x-1.96706x**.
- It improves the previous scored best by **3.979 ms (1.95%)** and moved the team from rank 25 to **rank 19 of 697**.
- Source SHA-256: `c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb`; judge content hash: `bef68ced286bf325cd328ddde1af93b1fac5b0690cf2d2efc4bb1a680541e526`.
- Exact backup: `submission_scored_d81535430cfd4b48.py`. Remaining quota: **10 checks and 6 scored attempts**.

## Checked route-shaper candidate `fa2c5aa06ac64db9`

- The conservative <=7% route shaper is now ported into the lean architecture, including a correct rebuild of per-expert positions after rerouting.
- The official unscored H200 check passed 5/5 with no flags at **194.842 ms / 1.98830x**, improving the current scored best by **5.058 ms (2.53%)** and clearing the current top-10 threshold.
- Active and checked source SHA-256: `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`; judge content hash: `9b3ec873a50e50a70d655b1d877dbbbcb135ed431f2a5894104c102b95241cf0`.
- Exact checked backup: `submission_checked_fa2c5aa06ac64db9.py`. No scored submission has been made. Remaining quota: **9 checks and 6 scored attempts**.

## Scored route-shaper result `910fa5d559e44d0d`

- The exact checked route-shaper source was submitted unchanged; judge content hash `9b3ec873a50e50a70d655b1d877dbbbcb135ed431f2a5894104c102b95241cf0` matches the check.
- It passed 5/5 with `checks.ok=true`, no flags, and no rejections at **194.310 ms / 1.99374x**, 95% CI **1.96383x-2.02365x**.
- This improves the previous scored best by **5.590 ms (2.80%)** and moves the team from rank 19 to **rank 7 of 700**.
- Active source SHA-256: `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`; exact scored backup: `submission_scored_910fa5d559e44d0d.py`.
- Remaining quota: **9 checks and 5 scored attempts**. The leaderboard row has no screening field yet, so the existing FP8 pooler/unpooler manual-screening caveat remains open.

## Boundary-fusion continuation and current resume point

- The exact scored rank-7 source is active again as `submission.py`: SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`.
- A fused residual-boundary/FP8-row-quantization kernel passed 5/5 in job `897df59a8de84a8d` at **194.597 ms / 1.99080x**. Its targeted 16K component saved about **103 us**, but aggregate performance did not beat the scored 194.310 ms result.
- Allowing the route planner to eliminate two experts globally regressed to **196.046 ms** in job `77e19c38998044b9` and is rejected.
- Restricting two-expert elimination to full-context layers passed 5/5 in job `9a42b0f30dc241da` at **194.312 ms / 1.99372x**, only **0.002 ms** from the scored incumbent. This is a statistical tie, not evidence for a scored submission.
- Preserved candidates: `submission_boundary_q8_candidate.py` (fusion only) and `submission_boundary_q8_full_route2_candidate.py` (fusion plus full-only two-expert planner). The globally aggressive route candidate is retained only as negative evidence.
- No scored attempt was used. Confirmed quota is **5 unscored checks and 5 scored attempts**.
- Next technical target: full-context attention, especially the longest cached-context cases. Do not spend a score unless an unscored candidate beats 194.310 ms by a meaningful margin, not merely a confidence-interval overlap.

## Full-context attention continuation

- Four H200 checks evaluated the remaining implemented attention alternatives; all passed 5/5 with no flags, and all were rejected on performance.
- `ATTN_TRIM=7` (`9cee59e73efe4d0d`) regressed to **204.671 ms**; isolated immediate-scale `ATTN_TRIM=4` (`365c8323c8e84a9e`) regressed to **206.044 ms**.
- The two-consumer-warpgroup PIPE3/exp-turn topology was directly tuned in `bb40fe5711d0490e`. It lost every class: **459 vs 305 us** sliding, **5425 vs 3836 us** full-long, **620 vs 365 us** single full, and **2233 vs 1393 us** batched full. The existing variants were retained, producing a control-like **194.405 ms** result.
- Late-max softmax (`b3cc203c38264dc5`) regressed to **203.642 ms**; 32K/65K full attention increased to approximately **5.158/8.567 ms**.
- No scored attempt was used. Exact experimental copies are preserved with `submission_attn_*.py` names.
- `submission.py` is restored exactly to scored job `910fa5d559e44d0d`, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`.
- A safer follow-up skipped only the initial shaped-route positions that are overwritten by the final rebuild. It passed the same invariant suite and improved the local 16K route from **20615.2/20613.1 us** to **20516.9/20537.3 us**. The 0.4-0.5% route-only gain projects to only a few H200 microseconds, so no check was spent. Candidate: `submission_route_dead_pos_candidate.py`, SHA-256 `e9611831265807de0979ef221678bd5ae2bba6aff34b42e4a7546725ed34746c`.
- Remaining quota before reset: **1 check and 5 scored attempts**. Current attention variants 8/43 are decisively best; the next attempt must use a genuinely different dataflow, not another existing scheduling flag.

## Route-support fusion audit

- A new candidate fused route-shape application, weight renormalization, and the final position/count rebuild, while moving the 32-count clear into the planner. It preserved the existing 7% policy and removed one launch, one memset, one traversal, and dead count atomics.
- A route-only local GPU probe passed eight cases through T16384, including exact final counts, dense positions, normalized/unique routes, one replacement per token, and the 7% budget. Candidate and control shaped count vectors matched on the paired 16K case.
- Paired timing rejected it: the scored control measured **20602.9/20612.1 us**, versus **20686.9/20681.7 us** for the best warp-parallel fused layout. The fusion adds a dependency inside the position kernel and is repeatably slower locally.
- No official check or score was used. The rejected source is `submission_route_apply_pos_candidate.py`, SHA-256 `7a21fa0031140584ac32e090d9355c3b59767bea9016e6c7a1e0d9d433dbd398`.
- `submission.py` is restored exactly to scored job `910fa5d559e44d0d`, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`.

## Availability-aware route-planner candidate

- New upstream prefill cohorts were audited first; none approaches the current 1.99374x architecture, and the only 1.073x path dequantizes NVFP4 expert weights during prepare, which conflicts with the challenge's storage rule.
- The scored planner left 14-25% of its requested removals unfulfilled in representative local cases because it planned from counts without knowing safe-candidate availability.
- `submission_route_avail_candidate.py` counts only routes that pass the existing score guards and plans only reductions that availability can fill. The 7% cap and one-expert-elimination limit are unchanged.
- Eight invariant cases passed. Quota shortfall fell to zero in seven cases and one route in the eighth. Extra padded-work savings were 0-32 rows at T753, 96-128 at T3012, and 128-448 at T16384 on the ordinary comparison seeds, with locally flat route latency.
- Candidate SHA-256: `30865f672c5c9a4c6aaddc84aa101472dc4c9398aef9b07b9e8b508b2e014485`.
- The H200 client timed out twice (64 s and 304 s) without returning a job id or result; no pass/fail claim is made. Do not score it until a deduplicated check succeeds and beats the incumbent materially.
- Active `submission.py` is still the exact scored 194.310 ms source, SHA-256 `c27ec2aa3ee57b7513789692ac5fe862cde36263d6fbc2f3c5ca01b448b06949`.

## 2026-11-01 conclusive legal-speedup sweep

- A four-consumer-warpgroup paired-output MoE kernel shared each activation stage across two output tiles. It was
  correct but decisively slower: job `3ee0956bbe97433c`, **218.572919 ms**, with large MoE spans of roughly
  4004/4114/3783/3865 us. It has been fully removed.
- FlashInfer's PRMT E2M1 decode is instruction-heavier than the retained fused-scale repack, so it was rejected by
  source-level instruction count without consuming a check.
- Widening route score guards failed 3/5 mandatory cases in `127d77b4d1044fad`; the BF16 hi+lo router can differ
  from the reference gap by about .010, so the original .010/.030 guards are necessary.
- Availability-aware two-expert elimination passed 5/5 in `049cfc0c136644aa`, but **210.286913 ms** and a
  3327.8-us weighted sliding MoE span did not beat the one-expert K64 control. It was reverted.
- Clean rollback `7f9c34f06cc44737` passed 5/5 with no flags at **210.578984 ms**. The best checked result remains
  `a1d2f61ed3534773` at **210.067014 ms**; no scored attempt was used.
- Current `submission.py` SHA-256 is `9FB5E1B6B6291A180D374DC5373C38E5E009A320480FD79C7217EAAD407D9BB4`.
  Quota is **16/20 checks used, 4 left, 0/10 scored used**.
- The remaining path to 198 ms is not an upstream scheduling transplant. It would require a new legal primitive or
  a library-quality fused dense/MoE dataflow that removes material math/traffic without changing BF16/NVFP4 storage.

## 2026-11-01 final attention conclusion

- Paired FP16 exponentials passed validation in `781f98b51b114059` but regressed full-long attention from 3836 to
  4718 us and were removed.
- A v43 cycle profile in `8cb23a6805644684` measured only **3.86% K/V readiness wait** and **1539/1997 consumer
  cycles in softmax**. Perfect K/V multicast cannot provide a significant end-to-end gain, so no cluster rewrite
  was attempted.
- FMA/MUFU-balanced v43 hybrids passed validation in `a36eb807d8964526`, but the smallest change regressed
  full-long from 3835 to 4414 us; larger substitutions were worse. All were removed.
- Full-context attention is softmax/dependency-bound rather than K/V-bandwidth-bound. The implemented legal
  alternatives are conclusively slower, and another scheduling microvariant is not justified.
- `submission.py` is restored exactly to SHA-256
  `9FB5E1B6B6291A180D374DC5373C38E5E009A320480FD79C7217EAAD407D9BB4`. Best checked result remains
  `a1d2f61ed3534773` at **210.067014 ms**. Quota: **19/20 checks used, 1 left, 0/10 scored used**.

## Final 16K cross-stage pipeline result

- `91e45f069d81425c` split only 16K full-context attention into two exact query ranges and pipelined each completed
  8K row slice through the legal BF16 O/boundary/Pooler path, with the exact FP32 hi+lo Router on a second stream.
- It passed **5/5** with no flags or rejections, proving the dataflow correct, but regressed to **212.609710 ms**.
- ctx32K attention increased from roughly 3835 to **4538 us** and ctx64K from roughly 6820 to **7922 us**. The
  profiler showed real 1.0-1.5 ms overlap, but split-grid load balance and SM contention cost more.
- The experiment was fully removed. Active source is behaviorally the retained legal K64 implementation, current
  SHA-256 `055353076DC539216B11963D26ADCF459654E2C15E6E59B9DB5E4A409D5B3520`; syntax and diff checks pass.
- Daily quota is **20/20 checks used, 0 left, 0/10 scored used**. Separate-kernel attention/dense overlap is now a
  measured rejection, not an open direction.

## Direct E4M3 softmax encoding

- `3b7bb46bb3fb4ae4` exposed a missing shared-memory opt-in and fell back to FA3; it was not a candidate timing.
- Corrected job `cf92c64009f94685` passed 5/5 with no flags and matched v43's 0.0151 validation error, but direct
  log2-to-E4M3 threshold encoding was 10-11x slower than MUFU plus native conversion on every attention class.
- The tuner retained v8/v43, so the scored aggregate remained safe at **212.159183 ms**. Variant 72 and all helper
  code were removed. Scored usage is now **2/10**.

## Affine and sparse attention follow-ups

- Affine log2-to-E4M3 encoding passed validation at 0.0191. Combining its two dependent FMAs and using PRMT packing
  improved full-long attention from 4494 to 4081 us, but v43 remained faster at 3830 us (`402166d6a7e7429c`).
- Round-robin affine scheduling regressed to 4555 us in `89e0b3e7dd764c04`; MUFU-specific turn-taking does not
  transfer to the integer/FMA encoder.
- Grouped activity voting with a shifted-probability cutoff passed at 0.0203 in `f2648bafbedd4f8b`, but control flow
  made full-long 6052 us versus 3803 us. It was removed.
- All candidates passed 5/5 with no flags, and all were rejected by the tuner. Active source is restored to fixed
  v8/v43. Best legal checked result remains **210.067014 ms**; scored usage is **6/10**.

## Final scored sweep: tile-aware MoE contribution pruning

- Four scored variants tested 12%, 4%, and 2% low-weight contribution budgets while retaining returned route IDs
  and weights. Pruning occurred only at real expert tile boundaries, with one- and two-route per-token protection.
- Jobs `9513d4974c7a4001`, `45cb73dfa0844d1e`, and `c46f58fef5234203` failed correctness as the budget fell from
  12% to 2%. The 2% case passed global error but exceeded the per-row cap by only 0.028.
- Final job `b298ae9c908340ef` protected two local routes per token and passed all five initial checks. It failed a
  timed turn-4 recheck on one row: 0.556 versus a 0.540 cap. No flags or static rejections occurred.
- The result closes contribution dropping: post-expert RMS normalization makes a single omitted expert vector too
  visible to the per-row guard, even when aggregate L2 is safe. All pruning code was removed.
- Active source is restored to SHA-256
  `11D3110980F27C67B05E410E8E5684C1CBD9F9F358F13D73BFF9CC9D21455BE5`; syntax and diff checks pass.
  Scored quota is **10/10 used**. Retained best: `a1d2f61ed3534773`, **210.067014 ms**, 5/5, no flags.

## 2026-11-02 FP8-on-K64 hybrid

- The proven FP8 Pooler/Unpooler boundary is now integrated into the retained 210 ms K64 architecture. QKV/O,
  routing, attention, NVFP4 expert math, and residual reconstruction are unchanged.
- Official check `9c8ab5f2abbd4847` passed **5/5** with no automated flags or rejections at
  **192.838943 ms / 2.0089505x** (95% CI **1.978816x-2.039085x**).
- This improves the 210.067014 ms control by **17.228071 ms (8.20%)**; all eight scored shapes improved, with the
  largest gain (**10.21%**) on the dominant 16K sliding case.
- Active source SHA-256: `20B6F0B7FF4E11F895B53BB6A2E32F3F0DCF98B1C41C947A8D1E7BE29929EBDE`; judge content hash:
  `b78ac6f14827be7f594fea2ba165eb77f5fe06c73ac31c0c280508d555545f1a`.
- The E4M3 copies of BF16 Pooler/Unpooler weights remain a known manual-screening risk under the written precision
  rule. This was an unscored check only; no leaderboard submission was made.

### Scored result

- Exact checked source submitted as job `78171469544c4a6a`; content hash stayed
  `b78ac6f14827be7f594fea2ba165eb77f5fe06c73ac31c0c280508d555545f1a`.
- Passed **5/5** with no automated flags or rejections at **191.218188 ms / 2.0259783x** (95% CI
  **1.995589x-2.056368x**).
- Live leaderboard moved from rank 12 to **rank 7 of 799**.
- Manual screening remains unresolved because the Pooler/Unpooler BF16 weight copies are stored lossily in E4M3.

### Further FP8 boundary fusion

- Rejected `fce39b5ce52e48a1`: enabling the FP8 Pooler for 16K ctx0 exceeded residual-update L2
  (**0.05331 > 0.0409475**), so the cached-context guard remains in place.
- Fused the accepted Pooler row quantization into the first CUDA residual boundary. `73b2715c12fd4cf0` passed
  **5/5**, no flags/rejections, at **192.385454 ms**; the weighted boundary-plus-quantizer component fell from
  roughly **365.0 us to 321.6 us**.
- Tightened the fused amax reduction and removed redundant BF16 repacking. Final check `4426cfbc6b234ef7` passed
  **5/5**, no flags/rejections, at **192.044796 ms / 2.0172580x**. This is **0.794148 ms** below the comparable
  pre-fusion check and **0.340658 ms** below fusion v1; the weighted fused boundary is about **317.8 us**.
- Retained source SHA-256: `A7DFD9292D56BB60076ED10603E9EB4AAE4B5BAF73E8756BDCCC10A2EA2C02F9`.
- Exact source submitted as scored job `2000e717879942e1`; content hash matched the check. It passed **5/5** with
  no flags/rejections at **192.860893 ms / 2.0087219x** (95% CI **1.978591x-2.038853x**).
- The scored draw did not beat job `78171469544c4a6a` at 191.218188 ms, so the leaderboard retained the old best.
  Team rank remains **7 of 802**.

### Split 16K FP8 Unpooler

- Replaced the cached-context 16K FP8 Unpooler with two M=8192 `_scaled_mm.out` calls into one BF16 output, while
  retaining one full, uncontended residual boundary.
- Overlapped prototype `c90671ac949f42ac` passed **5/5** at **190.935124 ms**, but was rejected structurally:
  competing boundary/GEMM work inflated the two half-boundaries to roughly **460-472 us**.
- Retained check `9af760ab349f4530` passed **5/5**, no flags/rejections, at **192.314027 ms / 2.0144339x**. Its
  headline was noise-level worse than the 192.044796 ms control, but all raw medians improved in this draw.
- The weighted 16K profile confirms the change: FP8 Pooler+Unpooler time fell **1676.8 -> 1543.1 us**, the final
  boundary remained **366.9 us**, and total profiled time fell **7904.7 -> 7726.9 us**.
- Retained source SHA-256: `C6434829398F23C402BD389385D515826461A46615D793B7BF88488F31050554`.
  This iteration is checked only; no scored submission was made.

### Confirmation and scored submission

- Independent comment-only confirmation `018c378b5d634cc9` passed **5/5**, no flags/rejections, at
  **191.073332 ms / 2.0275142x**; its weighted 16K profile repeated the total-time reduction at **7723.6 us**.
- Exact checked payload submitted as scored job `a1a1aa04a7fd4473`; content hash matched the check:
  `d0dc4c2a224308e71fd0cfd5b45d573b1d54313f3400c776380c5b7081b7deea`.
- The scored draw passed **5/5** with no flags/rejections but measured **193.318305 ms / 2.0039691x**. The board
  retained `78171469544c4a6a` at **191.218188 ms**, rank **7 of 805**.
- Active source SHA-256: `EE69F3D6FB259C4183721813A3723ACBB543F5879C59A44AA5ECAF5B2FC6D04D`;
  two scored attempts remain today.

### FP8 Unpooler geometry/accumulation sweep

- Four M=4096 calls regressed: `4b633e4fc0074d7c` passed **5/5** at **193.932464 ms**, but weighted FP8 time
  increased to **1694.8 us** and total profile time to **7836.9 us**. Rejected.
- Restored two M=8192 calls and enabled fast accumulation only for the FP8 Unpooler. Check
  `486a5224bc03495f` passed **5/5**, no flags/rejections, at **192.367744 ms**; weighted FP8/total profile times
  improved to **1486.6/7676.9 us**.
- Independent confirmation `d52ab1639f25445b` passed **5/5**, no flags/rejections, at **192.031227 ms** and
  repeated the structural gain at **1518.5/7595.3 us**.
- The full-profile gain is repeatable but only **0.05-0.13 ms**, while neither aggregate check beat the current
  **191.218188 ms** leaderboard best. No scored attempt was spent.
- Active source SHA-256: `69A7A3E13B6AD546D4D22A0DC21B384FD412F61B104C03DAFACEA13755C79964`.
  Checks are **10/10 used**; scored attempts are **3/5 used**, leaving two.

---

## 2026-11-05 final kernel-track checkpoint

- The final valid leaderboard incumbent is job `399cbda410c440ee`, **200.105694 ms / 1.935996x**. At the locked
  post-close query it was rank **68 of 899**. Exact source:
  `submission_scored_399cbda410c440ee.py`, SHA-256
  `7A5AEDF8B6DEF500D72463A5A12EEA7F9BB07C49A8977B400CCB27739D23676E`.
- The prior 186--191 ms FP8-weight experiments are not the live board entry and remain incompatible with the written
  BF16-weight-storage rule. Do not restore them as the production source.
- A 29-check final legal sweep tested tile-selective lossless K64, exact subgroup scale sharing, fine-tail policies,
  independent fair-BF16 sources, and fresh-seed confirmations. Every accepted check passed with no flags. Nothing is
  left queued or running.
- Tile-selective K64 was rejected. The best first draw was exact scale sharing on 128/160/192-row MoE tiles:
  `0eaac861cd8742ce`, **199.965750 ms**. Its executable-identical confirmation `e4906947348c4acf` measured
  **200.939688 ms**; the two-run mean (**200.452719 ms**) did not beat the incumbent.
- The kernel-track submission gate closed before a scored run could be accepted. A direct attempt with the best checked
  candidate returned the official track-closed error. No leaderboard mutation is possible.
- `submission.py` is restored byte-identically to the valid scored incumbent. The unfinished pre-session source is
  preserved as `submission_n72_tilekperm_scaleshare_v300.py`; the best checked candidate is
  `submission_n72_scaleshare128_160_192_v314.py`.
