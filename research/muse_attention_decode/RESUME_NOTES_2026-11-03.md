# Resume notes — 2026-11-03

## New scored best

- Active `submission.py` SHA256: `1EDEF4D178CDBEE3B256BA3D662738A0F5BA7788C1782ABCF012341F8A3F20C9`
- Scored job: `9408bd801dd04353`
- Correctness: 16/16
- Score: 1444.3444114015174 us
- Speedup: 2.6076190486626225x
- 95% interval: [2.568504762932683, 2.657126416995119]
- Change: the page-size-1 flat producer's cache copies use `cp.async.cg.shared.global.L2::64B` instead of `L2::128B`.
- Generated SASS contains `LDGSTS.E.BYPASS.LTC64B.128`; hot-kernel resources are unchanged.
- Previous best: 1447.5138910738851 us / 2.6019094001273095x (`722baa0be2354d5e`).

The remaining score gap to 2.7x is about 49.42 us (3.42%). Cross-run shape noise is material, so retain the
new scored source but do not infer a large isolated gain from individual shape deltas.

## Next exact candidate

`q1_occ3.patch` changes only the flat q_len=1 full-attention topology:

- 64 producer threads instead of 128;
- three resident CTAs per SM instead of two;
- launch 3 x SM-count CTAs for batched full q_len=1.

Static H200 analysis:

- block: 128 consumers + 64 producers = 192 threads;
- shared memory: approximately 62 KB per CTA, so three fit on H200;
- exact CUDA 12.4 `sm_90a` compilation passed;
- q_len=1 flat generated kernel: 96 registers, zero-byte stack (incumbent: 128 registers, zero-byte stack);
- q_len=8 flat remains 80 registers / 32-byte stack;
- three CTAs use 18 active warps/SM, including 12 consumer warps, versus 16 total / 8 consumer warps in the incumbent.

This candidate is exact and specifically targets the cold-HBM latency gap in B16/B32 full decode. Apply the patch
only after quota reset, run one unscored check first, and promote only if both q1 full shapes improve without a
material q8 or correctness regression.

## Quota

At 2026-11-03 00:20 IST (still the prior UTC quota day): attention decode had 0/5 scored and 0/10 checks left.
Quota resets at 00:00 UTC / 05:30 IST.

## Tomorrow startup checklist

1. Change directory to `D:\OpenKernelChallenge\muse_attention_decode`.
2. Confirm `submission.py` still has SHA256 `1EDEF4D178CDBEE3B256BA3D662738A0F5BA7788C1782ABCF012341F8A3F20C9`.
3. Query the Attention Decode quota before changing anything.
4. Apply `q1_occ3.patch` using the Codex apply-patch workflow.
5. Run the Python syntax check and exact `sm_90a` compile check.
6. After the quota reset, use one unscored check on the candidate.
7. Compare the B16/B32 q_len=1 full-layer shapes against **86.84 / 165.56 us** from scored job `9408bd801dd04353`; promote only if correctness remains 16/16 and the aggregate improves.
8. If it regresses, reverse the small changes in `q1_occ3.patch` and verify the scored-best SHA256 above.
