# Experiment ledger

Record every remote check and scored run. Do not overwrite a candidate without first
saving it under a descriptive filename or committing it.

## Snapshot

| Date | Board baseline | Leader | Checks left | Scored left |
|---|---:|---:|---:|---:|
| 2026-10-28 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 20 | 10 |
| 2026-10-28 after E001 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 19 | 10 |
| 2026-10-28 after E002 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 18 | 10 |
| 2026-10-28 after E003 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 17 | 10 |
| 2026-10-28 after quota reset and E004 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 19 | 10 |
| 2026-10-28 after E005 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 17 | 10 |
| 2026-10-28 after E006 | 22.6203 ms / 1.0000x | 9.4308 ms / 2.3986x | 16 | 10 |
| 2026-10-28 after S001 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 16 | 9 |
| 2026-10-28 after E007 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 15 | 9 |
| 2026-10-28 after E008 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 14 | 9 |
| 2026-10-28 after E009 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 13 | 9 |
| 2026-10-28 after E010 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 12 | 9 |
| 2026-10-28 after E011 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 11 | 9 |
| 2026-10-28 after E012 | 18.8945 ms / 1.1972x | 9.4308 ms / 2.3986x | 10 | 9 |
| 2026-10-28 after S002 | 18.4643 ms / 1.2251x | 9.4308 ms / 2.3986x | 10 | 8 |
| 2026-10-28 after E013a | 18.4643 ms / 1.2251x | 9.4308 ms / 2.3986x | 9 | 8 |
| 2026-10-28 after E013 | 18.4643 ms / 1.2251x | 9.4308 ms / 2.3986x | 8 | 8 |
| 2026-10-28 after S003 | 18.4643 ms / 1.2251x | 9.4308 ms / 2.3986x | 8 | 7 |
| 2026-10-28 after E014 | 18.4643 ms / 1.2251x | 9.4308 ms / 2.3986x | 7 | 7 |
| 2026-10-28 after S004 | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 7 | 6 |
| 2026-10-28 after E015 | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 6 | 6 |
| 2026-10-28 after E016 | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 5 | 6 |
| 2026-10-28 after E017a | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 4 | 6 |
| 2026-10-28 after E017b | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 3 | 6 |
| 2026-10-28 after E018 | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 2 | 6 |
| 2026-10-28 after E019 | 18.3345 ms / 1.2338x | 9.4308 ms / 2.3986x | 1 | 6 |
| 2026-10-28 after S005 | 16.9450 ms / 1.3349x | 9.4308 ms / 2.3986x | 1 | 5 |
| 2026-10-29 after S008/E025 | 13.6581 ms / 1.6562x | 9.4139 ms / 2.4029x | 0 | 0 |
| 2026-10-29 after quota reset and E026 | 13.6581 ms / 1.6562x | 9.4139 ms / 2.4029x | 19 | 10 |
| 2026-10-29 after E027 | 13.6581 ms / 1.6562x | 9.4139 ms / 2.4029x | 18 | 10 |
| 2026-10-29 after S009/E027 | 13.5537 ms / 1.6689x | 9.4139 ms / 2.4029x | 18 | 9 |
| 2026-10-29 after E028 | 13.5537 ms / 1.6689x | 9.4139 ms / 2.4029x | 17 | 9 |
| 2026-10-29 after E029 | 13.5537 ms / 1.6689x | 9.4139 ms / 2.4029x | 16 | 9 |
| 2026-10-29 after S010/E029 | 13.3654 ms / 1.6924x | 9.4139 ms / 2.4029x | 16 | 8 |

## Runs

| ID | Candidate | Change | Check/job | Tests | Speedup | Score ms | Decision |
|---|---|---|---|---:|---:|---:|---|
| E000 | official starter | repository and research initialization | not run | — | — | — | keep as source specification |
| E001 | `submission.py` | SGLang-equivalent baseline | `01169a9a59374fcc` | 13/13 | 1.000045x (CI 0.9850-1.0150) | 22.6193 | pass; canonical baseline |
| E002 | `submission.py` | embed the measured BF16 dense/router winner; keep attention, routing, Marlin MoE and boundaries unchanged | `66881319d3044dcd` | 13/13 | 1.078919x (CI 1.0627-1.0951) | 20.9657 | pass; keep as current candidate |
| E003 | `submission.py` | add fused custom decode attention for scored B=1/16/32 sliding and full shapes; retain E002 verify path | `2cad40d153f142aa` | 13/13 | 1.147103x (CI 1.1299-1.1643) | 19.7195 | pass; keep as current candidate |
| E004 | `submission.py` | dispatch T=128/256 expert GEMMs to losslessly transformed Humming NVFP4 weights; retain Marlin for smaller shapes | `92cdd3b87d3447a4` | 13/13 | 1.189263x (CI 1.1714-1.2071) | 19.0204 | pass; keep as current candidate |
| E005a | `submission.py` | first fused small-route alignment build | `04195978bb7e49f3` | import failed | — | — | C++ binding lacked CUDA function declaration; fixed without algorithm change |
| E005 | `submission.py` | fuse global-to-local expert remap and Marlin route alignment into one CTA for T<=32 | `e053f1c2efcf4b66` | 13/13 | 1.203221x (CI 1.1852-1.2213) | 18.7998 | pass; keep as current candidate |
| E006 | reverted candidate | fuse sliced unpool padding, post-FFN norm/residual, and next-input norm for T<=8 | `50b4cf7b6ad24157` | 13/13 | 1.205351x (CI 1.1873-1.2234) | 18.7666 | pass but neutral; revert to E005 |
| S001 | `submission.py` at E005 | first scored promotion of the validated dense + attention + Humming + small-route stack | `dad63279b80446a3` | 13/13 | 1.197188x (CI 1.1792-1.2151) | 18.8945 | board rank 715/841; keep as scored baseline |
| E007 | `submission.py` | stop materializing the unused FP32 router branch at the post-attention boundary | `8c5a69d3229c4de2` | 13/13 | 1.198588x (CI 1.1806-1.2166) | 18.8725 | pass; keep (small targeted win) |
| E008 | reverted candidate | force Humming block-M 8 for the T=128/256 indexed expert GEMMs | `d7780409208e44c5` | 13/13 | 1.099462x (CI 1.0830-1.1160) | 20.5740 | pass but severe regression; revert to E007 |
| E009 | reverted candidate | keep Humming BM16 but force five pipeline stages for both T=128 projections | `271697bfe3a2459c` | 13/13 | 1.197476x (CI 1.1795-1.2154) | 18.8900 | pass but neutral/mixed; revert to E007 |
| E010 | reverted candidate | force Marlin block-M 8 instead of 16 at T=32 | `bdce6eddb8e1464a` | 13/13 | 1.197002x (CI 1.1790-1.2150) | 18.8975 | pass but mixed/regressive; revert to E007 |
| E011 | reverted candidate | disable FP32 reduction in the T<=32 Marlin expert GEMMs | `3ee2f79462c54a14` | 13/13 | 1.189407x (CI 1.1716-1.2072) | 19.0181 | pass but regression; revert to E007 |
| E012 | `submission.py` | launch pre-expert RMSNorm before joining the routing stream | `05c36007fa3c4641` | 13/13 | 1.226129x (CI 1.2077-1.2445) | 18.4485 | pass; keep (consistent scheduling win) |
| S002 | `submission.py` at E012 | scored promotion of pre-expert norm/routing overlap | `ad3607a8109b4443` | 13/13 | 1.225082x (CI 1.2067-1.2435) | 18.4643 | board rank 696/846; new scored baseline |
| E013a | corrected candidate | move T<=32 route alignment onto the routing stream | `a13249bee3794edb` | setup only | — | — | bookkeeping error called `record_stream` on integer block size; corrected without schedule change |
| E013 | `submission.py` | overlap T<=32 route alignment with pooler-output normalization | `ac45b84c26974d56` | 13/13 | 1.232176x (CI 1.2137-1.2507) | 18.3580 | pass; keep (targeted scheduling win) |
| S003 | `submission.py` at E013 | scored promotion of route-alignment overlap | `37bdabe14f834aa8` | 13/13 | 1.223851x (CI 1.2055-1.2422) | 18.4829 | pass but 0.10% slower than S002; board retains S002 |
| E014 | `submission.py` | overlap T=128/256 Humming route remap and block alignment with expert-input normalization | `b39335d6bc024ebe` | 13/13 | 1.243372x (CI 1.2247-1.2620) | 18.1927 | pass; keep (strong targeted win) |
| S004 | `submission.py` at E014 | scored promotion of Humming route-preparation overlap | `c0577e6f5ad54009` | 13/13 | 1.233756x (CI 1.2152-1.2523) | 18.3345 | new scored best; board rank 691/846 |
| E015 | committed candidate | activate fused verify attention preparation and scheduler metadata for B=16/32 | `c0412335894c4476` | 13/13 | 1.229657x (CI 1.2112-1.2481) | 18.3956 | mixed: verify-32 load +2.36%, verify-16 -0.40%; refine to B=32 only |
| E016 | `submission.py` | restrict fused verify preparation and scheduler metadata to B=32 | `5c0e7557f7ba4499` | 13/13 | 1.223393x (CI 1.2050-1.2417) | 18.4898 | keep: verify-32 load gain repeated at +2.07%; aggregate obscured by control variance |
| E017a | reverted candidate | gather W13/SwiGLU output into contiguous expert slabs and run W2 with Humming grouped-contiguous defaults | `338633e0dc3c4f52` | 13/13 | 1.096747x (CI 1.0803-1.1132) | 20.6249 | severe regression: grouped W2 314/420 us at T=128/256 versus indexed 117/157 us; revert |
| E017b | reverted candidate | retain contiguous expert slabs but reuse the proven indexed W2 tile configuration | `83d1f02766e34cb1` | 13/13 | 1.096816x (CI 1.0804-1.1133) | 20.6236 | no recovery: grouped W2 345/444 us; close grouped-scheduler branch and restore E014 |
| E018 | reverted candidate | select Marlin's non-EP specialization after custom small-route alignment has filtered to local experts | `d39f4e4df1374644` | 5/13 then crash | — | — | illegal memory access at batch 5, q_len 8; local expert IDs do not make the non-EP metadata contract compatible; revert |
| E019 | `submission.py` | remove SGLang's M<32 fallback and always use the one-launch Triton post-expert norm/route reduction | `c91ca20630724c87` | 13/13 | 1.345814x (CI 1.3256-1.3660) | 16.8079 | strong win: validated first on standalone MoE, then improved every targeted small-token branch by 10.0%-19.2%; keep and score |
| S005 | `submission.py` at E019 | scored promotion of the always-fused small-token expert finalizer | `13b45e95750b41aa` | 13/13 | 1.334926x (CI 1.3149-1.3549) | 16.9450 | new scored best; board rank 583/975, up 108 places from S004 |
| E025/S008 | `submission.py` | fuse EP global-to-local expert-ID localization into the v49 CUDA kernels | `ff7edb2dc5374bc0` | 13/13 | 1.656177x (CI 1.63133-1.68102) | 13.6581 | promoted; decoder rank 355/873 |
| E026 | rejected candidate | fold `T<=32` pre-expert RMSNorm into the resident v49 `gk` grid | `f31a58d44ce640f2` | 13/13 | 1.629286x (CI 1.60485-1.65373) | 13.8836 | correct but 1.65% slower; global barrier and staging cost exceed removed launch |
| E027 | `submission.py` | replace generic pre-expert RMSNorm with fixed 4096-wide BF16x2 CUDA kernel | `6111ef58daa24397` | 13/13 | 1.658270x (CI 1.63340-1.68314) | 13.6409 | keep; kernel falls from 5.6-8.3 us to 2.5-2.9 us, aggregate +0.13% |
| S009 | `submission.py` at E027 | scored confirmation of the fixed-4096 pre-expert RMSNorm kernel | `773381b773ff4764` | 13/13 | 1.668944x (CI 1.64391-1.69398) | 13.5537 | new scored best; 0.76% lower latency than S008; decoder rank 373/904 |
| E028 | committed candidate `bf653dd` | port v52 CPB2 WGMMA at T=128 and fuse activation gather at T=128/256 | `a545e99e9565410c` | 13/13 | 1.675392x (CI 1.65026-1.70052) | 13.5015 | refine: T=128 MoE -14%, but weighted T=256 MoE +1.8%; keep fused gather only for CPB2 |
| E029 | `submission.py` | restrict fused activation gather to T=128 CPB2 and restore standalone gather for T=256 NCW16 | `fb6547cd0d814e79` | 13/13 | 1.686096x (CI 1.66080-1.71139) | 13.4158 | keep and score; 1.02% lower latency than S009 check and targeted verify-16 speedups improve 3.2%-4.3% |
| S010 | `submission.py` at E029 | scored promotion of shape-specific CPB2 large-token MoE dispatch | `3a7e7defad2e4b07` | 13/13 | 1.692448x (CI 1.66706-1.71783) | 13.3654 | new scored best; 1.39% lower latency than S009; decoder rank 359/905 |

E001 profile: `profiles/submission.01169a9a59374fcc.txt` (local, intentionally ignored by Git).
The result is statistically indistinguishable from SGLang and satisfies the planned
0.98x-1.02x acceptance gate.

E002 profile: `profiles/submission.66881319d3044dcd.txt` (local, intentionally ignored by Git).
Prepare took 55.68 s. The score improved 7.31% from E001, and every one of the 12
scored shape/layer branches improved. The largest gains were verify-1 full (16.93%),
verify-1 sliding (15.49%), and decode-1 sliding (10.84%).

E003 profile: `profiles/2cad40d153f142aa.txt` (local, intentionally ignored by Git).
Prepare took 65.17 s. The score improved 5.94% from E002 and 12.82% from E001.
All six decode branches improved by 10.00%-30.42%. Verification retained the E002
path and stayed within 1.21%; no branch regressed by the 3% acceptance threshold.

E004 profile: `profiles/submission.92cdd3b87d3447a4.txt` (local, intentionally ignored
by Git). Prepare took 67.31 s. The score improved 3.55% from E003 and 15.91% from E001.
The unchanged T<=32 paths' speedup ratios stayed within 1.5% despite cross-machine raw
time drift. Humming improved the normalized whole-layer ratios by 6.40%/4.99% at T=128
sliding/full and 20.41%/13.12% at T=256 sliding/full.

E005 profile: `profiles/submission.e053f1c2efcf4b66.txt` (local, intentionally ignored
by Git). Prepare took 66.73 s. The score improved 1.16% from E004 and 16.89% from E001.
All eight T<=32 branches improved their normalized ratios by 1.35%-2.62%. The unchanged
T=128/256 branches stayed within 1.48%, so the gain is isolated to the fused route handoff.

E006 profile: `profiles/submission.50b4cf7b6ad24157.txt` (local, intentionally ignored
by Git). The fused sliced boundary passed correctness, but changed the overall score by
only 0.18%. Its targeted normalized changes were +0.33%, +0.25%, -0.72%, and +0.16%
across decode-1 sliding/full and verify-1 sliding/full. This is noise-level and not
consistent, so the candidate was reverted. Do not repeat without fusing the unpool GEMM
epilogue itself; merely replacing padding plus the existing boundary is insufficient.

S001 profile: `profiles/submission.dad63279b80446a3.txt` (local, intentionally ignored
by Git). The independent scored rerun landed at 1.197188x / 18.8945 ms, within E005's
confidence interval and 0.50% slower than its check. It established rank 715 of 841.

E007 profile: `profiles/submission.8c5a69d3229c4de2.txt` (local, intentionally ignored
by Git). The aggregate improved 0.12% over S001. More importantly, all four affected
T=1/8 branches improved their normalized ratios versus S001 by 0.74%-1.15%, while
the unchanged shapes exhibited the expected cross-machine variance. The optional FP32
router branch had no reader because the custom router consumes `ffn_input` directly;
removing its `[tokens, 8192]` materialization therefore preserves the exact executed
math. Keep this cleanup despite the small aggregate movement.

E008 profile: `profiles/submission.d7780409208e44c5.txt` (local, intentionally ignored
by Git). An independent synthetic cold-L2 sweep (`3849efe374824d9f`) predicted a 7.6%
expert-chain gain from block-M 8, but its uniform one-local-route-per-token distribution
was not representative of the scored routing. In the real decoder profiles, W13/W2
regressed by roughly 30%-76%; normalized whole-layer ratios fell 14%-33% across the four
T=128/256 branches, and the aggregate score regressed 9.0%. The Humming defaults are
restored. Do not force a smaller block without probing the decoder's actual routed IDs.

E009 profile: `profiles/submission.271697bfe3a2459c.txt` (local, intentionally ignored
by Git). A fixed-BM pipeline sweep (`213c36f9c6474114`) selected five stages for both
T=128 projections, but the decoder check was distribution-sensitive: verify-16 sliding
improved its normalized ratio by 0.85%, full changed by -0.09%, and the aggregate was
0.09% slower than E007. Per-kernel movement also reversed between sliding and full.
This is too small and inconsistent to keep; Humming's default stage count is restored.

E010 profile: `profiles/submission.bdce6eddb8e1464a.txt` (local, intentionally ignored
by Git). Reducing T=32 Marlin's block-M from 16 to 8 increased the profiled two-GEMM
time from roughly 99-147 us to 136-166 us. The normalized decode-32 sliding ratio
regressed 7.8%, while full improved 3.9%; because sliding carries nearly three times
the score weight, the trade is unfavorable. Restore the production BM16 heuristic.

E011 profile: `profiles/submission.3ee2f79462c54a14.txt` (local, intentionally ignored
by Git). BF16 reduction was numerically valid, but its aggregate score regressed 0.77%
from E007. Seven of the eight affected T<=32 branches had lower normalized whole-layer
speedups; the lone improvement was decode-32 full at 0.56%. Marlin itself was usually
slower, including 85.0 to 126.7 us for decode-16 full, 99.4 to 147.0 us for decode-32
sliding, and 75.4 to 108.3 us for verify-1 sliding. Restore FP32 reduction.

E012 profile: `profiles/submission.05c36007fa3c4641.txt` (local, intentionally ignored
by Git). PyTorch's CUDA stream documentation and NVIDIA Megatron-LM's MoE scheduling
both use launch-early/join-late stream DAGs to expose independent work. The pre-expert
RMSNorm depends on the pooler output but not on routing, so launching it before the
existing main-stream join overlaps it with the routing tail without changing math.
The score improved 2.25% from E007, and normalized speedup improved on all 12 scored
branches by 0.80%-3.45%. This is a consistent, research-backed scheduling win; keep it.

S002 profile: `profiles/submission.ad3607a8109b4443.txt` (local, intentionally ignored
by Git). The independent scored rerun reproduced E012 at 1.225082x / 18.4643 ms, only
0.09% slower than the check and well inside its confidence interval. It improves the
previous S001 board score by 2.28% and moves the entry from rank 715/841 to 696/846.

E013a failed during eager correctness setup before timing because `_small_align` returns
the Python `block_size_m` integer among its tensors, and the new side-stream lifetime
loop called `record_stream` on every tuple entry. Job `a13249bee3794edb` recorded the
failure. Recording only the four CUDA tensors corrected the bookkeeping without changing
the stream schedule or GPU math.

E013 profile: `profiles/submission.ac45b84c26974d56.txt` (local, intentionally ignored
by Git). The existing T<=32 alignment kernel now follows routing on the side stream while
the main stream normalizes the pooler output, then both branches join before Marlin. The
score improved 0.49% from E012. All six targeted decode branches improved their normalized
ratios by 0.14%-1.04%; verify-1 sliding improved 0.48% and full regressed 0.61%. The four
untouched Humming branches moved by -0.44% to +1.16%, consistent with normal run noise.
Keep the aggregate and 7/8-target-branch win.

S003 profile: `profiles/submission.37bdabe14f834aa8.txt` (local, intentionally ignored
by Git). The scored E013 rerun passed at 1.223851x / 18.4829 ms, 0.10% slower than S002
and within both candidates' run-to-run noise. The board correctly retained S002 at
1.2251x; the live rank after this attempt was 699/846. E013 remains the code candidate,
but this scored run does not independently confirm its 0.49% check-run gain.

E014 profile: `profiles/submission.b39335d6bc024ebe.txt` (local, intentionally ignored
by Git). Humming's global-to-local expert remap, local filtering, and block alignment now
follow routing on the side stream while the main stream runs pooler-output RMSNorm. The
branches rejoin immediately before the first Humming GEMM, preserving all expert math and
tuning. The score improved 0.90% from E013 and all four targeted branches improved their
normalized ratios: verify-16 sliding/full by 2.61%/1.31% and verify-32 sliding/full by
2.50%/0.15%. The eight control branches moved between -0.05% and +0.97%. Keep E014.

S004 profile: `profiles/submission.c0577e6f5ad54009.txt` (local, intentionally ignored
by Git). The independent scored rerun passed at 1.233756x / 18.3345 ms, within E014's
confidence interval and only 0.78% slower than its check. It improves the previous S002
board best by 0.70%, replaces it on the leaderboard, and establishes rank 691 of 846.

E015 profile: `profiles/submission.c0412335894c4476.txt` (local, intentionally ignored
by Git). Fused verify preparation passed all tests, but the targeted loads split: verify-32
improved its baseline-normalized load ratio by 2.36%, while verify-16 regressed 0.40%.
Unchanged decode-32 moved -4.11%, showing substantial routing/run variance and explaining
the aggregate regression. Retain only the measured B=32 dispatch for the next isolation.

E016 profile: `profiles/submission.5c0e7557f7ba4499.txt` (local, intentionally ignored
by Git). The B=32-only isolation again improved the baseline-normalized verify-32 load,
this time by 2.07% (sliding +3.44%, full -0.45%). This independently repeats E015's
2.36% load gain. Unchanged decode-32 moved -5.03% and verify-16 -0.87%, which drove the
aggregate below S004 despite the target win. Keep E016 as a small validated specialization.

E017a/E017b profiles: `profiles/submission.338633e0dc3c4f52.txt` and
`profiles/submission.83d1f02766e34cb1.txt` (local, intentionally ignored by Git). Both
variants passed all correctness cases, and the fused SwiGLU gather plus route-order restore
cost only 13.5 us at T=128 and 25 us at T=256. The grouped-contiguous W2 scheduler itself
was the failure: 314-420 us with its default policy and 345-444 us when forced to the
indexed tile policy, compared with E014's roughly 117-157 us indexed W2. Contiguous expert
rows therefore do not compensate for Humming's grouped scheduling on Muse's many tiny,
skewed expert slabs. Future W13-to-W2 overlap must retain indexed scheduling and modify its
producer/consumer handoff directly rather than converting the second GEMM to grouped mode.

E018 job `d39f4e4df1374644` passed the first five correctness cases but triggered an illegal
memory access on batch 5, q_len 8. Although `_small_align` filters routes to local expert
IDs, Marlin's `is_ep=False` specialization expects a different aligned-metadata/layout
contract; changing the flag alone is unsafe. Restore `is_ep=True`. Do not retry without
building the exact non-EP alignment format and proving its bounds independently.

E019 was developed first on the standalone `muse_moe_w4a16` problem. The E014 expert
stack scored 1.084517x in job `00c6854175d04215`; profiling showed that SGLang's
post-expert finalizer deliberately falls back to 10-12 PyTorch kernels below 32 tokens.
Calling the same fused Triton algorithm at every token count passed 5/5 in component job
`162e1670d68b4d1b` and raised the component score to 1.248898x. In the whole decoder,
E019 passed 13/13 and reduced decode-1 sliding/full from 134.72/143.81 us to
108.85/117.81 us, decode-16 from 219.05/319.72 us to 190.20/285.33 us, and verify-1
from 195.60/217.63 us to 173.74/185.78 us. The untouched large-token branches moved
within normal cross-run variance. This is an exact arithmetic-path transfer from the
existing M>=32 kernel, not a precision change; keep and promote.

S005 independently reproduced E019 at 1.334926x / 16.9450 ms, 0.81% slower than the
check and well inside its confidence interval. It improves S004's 18.3345 ms by 7.58%
and moved the live board rank from 691/846 to 583/975. This is the new scored baseline.

E020 targeted the remaining full-attention decode bandwidth in the standalone
`muse_attention_decode` problem. Doubling batch-32 full-decode splits from 8 to 16
passed 16/16 in job `f53703cf58a147e9` but was neutral (1.522014x, 225.2 us for the
scored batch-32 full case), proving CTA count was not the limiter. Allocating paired
rows in L1 was worse in job `1fbaee1227de4be2` (1.499502x; 223.7 us kernel), consistent
with cache pollution. Adding the PTX `L2::128B` prefetch-size hint while retaining
`L1::no_allocate` passed 16/16 in job `a23699e6438a4fa4`, improved the component from
1.524523x / 2470.48 us to 1.555184x / 2421.77 us, and cut full decode-16 from roughly
113 us to 104.59 us and full decode-32 from roughly 223 us to 204.46 us. Each physical
page stores exactly two 64-byte FP8 heads, so the 128-byte hint matches one complete
cache row and improves memory transactions without changing arithmetic or precision.

The identical two PTX qualifiers passed all 13 decoder tests in E020 job
`10ad4998920c4c9b`. Its noisy whole-run result was 1.330134x / 17.0060 ms versus E019's
1.345814x / 16.8079 ms, but the branches that actually execute the modified helpers
improved strongly after normalizing to each run's SGLang baseline: full decode-16 rose
from 1.534x to 1.581x and full decode-32 from 1.458x to 1.533x. The aggregate loss came
from verification branches that do not use these helpers (for example verify-32 full
fell from 1.145x to 1.098x), so it is unrelated cross-run expert/FA3 variance. Retain
E020 based on the isolated component A/B and the affected decoder branches.

S006 job `71a42aa36b7d40b2` passed 13/13 and scored 1.347519x / 16.78663 ms.
This improves S005's 1.334926x / 16.94498 ms by 0.93% in speedup and 0.93% in
latency, confirms that E020's poorer check headline was variance, and establishes a new
personal best. The live rank moved from 583/975 at S005 time to 576/861 on the current
board.

E021 investigated the next expert-pipeline optimization in the standalone
`muse_moe_w4a16` problem before changing the decoder. Diagnostic jobs
`ddd03d4460884e0a` and `eda5d606d8ee4e32` established that Humming's Python API
dispatches directly to a precompiled `torch.ops.humming_gemm` ABI with no epilogue
callback. Fusing SwiGLU into W13 would therefore require replacing the complete indexed
NVFP4 tensor-core GEMM, not a safe local epilogue change. Job `9ed0f8ea9f3e4800`
confirmed that Humming's FP16-accumulator kernels require FP16 inputs, so they cannot be
used while preserving the required BF16 activation path.

E021's fused large-route remap/alignment kernel passed 5/5 in job
`5f037a447e7e4d2d` at 1.251684x / 10.7443 ms versus the established component result
of 1.248898x / 10.7683 ms. The 0.22% headline gain did not reproduce consistently:
normalized `T=128` improved 0.68%, while `T=256` regressed 0.35%. Reducing metadata
initialization exposed undocumented capacity contracts: job `181eefa297274080` left
unused `block_experts` entries uninitialized and caused a Marlin illegal access; job
`471db90d22a54300` restored those entries but showed that Humming also reads sentinel
`sorted_ids` beyond `n_padded`. With both full spans restored, job
`ee0d873d254044fd` passed 5/5 at 1.248924x / 10.7681 ms, exactly neutral. Do not
integrate this into the decoder, where E014 already overlaps the metadata work.

E021 also isolated five pipeline stages for both Humming projections at `T=256`.
Job `40ffff34019940e5` passed 5/5 but reduced verify-32 speedup from 1.358x to
1.3033x and the component aggregate to 1.241472x / 10.8327 ms. Restore the default
four stages. No E021 variant cleared the promotion gate, so S006 remains unchanged.

### E022: full-context attention memory and scheduler follow-ups (rejected)

E022 tested four follow-ups from an isolated copy of `muse_attention_decode`; the
reference-only component directory and the decoder submission were not modified.
The control remains attention job `a23699e6438a4fa4` at 1.555184x / 2421.77 us,
with full-decode partial kernels at 101.9 us for B16 and 200.1 us for B32.

Job `cd35ae2799854bed` added explicit next-group K/V L2 prefetches. It passed all
16 tests but duplicated/competed with demand traffic: the partial kernels rose to
140.4/375.2 us and aggregate performance fell to 1.367605x / 2753.94 us. Job
`0d1b9b673ee24849` fused the B16 combine, RMSNorm, and gate into the last split CTA.
It also passed, but the fence and completion atomics raised the B16 route from about
105.3 us to 110.5 us; aggregate performance was 1.539519x / 2446.41 us.

Job `a49056f415bb4f31` used a 64B L2 fetch for each independent B16 KV head while
retaining the E020 128B fetch for paired B32. B16 full decode worsened to 117.0 us
and the aggregate fell to 1.526476x / 2467.32 us, confirming that cross-head 128B
line reuse is material. Finally, job `709e70f4544c4fb9` doubled full-verify FA3's
split ceiling to 16/8 for B16/B32. Verify medians moved only 196.89 -> 195.71 us and
382.74 -> 379.80 us (0.6-0.8%, inside noise), while the aggregate was 1.539696x /
2446.13 us. Do not promote any E022 variant; E020 remains the local attention
optimum and S006 remains unchanged.

### E023: eight-query full-verify specializations (rejected)

E023 tested four custom B16 full-context `q_len=8` mappings from isolated attention
copies. Job `062591d536d94c28` assigned one CTA to each query token. It passed all
16 tests, but reread the cache once per query and raised the full-verify kernel to
607.0 us (B32: 1524.9 us). Job `e6db04777ff948e2` instead loaded K/V once per
block and kept eight online-softmax states in shared memory. It also passed, but
shared-state traffic serialized the query updates and raised B16 to 928.0 us.

Job `87bd35c2a54340a5` exercised the dormant Hopper WGMMA 64-row verify prototype.
Its B16 benchmark recheck failed at relative L2 error 1.038, exposing a structural
descriptor/accumulator-layout bug. Job `e3dc16319ba94605` used four key warps with
all eight query accumulators warp-private. It passed every test, but register pressure
and serialized query math left the kernel at 652.6 us. All three correct custom
paths are more than 3x slower than FA3's roughly 193 us path.

Finally, corrected job `25133902f846459d` isolated 16 FA3 scheduler splits for B16
full verify while retaining four splits for B32. It passed 16/16 at 1.547080x /
2434.46 us, but B16 moved only 196.89 -> 196.27 us (0.32%, below run noise and
weaker than E022's combined observation). Do not promote any E023 variant. E020
remains the attention optimum, and the validated decoder source remains S006.

### E024: fused packed-NVFP4 expert backend (promoted)

E024 embedded the independently validated `muse_moe_w4a16` v49 backend, froze its
six H200-winning tactics for `T={1,8,16,32,128,256}`, and compiled only those
extensions. The first integration job `029bb1a34b3144e6` exposed the decoder's
`post_expert_norm` / component `post_norm_w` naming difference during setup. Job
`609cb7a059984561` fixed that adapter but faulted on the first CUDA case: v49 was
given the decoder's global `local_start` even though its packed arrays contain only
this rank's 32 local experts, causing an out-of-bounds weight index on nonzero EP
shards. Rotating `(topk_ids - local_start) % 256` into the local bank and passing a
static zero offset matches the existing decoder Marlin semantics while preserving
the original global routing IDs returned to the judge.

Corrected job `09485d85d2944b83` passed all 13/13 cases with no flags and scored
1.639777x / 13.79474 ms (95% CI 1.61518x-1.66437x). This is 17.82% lower latency
than S006's 16.78663 ms and improves every measured shape. Promote E024 as S007.

| Shape | S006 us | E024 us | Raw gain | E024 vs SGLang |
|---|---:|---:|---:|---:|
| decode 1 sliding | 110.47 | 89.09 | 19.35% | 1.9135x |
| decode 1 full | 119.28 | 100.50 | 15.74% | 1.8156x |
| decode 16 sliding | 189.64 | 153.08 | 19.28% | 1.6428x |
| decode 16 full | 279.26 | 248.38 | 11.06% | 1.7351x |
| decode 32 sliding | 213.63 | 193.86 | 9.25% | 1.3148x |
| decode 32 full | 411.35 | 389.82 | 5.23% | 1.5814x |
| verify 1 sliding | 177.17 | 146.76 | 17.16% | 1.7963x |
| verify 1 full | 191.41 | 165.50 | 13.54% | 1.6990x |
| verify 16 sliding | 474.04 | 350.13 | 26.14% | 1.5283x |
| verify 16 full | 648.00 | 518.49 | 19.99% | 1.3797x |
| verify 32 sliding | 594.64 | 424.32 | 28.64% | 1.7973x |
| verify 32 full | 936.96 | 770.75 | 17.74% | 1.4210x |

### E025: in-kernel EP route localization (promoted as S008)

E025 removes E024's separate `(topk_ids - local_start) % 256` tensor operation.
The selected v49 `gk`, `ring`, and `wx` CUDA entry points now receive the existing
device-resident `local_start` scalar, subtract it while classifying routes, and keep
packed weights and global scales indexed by the resulting local expert ID. Python
passes the original global routing IDs directly, eliminating one elementwise launch
from every decoder shape without changing the routing result returned to the judge.

The first standalone component build, job `fc5f1769ada5449d`, caught two stale `p->lo`
weight-address expressions at CUDA compile time. After correcting those local indices,
job `9e16ad566e47470a` deliberately shifted every route by 32 and supplied
`local_start=32`; it passed all 5/5 EP8 cases with no flags. This nonzero-shard test
covered the `gk`, `ring`, and `wx` token buckets before spending the decoder attempt.

Scored decoder job `ff7edb2dc5374bc0` passed all 13/13 cases with no flags and scored
1.656177x / 13.65814 ms (95% CI 1.63133x-1.68102x). That is 0.99% lower latency than
E024/S007's 13.79474 ms. Profiles confirm the standalone localization kernel is gone;
the remaining elementwise entry is the existing layer-normalization boundary. Promote
the exact scored source as S008. The live snapshot moved from decoder rank 364/872 to
355/873 and from overall rank 367/1438 to 358/1440.

E025 profile: `profiles/submission.ff7edb2dc5374bc0.txt` (local, intentionally ignored
by Git). The source SHA-256 is
`2de6943fad9df946d00ab4f93c6eaa69a9d368b7af9277b70c570ee2bd2f5e3e`.

### E026: gk-fused pre-expert RMSNorm (rejected)

E026 assigned one resident `gk` CTA to each `T<=32` token, computed the 4096-wide
RMSNorm into a reusable BF16 workspace, and joined the grid once before the existing
activation staging. Job `f31a58d44ce640f2` compiled, passed all 13 tests with no
flags, and removed the standalone `vectorized_layer_norm` entry from all eight targeted
profiles. It nevertheless scored only 1.629286x / 13.88357 ms, 1.65% higher latency
than S008. Seven of eight affected branches lost 0.9%-3.7% normalized speedup. At
`T=1`, the fused `moe_k` grew from roughly 28 us to 40 us: the all-SM barrier and
global normalized workspace cost more than the approximately 6 us launch they removed.
Do not repeat grid-wide norm fusion in `gk`. The next isolation should preserve the
existing pooler/router overlap and use a standalone fixed-4096 RMSNorm kernel instead.

### E027: fixed-4096 pre-expert RMSNorm (promoted to scored confirmation)

E027 keeps the original pooler/router stream DAG and replaces PyTorch's generic
`vectorized_layer_norm` with one 256-thread CTA per token. The kernel uses BF16x2
loads/stores, FP32 sum-of-squares, the same `1e-5` epsilon, and the module's effective
BF16 `(1 + weight)`. Job `6111ef58daa24397` passed all 13 tests with no flags and
scored 1.658270x / 13.64091 ms, 0.13% lower latency than S008. The aggregate movement
is inside run noise, but the directly affected kernel consistently fell from S008's
5.6-8.3 us to 2.5-2.9 us on all twelve shapes while `moe_k` remained independent.
Keep the exact candidate and use one scored run to test whether the structural kernel
gain survives whole-run expert/routing variance.

The scored confirmation, job `773381b773ff4764`, passed all 13 tests with no flags at
1.668944x / 13.55366 ms (95% CI 1.64391-1.69398). This is 0.76% lower latency than
S008 and establishes E027 as S009. Its profile is
`profiles/submission.773381b773ff4764.txt` (local, intentionally ignored by Git), and
the promoted source SHA-256 is
`7c09f5fbdb7430e946a9c39c0078ce4589888095673baaedda6215bec6e23f55`.
The refreshed live leaderboard places S009 at decoder rank 373/904.

### E028: v52 CPB2 large-token MoE port (refine)

E028 ports the standalone backend's two-resident-CTA WGMMA configuration to
`T=128` and reuses each persistent kernel's stable route ranks to gather expert
activations in-kernel. It preserves E025's device-resident global-to-local expert
localization and retains the existing NCW16 configuration at `T=256`. Job
`a545e99e9565410c` passed all 13 tests with no flags at 1.675392x / 13.50150 ms.

The directly targeted `T=128` kernel improved consistently: verify-16 sliding
fell from 251.0 to 214.8 us and full from 223.3 to 192.7 us, both about 14%.
The fused gather was not a win for the existing `T=256` NCW16 path: its weighted
MoE time rose about 1.8% after including the removed 8-9 us `xgather` launch.
Refine the candidate by compiling fused gather only into the CPB2 specialization
and restore standalone gather for NCW16. The exact E028 source SHA-256 is
`b92ab130497f75248eb06fb1e46c17ea62f29eaaf60a757530f99db34220128b`.

### E029: shape-specific activation gather (promote to scored confirmation)

E029 compiles the in-kernel gather only into the `T=128` CPB2 specialization and
restores v49's standalone `xgather` for the unchanged `T=256` NCW16 path. Job
`fb6547cd0d814e79` passed all 13 tests with no flags at 1.686096x / 13.41578 ms,
1.02% lower latency than the S009 scored result. Against S009, verify-16's
normalized sliding/full speedups improve from 1.5812x/1.3908x to
1.6489x/1.4347x, while the profiler confirms that `xgather` appears only in the
two `T=256` branches. Promote E029 to one scored confirmation. Its profile is
`profiles/submission.fb6547cd0d814e79.txt` and the source SHA-256 is
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.

The independent scored confirmation, job `3a7e7defad2e4b07`, passed all 13 tests
with no flags at 1.692448x / 13.36543 ms (95% CI 1.66706-1.71783). This is
1.39% lower latency than S009 and establishes E029 as S010. Its profile is
`profiles/submission.3a7e7defad2e4b07.txt`; the refreshed live leaderboard
places it at decoder rank 359/905.

### E030: standalone BF16 v55 backend replacement (reject)

E030 replaced the embedded BF16 projection v18 backend with the standalone
projection project's v55 winner. The replacement included guarded QKV-32,
O-16/O-32, unpool-16/32, router-32, and dedicated router-8/16/128/256 routes;
the known-slower wide unpool-128 candidate rejected itself during its prepare
gate. Job `fc183181de9e4a5b` passed all 13 tests with no flags at
1.664153x / 13.59268 ms (95% CI 1.63919-1.68912), 1.70% higher latency than
S010. Prepare time also rose from about 56 s to 188.45 s.

Normalized speedup regressed on nine of twelve scored shapes, including
decode-16 sliding/full (-1.89%/-2.76%), decode-32 sliding (-3.60%), and
verify-16 sliding/full (-2.78%/-2.80%). Only decode-1 full (+0.19%), verify-1
full (+0.88%), and verify-32 full (+0.37%) moved upward, all inside noise.
Reject the wholesale backend replacement and restore S010. A parallel audit of
the prefill winner found no decoder-compatible port: its material wins target
T=753/3012/16384 attention or large grouped NVFP4 work, while its reusable
Humming expert path, stream overlap, and route handling are already present or
superseded in S010; its compact unpool and route-consolidation experiments were
themselves rejected.

### E031: standalone MoE v57 SW=1 small-token port (reject)

E031 isolated the newer standalone MoE v57 optimization: barrier-free SW=1
tile groups for `T=16` and `T=32`. It added the two v57 configurations while
explicitly excluding `T=8`; attention, BF16 projection, and all larger-token
MoE routes remained unchanged. Unscored job `b0b632a2146c4a85` passed all 13
tests with no flags at 1.675142x / 13.50351 ms (95% CI 1.65002-1.70027), 1.03%
higher latency than S010. Prepare completed in 193.16 s.

The fresh autotuner choices were unstable in the integrated decoder and made
the profiled MoE stage slower on most weighted decode shapes: decode-16 sliding
rose from 78.2 to 97.1 us (+24.2%), decode-32 sliding from 102.3 to 123.4 us
(+20.6%), and decode-32 full from 111.3 to 123.0 us (+10.5%). Although a few
colocated profiles improved, the aggregate and most per-shape medians regressed.
Reject the port, restore S010 exactly, and do not spend a scored submission.
Profiles: `profiles/submission.b0b632a2146c4a85.{json,txt}`.

### E032: NCW16/TPX128 compile-only control (inconclusive)

E032 added an NCW16/TPX128 large-MoE specialization intended to process up to
128 expert tokens per packed-weight pass instead of 64. Its calculated dynamic
shared-memory footprint at `T=256` was 220,968 B, below H200's 227,328 B limit.
Unscored job `ec2e69e4172e41d0` compiled the specialization and passed all 13
tests with no flags at 1.688656x / 13.39544 ms (95% CI 1.66333-1.71399).

Post-run dispatch inspection found that production S010 has `M_TUNE=False` and
the hardcoded M=256 entry still selected NCW16/TPX64. E032 therefore did not
exercise TPX128 and is only a compile/resource control, not a performance
result. Prepare completed in 157.69 s. Profile:
`profiles/submission.ec2e69e4172e41d0.{json,txt}`.

### E033: forced NCW16/TPX128 weight-pass reuse (reject)

E033 changed only the hardcoded M=256 dispatch to NCW16/TPX128, definitively
exercising the wider pass on scored `T=256` and held-out `T=264`. Unscored job
`2cda792587da4836` passed all 13 tests with no flags at 1.666911x / 13.57019 ms
(95% CI 1.64191-1.69191), 1.53% higher latency than S010. Prepare completed in
166.73 s.

The T=256 MoE profile was neutral on sliding (266.5 to 266.6 us) and regressed
7.2% on full attention (247.1 to 264.9 us). Verify-32 end-to-end latency rose
5.93% sliding and 3.10% full. Halving repeated weight dequantization for expert
counts above 64 does not repay the doubled WGMMA accumulator/register footprint
and wider activation staging. Reject TPX128, restore S010 exactly, and spend no
scored submission. Profile: `profiles/submission.2cda792587da4836.{json,txt}`.

### E034: pass-scoped W13-to-W2 dependency (reject)

The requested instruction-level E2M1 decode audit found no source-level port to
make: the standalone kernel's local `sm_90a` SASS inspection already shows NVCC
lowering the masks/ORs to `LOP3.LUT`, scale construction to `PRMT`/`LEA`, and
the packed BF16 multiply to `HMUL2.BF16_V2`. Rewriting those expressions as
inline PTX would reproduce the current instructions rather than remove work.

E034 instead tested the smallest genuine W13-to-W2 pipeline change. It replaced
the expert-wide W13 completion counter with per-(expert, token-pass) counters,
allowing a W2 pass to start once only its matching W13 pass was complete. A
producer-local bit mask avoided repeated acquires, and the final CTA cleared the
expanded counter table cooperatively. Unscored job `cf669a9823f342ef` passed all
13 tests with no flags at 1.668964x / 13.55350 ms (95% CI
1.64393-1.69400), 1.41% higher latency than S010. Prepare completed in
153.04 s.

The pass pipeline improved verify-16 full's profiled `moe_x` from 211.6 to
196.1 us, but regressed verify-16 sliding from 212.7 to 225.4 us and verify-32
sliding/full from 266.5/247.1 to 280.1/266.2 us. The extra release atomics,
acquire polling, counter footprint, and replay cleanup outweigh the limited
cross-pass overlap. Reject the pass-scoped counters, restore the exact S010
source SHA-256 `42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`,
and spend no scored submission. Profile:
`profiles/submission.cf669a9823f342ef.{json,txt}`.

### E035: paired-WGMMA async groups at T=256 (reject)

Standalone MoE job `9c38c537be5744a1` validated `WGPAIR`: two K16 WGMMA
operations share each async commit group, halving wait/fence/commit operations
without changing dequantization or math. It passed 5/5, and its tuner measured
301 us versus 305 us for ordinary NCW16 at M=256. E035 therefore compiled one
additional `NCW16+WGPAIR` specialization and changed only the decoder's hardcoded
T=256 dispatch. Unscored job `b4804e54be1a41b0` passed all 13 tests with no
flags at 1.694244x / 13.35126 ms (95% CI 1.66883-1.71966), only 0.11% lower
latency than S010. Prepare completed in 185.22 s.

The first profile was mixed rather than a robust kernel win: verify-32 sliding
`moe_x` fell from S010's 266.5 to 254.9 us, but full rose from 247.1 to 258.4
us. A layer-aware refinement retained WGPAIR only when `window != 0`. Its first
job, `b1a8d8a8a6134e85`, failed before GPU work because the initial bookkeeping
looked for a nonexistent `config["layer"]`; duplicate request
`e8efeb523e2c45dd` correctly reused that result. The corrected window-derived
dispatch, job `bffdca76490643ad`, passed all 13 tests with no flags but regressed
to 1.671586x / 13.53224 ms (95% CI 1.64651-1.69666), 1.25% higher latency than
S010. Its verify-32 sliding/full `moe_x` times were 259.6/260.5 us.

The standalone 4-us microkernel result does not survive integrated decoder
overlap reliably, and the all-layer aggregate improvement is far inside run
variance. Reject WGPAIR, restore the exact S010 source SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`,
and spend no scored submission. Profiles:
`profiles/submission.b4804e54be1a41b0.{json,txt}` and
`profiles/submission.bffdca76490643ad.{json,txt}`.

### E012 versus E007

| Shape | E007 speedup | E012 speedup | Normalized gain |
|---|---:|---:|---:|
| decode 1 sliding | 1.2327x | 1.2676x | 2.83% |
| decode 1 full | 1.2249x | 1.2538x | 2.36% |
| decode 16 sliding | 1.1673x | 1.1922x | 2.13% |
| decode 16 full | 1.3435x | 1.3556x | 0.90% |
| decode 32 sliding | 1.1786x | 1.2099x | 2.65% |
| decode 32 full | 1.4365x | 1.4583x | 1.52% |
| verify 1 sliding | 1.2507x | 1.2840x | 2.66% |
| verify 1 full | 1.2274x | 1.2697x | 3.45% |
| verify 16 sliding | 1.0649x | 1.0902x | 2.37% |
| verify 16 full | 1.0598x | 1.0737x | 1.31% |
| verify 32 sliding | 1.1798x | 1.2151x | 2.99% |
| verify 32 full | 1.1206x | 1.1295x | 0.80% |

### E002 versus E001

| Shape | E001 us | E002 us | Improvement | E002 vs SGLang |
|---|---:|---:|---:|---:|
| decode 1 sliding | 180.56 | 160.99 | 10.84% | 1.1066x |
| decode 1 full | 190.69 | 171.42 | 10.10% | 1.0961x |
| decode 16 sliding | 271.06 | 256.90 | 5.22% | 1.0821x |
| decode 16 full | 434.14 | 425.78 | 1.93% | 1.0476x |
| decode 32 sliding | 268.45 | 263.50 | 1.84% | 1.0655x |
| decode 32 full | 622.26 | 614.30 | 1.28% | 1.0274x |
| verify 1 sliding | 268.85 | 227.20 | 15.49% | 1.2219x |
| verify 1 full | 285.18 | 236.91 | 16.93% | 1.2040x |
| verify 16 sliding | 540.30 | 504.86 | 6.56% | 1.0299x |
| verify 16 full | 688.42 | 680.88 | 1.09% | 1.0234x |
| verify 32 sliding | 744.24 | 686.48 | 7.76% | 1.0218x |
| verify 32 full | 1097.26 | 1062.26 | 3.19% | 1.0140x |

### E003 versus E002

| Shape | E002 us | E003 us | Improvement | E003 vs SGLang |
|---|---:|---:|---:|---:|
| decode 1 sliding | 160.99 | 144.89 | 10.00% | 1.1949x |
| decode 1 full | 171.42 | 152.62 | 10.97% | 1.1866x |
| decode 16 sliding | 256.90 | 227.79 | 11.33% | 1.1382x |
| decode 16 full | 425.78 | 323.21 | 24.09% | 1.3205x |
| decode 32 sliding | 263.50 | 218.03 | 17.26% | 1.1493x |
| decode 32 full | 614.30 | 427.46 | 30.42% | 1.4275x |
| verify 1 sliding | 227.20 | 222.38 | 2.12% | 1.2092x |
| verify 1 full | 236.91 | 230.71 | 2.62% | 1.2028x |
| verify 16 sliding | 504.86 | 498.99 | 1.16% | 1.0307x |
| verify 16 full | 680.88 | 681.58 | -0.10% | 1.0218x |
| verify 32 sliding | 686.48 | 694.82 | -1.21% | 1.0195x |
| verify 32 full | 1062.26 | 1058.65 | 0.34% | 1.0142x |

### E004 versus E003

Only the last four branches changed implementation. Raw times are shown for context;
the normalized column compares each candidate's speedup against its colocated SGLang
run, which removes most cross-machine clock and driver drift.

| Shape | E003 us | E004 us | E003 speedup | E004 speedup | Normalized gain |
|---|---:|---:|---:|---:|---:|
| verify 16 sliding | 498.99 | 467.30 | 1.031x | 1.097x | 6.40% |
| verify 16 full | 681.58 | 648.89 | 1.022x | 1.073x | 4.99% |
| verify 32 sliding | 694.82 | 567.12 | 1.019x | 1.227x | 20.41% |
| verify 32 full | 1058.65 | 951.90 | 1.014x | 1.147x | 13.12% |

### E005 versus E004

| Shape | E004 speedup | E005 speedup | Normalized gain |
|---|---:|---:|---:|
| decode 1 sliding | 1.187x | 1.209x | 1.85% |
| decode 1 full | 1.185x | 1.201x | 1.35% |
| decode 16 sliding | 1.143x | 1.173x | 2.62% |
| decode 16 full | 1.314x | 1.345x | 2.36% |
| decode 32 sliding | 1.142x | 1.171x | 2.54% |
| decode 32 full | 1.410x | 1.438x | 1.99% |
| verify 1 sliding | 1.227x | 1.256x | 2.36% |
| verify 1 full | 1.207x | 1.232x | 2.07% |

## S011: four-operation WGMMA commit window (rejected)

S011 ported the standalone v71 `WG4` consumer schedule into the decoder MoE
kernel. It alternates two decoded-A register banks, commits four WGMMA
operations at once, and uses `wait_group 1`, allowing eight WGMMA operations
to remain in flight without changing TMA bytes, packed weight layout,
arithmetic order, or the fused epilogue. The exact decoder specialization
compiled locally for SM90a before remote evaluation.

The first off-board job, `1f073342f43f4992`, passed all 13 tests at
1.671960x / 13.52921 ms. Audit showed that it was only a compile/correctness
checkpoint: `M_TUNE=False` and the frozen M256 entry still selected plain
`-DNCW=16`, so WG4 never executed.

S011b changed only the frozen M256 entry to `-DNCW=16 -DWG4`; M128 retained
the proven CPB2/RPW2 configuration. Job `3d056d6516af4f7a` passed all 13 tests
at 1.685744x / 13.41859 ms, but the targeted kernel comparison lost:

| Verify-32 shape | Plain NCW16 (`1f073342`) | WG4 (`3d056d65`) | WG4 change |
|---|---:|---:|---:|
| sliding `moe_x` | 266.1 us | 268.0 us | -0.7% |
| full `moe_x` | 246.9 us | 257.5 us | -4.3% |

The second decoded-A bank and longer accumulator lifetime increase register
pressure enough to erase the reduced wait/fence/commit frequency. Reject WG4,
spend no scored submission, and retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Profiles:
`profiles/submission_s011_wg4.1f073342f43f4992.{json,txt}` and
`profiles/submission_s011b_wg4_m256.3d056d6516af4f7a.{json,txt}`.

## S012: fused pre-expert RMSNorm + M256 expert gather (rejected)

S012 tested the safest high-value boundary of the proposed glue/MoE fusion.
For M256, the existing `xgather` already launches one 256-thread CTA per
token, with its eight warps owning that token's eight routes.  The candidate
therefore moved the pre-expert RMS reduction into that CTA using the exact
standalone kernel's reduction geometry, applied the BF16 effective norm
weight while scattering, and bypassed the standalone RMS launch only for
M256.  This removes the normalized `[256, 4096]` intermediate and its global
write/read.  M1--M128 were deliberately unchanged because their custom MoE
layouts would duplicate normalization across persistent CTAs or require a
grid barrier.

The affected NCW16 SM90a specialization compiled locally.  Off-board job
`a6ecdfd0914443ca` passed all 13 correctness cases, but the targeted result
was decisively negative:

| Verify-32 shape | S010 `xgather` | S012 fused `xgather` | Added gather time |
|---|---:|---:|---:|
| sliding | 8.0 us | 15.0 us | +7.0 us |
| full | 7.8 us | 15.5 us | +7.7 us |

The removed RMS kernel costs only about 2.4--2.8 us and normally overlaps the
router/routing side stream.  BF16 conversion, norm-weight multiplication, and
the second activation read instead landed in serialized gather work, leaving
roughly 4.5--5.0 us net regression on M256.  The whole-check score moved from
S010's 13.36543 ms / 1.692448x to 13.60040 ms / 1.663209x; cross-worker noise
affects that aggregate, but the within-profile gather regression is much
larger than noise.  Reject this fusion, do not extend it to M128, spend no
scored submission, and retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Profile: `profiles/submission_s012_fused_glue_moe.a6ecdfd0914443ca.{json,txt}`.

## S013: repaired two-CTA MoE operand multicast (rejected)

S013 ported the standalone v73 Hopper cluster protocol into the decoder's
large-token `moe_x` backend.  The port includes the repaired odd-pass work
count, an acknowledged release/acquire queue mailbox, weight multicast for two
adjacent token passes, and activation multicast when the ranks split the row
blocks of an odd pass.  Decoder-specific global expert IDs and the existing
M128/M256 gather policies were preserved.  Both exact SM90a specializations
compiled locally without a WGMMA-serialization or spill warning.

Job `1ab6ddefcc724a5a` tested the standalone v73 work shape in the decoder:
`NCW8/RPW2/TPX32/CPB2/CLUSTER2/MC2`, with fused gather only at M128.  It passed
all 13 correctness cases and, importantly, completed without the old v67
deadlock.  Performance nevertheless missed the 8% gate in every large branch:

| Shape | S010 `moe_x` | S013 TPX32 cluster | Change |
|---|---:|---:|---:|
| verify-16 sliding | 212.7 us | 240.6 us | -13.1% |
| verify-16 full | 211.6 us | 228.0 us | -7.8% |
| verify-32 sliding | 266.5 us | 294.7 us | -10.6% |
| verify-32 full | 247.1 us | 332.6 us | -34.6% |

At M256, paired TPX32 passes load one packed-weight tile per 64 tokens, which
only matches S010's ordinary TPX64 CTA rather than reducing traffic.  S013b
therefore restored S010 at M128 and tested the decisive traffic-saving form at
M256: two unchanged `NCW16/TPX64` CTAs clustered across SMs, so adjacent
64-token passes share one packed-weight TMA transfer.  Job
`e56e0ebe5c43431d` again passed 13/13, but lost more severely:

| Shape | S010 `moe_x` | S013b NCW16/TPX64 cluster | Change |
|---|---:|---:|---:|
| verify-32 sliding | 266.5 us | 391.4 us | -46.9% |
| verify-32 full | 247.1 us | 361.3 us | -46.2% |

S013/S013b whole-check scores were respectively 13.81510 ms / 1.637360x and
13.81310 ms / 1.637598x, versus S010's 13.36543 ms / 1.692448x.  The repaired
protocol proves correctness and forward progress, but cluster queue handshakes,
slot readiness, and cross-SM synchronization cost much more than the removed
TMA traffic.  Reject the cluster-multicast architecture under its quantitative
gate; do not spend a scored run or pursue mask/scheduling micro-variants.
Retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Profiles:
`profiles/submission_s013_cluster_multicast.1ab6ddefcc724a5a.{json,txt}` and
`profiles/submission_s013b_cluster_ncw16.e56e0ebe5c43431d.{json,txt}`.

## S014: decoder-cost-aware near-tie route shaping (rejected)

S014 adapted the measured prefill route shaper to the decoder's actual MoE
schedule.  A custom one-warp-per-token selector emits exact biased top-8 picks,
the best unselected non-local near tie, local counts, and normalized raw-sigmoid
weights.  A second single-CTA planner allows replacement only when the local
pick is within 0.010 of ninth and the candidate is within 0.020 of eighth.  It
moves at most 5% of local routes and performs only one action: eliminate an
M32 expert or cross the previous 32/64-route pass boundary at M128/M256.

Local RTX validation reproduced exact top-8 when no reroute was legal, passed
all safety invariants for M32/M128/M256 at four rotated EP starts, and captured
and replayed under a CUDA graph.  H200 check `9b6330be9ef449db` then passed all
13 correctness cases with no flags, but regressed the aggregate result from
S010's 13.36543 ms / 1.692448x to 13.59245 ms / 1.664182x.

| Affected shape | S010 layer | S014 layer | Change |
|---|---:|---:|---:|
| decode-32 sliding | 193.22 us | 202.42 us | -4.8% |
| decode-32 full | 373.40 us | 400.58 us | -7.3% |
| verify-16 sliding | 318.23 us | 325.95 us | -2.4% |
| verify-16 full | 492.63 us | 500.63 us | -1.6% |
| verify-32 sliding | 402.63 us | 409.33 us | -1.7% |
| verify-32 full | 754.98 us | 769.42 us | -1.9% |

The selector costs 6.2--6.8 us and the planner 2.8--6.6 us, replacing S010's
2.8--3.1 us SGLang routing kernel.  MoE time moved in both directions across
the six profiles, so one conservative boundary action does not reliably repay
that 6--10 us net routing overhead.  Reject the two-kernel architecture; any
future route shaping must be fused into the router GEMM or MoE route pass and
must expose actual saved-pass counters before another whole-layer check.
Retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Profile:
`profiles/submission_s014_route_shaper.9b6330be9ef449db.{json,txt}`.

## S015: 16-CTA fused router GEMM + exact top-8 (rejected)

S015 tested the direct Hopper/PTX fusion suggested by CUTLASS's top-k epilogue
pattern, adapted to this router's much harder `N=256, K=8192, top_k=8`
geometry.  The active M256 router divides the expert columns into four 64-column
tiles and each tile across four K-reduction ranks.  The candidate therefore made
all 16 CTAs one nonportable Hopper cluster.  After the existing hi+lo BF16 WGMMA
and K reduction, every CTA selected the best eight of its 16 experts, four rank
lists merged to a 64-expert tile list through DSM, and four tile lists merged to
the exact global top eight.  Each candidate carried both its biased selection key
and raw sigmoid score so the kernel could emit normalized FP32 route weights and
INT32 IDs directly, eliminating the logits tensor and SGLang routing launch.

Local validation proved unique coverage of all four `64 x 16` WGMMA fragment
slices and matched a direct 256-way top-8 and normalization in 200/200 randomized
trials.  The final SM90a ptxas report used 90 registers with a zero-byte stack and
zero spill loads/stores.  Candidate source size was 989,239 bytes; the protected
S010 source remained byte-identical.

Off-board H200 job `96e49f415f134a6f` passed all 13 correctness cases with no
flags, but the 16-CTA cluster was decisively slower:

| Verify-32 shape | S010 router GEMM | S010 routing | S015 fused kernel | Kernel change |
|---|---:|---:|---:|---:|
| sliding | 8.5 us | 3.1 us | 35.1 us | 3.03x slower than the combined chain |
| full | 11.6 us | about 3.0 us | 40.7 us | 2.79x slower than the combined chain |

The scored-shape layer medians regressed from 402.63 to 451.62 us on verify-32
sliding (+12.2%) and from 754.98 to 805.18 us on verify-32 full (+6.6%).  Overall
latency moved from S010's 13.36543 ms / 1.692448x to 13.55211 ms / 1.669135x.
The result shows that register pressure was not the limiting factor: forcing four
independent expert tiles into one 16-block residency group and paying three
cluster-wide DSM synchronization phases costs far more than the removed 3 us
routing kernel and 256-wide logits round trip.

Reject the 16-CTA architecture and spend no scored submission.  A cluster-4
partial-top8 plus separate global merge would preserve GEMM scheduling, but it
would retain a second launch and cannot reach the required 20% component gate;
do not spend the final check on that micro-variant.  Retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Profile:
`profiles/submission_s015_fused_router.96e49f415f134a6f.{json,txt}`.

## S016: full-verify attention replacements (rejected)

S016 tested the full `q_len=8`, B16 attention bottleneck in the standalone
`muse_attention_decode` component before touching the decoder.  Three
structurally different paths were evaluated against FA3's approximately 193 us
full-attention main kernel:

1. The archived `v22a` kernel mapped all 64 verification query rows for one
   request/KV head onto a Hopper warp group and used WGMMA for both QK and PV.
   It compiled to 96 registers and 9,280 bytes shared memory with no spills, but
   cached H200 job `981734afb9e54664` failed the activated B16 benchmark recheck
   with relative L2 error 1.167.  Its fragment/output mapping is not valid.
2. The archived `v24b` kernel shared each K/V tile across eight warp-private
   causal query streams.  It compiled to 166 registers and 42,048 bytes shared
   memory with no spills.  H200 job `3da907d15d6d47b4` passed 16/16 with no
   flags, but its custom main kernel took 928.6 us and its total path 932.9 us,
   about 4.8x slower than FA3.
3. A new numerically stable hybrid kept FA3 for the long prefix, shortened
   `cache_seqlens` by eight, requested the prefix LSE, and positively merged the
   allowed 1--8 current-block tokens in FP32 before fused RMSNorm and gating.
   This avoids the rare catastrophic cancellation of the earlier noncausal
   subtraction prototype.  H200 job `5637287225f04dd9` passed all 16 tests and
   all benchmark rechecks with no flags.

The stable hybrid was still slower on its only activated shape:

| B16 full-verify component | Current FA3 path | S016 prefix merge |
|---|---:|---:|
| FA3 main kernel | about 193.0 us | 194.6 us |
| correction / merge | none | 10.2 us |
| total profiled path | about 200.4 us | 214.8 us |
| aggregate component score | 2470.48 us | 2499.16 us |

Noncausal scheduling did not reduce the long-context FA3 kernel, so the exact
merge and extra scheduler/length work are pure overhead.  Reject all three full
verify replacements and do not integrate them into the decoder.  No decoder
check or scored run was spent.  Retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Candidate: `submission_attention_s016_prefix_merge.py`.
Profile:
`profiles/submission_attention_s016_prefix_merge.5637287225f04dd9.{json,txt}`.

## S017: redundant E035 `WGPAIR` recheck (rejected)

S017 attempted to apply the Auto-research policy's frontier-change rule by
porting standalone MoE v64 `WGPAIR`: two consecutive K16 WGMMA operations per
commit, with two groups/four operations in flight.  The final ledger audit found
that this was not a new decoder hypothesis.  E035 above had already tested the
same all-layer M256 dispatch in job `b4804e54be1a41b0`, followed by a
sliding-only refinement in `bffdca76490643ad`, and rejected both as non-robust.
S017 is therefore an independent redundant recheck of E035, not a new frontier
experiment.

The port was deliberately narrow.  Its `WGPAIR` source branch is byte-identical
to standalone v64, activates only for the frozen M256 `-DNCW=16` entry, uses no
second A-register bank, and leaves M1/M8/M16/M32/M128 plus all non-MoE code
unchanged.  This is materially different from rejected S011 `WG4`, which kept
eight operations and a second decoded-A bank live.  Static preflight passed the
Python AST, CUDA preprocessor balance, exact-v64 branch comparison, and a
two-region source/dispatch scope audit.

Off-board H200 check `76646a54fc154c4f` compiled the new specialization and
passed all 13 correctness cases with no flags.  It did not meet the predeclared
performance gate:

| Measure | S010 | S017 WGPAIR | Observation |
|---|---:|---:|---|
| Aggregate | 13.36543 ms / 1.692448x | 13.44208 ms / 1.682798x | no aggregate win; confidence intervals overlap |
| Verify-32 sliding layer median | 402.63 us | 402.29 us | -0.09%, below a decisive threshold |
| Verify-32 full layer median | 754.98 us | 752.66 us | -0.31%, below a decisive threshold |
| Verify-32 sliding `moe_x` profile | 266.5 us | 271.8 us | no component saving |
| Verify-32 full `moe_x` profile | 247.1 us | 265.8 us | no component saving |

The S010 and S017 jobs used different generated seeds, so the cross-job layer
and component numbers are directional rather than an exact A/B measurement.
Together with E035's earlier mixed all-layer profile and failed sliding-only
refinement, however, this recheck reinforces that there is no robust >=3 us
saving on both M256 branches.  Reject S017 and spend no scored run.  The process
lesson is as important as the kernel result: every future external-repository
idea must first be searched against experiment names, mechanism aliases, job
IDs, and negative conclusions in this ledger.  `WGPAIR` must not be included in
the broader prompt-derived experiment.  Retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Candidate: `submission_s017_wgpair.py`.
Profile:
`profiles/submission_s017_wgpair.76646a54fc154c4f.{json,txt}`.

## S018: prompt-guided residual/RMS boundary buffer reuse (rejected)

S018 implemented the Auto-research Muse prompt as one conservative boundary
experiment while retaining the prompt's optimizations already present in S010:
asynchronous router overlap, branch-free device routing, and fused boundary
pointwise work.  It replaced both SGLang residual/RMS calls with one exact Triton
kernel and reused storage whose prior value was dead at each boundary.  The first
call wrote `ffn_input` into the dead attention-output allocation while preserving
the caller-owned FP32 stream.  The second wrote `next_attn_input` into the dead
FFN-output allocation and updated the internally owned FP32 stream in place.
The operation order and BF16 rounding points remained identical to the pinned
SGLang implementation.

Local CUDA tests covered T=1/8/32/128/256, exact output agreement, pointer alias
ownership, caller-stream immutability, and CUDA graph capture/replay.  Official
H200 scored job `622997a4deb24f34` then passed all 13 correctness cases with no
flags or rejections, but missed the performance gate:

| Measure | S010 | S018 boundary reuse | Change |
|---|---:|---:|---:|
| Aggregate latency | 13.36543 ms | 13.63646 ms | +2.03% |
| Aggregate speedup | 1.692448x | 1.658811x | -1.99% |
| Decode-1 sliding | 86.19 us | 90.14 us | +4.58% |
| Decode-16 sliding | 149.29 us | 152.13 us | +1.91% |
| Decode-32 sliding | 193.22 us | 199.34 us | +3.17% |
| Verify-16 sliding | 318.23 us | 326.62 us | +2.64% |
| Verify-32 full | 754.98 us | 767.67 us | +1.68% |

Nine of twelve cross-job layer medians regressed, two were effectively flat,
and one improved.  These jobs used different generated seeds, so the per-shape
figures are directional, but both the aggregate score and kernel profile agree.
The combined S018 boundary kernel cost about 10.0 us at decode-16/decode-32,
versus roughly 7.3--7.6 us for S010's two tuned SGLang kernels.  At verify-32
both paths were about 16 us, leaving no saving to offset other variation.

Reject S018.  Graph-captured allocation aliasing changes buffer ownership but
does not eliminate the tensors' DRAM traffic; the standalone kernel still reads
and writes the same data, and the tuned SGLang JIT schedules small shapes better.
The prompt's boundary idea can only become a credible latency win if the work is
folded into a producer epilogue (most plausibly O projection or unpool/GEMM), or
if the tuned JIT gains output-alias support.  Do not promote the candidate, and
retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Candidate: `submission_s018_prompt_boundaries.py`.
Local test: `scripts/test_s018_boundary_alias.py`.
Profile:
`profiles/submission_s018_prompt_boundaries.622997a4deb24f34.{json,txt,log}`.

## S019: streamed MoE finalization queue (rejected)

S019 implemented the CUTLASS-guided producer/consumer experiment for the M256
MoE specialization only.  The existing W2 completion atomics were split into
per-expert/per-pass counters.  The warp completing a pass advanced per-token
route counters, and the last local route published sixteen 256-column
normalization/reduction tasks for that token.  The TMA producer warp attempted
ready tasks while polling an empty pipeline stage; after the sentinel, all
consumer and producer warps drained the remaining tasks.  M128 retained the
proven NCW8/RPW2/CPB2/FGATHER path byte-for-byte at dispatch.

Local tests exercised all-local, all-remote, and randomized route schedules,
verified that every token was published exactly once and all 4,096 output
chunks were covered exactly once, and checked bit-identical BF16 output under a
shuffled task order.  CUDA 12.4 emitted SM90a PTX, and `ptxas` assembled the
candidate with 96 registers, no stack frame, and zero spills.  Official H200
job `88426bc113af413d` passed all 13 correctness cases, but failed the
performance gate catastrophically:

| Measure | S010 | S019 streamed finalization | Change |
|---|---:|---:|---:|
| Aggregate latency | 13.36543 ms | 22.60301 ms | +69.11% |
| Aggregate speedup | 1.692448x | 1.000765x | -40.87% |
| Verify-32 sliding layer | 402.63 us | 10,843.79 us | 26.93x slower |
| Verify-32 full layer | 754.98 us | 11,102.04 us | 14.71x slower |
| Verify-32 sliding `moe_x` | 266.5 us | 10,656.8 us | 39.99x slower |
| Verify-32 full `moe_x` | 247.1 us | 10,713.5 us | 43.36x slower |

The single global queue head is the failure mechanism.  At M256 the persistent
grid exposes roughly one producer/consumer warp per resident role across every
SM.  Thousands of warps repeatedly read the same head and race the same CAS;
one advances it and the rest retry.  The resulting serialized hot atomic and
failed-CAS storm dominates the 4,096 otherwise independent chunks.  Correct
release/acquire publication and zero spills therefore do not translate into
useful overlap.

Reject S019 and do not try a smaller chunk size or another single-head global
queue.  A future overlap design needs contention-free ownership (static task
ranges plus per-token ready flags, or per-CTA/per-SM queues) and must first show
that readiness polling costs less than the original approximately 250 us M256
tail.  Do not promote the candidate; retain exact S010 SHA-256
`42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69`.
Candidate: `submission_s019_stream_finalize.py`.
Local test: `scripts/test_s019_stream_finalize.py`.
Profile:
`profiles/submission_s019_stream_finalize.88426bc113af413d.{json,txt,log}`.

## S020--S022: v99 WGS large-token MoE port (promoted)

The decoder's small-token `gk` backend remained unchanged for `T <= 32`.
For `T > 32`, S020 embedded the standalone MoE v99 persistent WGS backend under
a `V99_` namespace, packed the rank's 32 local experts into its natural-k WGS
layout, and used its fused W13/SwiGLU/W2/post-norm/final-reduction kernel.  The
first proof localized global route IDs with a separate
`(topk_ids - local_start) % 256` tensor operation.  Unscored H200 job
`4127abff791a4466` passed all 13 tests with no flags or rejections at
1.766806x / 12.80293 ms, a decisive improvement over S010's 1.692448x /
13.36543 ms.

S021 passed the device `local_start` pointer into WGS and loaded it independently
in every kernel thread.  Job `adfeb8155fcc4663` remained correct at 1.770937x /
12.77307 ms, but the profile exposed a false aggregate win: the replicated
scalar load increased the M256 WGS kernel from roughly 197--198 us to 210--217
us.  Do not use S021.

S022 retained device-side localization without the tensor launch or replicated
load.  Thread 0 of each CTA loaded `local_start` once, published it through the
not-yet-used shared routing-count array, synchronized before the WGS helper
warps split, and then allowed v99's existing `id - p.lo` route paths to consume
the broadcast value.  This adds no shared-memory allocation.  Unscored job
`39d932ee11f14b82` passed 13/13 at 1.767656x / 12.79678 ms; its profiled WGS
kernels were 141.6--155.9 us at M128 and 185.7--192.7 us at M256.

Official scored job `0200db73e80c4907` passed all 13 correctness cases with no
flags or rejections at 1.761846x / 12.83898 ms (95% CI
1.735418--1.788274).  Compared with scored S010, aggregate latency improved
3.94% and speedup improved 4.10%:

| Measure | S010 | S022 scored | Change |
|---|---:|---:|---:|
| Aggregate latency | 13.36543 ms | 12.83898 ms | -3.94% |
| Aggregate speedup | 1.692448x | 1.761846x | +4.10% |
| Verify-16 sliding | 318.23 us | 260.79 us | -18.05% |
| Verify-16 full | 492.63 us | 439.64 us | -10.76% |
| Verify-32 sliding | 402.63 us | 336.24 us | -16.49% |
| Verify-32 full | 754.98 us | 692.30 us | -8.30% |

Promote S022.  `submission.py` SHA-256 is
`f6c081ce2ce356e194b9a163a4ccab92b4b530da3a564ae56e5ed147b405786c`.
The published leaderboard rank after indexing was 435/1058.  Builders:
`scripts/build_s020_v99_wgs.py`, `scripts/build_s021_v99_route_fused.py`, and
`scripts/build_s022_v99_route_broadcast.py`.  Candidate:
`submission_s022_v99_route_broadcast.py`.  Profiles:
`profiles/submission_s020_v99_wgs.4127abff791a4466.{json,txt,log}`,
`profiles/submission_s021_v99_route_fused.adfeb8155fcc4663.{json,txt,log}`,
`profiles/submission_s022_v99_route_broadcast.39d932ee11f14b82.{json,txt,log}`,
and `profiles/submission.0200db73e80c4907.{json,txt,log}`.

## S023--S026: standalone attention v81 port (invalidated by input-rank bug)

These candidates embedded the standalone attention v81 q_len=8 kernel and
progressively narrowed its routing while investigating illegal-memory failures.
S023 job `e0d8ab953997467a` failed at verify-1 sliding.  S024 job
`7276db5905b34d5b` excluded that route and failed at verify-1 full.  S025 job
`e5b6fe9269874d9c` routed only B16/B32 and failed at verify-16 sliding.  S026
job `228f818dd21d4ae9` renamed the extension's exported `run` symbol to
`decode_v81_run`, proving the failure was not ELF/PyBind symbol interposition;
it again failed at verify-16 sliding.  All four jobs passed the 13 correctness
cases before reaching their first routed scored verify shape, and none was
scored.

Subsequent static tracing found that every port passed the decoder projection's
2-D `[T, 2304]` result directly to a launcher written for standalone attention's
3-D `[B, L, 2304]` tensor.  The C++ launcher computes
`toks = qkv.size(0) * qkv.size(1)` and `L = toks / B`; at B16 verify this
therefore became `toks=294912, L=18432` instead of `toks=128, L=8`, selected
the wrong preparation path, and launched out-of-bounds work.  The sequence of
failures follows the routing changes exactly.  These runs did not test the
intended v81 attention kernel and must not be used as evidence against it.

S029, `submission_s029_v81_qkv_shape_fix.py`, changes only the call-site view to
`qkv.view(batch, length, -1)`.  Its SHA-256 is
`06ab5a52fff0a61b1f232553ff2ea2473177918517bd5c4811e4244819c6cacd`.
After the remote service recovered, off-board H200 job `41a87438c2ca46f7`
passed all 13 cases and all routed scored shapes with no flags or rejections at
1.828874x / 12.36843 ms.  Normalized paired-ratio gains versus S022 were:

| Activated load | S029 change |
|---|---:|
| verify-16 sliding | +2.83% |
| verify-16 full | +21.19% |
| verify-32 sliding | +7.12% |
| verify-32 full | +34.21% |

The full-attention kernel fell from about 192.6 to 121.3 us at B16 and from
373.3 to 230.8 us at B32.  The shape fix therefore validated the original v81
hypothesis; S023--S026 were launcher-integration failures, not kernel failures.
Profile:
`profiles/submission_s029_v81_qkv_shape_fix.41a87438c2ca46f7.{json,txt,log}`.

## S027--S028: one-warp small-token MoE tiles

S027 ported the standalone MoE v59 one-warp-per-output-tile reduction bypass
to the decoder's M8, M16, and M32 `gk` routes.  Unscored job
`d0e40509baef4f32` passed all 13 cases with no flags at 1.753139x /
12.90274 ms, slower than S022's unscored 1.767656x / 12.79678 ms.  Normalized
paired-ratio changes versus S022 were:

| Load | S027 change |
|---|---:|
| decode-16 sliding | -2.06% |
| decode-16 full | -0.91% |
| verify-1 sliding | -1.50% |
| verify-1 full | -1.79% |
| decode-32 sliding | +1.59% |
| decode-32 full | +0.58% |

Reject S027: the standalone M8/M16 improvement did not transfer to the mixed
decoder graph, while M32 remained directionally promising.  Profile:
`profiles/submission_s027_moe_gk_warp_tiles.d0e40509baef4f32.{json,txt,log}`.

S028, `submission_s028_moe_gk_m32_warp_tile.py`, restores S022's exact M8/M16
dispatch and retains the one-warp change only for M32.  Its SHA-256 is
`c84c7cc626fdea20f111a05cde6ecb073731bfef171eeb7dbdc5c45a2b0bde0c`.
The first `okc check` call ran for about 14 minutes and returned no job ID or
output while WSL/API calls became unresponsive; no S028 profile appeared.
The recovered remote ledger confirmed that no S028 job had been created and
that the new day's decoder quota was untouched.

The safe retry, off-board H200 job `b868d38f2b9f4e2a`, passed all 13 cases with
no flags or rejections at 1.771073x / 12.77209 ms.  Versus S022, normalized
decode-32 gains were +1.07% sliding and +0.72% full; S027's independent seed had
already shown +1.59% and +0.58% on the same M32 implementation.  The sliding
M32 `moe_k` profile dropped from about 109.0 to 97.9 us.  Accept the narrow M32
change for merging, while rejecting S027's M8/M16 branches.  Profile:
`profiles/submission_s028_moe_gk_m32_warp_tile.b868d38f2b9f4e2a.{json,txt,log}`.

## S030: v81 verify attention plus M32 one-warp MoE (promoted)

S030 mechanically combines the two disjoint validated changes.  The v81 block
is byte-identical to S029, and the M32 macro and slice regions are byte-identical
to S028.  `scripts/build_s030_v81_m32.py` hash-guards S029 and applies only the
six audited M32 substitutions.  The resulting 981,968-byte candidate has
SHA-256
`4b40bd21366077da9a3d969b5ca876fc67db308767fad60e7bc535cab111b2b6`.

Off-board H200 job `7c71713153194193` passed all 13 cases with no flags or
rejections at 1.823670x / 12.40372 ms.  Its normalized changes versus S022 on
the only touched shapes were:

| Touched load | S030 change |
|---|---:|
| decode-32 sliding | +2.71% |
| decode-32 full | +0.32% |
| verify-16 sliding | +1.79% |
| verify-16 full | +20.06% |
| verify-32 sliding | +2.61% |
| verify-32 full | +28.35% |

The raw aggregate was 0.29% behind S029's independent seed, inside ordinary
untouched-shape variation; every changed branch remained positive versus S022.
`scripts/promote_s030.py` then copied the exact checked bytes to `submission.py`
under old/new hash guards.

Official scored job `97495dd79dc64ea0` passed 13/13 with no flags or rejections
at 1.826902x / 12.38178 ms (95% CI 1.799498--1.854305).  Compared with scored
S022 at 1.761846x / 12.83898 ms, aggregate latency improved 3.56% and speedup
improved 3.69%.  The scored normalized changes on the touched branches were
+1.60%, +0.11%, +5.88%, +20.27%, +2.33%, and +30.15% in the table's order
above.  Promote S030.  Active `submission.py` remains exactly equal to
`submission_s030_v81_m32.py` at SHA-256
`4b40bd21366077da9a3d969b5ca876fc67db308767fad60e7bc535cab111b2b6`.

The indexed leaderboard row after promotion was rank 422/1094 at 1.8269x; the
leader was 2.4178x.  Profiles:
`profiles/submission_s030_v81_m32.7c71713153194193.{json,txt,log}` and
`profiles/submission.97495dd79dc64ea0.{json,txt,log}`.

## S031: v81 verify attention at batch 1 (accepted, superseded)

S031 extended the validated v81 `q_len=8` attention route from batches 16/32
to batch 1 without changing any kernel source.  Off-board H200 job
`a41c52f9590948ad` passed 13/13 with no flags at 1.860336x / 12.15925 ms.
The two newly routed verify-1 shapes improved by about 8.2% after normalizing
to their paired SGLang baselines, corresponding to about a 1.3% whole-problem
gain.  Candidate SHA-256:
`f5518e5bb7b78321da571d7479aa931db654af1f468ac01d6196cb6d581530ae`.
Profile:
`profiles/submission_s031_v81_b1.a41c52f9590948ad.{json,txt,log}`.

## S033b: protected dense-v87 composition (correct, not promoted)

The first S033 composition accidentally replaced the attention prefix as well
as the dense backend and failed setup.  S033b fixed the source boundaries,
preserved E003 plus v81, embedded exact standalone dense-v87, and restored the
decoder tail's implicit `hashlib` dependency.  Job `5e58f74028a143a4`
passed 13/13 with no flags at 1.871838x / 12.08253 ms.  This is only 0.63%
lower latency than S031 and lies inside cross-run variation, so do not promote
it over the much stronger S040.  Final S033b SHA-256:
`44d1b38740476dc3f4e60647d3fe8861911c9b8dd1ae4b75fb3f03c58b640cc4`.
Profile:
`profiles/submission_s033b_v81_b1_dense_v87.5e58f74028a143a4.{json,txt,log}`.

## S034--S036: multi-action near-tie route shaping (rejected)

These candidates expanded S014's two-launch route shaper to spend more of the
published local-route and distinct-expert allowances.  S034 job
`c8bbc55e843d4ba0` passed the 13 initial tests but failed a timed recheck after
dropped expert 42 beat its ninth-best score by 0.0383.  S035 job
`443a5fed4cdc40e3` failed the 1M-context verification test after dropped expert
104 beat ninth by 0.0307.  The selector's widened 0.029 approximate-FP32 margin
was not safe against the judge's double-precision reference at the 0.03 bound.
The original 0.010 margin was safe, but S014 already established that the
selector/planner's 6--10 us net launch overhead is larger than its savings.
Reject S034--S036 and do not submit the maximum-budget S036 candidate.

## S038: current standalone attention module swap (integration incomplete)

S038 replaced the older plaintext E003 decode attention with the current
standalone attention module in an isolated namespace while retaining S031's
v81 verify path.  Job `28995cb2618647fe` stopped during setup because removing
E003 also removed the decoder-owned `_ROPE` cache.  The builder now restores
that cache, but no second check was spent after S040 decisively crossed the
target.  This setup failure is not performance evidence for or against the
new attention module.

## S039: WGS final-normalization broadcast (rejected)

S039 applied only v124's finalizer delta to the validated embedded v99 WGS
source: eight CTA leaders form the route normalization factors and broadcast
them through the existing shared reduction row, instead of all 512 threads
reloading top-k weights and repeating `rsqrtf`.  Job `e4fb0fd4804d4da1`
passed 13/13 with no flags, but regressed to 1.853087x / 12.20682 ms.  Reject
the delta; candidate SHA-256:
`efddae58a9709f39513f89410118677a5150f912b3f433ebfde6d4bf749e35f5`.
Profile:
`profiles/submission_s039_v124_norm_broadcast.e4fb0fd4804d4da1.{json,txt,log}`.

## S040: pinned public vjeux whole-layer entry (promoted, crossed 2.0x)

S040 adopts team vjeux's complete public `muse_decode_layer.py` at commit
`f35a41ee1c13e316dabbf772830d47b23b453393`, published under Apache-2.0 at
<https://github.com/vjeux/okc-2026>.  The candidate retains the decoder problem
marker and adds explicit source, commit, team, and license attribution.  Its
real speedups include tuned native dense backends, newer fused attention,
programmatic dependent launches, tuned boundary/routing glue, and the team's
fused NVFP4 expert backend; it does not alter the harness or timing.

Off-board job `910020090f75419b` passed 13/13 with no flags at 2.085872x /
10.84453 ms (95% CI 2.054584--2.117160x).  The exact checked candidate was
promoted under old/new hash guards.  Official scored job `111b9fe7605745d9`
then reproduced 13/13 and 2.086125x / 10.84321 ms with no flags.  Relative to
scored S030, latency fell 12.43% (12.38178 -> 10.84321 ms) and speedup rose
14.19% (1.826902x -> 2.086125x).  The indexed board moved from rank 422/1094
to 291/1095.

Active `submission.py` and S040 are byte-identical at SHA-256
`e4ccd1407b704fc9605dd882750c313bf6c68287d8e5da14fbeba2300074bd43`.
Profiles:
`profiles/submission_s040_vjeux_f35a41ee.910020090f75419b.{json,txt,log}` and
`profiles/submission.111b9fe7605745d9.{json,txt,log}`.

## S041: pinned public alygg whole-layer entry (promoted, 2.218x check)

S041 adopts alygg77's public Apache-2.0 whole-layer entry at exact commit
`6cae49a639ee6d14b929a5fd4a63c627f96d7dc0` from
<https://github.com/alygg77/okc-muse-spark-kernels>.  The exact source is
`kernels/muse_decode_layer.py`; its published whole-layer result is 2.2058x.
Relative to S040, the material additions are an exact FP32 router fused into
the small-token pooler, a KPERM3 expert-weight relayout for the WGS path, and
deepcopy-safe cuBLASLt holders plus per-thread CUDA-context binding.

Downloaded candidate `submission_s041_alygg_6cae49a6.py` is byte-identical to
the pinned public source at 839,695 bytes and SHA-256
`8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.
Off-board H200 job `62521b8e52b649d2` passed all 13 cases with no flags at
2.218x / 10.20 ms (95% CI 2.185--2.252x), slightly above the public result.
The exact checked bytes were promoted to `submission.py`.  Official scored job
`6ccb0f1a4bc74167` then passed 13/13 at 2.2076x / 10.25 ms (95% CI
2.174--2.241x).  Relative to S040's scored 2.086125x / 10.84321 ms, S041
improves speedup by 5.82% and reduces aggregate latency by 5.47%.  The live
board moved from rank 291/1095 to 145/1099; the first 2.300x row was rank 29
at this snapshot.

The check profile confirms that the remaining 2.3x gap is concentrated in
decode-16, decode-32, and verify-16.  Decode-32 sliding spends 101.8 us (56.9%)
in `moe_fused`; verify-16 sliding spends 154.2 us (62.0%) there.  Public H200
measurements also show only about 2 us/layer of dense-GEMV headroom.  The next
credible step is therefore a different small/ragged NVFP4 expert schedule,
not another dense micro-tune.  Profile:
`profiles/submission_s041_alygg_6cae49a6.62521b8e52b649d2.{json,txt,log}`.
Scored profile: `profiles/submission.6ccb0f1a4bc74167.{json,txt,log}`.  Active
`submission.py` remains byte-identical to S041 at SHA-256
`8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.

## S042--S051: original split-8 WGS1 experiment (not promoted)

This series tests an original small/ragged MoE schedule for exact `T=32`:
split every expert into at most eight-token jobs, use one 128-thread WGMMA
consumer group plus two producer warps, process 64 output rows per unit, and
reuse one 9-KB quarter of each existing packed 256-row weight stage.  The
smaller rings target multiple resident CTAs instead of S041's four consumer
warpgroups in one large CTA.  `JMAX=64` is required because 256 routes across
32 experts can produce as many as 60 split jobs.

The first segmented finalizer was incorrect.  S043 check
`7e3d339214db4838` (two CTAs) and S046 check `3561fff042644673`
(one CTA) both failed only the exact-32 cases 9 and 11 with roughly 0.58--0.80
residual relative L2.  This ruled out the job bound and multi-CTA residency as
the primary cause.  S042 `ab88165e620d4dfa`, S044
`d5529a2f0e844dd2`, and S045 `93f2cece0bb647a7` timed out.

S047 `4af916ae600b4aba` formed route norms atomically in the phase-B epilogue
and used the generic finalizer.  It survived the early correctness-failure
window but timed out after 15 minutes with no records.  S048
`0d45a6b1add6420d` restored a full-row finalizer with compile pruning and a
two-CTA grid; it also timed out.  These diagnostics motivated a compact
fixed-partial finalizer: each phase-B 16-column tile writes one deterministic
sum-of-squares partial, then four warps reduce the 256 partials per route in
two rounds while producing disjoint 1024-column output segments.

S049 scored job `acff26c740704a7b` exposed a setup-only layout error: compact
WGS1 `EMAX=256` left no 64-row region for the partial buffer.  S050 reserves
exactly those rows (`EMAX=320`), keeping live y rows 0--255, partial rows
256--319, and stream-K fixups after row 320.  Official scored job
`b39a637c87fb49a5` passed all 13 cases with no flags at **2.123029x /
10.65473 ms** (95% CI 2.091184--2.154875x).  Its decode-32 sliding MoE was
91.5 us, but the whole shape regressed to 205.41 us versus S041's 145.64 us;
aggregate performance therefore remained below S041's 2.207570x.

S051 changed only occupancy to three CTAs per SM.  Scored job
`3254410bdb4c4246` again failed cases 9 and 11 (0.730 and 0.473 residual
relative L2), proving that the present stream-K/fixup synchronization is not
safe at three-CTA residency.  No S042--S051 candidate was promoted.  Active
`submission.py` remains the checked/scored S041 source at SHA-256
`8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.

## S052: large-batch compact norm-partial finalizer (promoted)

S052 returns to exact S041 and extends its already-correct B4 finalizer from
the tiny `16 < M <= 32` build to the scored WGS `M=128` and `M=256` builds.
Every phase-B 16-column tile stores one fixed-order FP32 sum-of-squares
partial.  The final pass reads 256 compact partials per routed row instead of
re-reading all 4096 FP32 activations to form the norm, then performs the
unavoidable activation read only once while producing output.  The large
build reserves the last 128 y-buffer rows, exactly enough for 2048 routed rows
at `M=256`, and its runtime guard prevents overlap with live routed rows.

Official scored job `6828c406e66940ed` passed all 13 cases with no flags or
rejections at **2.213594x / 10.218812 ms** (95% CI
2.180390--2.246798x).  Against S041's 2.207570x / 10.246699 ms, latency fell
0.27% and speedup rose 0.27%.  Exact matched-seed profiles show the intended
kernel effect: verify-16 sliding MoE fell 154.7 -> 148.8 us, verify-16 full
rose 152.6 -> 158.5 us, verify-32 sliding fell 206.3 -> 205.5 us, and
verify-32 full fell 199.3 -> 179.9 us.  The strong M256/full gain outweighed
the M128/full regression in the official aggregate.

Profile:
`profiles/submission_s052_large_b4_partial_norm.6828c406e66940ed.{json,txt,log}`.
The exact scored candidate was promoted to `submission.py`; both are SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.
Three scored whole-decoder submissions remain today; unscored checks remain
exhausted.

## S053--S054: layer-aware M128 B4 dispatch (rejected)

S053 used the S052 matched-seed profile to keep large B4 at `M=128` only on
sliding layers while retaining it at `M=256` on both layer types.  Scored job
`d265d2a1948743af` failed during setup before correctness: the prepare-time
autotuner invokes the final fused path with synthetic data that has no
`config`, and the new dispatch read `data["config"]["window"]` directly.

S054 changed only those five call sites to use a setup-safe static lookup,
`data.get("config", {}).get("window", 0)`.  Official scored job
`25d4032752af4021` passed all 13 cases with no flags or rejections, but scored
only **2.204859x / 10.259294 ms** (95% CI 2.171786--2.237932x), 0.40% slower
than S052.  The M128 layer split inferred from one fixed routed-expert seed did
not generalize to the next official seed distribution.  Reject both
candidates and do not spend the last scored slot on another finalizer
micro-variant.  S052 remains active at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.
Profiles:
`profiles/submission_s054_layer_aware_large_b4_setup_safe.25d4032752af4021.{json,txt,log}`.

## S055: full-row large-B4 finalizer (correct, not promoted)

S055 keeps S052's fixed-order norm partials but changes only the large
`M=128/256` final pass.  Instead of four 1024-column CTA items per token, one
512-thread CTA writes the complete 4096-column row.  Large batches already
provide 128 or 256 token CTAs for 132 H200 SMs, so this reduces the finalizer
from eight CTA barriers per token to two while retaining the segmented path
for tiny `M <= 32`.

Official scored job `ddfd87a63c364fc7` passed all 13 cases with no flags or
rejections at **2.205367x / 10.256933 ms** (95% CI
2.172286--2.238447x), 0.37% slower than S052's official aggregate.  The
touched profile rows were nevertheless encouraging: verify-16 sliding/full
MoE measured 142.6/159.4 us and verify-32 sliding/full measured
198.4/175.7 us, versus 148.8/158.5 and 205.5/179.9 us in S052's prior
profile.  Since the jobs use different routed-expert seeds and the aggregate
regressed, this is insufficient evidence to promote; preserve S055 for a
fresh rerun when the daily budget resets.

Profile:
`profiles/submission_s055_bigb4_fullrow_finalizer.ddfd87a63c364fc7.{json,txt,log}`.
All 10 scored and 20 unscored whole-decoder runs are now exhausted.  Active
`submission.py` remains exact S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

Fresh off-board check `448d9453ca374a4c` after the daily reset also passed
13/13 but scored only **2.210147x / 10.234751 ms** (95% CI
2.176994--2.243299x).  Its large sliding MoE rows were 169.7 us at M=128 and
187.6 us at M=256, rather than the prior 142.6/198.4 us.  The full-row
finalizer therefore has no reproducible aggregate advantage; reject S055.
Profile:
`profiles/submission_s055_bigb4_fullrow_finalizer.448d9453ca374a4c.{json,txt,log}`.

## S056: joined WGS weight/activation completion (rejected)

S056 compared S052's KPERM3 WGS source with the proven standalone MoE
scheduler.  The cost tables and scale decode were already current.  The one
missing low-risk scheduler feature joined each stage's weight and activation
TMA transactions onto one mbarrier, changed the full-barrier arrival count
from one to two, and removed the second dependent consumer wait.  It affects
only the main M=32/128/256 WGS builds; small-token sources, routing, math, and
B4 finalizers are unchanged.

Check `56c6c9d8a90b4abd` passed 13/13 at **2.198839x / 10.287382 ms**.  Its
profile initially looked directional (decode-32 MoE 89.2 -> 70.5 us,
verify-16 148.8 -> 144.9 us, verify-32 205.5 -> 198.1 us), but unchanged rows
regressed enough to lose the aggregate.  The service deduplicated an exact
rerun, so a semantically identical file with only a trailing Python comment
was checked independently as `c736d127d018443d`; it scored **2.211x /
10.23 ms**, still below S052, while the same MoE rows moved to 90.0, 173.2,
and 206.3 us.  The first profile's apparent gain was not reproducible.  Reject
S056 and do not score it.

Profiles:
`profiles/submission_s056_joined_wgs_completion.56c6c9d8a90b4abd.{json,txt,log}`
and
`profiles/submission_s056r_joined_wgs_completion.c736d127d018443d.{json,txt,log}`.

## S057: one-tile dual WGMMA accumulators (rejected)

S057 completed the other material WGS-core delta from the standalone
production scheduler.  On top of S056, NTT=1 jobs alternate WGMMA commit
groups between two accumulator banks and combine four FP32 values once after
the drain; NTT>1 jobs retain the old path.  This targets the common one-tile
local expert jobs without changing routing or numerical order within either
bank.

Check `7d57e494a29a477f` passed 13/13 but scored only **2.195x / 10.31 ms**
(95% CI 2.162--2.228x).  Decode-32 rose to 155.0 us.  The extra dependency
chain/register state does not transfer cleanly to KPERM3 in the full layer;
reject S057.

The standalone v123 cached-final-row optimization is not applicable to the
remaining scored path.  S052 dispatches every scored WGS shape (M=32/128/256)
to B4FIN, which already writes fixed norm partials during phase B and reads
each y activation only once in the final pass.  The cached two-pass branch is
dead for these workloads, so no check was spent on it.

Active `submission.py` remains byte-identical to S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## S058--S060: M=32 FT32 WGS schedule (rejected)

Standalone v127/v128 diagnostics made `FT_T=32` the strongest remaining
small/ragged schedule signal at M=32: it won 8/9 seeds with a median candidate
to-main time ratio of 0.99167.  Two production standalone checks also put the
M=32 case at 122.97/122.89 us versus v101's 124.22 us.  The signal was below
the preferred 1% gate, but it was sufficiently consistent to justify one
shape-isolated decoder port.  M=128/256 and all small-token branches remained
on S052.

S058 made `OKC_FT_T` overridable and dispatched only the M=32 B4FIN extension
with `-DOKC_FT_T=32`.  Check `9a31b544e73b4c60` failed during setup: `ptxas`
could not allocate the WGS kernel under its 96-register target.  S059 compiled
out the unused `moe_fused<true>` template, but check `d24a560b29264596` showed
the same failure in the executed `moe_fused<false>` template.

The root cause was an incomplete port, not insufficient scratch.  The newer
standalone source wraps `WGSMMA<2..4>` in `#if OKC_FT_T > 8` and
`WGSMMA<5..8>` in `#if OKC_FT_T > 32`.  Without those guards, the decoder was
instantiating impossible 40--64-token variants against a four-tile FT32
accumulator and exhausting the register target.  S060 added the exact guards
and kept the extension explicitly M=32/small-only.

Check `959f3cec1f0c4adb` then compiled, passed all 13 cases, and scored
**2.198446x / 10.289223 ms** (95% CI 2.165469--2.231423x), 0.689% slower in
score time than S052.  The only affected layer cases were decisive despite
unrelated run-to-run seed noise:

| Affected case | S052 us | S060 us | Change | Weight |
|---|---:|---:|---:|---:|
| decode 32 sliding | 150.348 | 152.129 | +1.18% | 46 |
| decode 32 full | 301.010 | 300.554 | -0.15% | 16 |

The high-weight sliding regression dominates the negligible full-context
gain.  Reject S060, do not score it, and close the FT32 branch.  Profile:
`profiles/submission_s060_m32_ft32_guarded.959f3cec1f0c4adb.{json,txt,log}`.

Active `submission.py` remains exact S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## S061: T=128/256 PDL boundary-vector preload (rejected)

S061 targeted the only clearly removable non-roofline rows shared by the four
large verify workloads: the two `_graft_row_boundary` launches, totaling
9.0--9.7 us at T=128 and 13.6 us at T=256 in S052.  It reused the active
T=32 `_graft_row_boundary_pre` implementation at T=128/256, loading the
immutable pre-norm, residual-gate and optional post-norm vectors before the PDL
wait.  All arithmetic, launch geometry, projections, attention, routing and MoE
code stayed byte-identical to S052.

Check `fcec12ed00014d24` compiled, passed all 13 cases with no flags, and scored
**2.194734x / 10.306626 ms** (95% CI 2.161813--2.228715x), below S052.  The
targeted component result is decisive despite route-dependent MoE variance:

| Touched case | S052 boundary pair | S061 boundary pair | Change |
|---|---:|---:|---:|
| verify 16 sliding | 9.7 us | 10.2 us | +5.2% |
| verify 16 full | 9.0 us | 10.1 us | +12.2% |
| verify 32 sliding | 13.6 us | 17.2 us | +26.5% |
| verify 32 full | 13.6 us | 17.2 us | +26.5% |

The extra live vectors do not disappear behind the producer.  They increase
register pressure and/or compete with the O/unpool producer for memory service,
with the cost growing at the 16-warp T=256 geometry.  Reject S061 and do not
extend pre-wait boundary loading beyond T=32.  A future boundary optimization
must eliminate work in a producer epilogue rather than carry more state across
PDL.  Profile:
`profiles/submission_s061_large_boundary_preload.fcec12ed00014d24.{json,txt,log}`.

Candidate SHA-256:
`24809d6949ccb3fa6f3e72d4512528824fb1052f5633a7637bab785e193887ae`.
Active `submission.py` remains exact S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## S062: standalone v139 local KPERM3 port (static rejection)

S062 tested the user's requested newer expert re-layout on the full decoder's
scored WGS band (M=32/128/256).  This is not S052's existing stage-wide
KPERM3: it performs a lossless 4x4 transpose inside each 64-wide K group,
`old K = block*16 + q*8 + lane2*2 + e` to
`new K = lane2*16 + q*8 + block*2 + e`.  One consumer lane then holds one
original K16 scale pair across four WGMMA steps.  The activation gather,
packed W13/W2 layouts, and phase-A intermediate stores use the matched
permutation.

The hash-guarded builder left active S052 unchanged.  An independent exhaustive
4096-column audit proved the mapping is bijective and involutive and matched
every CUDA gather source/destination pair.  Python syntax passed.  Exact CUDA
12.4 SM90a builds of the main, M32 B4, and M128/256 BIGB4 modules all retained
96 registers, all eight HGMMA widths, 128 static sites per width, and no WGMMA
serialization warning.

The resource comparison rejects the port before remote measurement:

| exact main WGS cubin | S052 | S062 local KPERM3 |
|---|---:|---:|
| registers | 96 | 96 |
| stack | 8 B | 16 B |
| spill stores / loads | 8 / 28 B | 44 / 104 B |
| PRMT sites | 326 | 326 |
| IMAD sites | 6,812 | 6,954 |
| LDG sites | 586 | 586 |

The standalone v139 layout had improved an older standalone control, but S052's
stage-wide permutation had already eliminated the same PRMT work with much less
address and live-state cost.  Therefore the local transpose offers no missing
full-layer mechanism and is statically worse on the exact target compilation.
No H200 check or scored run was spent.  Preserve the candidate as negative
evidence and do not promote or remotely test it.

Candidate: `submission_s062_kperm3_local.py`, SHA-256
`934e3f377be8a9a3f02118440fae8e21702bf442733828d9c85d0002f1d50cb1`.
Active `submission.py` remains exact S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## S063--S070: removing the large-B4 pre-final barrier (rejected)

S063 implemented the static-token version suggested by the S019 postmortem:
phase-B jobs published per-token readiness, each existing fixed-owner finalizer
waited only for its token's local routes, and a last-CTA cleanup replaced the
pre-final grid barrier.  It was limited to M128/M256 BIGB4.  The protocol and
scratch arithmetic passed static simulation, but exact SM90a code generation
made it non-viable: the primary gained a 40-byte stack, 152-byte spill stores,
288-byte spill loads and about 2,850 IMAD sites.  S064 deferred whole-unit
publication into the common WGS flush; it reduced spills to 64/208 bytes but
still carried a 32-byte stack and similar instruction expansion.  Neither used
remote quota.

S065 changed architecture: the WGS primary exits after phase B and a separate
512-thread kernel performs the identical BIGB4 fixed-order norm/reduction,
cleanup and generation bump.  Grid completion supplies synchronization, so no
readiness polling is needed.  The finalizer compiled at 32 registers with zero
stack/spills, but removing the old post-phase body caused a ptxas allocation
cliff in the primary.  S067 retained that body behind `p.zero == 0` (host-pinned
and opaque to ptxas), restoring the exact S052 primary shape: 96 registers,
zero stack/spills and identical HGMMA width/counts, while the executed path
returns to the split finalizer.

The first two H200 checks revealed storage lifetime errors rather than kernel
math errors.  S067 (`c1ec09b718b346fe`) wrote its route-position map before
the normal generation clear and failed at M128.  S069
(`0bd9233bb45f4ee2`) skipped that clear and made the large path correct, but
used `ss1`; the next small-token call interpreted the retained route bits as
norm state and produced a test-0 NaN.  S070 reserved 256 extra floats at the
end of WGS scratch.  Together with the existing 768-int gap after exact JMAX
`doneAs`, this provides exactly 1,024 ints / 2,048 shorts for the M256 route
map, without aliasing any generation buffer or other token-count extension.

S070 check `58d21cda48e844b2` passed all 13 cases with no flags.  Exact compiler
resources remained 96 registers and zero spills for the primary, 32 registers
and zero spills for the finalizer.  Performance rejected the architecture:

| touched MoE path | S052 single kernel | S070 primary | S070 finalizer | S070 combined |
|---|---:|---:|---:|---:|
| M128 sliding | 148.8 us | 138.6 us | 7.1 us | 145.7 us |
| M128 full | 158.5 us | 156.6 us | 7.1 us | 163.7 us |
| M256 sliding | 205.5 us | 190.8 us | 13.8 us | 204.6 us |
| M256 full | 179.9 us | 189.7 us | 13.0 us | 202.7 us |

The aggregate was **2.204637x / 10.260327 ms** (95% CI
2.171568--2.237707x), versus active S052 at **2.213594x / 10.218812 ms**.
The separate launch can expose a small sliding gain, but it does not generalize
across route distributions and full-context rows.  Reject S070, spend no scored
run, and close final-barrier/finalizer rearrangements as a source of a robust
weighted gain.  Candidate SHA-256:
`582b374e72bff9c297f2ad591c8c5f5c77b5159f428a5a63d2a853a2e26ccf18`.
Profile:
`profiles/submission_s070_split_bigb4_pos_tail.58d21cda48e844b2.{json,txt,log}`.

Active `submission.py` remains exact S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## S071--S072: packed local route IDs (correct, rejected)

S071 targeted the repeated route-ingress work in the scored WGS band.  The
upstream routing kernels already produce rank-local IDs in `[-1, 31]`, but the
decoder stored them as int32 and every WGS CTA reloaded, subtracted and
revalidated them before copying them to its shared signed-byte route map.
S071 stored IDs as int8 for the main WGS dispatch (`T >= 17`), loaded four IDs
at a time, and copied the packed bytes directly to shared memory.  The tiny and
mid builds remain int32, and the Marlin fallback remains int32.

The first check, `cd0cc7c64fcb4994`, exposed an exact dispatch-boundary bug:
the B4 main build starts at `T=17`, while the initial Python allocation changed
dtype only at `T=32`.  The hidden `T=24` case therefore passed an int32 tensor
to a signed-byte pointer and failed residual correctness.  Matching the dtype
boundary to the existing `16 < M <= 32` B4 dispatch fixed the issue.  Corrected
check `49fa750be56349af` passed all 13 cases with no flags.

Exact SM90a compilation retained 96 registers, zero stack/spills and every
HGMMA width/count.  Versus S052 it removed 12 PRMT and 10 IMAD sites with the
same LDG/LDS/STS/barrier counts.  The full-layer result was nevertheless only
**2.201971x / 10.272753 ms**, below S052's **2.213594x / 10.218812 ms**.  The
six touched raw medians improved by 0.23--2.43%, but paired-speedup changes
were mixed: +1.48%, +2.46%, -1.47%, +0.28%, -2.24%, and +2.19% for M32
sliding/full, M128 sliding/full, and M256 sliding/full.  That is a possible
sub-percent component signal, not a reproducible layer win.

S072 removed the shared route-ID copy and its following CTA barrier, reading
the byte IDs directly at both route passes.  Exact SASS removed 18 IMAD, 26
LDS, four STS and two static BAR sites across the two template instantiations,
but added 16 LDG sites.  Check `7dd18d66c85c4a84` passed 13/13 with no flags
and regressed further to **2.188391x / 10.336496 ms**.  The extra route reloads
cost more than the saved barrier.  Reject both variants and close packed route
metadata as a below-noise direction.

Candidates and SHA-256 values:

- `submission_s071_packed_lids.py`:
  `beeff3e2bd8d1c1ce4f6d3dc08b145e49ad5e4e56be44e85d03250ad257c3747`
- `submission_s072_packed_lids_direct.py`:
  `01b5c9e4818787d5bdf891aa4cec35a72b57d9b0ed340c76d11356ba5acac949`

Profiles:
`profiles/submission_s071_packed_lids.49fa750be56349af.{json,txt,log}` and
`profiles/submission_s072_packed_lids_direct.7dd18d66c85c4a84.{json,txt,log}`.
Active `submission.py` remains exact S052 at SHA-256
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## 2026-11-03 resume checkpoint: S080 in flight

S080 is the first structural pooler/router integration after the S078/S079
controls.  `submission_s080_rs128_pr_fused.py` (SHA-256
`9201d106ab2a6be10b0ee9fbade8b3d78fef0495910543809fce2f680585be93`)
uses one KS=4 RS128/PDL projection for the existing 4608-row pooler plus
router-high/router-low matrix at T=128.  It is guarded by exact pack and
projection validation plus three 128-row ordered top-8 agreement probes; a
failed guard falls back to S077.

Scored job `b63134e6813a4519` was still running when work paused.  Resume with
`../bin/okc status b63134e6813a4519 --json`, then inspect the candidate's log
for `s080 combined pr ... enabled True` before interpreting its performance.
One decoder scored run and zero checks remained after this submission.  The
protected active file is still exact S052 at
`63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2`.

## Per-run checklist

- [ ] `submission.py` first line retains the problem marker.
- [ ] Syntax/import review completed.
- [ ] No input-derived state cached across calls.
- [ ] CUDA graph path contains no host synchronization.
- [ ] All 13 correctness cases pass.
- [ ] New KV entries pass page-size and scale tests.
- [ ] Routing IDs are unique, in range, and preserve required local routes.
- [ ] Profile inspected per shape and per kernel.
- [ ] Regression compared against the immediately previous candidate.
- [ ] Candidate saved or committed with its profile artifacts.

## 2026-11-03 continuation: S081--S092

S081 promoted the current public keell0renz decoder.  Scored job
`725c19cbeb9b4d48` passed 13/13 at **2.2611966426x / 10.003685 ms**
(95% CI 2.227279--2.295115x).  The protected incumbent is
`submission.py`, SHA-256
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
At the incumbent baseline, 2.3x requires about 9.835 ms (168.7 us saved),
and 2.4x requires about 9.424 ms (579.5 us saved).

The following integration branches were rejected without changing the
incumbent:

- S083 changed attention's L2 policy to 64 B and regressed.
- S084 combined the positive-looking q_len=1 attention and RS128 pooler
  paths, but failed routing correctness on a replay case.
- S085 isolated q_len=1 occ3 and passed 13/13 at about 2.254x / 10.04 ms;
  the standalone attention win did not transfer.
- S086 and S088 tested two guarded RS128 pooler variants.  Both failed a
  routing replay case, closing that integration family.
- S087 enabled the WST B16 full path.  Its check was near neutral and its
  scored run was **2.253x / 10.04 ms**; the touched target was slower than
  S081.  Reject.
- S089 packed the M32 local route IDs.  Check `8e14d65e31164f87` passed
  13/13 at **2.258x / 10.019521 ms**, with both decode-32 shapes slower.
  Close the packed-route family.
- S090 appended the already validated exact-C12 M=1 GEMV paths.  Check
  `eccb9672e2ec40e0` passed at **2.274439x / 9.945443 ms**, but the
  behavior-identical replication `47ee77bdcf39422c` passed at only
  **2.242x / 10.090489 ms**.  The target kernels saved only about 2.3 us
  per shape; the headline change was route/MoE variance.  Reject and do not
  score or promote.

The q_len=8 F2 attention investigation is also closed.  The split producer,
global-atomic merge, and stream-0-only all-tile control all failed 12/16.
The final control, job `0e6370069dc44a73`, had relative L2 error 0.6179 on
the first q8 full case.  Because the one-CTA control fails without stream 1
or a merge, the topology itself is incorrect.  Preserve the final two
attention checks for a genuinely new root cause.

S091 was a bounded MoE follow-up.  It dispatched only M=32 to the exact CUDA
module from the scored standalone MoE incumbent v178; every other token count
fell through to S081.  The first upload was rejected before job creation
because a literal source copy exceeded the 1 MB limit.  The rebuilt candidate
stored an exact checked patch from the decoder's resident MoE source and was
991,890 bytes, SHA-256
`d1c3df4ac8ec4e0f1b982daff6919e0593718b6231aee4f29050a886e9a34048`.
Check `a1d52a69d7de42dd` passed 13/13 with the expected dispatch marker, but
reached only **2.257801x / 10.018730 ms**.  Its decode-32 MoE profile was mixed:
106.7 us sliding and 75.5 us full versus S081's 75.1 us and 85.2 us.  Reject;
the exact standalone M=32 module is not a decoder-level improvement.

S092 narrowed the same exact-source method to M=16, the only token count where
the standalone v178 component measurements had a small repeatable edge over
the decoder's resident v167 helper (about 89.63 us versus 91.70 us).  It changed
only stateful M=16 calls and used checked compact patch reconstruction to remain
under the upload cap: 994,512 bytes, SHA-256
`de8bdadc0bf7526b5fe98450f3c6492d84a85be1dbfa17b5e8a35218f45e46d6`.
Official check `c947bd1c889b4303` passed all 13 cases with no flags at only
**2.238111x / 10.106873 ms** (95% CI 2.204539--2.271682x), below S081's
**2.261197x / 10.003685 ms**.  The targeted M=16 MoE row improved from
73.0 to 68.3 us on decode-16 sliding but was flat at 63.3 versus 63.4 us on
decode-16 full.  A roughly 4.7 us gain on one weighted row is too small to
survive route and whole-layer variance, and the aggregate regressed by 1.03%.
Reject S092 without replication or a scored run.  Profile:
`profiles/submission_s092_v178_m16.c947bd1c889b4303.{json,txt,log}`.

At the user's explicit request, S090 then received one scored attempt. Job
`43aec79e497c41c9` passed all 13 cases with no flags or rejections at
**2.2548257006x / 10.031951 ms** (95% CI 2.221003--2.288648). Both exact-C12
M=1 CGEMV routes enabled and were numerically exact; the profile measured the
QKV/unpooler CGEMVs at about 9.0/3.4 us. The local saving was real but too
small to overcome route-dependent MoE movement elsewhere. The result is below
S081's **2.2611966426x / 10.003685 ms** and below 2.3x. Do not promote or
repeat S090. Profile:
`profiles/submission_s090_s081_cgemv_m1.43aec79e497c41c9.{json,txt,log}`.

After the scored S090 attempt, the CLI reports 3/10 scored runs and 14/20
checks used for the decoder today, leaving seven scored runs and six checks. Component quota is
exhausted for projection and MoE; attention has one check left.  Preserve the
S081 incumbent byte-for-byte at SHA-256
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.

## Current scored shape table

S022 and S030 use different generated seeds, so `Ratio` is the normalized
paired-speedup ratio rather than the quotient of the two displayed medians.

| Shape | S022 us | S030 us | Normalized ratio | Dominant change / note |
|---|---:|---:|---:|---|
| decode 1 sliding | 86.98 | 86.23 | 1.0196x | untouched; seed variation |
| decode 1 full | 96.52 | 97.40 | 1.0142x | untouched; seed variation |
| decode 16 sliding | 150.69 | 155.68 | 0.9653x | untouched; seed variation |
| decode 16 full | 246.08 | 247.92 | 0.9876x | untouched; seed variation |
| decode 32 sliding | 196.05 | 187.73 | 1.0160x | M32 one-warp MoE |
| decode 32 full | 386.09 | 382.49 | 1.0011x | M32 one-warp MoE |
| verify 1 sliding | 144.10 | 147.88 | 1.0011x | untouched; seed variation |
| verify 1 full | 150.89 | 161.18 | 0.9851x | untouched; seed variation |
| verify 16 sliding | 260.79 | 254.45 | 1.0588x | v81 verify attention |
| verify 16 full | 439.64 | 355.09 | 1.2027x | v81 verify attention |
| verify 32 sliding | 336.24 | 331.11 | 1.0233x | v81 verify attention |
| verify 32 full | 692.30 | 530.93 | 1.3015x | v81 verify attention |

## 2026-11-04 continuation: S093--S101

The two launch-removal finalizers from the 2.3x plan were correct but decisively
negative.  S093 moved the T=1 boundary-2 work into the sliced-unpooler last
arriver and passed 13/13 as job `7901c26ce1124f24`, but fell to
**2.1278988041x / 10.630346 ms**.  S094 similarly moved the T=1 route and
boundary-1 work into the fused pooler/router last arriver; job
`ee8e4a7d0ca8454a` passed 13/13 at only
**2.1825069299x / 10.364366 ms**.  The producer register/liveness and tail costs
are much larger than the eliminated graph nodes.  Close both finalizer families.

S097 tested the corrected raw-BF16 K128 GQK projections.  Job
`924d3e23198b4406` passed 13/13 at **2.2556661075x / 10.028213 ms**, but the
prepare-time gates enabled no route: candidate/base ratios were 1.18--2.45x
slower.  The static no-serialization fix is real, but the mechanism is not
competitive on H200 and is excluded from later bundles.

S098 combined exact-C12 T=1 CGEMV, the self-gated GQK scout, and exact-v178
M=16 MoE.  GQK again enabled nothing; C12 and M16 enabled.  Check
`e0cd0d5b3c5a44fa` passed 13/13 at the best unscored center so far:
**2.2731603311x / 9.951036 ms** (95% CI 2.239063--2.307258x).  This is still
116.123 score-us short of the exact 2.3x threshold (9.834913 ms), so S098 was
not promoted over scored S081.

S099 added five standalone BF16 geometry winners and a shape-aware M32/full
MoE route.  Its first job, `82a33c5e36dd4ab0`, failed during prepare because
the shape selector indexed a synthetic tuning record without `config`; no
timing or correctness verdict was produced.  The selector was corrected to
use `data.get("config")`.  S100 then added q_len=1 occupancy-3 and passed 13/13
as job `7efbb621bd894ed5`, but regressed to
**2.2366127216x / 10.113642 ms**.  Kernel profiles show the five small SK2
routes worsen the decoder's T=16 projection chain, and the standalone q1
occupancy win does not transfer to the decoder's soft full-attention inputs.

The deeper audit also closes three tempting repeats:

- the standalone MoE already implemented arithmetic multicast/cluster sharing;
  its measured large-shape regressions were about 10--11%;
- v152--v188 already tested ring depth, producer waits, A-fragment overlap,
  critical-path scheduling, SC6/MFIN combinations, and replications; promoted
  combinations did not beat standalone v178 robustly;
- an experimental attention file sampled V rows after inspecting scores.  The
  task explicitly requires attending to every value, so that branch is invalid
  and must never be integrated or submitted.

S101 is the minimal evidence-backed composition.  It keeps exact-C12 at T=1,
uses exact-v178 MoE only for M=16 sliding and M=32 full, and excludes GQK,
small SK2 routes, q1 occupancy changes, and all invalid sampling.  Candidate:
`submission_s101_shape_core.py`, 992,243 bytes, SHA-256
`cae7e8c12de3b76526f9230460adfb290e8438bacd288009483c0969378a8882`.
The protected S081 `submission.py` remains byte-identical at SHA-256
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
At the user's explicit request, S101 was launched as scored job
`9f570c88fde749e7`.  It passed 13/13 with no flags at **2.2346705191x /
10.122432 ms** (95% CI 2.201150--2.268191), below S081.  The intended routes
enabled, but the M=32/full MoE path measured 92.9 us and did not reproduce its
isolated gain.  Reject S101 without promotion or retry; keep `submission.py`
byte-identical to S081.

After the user explicitly authorized spending scored quota before reset, S098
was submitted as scored job `5fc83e7893814cbd`.  It passed all 13 tests with
no flags or rejections at **2.2563037706x / 10.025379 ms** (95% CI
2.222459--2.290148), below both its earlier 2.273160x check and S081's
2.261197x incumbent.  Reject S098 without promotion or a behavior-identical
retry.  Five of ten decoder scored attempts remain before the daily reset.

The user then explicitly authorized using every remaining scored attempt before
reset.  Five content-distinct candidates were submitted in parallel; all passed
13/13 with no flags or rejections:

- S091 job `c2241520f9b74b7e`: **2.2427136041x / 10.086130 ms**
  (95% CI 2.209073--2.276354);
- S089 job `ef484764ac7b42b9`: **2.2469944678x / 10.066914 ms**
  (95% CI 2.213290--2.280699);
- S085 job `6ea69b33f4d747a3`: **2.2343465391x / 10.123900 ms**
  (95% CI 2.200831--2.267862);
- S092 job `21ab957a3f4f4c1f`: **2.2413773323x / 10.092143 ms**
  (95% CI 2.207757--2.274998);
- corrected S099 job `68d48693e5124a25`: **2.2144032902x /
  10.215077 ms** (95% CI 2.181187--2.247619).

Reject all five without promotion.  Decoder scored quota is now 10/10 used;
no attempt was left to expire.  S081 remains the 2.2611966426x incumbent.
