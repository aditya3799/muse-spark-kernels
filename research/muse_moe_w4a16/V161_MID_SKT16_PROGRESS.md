# v161/v162 M16 SKT16 diagnostic

## Local opportunity and compile gates

Across 1,024 deterministic M16 routing seeds, 348 (33.98%) had at least one local expert with more than eight
routes.  Moving from eight-token to sixteen-token skinny jobs reduced the mean number of local expert-weight passes
from 12.3125 to 11.9482, a 3.30% mean reduction.  The median seed saved no pass; the p90 saved 10%.

`analyze_v161_decode_layout.py` separately proved that the desired eight E2M1 output-bit positions do not admit a
four-translation tiling of the 32-bit packed word.  The best four translations cover only 30/32 bits, so a
rotate-only seven-instruction decoder is impossible.  Frozen v139's multiply decoder already uses the necessary
eight-instruction form.

Exact CUDA 12.4 `sm_90a` compilation from `compile_v161_skinny_static.py` measured:

| build | registers | stack | spills | static HMMA | LDG | IMAD |
|---|---:|---:|---:|---:|---:|---:|
| active MID8 helper-warp path | 145 | 0 B | 0 / 0 B | 64 | 106 | 1,030 |
| generic SKT16 path | 148 | 0 B | 0 / 0 B | 192 | 113 | 1,234 |

SKT16 therefore cleared the resource gate.  Its larger static body contains both one- and two-token-tile paths;
only experts with more than eight routes execute the second tile.  It loses the active helper-warp epilogue, so
runtime evidence is required.

## v161 remote outcome: diagnostic assertion too strict

Frozen v161 SHA256 was `4832D00FCA83D81782528F3B056C2063D80BDB2207CC3BA77F1DE6E19F3AAA51`.
Authorized check job `60cba60d31014d88` compiled both variants but stopped on the first embedded comparison:

- MID8 versus SKT16 relative L2: `2.049e-7`;
- bit-exact equality: false;
- diagnostic assertion: required both exact equality and zero relative L2.

This is an over-strict diagnostic failure, not evidence of numerical invalidity: a different legal reduction
schedule is allowed to round differently, and `2.049e-7` is far below the task tolerance.  The job failed during
prepare, before official correctness cases or paired timings, so it provides no performance verdict.  Do not infer
that SKT16 won or lost.

## v162 prepared correction

v162 changes exactly one line from frozen v161: the embedded parity assertion now requires relative L2 no greater
than `1e-5` instead of bit-exact equality.  CUDA source, official v139 dispatch, 21 seeds, cold-L2 paired timing, and
promotion gates remain unchanged.  `audit_v162_mid_skt16.py` proves that one-line delta and CUDA identity.

Do not run v162 off-board without fresh user authorization.  No scored submission is authorized.  After v161, the
service should have three checks and five scored runs remaining for `muse_moe_w4a16` today.

## v162 remote outcome: diagnostic and performance rejected

Authorized check `d64205beaa584d68` compiled both M16 variants in 33 seconds.  A delayed CLI response caused the
follow-up request `80e50e6bce204efd` to be marked as a duplicate of that canonical job; only one check was charged.
The diagnostic produced:

- seed 1250: relative L2 `2.896e-7`, MID8 `88.2 us`, SKT16 `107.0 us`, ratio `1.21291x`;
- seed 1250 had `jobs8=10`, `jobs16=10`, so SKT16 saved no expert-weight pass;
- seed 2259: relative L2 `9.531e-5`, which exceeded the diagnostic's `1e-5` gate and stopped setup.

The job therefore did not reach the official cases or the remaining paired seeds.  Nevertheless v162 itself is
rejected: its only completed paired timing is a 21.29% regression in a no-savings case, so relaxing the assertion
and rerunning the same unconditional candidate is not justified.  Any successor must first keep MID8 whenever the
route histogram predicts no material pass reduction, then validate the numerical comparison against the task's
actual tolerance before spending another check.  No scored run was used.  The service reports two checks and five
scored runs remaining today.

## v163 profitability-gate model: host-rejected

The task contract forbids a host branch on input route values during CUDA-graph replay, so a legal successor would
need both MID8 and SKT16 device paths in one compiled kernel and select between them from the device-built route
histogram.  Before paying that integration cost, `analyze_v163_profit_gate.py` sampled 16,384 deterministic M16
routings from the task's distributional recipe.  Its job-saving histogram was:

| MID8 jobs saved | routings | share |
|---:|---:|---:|
| 0 | 11,189 | 68.292% |
| 1 | 4,966 | 30.310% |
| 2 | 226 | 1.379% |
| 3 | 3 | 0.018% |

The observed no-savings H200 pair gives an equal-job SKT16/MID8 ratio of `107.0 / 88.2 = 1.21315`.  Scaling that
ratio by `jobs16 / jobs8` and requiring a predicted ratio no worse than `0.995` makes only 133/16,384 routings
(0.8118%) eligible.  Even granting the model's predicted gain on every eligible routing, the average improvement
over all M16 calls is just 0.03747%.  That is below measurement resolution before accounting for the larger dual
path, extra second-tile shared memory, runtime branch, and changed register allocation.

Reject v163 without a CUDA candidate or remote run.  This closes the SKT16 branch: the potential pass reuse is too
rare to repay the measured loss of the helper-warp path.  Quota remains two checks and five scored runs.
