# Resume handoff: muse_moe_w4a16

Snapshot saved 2026-11-03 (Asia/Calcutta). Daily OKC limits reset at 00:00 UTC.

## Production baseline

- Active/frozen winner: `submission_v139_wgs_kperm3.py`.
- Frozen v139 SHA256:
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`.
- Live `submission.py` SHA256:
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`.
- Best recorded v139 job: `13cca0e4992b4983`, 5627.316 us / 2.389860x.
- Do not overwrite or promote into `submission.py` without a new passing gate.

## Quota at save time

- Checks remaining: **1**.
- Scored runs remaining: **5**.
- No scored run was consumed in the v164 work.

## Latest experiment: v164 WGS polynomial SiLU

Candidate: `submission_v164_wgs_polysilu_diag.py`.

The local audit and exact SM90a build passed:

- 96 registers, 48-byte stack, 44/104-byte spills;
- unchanged 1,024-site HGMMA inventory and no WGMMA serialization;
- MUFU reduced from 466 to 34 across the two templates;
- macro-off control SASS byte-identical to frozen v139;
- scalar audit relative L2 0.267579%.

Authorized H200 check `a85addbaf44340c3` passed all five official cases and
all 21 embedded output comparisons. Candidate/control maximum relative L2 was
0.9994%; maximum row error was 1.1305%.

Performance failed the predeclared gate:

| Shape | paired median ratio | wins |
|---|---:|---:|
| M32 | 0.9976x | 4/7 |
| M128 | 0.9994x | 4/7 |
| M256 | 0.9999x | 5/7 |

The 21-case mean was 0.9989x versus the required <=0.9975x. Reject v164
without promotion or scoring. Its macro-off aggregate of 5701.33 us / 2.35884x
is control-only and is not evidence for the candidate.

Conclusion: exact-SiLU special-function latency is not a dominant WGS wall-time
bottleneck. Removing 432 MUFU sites was numerically safe but timing-neutral.

## Closed branches that should not be repeated

- WGMMA removal/raw-dequant/A-fragment overlap: neutral, invalid, or slower;
  overlap doubled spills.
- TMA/gather/ring/readiness/scheduling/warp-role/CTA permutation families.
- Segmented final pass and paired norm-weight load.
- M16 SKT16 and its profitability gate.
- Decoder translation search: current multiply decoder already meets the
  eight-instruction lower bound.
- WGS polynomial SiLU (v164): correct but below the timing promotion gate.

See `OPTIMIZATION_CHECKPOINT.md` for the complete rejection ledger and
`V164_WGS_POLYSILU_PROGRESS.md` for the latest evidence.

## Recommended restart plan

1. Re-run `audit_v139_kperm3.py` and verify the two protected hashes above.
2. Treat v139 as immutable and begin from a new default-off diagnostic branch.
3. Search outside WGMMA/dequant/SiLU arithmetic. The remaining credible area is
   WGS non-WGMMA bookkeeping and memory dependency latency on the launch-critical
   path: address generation, job lookup, phase handoff, and epilogue stores.
4. Use source-attributed SASS plus dependency-chain analysis to choose one
   concrete instruction/dependency target before writing CUDA.
5. Require unchanged-or-better registers/stack/spills and macro-off SASS
   identity before using the next remote check.
6. Do not spend a scored run until a candidate passes correctness and a paired
   H200 promotion gate by more than measurement noise.

## Latest artifacts

- `analyze_v164_v139_sass.py`
- `make_v164_wgs_polysilu_diag.py`
- `audit_v164_polysilu.py`
- `compile_v164_polysilu_static.py`
- `submission_v164_wgs_polysilu_diag.py`
- `V164_WGS_POLYSILU_PROGRESS.md`
- `profiles/submission_v164_wgs_polysilu_diag.a85addbaf44340c3.log`
- `profiles/submission_v164_wgs_polysilu_diag.a85addbaf44340c3.json`
- `profiles/submission_v164_wgs_polysilu_diag.a85addbaf44340c3.txt`

