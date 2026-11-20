# v160 paired phase-B norm-weight loads

Status: H200 check passed correctness but failed the predeclared performance gate; reject without scoring.
One remote check and no scored run were used.

## Target

With `OKC_WGSFINSS=1`, the phase-B epilogue only scales and stores each FP32 output fragment.  Four lanes in each
subgroup use the same two BF16 post-norm weights `(g, g+8)`.  The original layout makes those two loads perfectly
coalesced, so leader-load/broadcast variants do not reduce memory transactions and add shuffle/control work.

The useful representation change is instead lossless untimed repacking from each 16-weight row block
`[0..7, 8..15]` to eight adjacent BF16 pairs `[(0,8), ... (7,15)]`.  The epilogue then issues one `LDG.32` and
reconstructs the two FP32 values by placing each original BF16 bit pattern in the high half of an FP32 word.  The
kernel still forms `1 + weight` in FP32.  No weight is converted, rounded, or discarded; the extra packed tensor is
exactly 4,096 BF16 values / 8 KB and is built in `prepare()`.

## Rejected subgroup broadcasts

`compile_v160_epinorm_static.py` first tested two no-repack broadcasts.  Both retained the same 96-register,
48-byte-stack, 44/104-byte spill shape and HGMMA mix, but ptxas expanded their instruction shape:

| variant | LDG | SHFL | IMAD | PRMT | FADD |
|---|---:|---:|---:|---:|---:|
| v139 control | 540 | 308 | 5,966 | 326 | 444 |
| two-FP32 broadcast | 652 | 524 | 6,130 | 326 | 560 |
| packed-word broadcast | 656 | 416 | 6,110 | 434 | 560 |

These variants are rejected locally.  Repeated lane addresses already coalesce, so their reduced active-lane count
does not repay the new shuffles and predicates.

## Frozen diagnostic

`submission_v160_wgs_epinorm_diag.py` is derived from the v150 multi-seed harness.  The official `kernel_fn` and
the diagnostic control compile with `OKC_WGSEPINORM=0`; only the embedded `normpair` variant enables the new load.
Preparation appends the exact paired BF16 tensor as `w[14]`.  The diagnostic checks bit-exact candidate/control
outputs for seven seeds at each of M32/M128/M256, then runs alternating-order, cold-L2 CUDA-graph timings.

`audit_v160_epinorm.py` passes.  It checks the protected hashes, exact one-to-one 4,096-element pair mapping,
finite/infinity BF16-to-FP32 bit reconstruction, macro-off production dispatch, candidate-only define, and the
remote exact-parity assertion.

The actual generated candidate was then compiled—not only the earlier in-memory scout.  Both builds retain 96
registers, a 48-byte stack, 44-byte spill stores, 104-byte spill loads, no WGMMA serialization, and all eight HGMMA
N widths with 128 sites each (1,024 total):

| build | LDG | SHFL | IMAD | PRMT | HMUL2 | FADD | LDL/STL |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 540 | 308 | 5,966 | 326 | 4,096 | 444 | 54/26 |
| normpair | 492 | 308 | 5,952 | 326 | 4,096 | 448 | 54/26 |

Thus the candidate removes 48 static LDGs and 14 IMADs at the cost of four FADDs, with no resource or pipeline
regression.  This clears the local plausibility gate but is not runtime proof; v146 showed that removing 48 LDGs
from a producer path did not guarantee an H200 win.

## Remote gate

One off-board check should be used only with fresh authorization.  Promotion requires:

- all five official correctness cases pass;
- exact BF16 candidate/control equality for all 21 embedded shape/seed cases;
- at least 5/7 paired wins at both M128 and M256;
- median candidate/control ratio <= 0.995 at M128 and M256;
- no M32 median regression beyond 0.2%.

Do not score or replace either protected file without a separate post-check decision.

## Remote H200 verdict

Authorized check job `3d76440380a84917` passed all five official correctness cases with no flags or rejections.
The official macro-off/control payload measured 5,642.55 us / 2.38341x (95% CI 2.34766..2.41916x); because the
candidate is default-off in `kernel_fn`, that aggregate is not candidate performance evidence.  All 21 embedded
candidate/control comparisons were bit-exact.

Paired candidate/control timing ratios (below 1.0 favors `normpair`) were:

| shape | seven paired ratios | median | wins | gate |
|---|---|---:|---:|---|
| M32 | 1.0021, 0.9963, 0.9956, 1.0004, 1.0026, 1.0019, 0.9904 | 1.0004x | 3/7 | pass: <= 1.002 |
| M128 | 0.9962, 1.0005, 0.9897, 1.0021, 1.0007, 1.0008, 1.0066 | 1.0007x | 2/7 | fail |
| M256 | 0.9989, 1.0003, 0.9957, 0.9963, 0.9982, 1.0030, 1.0008 | 0.9989x | 4/7 | fail |

The diagnostic's mean of shape medians was 0.9990x, only about 0.1% faster and well inside the noisy/neutral
region.  M128 and M256 both miss the required >=5/7 wins and <=0.995 median ratio.  Reject v160: do not score,
promote, or replace a protected file.  The load-count reduction is real but is not on a meaningful runtime
critical path.

## Frozen hashes

- `submission_v160_wgs_epinorm_diag.py`:
  `1911260BA9E70C12F9B5DA44DE97F2A0A9BA8CD2C542E67CC998F9E8DE5960FD`
- `audit_v160_epinorm.py`:
  `BB36912AD883C368E80900A8C05C33732DCB1CBDE9FF8AF8D560A0384A1CFD48`
- `compile_v160_candidate_static.py`:
  `9FF2D78F2E8EF912F58CC6C26485C4C23242589745448B8D10CDFB58CB8748B9`
- protected v139:
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`
- protected `submission.py`:
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`

Quota after the check is 4 checks and 5 scored runs.  v139 remains the provisional board candidate at
5,627.32 us / 2.38986x.
