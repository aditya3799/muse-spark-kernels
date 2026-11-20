# v159 phase-A dependency-pair tail reorder

Status: rejected by the corrected host feasibility gate; no CUDA candidate, H200 check, or scored run was used.

## Source correction and narrowed hypothesis

Direct inspection of frozen v139 shows that each phase-A unit `ub` releases `doneAs[ub >> 1]`; phase-B stage `st`
therefore waits for adjacent A units `2*st` and `2*st+1`.  The originally proposed `(st, st+16)` pair-major order
would be semantically wrong, while the correct adjacent pair-major order is already v139's native unit order.

One narrower reorder remains feasible.  When a CTA's range ends partway through an odd unit, v139 moves that
partial to the front but leaves its complete preceding even mate at the end of the remaining span.  v159 models
pulling the mate next to the partial in both implementable orders:

- `partial_first`: odd partial, complete even mate, then the prefix;
- `mate_first`: complete even mate, odd partial, then the prefix.

Both variants preserve exact stage ownership, phase-A work, cost-weighted boundaries, and split incidence.

## Model and result

`analyze_v159_pairtail.py` reuses the corrected v158 cost/range and max-plus B model.  It covers 3,456 launches:
288 routing/count schedules across M32/M128/M256 times 12 epilogue/split-cost assumptions.  Per-CTA assertions
prove every owned stage appears exactly once, and every candidate phase-A finish time exactly equals baseline.

Both orders produced identical launch-level results despite changing readiness timestamps in all 3,456 launches:

| shape | median | p90 | worst | wins | losses |
|---|---:|---:|---:|---:|---:|
| M32 | 1.00000x | 1.00000x | 1.00425x | 0/960 | 80/960 |
| M128 | 1.00000x | 1.00000x | 1.00731x | 66/1,344 | 78/1,344 |
| M256 | 1.00000x | 1.00000x | 1.00781x | 62/1,152 | 73/1,152 |

The launch is usually limited by a different CTA/range, so earlier completion of one dependency pair does not move
the maximum.  In the non-neutral cases the reordered work is slightly more likely to delay a critical earlier B
dependency than to help the pulled pair.  The required p90 <= 0.98 and worst <= 1.0 gate fails decisively.

## Verdict and artifacts

Reject v159 before CUDA implementation.  Do not change `wgs_split`, spend an H200 check, or score this direction.
The corrected dependency audit also establishes that there is no missing global pair-major permutation to test.

- `analyze_v159_pairtail.py` SHA256:
  `284319BC876AFA79AF925FDF2B5146E0C020C54016853AD4C68EF1A2DAB7D5CD`
- corrected `analyze_v158_phase_permute.py` SHA256:
  `27F55CA5E3C895A746BE6BD21DE2D7DF67B9EE79567E4B2D9CCC506FA8F03604`
- protected v139 SHA256:
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`
- protected `submission.py` SHA256:
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`

Quota remains 5 checks and 5 scored runs.  v139 remains the provisional board candidate at 5,627.32 us / 2.38986x.
