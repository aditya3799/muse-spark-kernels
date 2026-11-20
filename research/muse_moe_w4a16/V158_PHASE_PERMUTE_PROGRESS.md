# v158 phase-coupled CTA permutation scout

Status: rejected by the host feasibility gate; no CUDA candidate, H200 check, or scored run was used.

## Hypothesis

Frozen v139 lets every physical CTA finish its cost-partitioned phase-A range and immediately execute the
same-index phase-B range.  v152 measured a small but persistent phase-A CTA tail, so v158 tested whether a fixed
bijection from physical CTAs to logical phase-B ranges could pair late A finishers with lighter B work and reduce
the launch tail.  A successful implementation would preserve v139's exact phase partitions and need no new global
barrier, atomic, or scratch allocation; only logical phase-B ownership and its fixup owner would change.

## Exact host model

`analyze_v158_phase_permute.py` reconstructs v139's two cost-weighted partitions with the production cost tables,
32 A units and 16 B units per job, 16 stages per unit, and the phase-A trailing-partial-first reorder.  It models
split-edge overhead, unit epilogues, and B's exact stage dependency on adjacent A units `2*st` and `2*st + 1`.
This dependency was corrected after a direct source audit of the `doneAs[ub >> 1]` release.  Every serial B
range is reduced exactly to a max-plus function `finish(start) = max(start + work, readiness_intercept)`.

The sweep covered:

- 93 bijections: identity, reverse, eleven rotations, and 80 affine permutations;
- 288 distinct routing/count schedules built from the observed M32/M128/M256 touched-job counts, two count
  patterns per shape, cyclic offsets, and reversals;
- 12 epilogue/split-cost settings, for 3,456 modeled launches in total;
- one mapping shared by all shapes and the stronger option of independently selecting a mapping for each shape.

The promotion gate was p90 ratio <= 0.98 for every shape with no modeled regression in any schedule.

## Result

The existing A tail is only 2.48-2.77% of modeled launch time at the median, and B readiness couples each logical
range back to its corresponding A producers.  Permuting B ownership therefore converts the small load imbalance
mostly into readiness stalls instead of removing it.

The best single mapping was affine permutation `25*c + 66 (mod 132)`:

| shape | median | p90 | worst | wins |
|---|---:|---:|---:|---:|
| M32 | 1.00000x | 1.00000x | 1.00000x | 352/960 |
| M128 | 1.00000x | 1.00255x | 1.00953x | 250/1,344 |
| M256 | 1.00000x | 1.00247x | 1.01682x | 424/1,152 |

Allowing a different mapping per shape did not reveal a hidden candidate.  M32's best remained exactly neutral at
p90 and worst; M128's best p90 was 1.00242x; M256's best p90 was 1.00168x.  All medians were exactly 1.00000x.
This is far from the required 2% robust improvement and introduces up to a 1.68% modeled regression.

## Verdict and frozen artifacts

Reject the fixed phase-B permutation direction.  Do not implement or remotely check it.  The result also rules out
shape-specific fixed mappings; a routing-dependent permutation would require new grid coordination on the timed
path and has no modeled benefit large enough to repay that cost.

- `analyze_v158_phase_permute.py` SHA256:
  `27F55CA5E3C895A746BE6BD21DE2D7DF67B9EE79567E4B2D9CCC506FA8F03604`
- protected v139 SHA256:
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`
- protected `submission.py` SHA256:
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`

Quota remains 5 checks and 5 scored runs.  v139 remains the provisional board candidate at 5,627.32 us / 2.38986x.
