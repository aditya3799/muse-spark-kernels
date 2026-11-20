# v156 tail-balance feasibility result

Date: 2026-11-02

## Protected state

- Public/scored v139: `submission_v139_wgs_kperm3.py`, SHA256
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`.
- Active recovery `submission.py`, SHA256
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`.
- Neither file was modified.  No remote check or scored run was used.

## Dynamic tail-pool feasibility

The proposed adaptive tail pool is not a small scheduler-only change in the WGS kernel.  Its weight and activation
producer warps pre-stream a fixed per-CTA stage sequence through three-slot rings, while four consumer warpgroups
derive unit identity and N width from the same static range.  Runtime stealing would require a producer-visible
work-ID ring, a globally contended claim protocol, and dynamic split completion.  It would also give up the proven
phase-B designated-finisher order.  That synchronization rewrite is too intrusive without a strong local bound.

## Bounded static correction

`analyze_v156_unit_bias.py` evaluates the safe alternative: charge the final stage of each 16-stage output unit an
extra 1..4 stage-equivalents when computing the existing cost-weighted CTA boundaries.  This keeps the exact v139
producer order, arithmetic, and fixup protocol.  The model covers both phases, all observed job counts, narrow/wide
job patterns, epilogue costs of 0.5..4 stage-equivalents, and split-edge penalties of 0..1 stage-equivalent.

Every positive bias is worse in the worst modeled case:

| Bias | Best worst-case max-load ratio | Additional split edges |
|---:|---:|---:|
| 0 (v139) | 1.00000x | 0 |
| 1 | 1.01724x | up to 4 |
| 2 | 1.02778x | up to 5 |
| 3 | 1.03932x | up to 5 |
| 4 | 1.06819x | up to 6 |

Other plausible epilogue/fixup assumptions make the biased schedules as much as 10.17% worse.  The quantized
stage boundary absorbs the extra unit-final weight poorly and increases rather than reduces imbalance.

The exhaustive ownership audit passes 217,600 schedules: jobs 1..136, grids 1..160, both phases, and biases 0..4
are monotone, gap-free, and exact.  Correctness is therefore not the rejection reason; modeled performance is.
The analysis script SHA256 is `120FB37000CD7360D32D90CCE2B5260DB29FA324C79029F218567B0B8AA987E0`.

## Verdict

Reject the v156 tail-pool/unit-bias direction before CUDA implementation.  It misses the required 2% local
max-load improvement and creates more split boundaries.  Do not spend an H200 check, do not score, and do not
revive this mechanism without a fundamentally different producer protocol.
