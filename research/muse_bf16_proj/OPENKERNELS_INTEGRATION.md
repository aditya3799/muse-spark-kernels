# Public-kernel integration

## Current leaderboard candidate

Source: https://github.com/NathenFernandes/muse-bf16-proj

- Upstream revision: `7385e5d8f70980cfd5322ea254edc0b2b8ca90f7`
- Upstream license: MIT; see `NATHEN_FERNANDES_MIT_LICENSE.txt`
- Imported candidate: upstream `submission.py` (v18, unchanged bytes)
- Local working file: `submission.py`
- Preserved prior candidate: `submission_openkernels_v22_local.py`
- Local scored validation: job `e3182690ad2446eb`, 9/9 tests, no flags,
  `1.509493x`, `3200.28 us`

## Previous openkernels candidate

Source: https://github.com/lee101/openkernels

- Upstream revision: `ba37f4ff2bb83519d529d113a41a61d2aab865ee` (`master`, resolved 2026-10-27)
- Upstream license: Apache-2.0
- Imported candidate: `muse_bf16_proj/submission.py`
- Preserved local file: `submission_openkernels_v22_local.py`
- Preserved previous local implementation: `submission_coldtune.py`

## Judge validation history

| Candidate | Job | Tests | Speedup | Score |
|---|---|---:|---:|---:|
| Nathen Fernandes v18 | `e3182690ad2446eb` | 9/9 | 1.509493x | 3200.28 us |
| Published upstream v11 | `e1b26e514c6042ed` | 9/9 | 1.144744x | 4219.98 us |
| Upstream working copy | `e2d047db819b4ebb` | 9/9 | 1.451120x | 3329.01 us |
| Previous local best check | `4409b16f33fa4bc3` | 9/9 | 1.165664x | 4144.25 us |
| Scored upstream working copy | `2d3105eb570c454f` | 9/9 | 1.455999x | 3317.86 us |
| Dense-budget + paired tuning check | `d155faa172764295` | 9/9 | 1.453648x | 3323.22 us |
| Focused m32 candidate check | `6d14586dedc3418a` | 9/9 | 1.457593x | 3314.23 us |
| Final pruned candidate check | `435ef2b972bb4bcf` | 9/9 | 1.465147x | 3297.14 us |
| Final pruned scored submission | `b5469e7c4f504224` | 9/9 | 1.463870x | 3300.02 us |

The previous openkernels working copy was promoted because it improved the verified check
speedup by about 24.5% relative to the previous local best. It uses exact
12-bit-packed BF16 projection weights for small token counts, specialized
router kernels, a WGMMA router path for larger token counts, and prepare-time
cuBLASLt/autotuning fallbacks.

The preserved openkernels candidate keeps the denser cuBLASLt search and paired
winner confirmation from `d155faa172764295`.  From the focused m32 check it
retains only the winning pooler ring-depth-1 variant; the losing exploratory
QKV and split-K variants were removed after the check.
