# Resume Notes — Muse Attention Decode

Date saved: 2026-10-31 (Asia/Calcutta)

## Current safe workspace state

- Working directory: `D:\OpenKernelChallenge\muse_attention_decode`
- Active source: `submission.py`
- Current SHA-256: `A07CB2019246C85418DA24F3AD73321FF01A74C82590AFFE2727D6B013D47C46`
- Python syntax passes.
- `_DBG = ["-DFD_L2H=1"]`: the validated `fdec` L2 cache-load hint is active.
- `AP_LDMT` is default-off; its staged code does not alter the validated runtime path.
- Do not overwrite or revert the current FP8-QK, buffer-preallocation, or `FD_L2H` changes.

The current file contains three retained changes:

1. Full-decode QK uses native Hopper E4M3 × E4M3 tensor-core MMA with FP32 accumulation. Softmax and P×V remain on the original higher-precision path.
2. Output tensors and the empty profiler tensor are cached per route/shape/device. Existing `wo` and `wml` max/logsumexp scratch buffers were already cached.
3. Full-decode `fdec` K/V cache copies use the `.L2::128B` prefetch hint (`FD_L2H`).

## Best validated results

### Latest unscored validation

- Job: `432081cc16904fbf`
- Profile: `profiles/submission.432081cc16904fbf.txt`
- Correctness: 16/16 passed
- Flags/rejections: none
- Score: 1854.5503070936081 µs
- Speedup: 2.0308427253733683×
- This validates the current source state, including output/scratch preallocation and `FD_L2H`.

### Official scored submission

- Job: `bb717dcdcac54bfb`
- Profile: `profiles/submission.bb717dcdcac54bfb.txt`
- Correctness: 16/16 passed
- Flags/rejections: none
- Score: 1856.9 µs
- Speedup: 2.028× (95% interval: 1.998× to 2.059×)
- This validates the current active runtime path, including output/scratch preallocation and `FD_L2H`; the staged `AP_LDMT` candidate remained disabled.
- Previous official scored job: `98c816882fed492d`, 1906.7175795351582 µs, 1.9752794228278907×.

### Strong FP8-QK validation run

- Job: `3beb92100ee149b5`
- Profile: `profiles/submission.3beb92100ee149b5.txt`
- Correctness: 16/16 passed
- Score: 1870.5524405204585 µs
- Speedup: 2.013469346495345×
- Treat this as a strong run with normal H200 variance, not a guaranteed scored latency.

## Experiments already completed

| Experiment | Result | Decision |
|---|---:|---|
| Original production baseline | 1909.27 µs | Reference |
| Force full-context decode through `attnd` | 1931.52 µs | Reverted |
| Native FP8 QK in `fdec`, first run | 1900.88 µs | Promising |
| Native FP8 QK in `fdec`, repeat | 1870.55 µs | Retained |
| FP8 QK + FP8 P×V | 1917.29 µs; B32 kernel about 208 µs | Reverted |
| Persistent output/profile buffers | 1896.82 µs, 16/16 | Retained; aggregate difference is within run variance |
| `FD_L2H` on `fdec` K/V copies | 1854.55 µs, 16/16; B16 full 110.24 → 106.30 µs; B32 full 201.46 → 197.95 µs | Retained |
| `PFF=1` explicit L2 bulk prefetch | 2195.27 µs; B16 full 177.83 µs; B32 full 341.55 µs | Reverted; duplicate traffic/L2 contention |
| `RBF=6` deeper `fdec` ring | 1871.00 µs; B16 full 108.62 µs; B32 full 201.35 µs | Reverted; five slots are already sufficient |
| Balanced 132-CTA full-decode scheduling | 1906.34 µs; B16 full 110.62 µs; B32 full 203.07 µs | Reverted; uneven split/merge overhead exceeds extra-SM benefit |
| Sequence-independent B16/B32 full splits | 1876.71 µs; B16 full 108.55 µs; B32 full 201.51 µs | Reverted; capacity-range imbalance exceeds dependency overlap benefit |

Why FP8 P×V lost: `fdec` processes 16 keys per ring iteration, but FP8 MMA consumes K=32. Zero-padding kept the same effective MMA count while adding probability quantization and a more expensive V transpose.

## Suggested optimizations already present

The following ideas were audited and are already implemented on every scored route:

- Fused Q/K RMS normalization, RoPE, FP8 quantization, cache update, and attention occur in one kernel launch.
- Cache reads use aligned 16-byte lane copies. Eight lanes collectively load each 128-byte K or V row.
- Scored `attnd` and `attnp` variants already enable their `.L2::128B` hint through bit 64 in `x39`.
- Decode boundary masks use predicated selection. Verify's outer `if (!full)` is warp-uniform and avoids mask work for full tiles; the small lane condition should compile to predicated assignment.

Do not split RoPE/cache update into a separate kernel and do not force mask comparisons on every full tile.

## Resolved `FD_L2H` experiment

The existing `fdec` cache-copy hint is now enabled:

```python
_DBG = ["-DFD_L2H=1"]  # fdec: eight aligned 16 B lane copies form each 128 B K/V cache row
```

Validated unscored job: `432081cc16904fbf`.

- Correctness: 16/16; no flags or rejections.
- Aggregate score: 1854.5503070936081 µs, 2.0308427253733683×.
- B1 full decode: benchmark 17.29 → 17.00 µs; profiler 15.5 → 15.6 µs.
- B16 full decode: benchmark 110.24 → 106.30 µs; profiler 111.8 → 106.0 µs.
- B32 full decode: benchmark 201.46 → 197.95 µs; profiler 202.3 → 197.8 µs.

The B16 and B32 gains are consistent in both benchmark and profiler measurements, and the aggregate improvement is 2.23%, so the flag is retained. The earlier candidate SHA recorded as `59106CFA...` was stale or mistyped: toggling only this flag from the exact safe hash deterministically produces current SHA `E8523607...`.

WSL is functioning again. The two prior failed client attempts never registered a remote `muse_attention_decode` job.

Do not make another scored submission without explicit user approval.

## `fdec` latency decomposition

Diagnostic check `c4e15a971141427f` used the existing `PROF` timestamps on B16/B32 full decode and intentionally ended with `FDIAG_COMPLETE`; production source was then restored byte-for-byte.

Main-loop times (`sync` to `loop-end`, median CTA):

| Variant | B16 | B32 | Interpretation |
|---|---:|---:|---|
| Base | 87.42 µs | 172.70 µs | Normal FP8-QK + `FD_L2H` path |
| No KV loads | 65.37 µs | 129.60 µs | Page-table/loop + compute |
| No compute | 81.76 µs | 166.47 µs | Page-table/loop + KV gather |
| Neither | 10.37 µs | 21.89 µs | Page-table/loop overhead |

KV gathering contributes about 71.39 µs at B16 and 144.58 µs at B32 beyond page/loop overhead. Compute contributes about 55.00/107.71 µs in isolation, but only about 5.66/6.23 µs remains exposed when overlapped with KV IO. Page-table/loop work is roughly 13% of the IO-only path, so optimizing page-table lookup alone is not the priority; target KV latency/bandwidth while preserving overlap.

## Remaining optimization direction

The next meaningful kernel change is not more source-level branch rewriting. Better candidates are:

- Consider a true 32-key `fdec` ring step only if shared-memory/occupancy analysis shows it can reduce FP8 PV instruction count without losing residency.
- Preserve the current fused prologue and native FP8-QK implementation.

## Full-verify structural experiments (later 2026-10-31 session)

Production was restored after every rejected experiment. The earlier retained validation used SHA256:

`E852360748ECC58126F091DD11C1DF8014A1DB0CAF47910FAC794EBFCAC7EA2F`

After the later edit/revert cycle, the restored semantic baseline was SHA256 `DF9FF6AB5E79B3D2E2E9F4AE88789238AC8A0B96A24907073BC22BD834B662A9`. The current file is SHA256 `A07CB2019246C85418DA24F3AD73321FF01A74C82590AFFE2727D6B013D47C46` only because it also contains the default-off `AP_LDMT` candidate described below; the active runtime path remains the same.

The retained best unscored validation remains job `432081cc16904fbf`: 1854.5503070936081 us, 2.0308427253733683x, 16/16, no flags. The current official scored job `bb717dcdcac54bfb` is 1856.9 us, 2.028x, 16/16, with 9 of 10 scored runs and 0 of 20 unscored checks left for the day.

### Verify bottleneck ablation

Job `e1c2aa2d038e4723` measured B32 full verify under a CUDA graph (diagnostic intentionally failed after recording):

| Variant | Median | Exposed saving |
|---|---:|---:|
| Base | 224.74 us | - |
| No exponentials | 208.19 us | 16.55 us |
| No consumer V transpose | 191.04 us | 33.70 us |
| No KV loads | 187.39 us | 37.35 us |
| No WGMMA | 187.52 us | 37.22 us |

Full verify is balanced across IO, tensor-core work, and consumer transpose. The V transpose is the largest removable serial block, but no single block is large enough to reach the approximately 30% full-shape reduction needed for 2.6x aggregate.

### Rejected full-verify experiments

| Experiment | Job | Result | Decision |
|---|---|---|---|
| Producer-side FP8 V transpose, old and new templates both instantiated | `1931a7500ee8450e` | 16/16 checks passed, then global timeout before B16 full verify | Reverted; compile/template bloat made result inconclusive |
| Producer transpose forced through old runtime key | `54b6a882034f4904` | Illegal address at B16 full verify | Diagnostic only: runtime flags under-allocated shared memory by 4 KB |
| Producer transpose with correct allocation and one specialization | `cdfcc0ff0bdc4534` | Correctness passed, then timeout before B16 full verify | Reverted; large-batch producer protocol/path is unusable |
| One-quarter FMA polynomial exponential offload (`PEX`) | `bbe62b92d14f4d46` | 1930.76 us; B16/B32 full verify 129.8/244.0 us | Reverted; FMA contention exceeds MUFU relief |
| Issue QK before consumer V transpose | `78f5990d9a7648a1` | 1897.31 us; B16/B32 full verify 127.7/241.1 us | Reverted; WGMMA and transpose shared-memory traffic contend |
| Extend `QA` next-step QK/PV pipeline to FP8 V transpose | `297682ef957e4ecc` | 1946.41 us; B16/B32 full verify 129.9/241.5 us | Reverted; WGMMA queueing exceeds hidden transpose benefit |
| One split-lane CTA, two CTAs per SM (264 total lanes) | `95db9be698f94c23` | 2158.45 us, 1.7449x; B16/B32 full verify 199.20/368.25 us; profiler 171.1/312.6 us | Reverted; co-resident CTAs heavily contend for memory/tensor resources |
| Direct consumer V gather, first dispatch diagnostic | `0e151fb250404532` | 16/16 checks passed; B16 full verify cache recheck failed because the new runtime key had no template-switch case | Diagnostic only; corrected by replacing the old specialization instead of adding another one |
| Direct consumer V gather, fixed specialization | `78947a029a7444eb` | 2056.53 us, 1.83138x; B16/B32 full verify 161.5/307.0 us | Reverted; register-visible random loads stall the consumer warps and lose the producer's cp.async latency hiding |
| Scalar FP32 probability row sums instead of auxiliary `m64n8` WGMMA | `5693c8a4e54f42ce` | 1911.79 us, 1.97003x; B16/B32 full verify 127.0/233.7 us | Reverted; the small tensor-core row-sum products are cheaper than scalar adds plus shuffles |
| Compact FP8 V buffers, recovered space used for a fifth 64-key ring slot | `995925cb1d51416b` | 1891.02 us, 1.99168x; B16/B32 full verify 123.0/231.7 us | Reverted; four slots already hide gather latency and the fifth adds bookkeeping |
| Dedicated four-CTA-cluster sliding kernel, routed only to B16 verify | `d4371a1056264165` | 1926.43 us, 1.95506x; 16/16 checks passed; B16 verify sliding profiler 24.7 us | Reverted; fewer compact CTAs do not compensate for replacing the FP8/WGMMA path with the older f16/warp-MMA implementation |
| Fold V-transpose row permutation into first-level PRMT selectors (24 -> 16 PRMT/thread/step) | `0cd304a6f8144109` | 1987.49 us, 1.89500x; 16/16 checks passed; B16/B32 full verify 142.5/261.4 us | Reverted; fewer instructions created a longer selector-dependent critical path and regressed the active FP8 verify kernels |

Mining every saved successful profile showed that an oracle selecting the best observed timing independently for all 12 shapes reduces the weighted kernel medians by only about 1.8%. There is no hidden configuration/cherry-pick path to the requested +0.5x.

Current conclusion: attnp instruction reordering, dependency folding, and CTA repacking all regress. A worthwhile next attempt must change the V/P tile representation or the page-1 gather architecture; merely reducing source instruction count or overlapping the existing QK, transpose, and PV stages is not enough on H200.

## Pending `ldmatrix` FP8 V-transpose candidate

`submission.py` now contains a default-off `AP_LDMT` path for the full-verify consumer V transpose. Do not add `-DAP_LDMT=1` to `_DBG` until an unscored check is available; there are 0 unscored checks remaining in the current daily window.

The candidate treats pairs of adjacent FP8 dimensions as raw b16 carriers and uses two `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` operations per consumer warp. Four fixed pairs of `prmt` selectors then reconstruct the exact byte layout expected by the existing WGMMA K-major V tile. A symbolic lane/layout emulator compared all 4096 bytes of the current 64 x 64 shared V-transpose tile and found zero mismatches.

The store schedule uses `odd = pass ^ (dg >> 2)`. For each store instruction, the 32 lanes address all 32 shared-memory banks exactly once, preserving the conflict-free property of the current implementation.

Static per-thread/per-step instruction model:

- Current path: 4 shared loads + 24 `prmt` + 8 shared stores = 36 instructions.
- Candidate path: 2 `ldmatrix` + 8 `prmt` + 8 shared stores = 18 instructions.

A CUDA 12.4 NVRTC probe compiled the exact inline-PTX pattern for `compute_90` successfully. The generated PTX contains exactly 2 transposed `ldmatrix`, 8 `prmt.b32`, and 8 `st.shared.u32` instructions. This validates syntax and compiler lowering only; it does not establish H200 correctness or performance.

Next action after the unscored budget resets:

1. Temporarily append `-DAP_LDMT=1` to `_DBG`.
2. Run one unscored remote check.
3. Retain the path only if all 16 correctness cases pass and the full-verify profile improves, especially B16 and B32.
4. Otherwise remove the debug flag and leave the default-off candidate staged for reference. Do not use a scored submission to test it.

## 2026-11-01 continuation: `AP_LDMT` enabled and tightened

The user asked to proceed without using Super Experiments. `_DBG` now contains
`-DAP_LDMT=1` in addition to the retained `-DFD_L2H=1`. Current `submission.py`
SHA256:

`80B8643B62BA86C36535F457AACCC0D0E706BE80DA5EAE19BF350DC8380A3D44`

The original array-indexed candidate compiled, but SASS inspection showed that
ptxas materialized the `aa`, `bb`, and dynamically indexed `y` temporaries as
local-memory operations. The implementation was rewritten as four explicit
register-only pairs. Its destination expression was simplified from the generic
`d / 8` and `d % 8` form to one common base plus 512-byte pair strides. An
exhaustive check of all `m x dg x pass` combinations reported zero destination
mapping mismatches. The first/second store still use `odd = pass ^ (dg >> 2)`,
so the original conflict-free per-instruction bank schedule is preserved.

CUDA 12.4 compiled the exact active `attnp<2, 537002308>` specialization for
`sm_90a`. Static resource/SASS comparison against the same source with only
`AP_LDMT` disabled:

| Metric | Baseline | `AP_LDMT` | Delta |
|---|---:|---:|---:|
| Registers | 96 | 96 | 0 |
| Stack frame | 32 B | 16 B | -16 B |
| Spill stores | 36 B | 24 B | -12 B |
| Spill loads | 60 B | 52 B | -8 B |
| Static SASS instructions | 4655 | 4615 | -40 |
| `PRMT` | 82 | 66 | -16 |
| `LDS.64` | 4 | 0 | -4 |
| `LDSM.16.MT88.4` | 0 | 2 | +2 |

The local RTX 3050 cannot execute Hopper instructions, so correctness and
performance still require an official H200 check. At 2026-10-31 18:39 UTC,
`okc problems --json` reported attention-decode checks 20/20 used, 0 left, and
scored runs 1/10 used, 9 left. Checks reset at 00:00 UTC (05:30 IST). No scored
run and no Super experiment were used.

`compile_submission_cuda.py` now reads `_DBG` from `submission.py`, so the
resource probe compiles the exact active flags. Pass `--baseline` to remove only
`-DAP_LDMT=1` for a controlled comparison. Its SHA256 is
`DF4FB9B1E98710A1942DE431691E675933A2EA04115949D0C6FD1849F6174E01`.

Next action after 05:30 IST: run exactly one `okc check submission.py --json`.
Retain `AP_LDMT` only if all 16 tests pass and the B16/B32 full-verify timings
improve. Do not make a scored submission without explicit user approval.

### Scored H200 validation (explicitly requested)

The user explicitly requested a scored submission before the unscored quota
reset. Job `f99510bd0c4b457a` completed successfully:

- Correctness: 16/16, no flags or rejection.
- Score: 1887.6023201490666 us, 1.9952825655048836x.
- 95% speedup interval: 1.9653533270223102x to 2.0252118039874567x.
- Remaining scored runs: 8/10; remaining checks: 0/20 in this UTC window.
- The existing best `bb717dcdcac54bfb` remains 1856.928158301949 us,
  2.028242171438694x.

The aggregate run was slower, but the only two important shapes changed by
`AP_LDMT` both improved against the existing best run:

| Shape | Existing best | `AP_LDMT` | Change |
|---|---:|---:|---:|
| B16 full verify benchmark | 120.68 us | 118.76 us | -1.60% |
| B32 full verify benchmark | 225.85 us | 219.67 us | -2.74% |
| B16 full verify profiler | 119.9 us | 119.1 us | -0.8 us |
| B32 full verify profiler | 228.5 us | 221.6 us | -6.9 us |

The aggregate regression came primarily from the untouched `fdec` full-decode
shapes: B16 106.53 -> 110.65 us and B32 198.10 -> 203.31 us (profiler deltas
+6.5 and +3.4 us). `AP_LDMT` is compile-time code only inside `attnp`, so it
cannot affect `fdec`; treat those decode changes as cross-run variance. Retain
`-DAP_LDMT=1` because correctness passed and its targeted B16/B32 full-verify
timings improved. Do not resubmit identical source: OKC will return the same
job. No Super Experiment was used.
