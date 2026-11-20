# v139 WGS compute investigation progress

Date: 2026-11-02 (Asia/Calcutta)

## Scope and preserved baseline

This investigation targeted tensor-core and dequantization cost in
`submission_v139_wgs_kperm3.py`, prioritizing M=256, then M=128, then M=32.

The following closed mechanisms were not changed: CTA scheduling, cost tables,
boundary snapping, L2/eviction hints, TMA ring depth or box size, CTA roles,
and prefetch.

- Production baseline: `submission_v139_wgs_kperm3.py`
- Baseline SHA256: `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`
- No commits were created.
- No experimental change was promoted into v139.

## Current WGS+KPERM3 architecture

For a job with `t = ceil(tokens / 8)` token tiles, `t` in 1 through 8,
`wgs_phase` dispatches once to `wgs_seg<PB, t>`.

Each K16 step issues one shape-specialized instruction:

| Tile count `t` | WGMMA shape | Instructions per K16 per consumer warpgroup |
|---:|---|---:|
| 1 | m64n8k16 | 1 |
| 2 | m64n16k16 | 1 |
| 3 | m64n24k16 | 1 |
| 4 | m64n32k16 | 1 |
| 5 | m64n40k16 | 1 |
| 6 | m64n48k16 | 1 |
| 7 | m64n56k16 | 1 |
| 8 | m64n64k16 | 1 |

The kernel does not issue `t` separate N=8 WGMMAs. The WGMMA instruction
count stays constant while its N width grows with `t`.

- One K=256 stage: 16 WGMMAs per consumer warpgroup.
- One complete K=4096 output unit: 256 WGMMAs per consumer warpgroup.
- Phase A: 32 output units/job, or 8192 instructions per fixed consumer-WG
  position across the job.
- Phase B: 16 output units/job, or 4096 instructions per fixed consumer-WG
  position across the job.
- There are four consumer warpgroups per CTA.

The baseline declares two A-fragment banks, `af[2][GS][4]`, but its ordering is:

1. Dequantize the current A bank.
2. Execute `wgmma.fence`.
3. Issue the current WGMMA group.
4. Commit the group.
5. Execute `wgmma.wait_group<1>`.
6. Begin dequantizing the next bank.

Therefore the baseline does not overlap the next dequantization with a pending
WGMMA group. A final `wait_group<0>` occurs before the epilogue.

## Step 1: WGMMA and dequantization ablations

Diagnostic file: `submission_v150_wgs_compute_diag.py`

Off-board H200 job: `4f0f76de8a1040b9`

- Official correctness: 5/5 passed.
- Measurements: seven routing seeds per shape, 15 alternating cold-L2 CUDA
  graph repetitions per seed.
- Official control-payload aggregate: 5615.32 us / 2.39497x.

### Seven-seed medians

| Shape | v139 control | WGMMA removed, dequant kept | Paired ratio | Raw-bit A, WGMMA kept | Paired ratio |
|---|---:|---:|---:|---:|---:|
| M=32 | 133.0 us | 133.1 us | 0.9969x | 130.8 us | 0.9788x |
| M=128 | 201.0 us | 201.1 us | 1.0027x | 199.9 us | 0.9884x |
| M=256 | 216.6 us | 221.9 us | 1.0153x | 215.6 us | 0.9850x |

### Interpretation

The no-WGMMA build retains dequantization through a never-taken opaque-zero
trap sink. Its SASS retains the baseline's 4096 HMUL2 instructions and full
PRMT count. The sink adds a few integer instructions, so the latency is a
conservative bound rather than a perfectly instruction-free WGMMA subtraction.

Removing WGMMA is neutral at M=32/M=128 and slower at M=256. Tensor-core work
is therefore already hidden by the wider pipeline and is not the dominant
wall-time limiter.

The raw-dequant build feeds four rotations of each packed code word directly
to the original WGMMA instructions. It removes unpacking, scale construction,
and HMUL2 work, but produces intentionally invalid output. Ptxas responds by
serializing the WGMMAs with warning C7512. Despite this optimistic arithmetic
removal, the observed gain is only about 1.2% at M=128 and 1.5% at M=256.

Diagnostic aggregate proxies, assuming the other three scored loads remain
unchanged:

- WGMMA removed: 5629.16 us, approximately 0.25% worse.
- Raw dequant: 5570.38 us, approximately 0.80% better as an invalid-output
  upper bound.

### Static resources

| Variant | Registers | Stack | Spill stores | Spill loads | BF16 WGMMA serialized |
|---|---:|---:|---:|---:|---|
| v139 control | 96 | 48 B | 44 B | 104 B | No |
| WGMMA removed, dequant kept | 96 | 32 B | 0 B | 0 B | Not applicable |
| Raw-bit A, WGMMA kept | 96 | 32 B | 0 B | 0 B | Yes, C7512 |

## Step 2: single wide WGMMA

Status: skipped because its prerequisite is false.

The requested single `N = 8*t` implementation already exists in v139 for all
`t` from 1 through 8. Adding another dispatch path would duplicate the current
architecture rather than replace a `t x N=8` loop.

## Step 3: A-fragment double-buffer overlap

Diagnostic file: `submission_v151_wgs_afpipe_diag.py`

Flag: `OKC_WGSAFPIPE`, default 0.

Off-board H200 job: `3b398913d4d147b1`

- Per-seed candidate/control correctness checks passed.
- Official control correctness: 5/5 passed.
- Official control-payload aggregate: 5684.40 us / 2.36586x.

The flagged implementation leaves group `g` pending, retires `g-1` with
`wait_group<1>`, fills the opposite A bank for `g+1`, and delays the previous
stage's activation-slot release until its final group has retired.

### Seven-seed medians

| Shape | v139 control | A-fragment pipeline | Paired ratio | Result |
|---|---:|---:|---:|---|
| M=32 | 134.1 us | 135.2 us | 1.0048x | Regressed |
| M=128 | 200.6 us | 201.7 us | 1.0035x | Regressed |
| M=256 | 216.0 us | 216.4 us | 1.0037x | Regressed |

The three-shape aggregate proxy is 5695.76 us, approximately 0.20% worse.
The candidate remains disabled and was not promoted.

### Resource impact

| Variant | Registers | Stack | Spill stores | Spill loads | WGMMA serialized |
|---|---:|---:|---:|---:|---|
| v139 control | 96 | 48 B | 44 B | 104 B | No |
| A-fragment pipeline | 96 | 48 B | 84 B | 176 B | No |

The deeper live window nearly doubles spill traffic at the existing 96-register
cap. Consumer-warpgroup ping-pong with named barriers was not attempted because
the prerequisite register headroom does not exist.

## Current conclusion

The per-tile cost is not primarily limited by WGMMA issue or the existing
dequant arithmetic:

- Removing WGMMA does not reduce wall time.
- Removing all dequant arithmetic exposes WGMMA serialization and saves only a
  small amount.
- Overlapping dequant with a pending group increases spills and regresses all
  three target shapes.

The next productive target should be outside this closed arithmetic family,
while continuing to respect the already rejected scheduling, TMA, cache-hint,
role, and prefetch experiments. Likely candidates are reducing non-WGMMA
consumer bookkeeping/epilogue cost or finding a representation that removes
bytes without adding dependent decode work.

## Artifacts

- `submission_v139_wgs_kperm3.py`: unchanged production baseline.
- `submission_v150_wgs_compute_diag.py`: flagged Step 1 ablations.
- `compile_v150_compute_static.py`: Step 1 ptxas/cuobjdump audit.
- `submission_v151_wgs_afpipe_diag.py`: flagged Step 3 candidate.
- `compile_v151_afpipe_static.py`: Step 3 ptxas/cuobjdump audit.
- `profiles/submission_v150_wgs_compute_diag.4f0f76de8a1040b9.log`: Step 1
  seven-seed measurements.
- `profiles/submission_v151_wgs_afpipe_diag.3b398913d4d147b1.log`: Step 3
  seven-seed measurements.
- `OPTIMIZATION_CHECKPOINT.md`: cumulative project ledger.
