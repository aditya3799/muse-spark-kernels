# v152 readiness diagnosis and v153 whole-unit cyclic scout

Date: 2026-11-02

## Frozen state

- Public/scored v139: `submission_v139_wgs_kperm3.py`, SHA256
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`.
- Active recovery `submission.py`, SHA256
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`.
- Neither file was modified.

## v152 H200 diagnosis

Off-board job `80547849e5e14294` passed all five correctness cases at 5,683.97 us / 2.36604x.  Its aggregate
number is an unscored variance sample; the embedded paired and counter data are the useful result.

The diagnostics compiled the exact v139 control, a timeline build, and an isolated producer-readiness profiler.
Median paired overhead was only 0.46% for the timeline build and 0.61% for the producer profiler.  Across seven
seeds per shape, median cold-launch counters were:

| M | span (us) | A tail (us) | B tail (us) | weight-ring empty wait (us) | x-ring empty wait (us) | x gather wait (us) | doneAs wait (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 123.9 | 13.3 | 7.4 | 95.5 | 70.2 | 3.6 | 25.1 |
| 128 | 191.7 | 18.9 | 16.1 | 143.1 | 108.8 | 5.8 | 36.3 |
| 256 | 205.8 | 18.7 | 14.6 | 149.4 | 110.6 | 4.9 | 35.9 |

Both producer warps spend most of their lifetime waiting for consumers to release ring slots.  The activation
producer remains ahead despite its `doneAs` waits, so TMA, ring depth, gather, and phase-B readiness are not the
next critical-path levers.  The actionable residual is the 6-23 us per-phase CTA completion spread plus split-unit
bookkeeping/fixup work.  Intrusive consumer/fixup timers were rejected locally before remote use: they raised spill
traffic from 44/104 B to as much as 1,820/428 B and would not have produced trustworthy measurements.

## v153 structural scout

`submission_v153_wgs_cyclic_diag.py` (SHA256
`5E0E501150E594A83FAC83AD4F25F8817FCC4903E9F1BBA908E30F91AD3CB5F6`) adds default-off
`OKC_WGSCYCLIC=1`.  CTA `c` owns complete phase units `c, c+G, ...`; each owned unit retains all sixteen K stages.
Producer and consumer mappings are identical, and whole units use fixup slot `-1`, so no cross-CTA partial buffer,
count, or fixup is reachable.

`audit_v153_contract.py` exhaustively checks job counts 1..136, both phases, and grids 1..160.  Every unit is owned
exactly once and each producer emits exactly sixteen stages per owned unit.  The active and frozen baseline hashes
are also enforced.

Local CUDA 12.4 SM90a results:

| variant | registers | stack | spill stores/loads | HGMMA shapes |
|---|---:|---:|---:|---|
| v139 control | 96 | 48 B | 44 / 104 B | unchanged eight N widths |
| v153 cyclic | 96 | 48 B | 8 / 8 B | identical to control |
| v153 cyclic timeline | 96 | 64 B | 48 / 88 B | identical to control |

No build emitted WGMMA serialization warnings.  The cyclic SASS also removed dead fixup-path loads, but runtime
performance is not yet known.  An attempted off-board check was stopped by the external-upload approval gate; do
not upload or score v153 without explicit user authorization.

The user subsequently authorized one unscored upload.  Job `8f0923163b4b4bd1`, payload content hash
`8c2d2b8a72e036c6a9e5d5e85634643db0ede6c25df763fa271b0f02fa067a71`, passed all five correctness cases at
5,668.73 us / 2.37240x.  The embedded paired experiment decisively rejects the cyclic schedule:

| shape | cyclic/control median | wins | cyclic A-tail median | cyclic B-tail median | v152 control A/B tails |
|---:|---:|---:|---:|---:|---:|
| M32 | 1.0757x | 0/7 | 22.8 us | 38.1 us | 13.3 / 7.4 us |
| M128 | 1.0949x | 0/7 | 34.8 us | 45.8 us | 18.9 / 16.1 us |
| M256 | 1.0802x | 0/7 | 27.4 us | 34.6 us | 18.7 / 14.6 us |

Mean ratio across all 21 cases was 1.0907x.  Removing fixups and nearly all spills did not compensate for the much
coarser whole-unit load balance.  This proves the current cost-weighted fractional stream-K partition is critical;
do not promote, score, or retry v153 cyclic scheduling.
