# v154 split-fixup progress

Date: 2026-11-02

## Frozen state

- Public/scored v139: `submission_v139_wgs_kperm3.py`, SHA256
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`.
- Active recovery `submission.py`, SHA256
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`.
- Neither file was modified.

## Motivation and schedule incidence

v153 proved that removing fixups by assigning whole units loses 7.6-9.5% because it destroys v139's fine-grained
cost balance.  v154 therefore preserves every v139 stage boundary and changes only how a split unit combines its
partial accumulators.

The offline model mirrors the exact v139 cost tables and `wgs_stage_at` partition.  Across the 21 routing seeds used
by v152/v153, 64.3-82.8% of stages belong to whole units.  Each launch has 240 split units across phases A+B; every
observed split unit has exactly two contributors.  The general contract was also checked for all 1..136 jobs, both
phases, and grids 1..160 (43,520 schedules), including units spanning more than two CTAs.

## Candidate protocol

`submission_v154_wgs_dualfix_diag.py` adds default-off `OKC_WGSDUAL=1`:

- The share containing stage 0 overwrites edge buffer 0 with ordinary vector stores.
- All later shares reduce into independently zero-based edge buffer 1.
- The existing stage-count/release protocol selects or waits for the finisher exactly as before.
- The finisher loads edge 0 and edge 1, adds them, clears only edge 1, and runs the unchanged epilogue.

Separate edge buffers make the protocol exact for any number of contributors without an initialization race.  On
the observed two-contributor units it replaces one of two FP32 global reductions with a plain store.  Stage ranges,
phase order, WGMMA math, and output arithmetic are unchanged.  The Python scratch allocation reserves the larger
diagnostic buffer, while the default control compile keeps the original v139 offsets.

## Local gates

`audit_v154_contract.py` passes 43,520 phase schedules and enforces the frozen v139/recovery hashes.  Exact CUDA
12.4 SM90a compilation from `compile_v154_dualfix_static.py` reports:

| variant | registers | stack | spill stores/loads | HGMMA |
|---|---:|---:|---:|---|
| v139 control | 96 | 48 B | 44 / 104 B | unchanged eight N widths |
| v154 dual fixup | 96 | 48 B | 4 / 4 B | identical counts and widths |

Neither build emits a WGMMA serialization warning.  The candidate adds the expected second fixup-buffer loads and
stores, so the static result is a plausibility gate rather than proof of a runtime win.

Candidate SHA256:
`C0509FF867D49CF799FDC39AE98CE9C3494F936C29CBF6FCD503442922558871`.

## Remote gate

Authorized off-board job `06abfff3d3604c4a` accepted the frozen v154 payload, compiled both variants in 45 s,
then hit the service's roughly 15-minute execution timeout before printing the first paired diagnostic record.
It produced no correctness or performance samples and is therefore inconclusive, not a numerical rejection.

The diagnostic harness reused `state["scratch"]` for both graph captures even though the control places its
count/done flags after two fixup sets while `OKC_WGSDUAL=1` places them after four.  Candidate phase-B fixup writes
could therefore overwrite the control's synchronization state and deadlock its next replay.  This is outside the
candidate protocol itself.

`submission_v155_wgs_dualfix_isolated_diag.py` repairs only that harness alias: every compiled variant receives a
separate persistent scratch allocation.  Its CUDA source is byte-identical to v154 (176,318 bytes, SHA256
`8AA3CF42352FE3679C6E8B90453337818D0EC88A0BB5F5A71EB9166C6E5B8BA2`).  The full v155 file SHA256 is
`C8CD1C9EC37902F1F3B1A71C6295F7CA0CC2B378BE6ED14D3D7E3186A16DC1F2`.

`audit_v155_contract.py` passes all 43,520 schedules, proves v154/v155 CUDA-source equality, and rechecks both
protected hashes.

Authorized replacement check `f03fbceb0f614e7d` completed normally on H200.  The default v139 path passed all five
official correctness cases and the internal candidate comparisons did not raise a correctness failure.  The paired
performance result is nevertheless a conclusive rejection:

| Shape | v139 control median | dualfix median | median paired ratio | paired wins |
|---|---:|---:|---:|---:|
| M=32 | 132.1 us | 140.1 us | 1.0226x | 0/7 |
| M=128 | 201.0 us | 208.3 us | 1.0415x | 0/7 |
| M=256 | 216.8 us | 230.2 us | 1.0588x | 0/7 |

Dualfix lost all 21 seeds; individual ratios ranged from 1.0187x to 1.0756x and the three-shape mean ratio was
1.0416x.  The extra second-buffer traffic and finisher work cost more than replacing one global reduction with a
plain store saves.  Reject v154/v155, keep `OKC_WGSDUAL=0`, and do not score, promote, or retry this mechanism.
The unchanged-control payload's incidental official aggregate was 5657.84 us / 2.376966x; it is not a dualfix
score because the diagnostic flag is default-off.
After this check, the problem has 6/10 off-board checks and 5/5 scored runs remaining for the day.
