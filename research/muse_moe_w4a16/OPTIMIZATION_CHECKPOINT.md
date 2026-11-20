# muse_moe_w4a16 optimization checkpoint

## 2026-11-02 v156 tail-balance feasibility (rejected locally; no CUDA candidate)

The adaptive tail-pool plan fails the bounded-implementation test because WGS's two producer warps pre-stream a
fixed per-CTA sequence through three-slot rings.  Runtime stealing would require a new producer-visible work-ID
ring, global claims, dynamic consumer dispatch, and loss of the proven phase-B designated finisher.

The safe static alternative was modeled instead: charge each 16-stage unit's final stage an extra 1..4
stage-equivalents in the existing cost partition.  `analyze_v156_unit_bias.py` passes 217,600 exact ownership
schedules, but every positive bias worsens the worst modeled maximum CTA load by at least 1.72% and creates up to
six more split edges; plausible cases regress by as much as 10.17%.  This misses the required 2% improvement gate.
No CUDA candidate, remote check, or score was created.  v139 and `submission.py` retain their protected hashes.
See `V156_TAIL_BALANCE_PROGRESS.md`.

## 2026-11-02 v154/v155 split-fixup candidate (rejected: 0/21 paired wins)

v154 preserves v139's exact cost-weighted fractional stage ranges and changes only split-unit accumulation.  The
stage-0 share writes a private edge buffer with ordinary vector stores; later contributors reduce into a second
buffer; the existing finisher adds both and clears the second.  This is exact for arbitrary contributor counts and
removes one contended FP32 global reduction from every split unit without reviving v153's coarse scheduling.

The exact schedule model finds 240 split units per observed launch, with 64.3-82.8% of stages still on whole-unit
paths and two contributors per observed split.  `audit_v154_contract.py` passes 43,520 schedules covering 1..136
jobs, both phases, and grids 1..160.  Exact SM90a compilation retains 96 registers, the 48-B stack, all eight HGMMA
widths, and no serialization; spills improve from v139's 44/104 B to 4/4 B.  Frozen candidate SHA256 is
`C0509FF867D49CF799FDC39AE98CE9C3494F936C29CBF6FCD503442922558871`.

Authorized v154 check `06abfff3d3604c4a` compiled both variants in 45 s but timed out before its first paired
record, so it yielded no correctness or performance verdict.  The cause is a diagnostic-only scratch alias:
the control's two-set layout and the candidate's four-set layout shared one allocation, allowing candidate phase-B
fixups to overwrite control synchronization flags.

v155 isolates scratch per captured variant and changes no CUDA code.  The v154/v155 CUDA source is byte-identical
(SHA256 `8AA3CF42352FE3679C6E8B90453337818D0EC88A0BB5F5A71EB9166C6E5B8BA2`); the full v155 file SHA256 is
`C8CD1C9EC37902F1F3B1A71C6295F7CA0CC2B378BE6ED14D3D7E3186A16DC1F2`.  `audit_v155_contract.py` passes all
43,520 schedules and both protected hashes.

Replacement H200 check `f03fbceb0f614e7d` completed normally.  The default path passed 5/5 official correctness
cases, and the candidate passed its internal comparisons, but dualfix lost all 21 paired seeds.  Median paired
ratios were 1.0226x at M32, 1.0415x at M128, and 1.0588x at M256 (three-shape mean 1.0416x).  Reject v154/v155:
the second-buffer loads, clears, and finisher work outweigh eliminating one global atomic reduction.  Keep
`OKC_WGSDUAL=0`; do not score, promote, or retry.  The diagnostic payload's incidental unchanged-control aggregate
was 5657.84 us / 2.376966x.  See `V154_SPLIT_FIXUP_PROGRESS.md`.
Six off-board checks and all five scored runs remain for the day.

## 2026-11-02 v152 readiness diagnosis and v153 cyclic tail scout

v152 off-board job `80547849e5e14294` passed correctness and measured the WGS producer/readiness path across seven
seeds at M32/M128/M256.  Weight-producer empty-ring waits were 95.5/143.1/149.4 us and x-producer empty-ring waits
were 70.2/108.8/110.6 us (shape medians), while x-gather readiness was only 3.6/5.8/4.9 us.  `doneAs` waits were
25.1/36.3/35.9 us but remained largely hidden because the x producer was still throttled behind consumers.  This
rules out TMA, gather, ring-depth, and phase-B readiness as the next primary lever.  Baseline CTA completion tails
were A=13.3/18.9/18.7 us and B=7.4/16.1/14.6 us.

v153 replaced fractional stream-K ranges with whole 16-stage units assigned CTA-cyclically.  Exhaustive host audit
proved exact ownership for 1..136 jobs, both phases, and grids 1..160.  Locally it retained 96 registers and identical
HGMMA shapes while reducing spill stores/loads from 44/104 B to 8/8 B and making the fixup path unreachable.

H200 job `8f0923163b4b4bd1` passed all correctness cases, but cyclic lost every one of 21 paired seeds.  Median
candidate/control ratios were 1.0757x at M32, 1.0949x at M128, and 1.0802x at M256 (mean 1.0907x).  Cyclic median
tails expanded to A=22.8/34.8/27.4 us and B=38.1/45.8/34.6 us.  Whole-unit granularity therefore creates far more
imbalance than split/fixup overhead costs.  Do not promote, score, or retry v153.  Preserve the fine-grained,
cost-weighted fractional partition; lower spill counts are not predictive when scheduling balance worsens.

## 2026-11-02 v139 WGS tensor-core/dequant diagnosis (v150-v151)

Scope was deliberately limited to the WGS+KPERM3 consumer arithmetic.  CTA scheduling, cost tables, boundary
snapping, cache hints, TMA geometry/ring depth, CTA roles, and prefetch were not changed.  The production v139 file
remained byte-identical at SHA256 `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`.

### Static architecture result

- `ntt = ceil(job_tokens / 8)` dispatches once to `wgs_seg<PB, t>` for `t=1..8`.
- Every K16 step issues exactly one `WGSMMA<t>` instruction, with PTX shape `m64n(8*t)k16`.  It does **not** loop
  over `t` independent N=8 WGMMA instructions.  One K=256 stage therefore has 16 WGMMA instructions per consumer
  warpgroup; one complete K=4096 output unit has 256, independent of `t` (the instruction width grows with `t`).
- Phase A has 32 256-row output units/job and phase B has 16.  Across a fixed consumer-warpgroup position, that is
  8192 and 4096 WGMMA instructions/job respectively; across all four consumer warpgroups it is 32768 and 16384.
- v139 already declares two A-fragment banks, `af[2][GS][4]`, but it is not a compute/dequant pipeline.  The order is
  dequant current bank -> fence -> issue -> commit -> `wait_group<1>` -> dequant next bank.  Thus the wait precedes
  the next dequant.  A final `wait_group<0>` precedes the epilogue.
- Consequently the requested single-wide-WGMMA Step 2 prerequisite was false; no redundant second implementation
  was added.

### Step 1: seven-seed H200 ablations

Off-board job `4f0f76de8a1040b9` passed 5/5.  Diagnostic source:
`submission_v150_wgs_compute_diag.py`.  Each cell is the median of seven routing-seed medians; paired ratios are the
median of the seven candidate/control ratios.

| Shape | v139 control | no WGMMA, dequant kept | paired ratio | raw-bit A, WGMMA kept | paired ratio |
|---|---:|---:|---:|---:|---:|
| M=32 | 133.0 us | 133.1 us | 0.9969x | 130.8 us | 0.9788x |
| M=128 | 201.0 us | 201.1 us | 1.0027x | 199.9 us | 0.9884x |
| M=256 | 216.6 us | 221.9 us | 1.0153x | 215.6 us | 0.9850x |

The no-WGMMA variant uses a never-taken, opaque-zero trap sink so ptxas cannot delete the dequant.  SASS retains the
control's 4096 HMUL2 and full PRMT count, so its small sink overhead makes the result a conservative bound.  The
raw-bit variant uses four rotations of the packed code word as the four BF16 A registers and removes scale decode,
unpack, and HMUL2.  Ptxas then serializes WGMMA (C7512), so even this deliberately optimistic/incorrect-output bound
saves only 1.2-1.5% at M=128/256.  Tensor-core removal is neutral-to-worse: WGMMA is hidden under the wider pipeline,
not the primary wall-time limiter.

Official aggregate for the unchanged-control payload was 5615.32 us / 2.39497x.  Treating the three paired ratios as
the only affected loads gives diagnostic-only aggregate proxies of 5629.16 us for no-WGMMA (0.25% worse) and
5570.38 us for raw-dequant (0.80% upper-bound improvement, invalid output and serialized WGMMA).

Ptxas/cuobjdump resources for both BIG specializations:

| Variant | Registers | Stack | Spill stores | Spill loads | BF16 WGMMA serialized |
|---|---:|---:|---:|---:|---|
| v139 control | 96 | 48 B | 44 B | 104 B | no |
| no WGMMA / dequant kept | 96 | 32 B | 0 B | 0 B | n/a (BF16 WGMMA absent) |
| raw-bit A / WGMMA kept | 96 | 32 B | 0 B | 0 B | yes, C7512 |

### Step 3: A-fragment overlap

`submission_v151_wgs_afpipe_diag.py` adds `OKC_WGSAFPIPE` (default 0).  When enabled, it leaves group `g` pending,
retires `g-1` with `wait_group<1>`, fills the opposite A bank for `g+1`, and delays the previous stage's x-slot
release until its final group has retired.  Off-board job `3b398913d4d147b1` passed all per-seed candidate/control
checks and the official 5/5 control checks, but the candidate lost consistently:

| Shape | v139 control | A-fragment pipeline | paired ratio |
|---|---:|---:|---:|
| M=32 | 134.1 us | 135.2 us | 1.0048x |
| M=128 | 200.6 us | 201.7 us | 1.0035x |
| M=256 | 216.0 us | 216.4 us | 1.0037x |

The control payload's official aggregate was 5684.40 us / 2.36586x.  Applying the three paired ratios gives a
candidate proxy of 5695.76 us (0.20% worse).  Ptxas stayed at 96 registers but spill stores/loads rose from
44/104 B to 84/176 B; stack remained 48 B and WGMMA did not serialize.  The candidate is rejected and the flag stays
off.  Consumer-warpgroup ping-pong with named barriers was not attempted because the prerequisite register headroom
does not exist: the simpler overlap already hits the 96-register cap and nearly doubles spill traffic.

Checkpoint date: 2026-10-30 (Asia/Calcutta)

## 2026-10-30 repository-candidate promotion (v99)

- Active kernel: `submission.py`, byte-identical to `submission_v99.py`.
- Active/v99 SHA256:
  `B59196AC2365CBA30CC666CC74F38051FF8F5E19C8FBD9E771ADFD4696C40765`.
- Source: the audited Gemini 3.8 Flash/OpenCode `muse_moe_w4a16`
  candidate from Auto-research-agent commit
  `cf8ca67c07f9eaf96e544f1bc4e6ba333936d859`.
- Official scored job: `698ba06f353d47a4`.
- Result: 5/5 passed, **5,701.94 us, 2.358583x**, 95% CI
  2.323204-2.393962x.
- Correctness: aggregate relative L2 approximately 0.255% across all five
  official cases; no source-audit flags.
- Leaderboard immediately after the run: **rank 131/1,180**.
- Previous active recovery point: `submission_v96.py`, SHA256
  `59FB967274560635CA5425EB31F59C784271317425217DBFF4E7A43D501975CA`.
- Previous scored board result: job `c93a5a6e13b54674`, 7,096.58 us /
  1.895069x, rank 482/1,180 immediately before v99.
- Daily quota after v99: 8 scored submissions and 0 off-board checks left.
  Do not spend another scored run on an unvalidated knob change.

The v99 gain comes from a different fused Hopper design, not lower-precision
activations: BF16 activation semantics are retained. Its key mechanisms are
lossless schedule-specific NVFP4 repacking, DQ7 code decode, exact guarded
PRMT E4M3 scale conversion with LUT fallback, shape-specific tiny/mid/WGS
paths, and a persistent fused route/GEMM1/SwiGLU/GEMM2/post-norm kernel.

This file is the short, operational handoff for continuing optimization. For
the longer design history and architecture discussion, see
`IMPLEMENTATION_PLAN.md`.

## Resume state

- Active kernel: `submission.py`, scored v77 polynomial-SiLU composite.
- Reproducible copy: `submission_v77.py`.
- Active/v77 SHA256:
  `71A79837CC7ED8C8E6A7CBF0F650A0FF96583F15AA7F509ECA346740D32CD19B`.
- Last scored control: `submission_v52.py`.
- Last control: `submission_v50.py`.
- v50 SHA256:
  `0918A205BC7291081205DC203359D456BC313BE4D85D503FEA9F9A847B459DA1`.
- Best scored job: `c6413be663324360`, v77.
- Result: 5/5 passed, 7,137.57 us, 1.884184x, 95% CI
  1.855922-1.912447x.
- Prior best: job `09cc04a74eeb4016`, v64, 7,151.14 us / 1.880608x.
- Prior off-board job: `93b4280a6f9a4d9d`, 5/5 passed,
  7,204.6 us, 1.86665x, 95% CI 1.839-1.895x.
- Leaderboard rank was not re-queried after v77; the last observed state was
  rank 387/1,051 with a 5,001.7 us / 2.6888x leader.
- The latest scored run left 1 scored submission and 0 off-board
  checks left.  Re-query the official state before any authorized remote run;
  never spend scored quota without explicit permission.

Do not modify sibling challenge folders. All work described here is confined
to `muse_moe_w4a16`.

## Best scored shape results

| Shape | SGLang | v77 | Speedup | Floor |
|---|---:|---:|---:|---:|
| M=1 decode | 76.51 us | 29.29 us | 2.612x | 11.4 us |
| M=16 decode | 188.02 us | 111.87 us | 1.681x | 52.5 us |
| M=32 decode | 204.78 us | 151.95 us | 1.348x | 70.8 us |
| M=8 verify | 143.46 us | 79.14 us | 1.813x | 37.3 us |
| M=128 verify | 454.99 us | 236.51 us | 1.924x | 101.3 us |
| M=256 verify | 619.66 us | 285.72 us | 2.169x | 108.9 us |

The scored profile is in:

- `profiles/submission_v77.c6413be663324360.txt`
- `profiles/submission_v77.c6413be663324360.json`
- `profiles/submission_v77.c6413be663324360.log`

## Shape dispatch selected by v52 tuning

| M bucket | Selected family/configuration | Synthetic median |
|---:|---|---:|
| 1 | `gk SW=4, KS2=1, MAXM=8` | 36 us |
| 8 | `gk SW=2, KS2=2` | 83 us |
| 16 | `gk SW=2, KS2=2` | 114 us |
| 32 | `gk SW=2, KS1=2, KS2=4, MAXM=32` | 163 us |
| 128 | `wx NCW=8, RPW=2, TPX=32, NSL=2, CPB=2` | 242 us |
| 256 | `wx NCW=16, RPW=1, TPX=64` | 303 us |

The important v52 gain is shape-specific. At M=128 the new two-CTA shape
beat NCW16 by 242 versus 255 us (about 5%). At M=256 it measured 306 us versus
303 us, so the tuner correctly retained NCW16.

## v57 SW=1 result

v57 adds a barrier-free SW=1 epilogue for the small persistent `gk` kernel.
The initial v56 attempt used named barrier `1 + grp`; with 16 single-warp tile
groups this reached illegal barrier ID 16 and poisoned the CUDA context. v57
directly consumes each warp's accumulator fragments, so no group reduction or
named barrier is needed.

Off-board H200 job `93b4280a6f9a4d9d` passed 5/5. The tuner selected SW=1
for both intended buckets:

| M | v52 SW=2 | v57 SW=1 | improvement |
|---:|---:|---:|---:|
| 16 | 116 us | 114 us | 1.7% |
| 32 | 164 us | 157 us | 4.3% |

The aggregate check was 7,204.6 us / 1.86665x. Scored job
`52b146a8c79e431c` then reproduced both selections (113 us and 157 us) and set
the new best at 7,188.2 us / 1.87091x.

## v58-v60 diagnostic and optimization results

v58 measured the selected `gk` path directly. TMA wait fractions were only
2-9%; W13 plus W2 consumed roughly 80-90% of each CTA timeline at about
3.1-3.8 TB/s. Barrier/epilogue tails were only 5-14 us. This rejects deeper
small-path buffering as the primary lever.

The separate large-path diagnostic (`221fa2914c1e4693`) showed the inverse
ring-pressure pattern: NCW16 consumers waited for full slots only about 4%,
while producers waited for empty slots about 55-57%. At M=256, the full path
was about 287 us, stream-only 195 us, and dequant-without-MMA 241 us. Exact
decode and MMA therefore cost roughly 46 us each and dominate the residual.

v59 tested constant-footprint TMA coalescing and an M8 SW=1 extension:

- Larger TMA boxes (`PPS=2/4, SG=1`) were 8-13% slower and are removed.
- `gk SW=1, KS2=2, MAXM=8` repeated a 82 us result versus 84 us for the v57
  M8 path and cleared the tuner's 2% selection gate.
- Cleaned job `f387cb04d4f34ef7` passed 5/5 at 7,150.7 us / 1.88072x.
- The cleaned configuration also reduced prepare time from 157 s to 112 s by
  pruning legacy candidates that had already lost paired tuning.

Scored v59 job `fd0df19766364246` passed 5/5 at 7,193.4 us / 1.86956x
(95% CI 1.8415-1.8976x). The M8 selection repeated at 82 versus 84 us, but
aggregate variance made this sample 5.2 us slower than scored v57. The board
therefore retains v57's 7,188.2 us / 1.87091x as the team best; current rank is
392/1,046.

v60 attempted to double resident large-path consumers with NCW16/WCH2/CPB2.
TPX64 could not compile under the two-CTA 56-register cap (ptxas required at
least 62). TPX32 compiled and passed correctness, but lost decisively: 326 us
at M=128 and 406 us at M=256 versus 241/303 us for the retained paths. This
direction is rejected.

Local sm_90a SASS inspection of the exact E2M1 decode showed that NVCC already
fuses the masks/ORs into `LOP3.LUT`, uses `PRMT`/`LEA` for scales, and lowers
the BF16 multiply trick to `HMUL2.BF16_V2`. A superficial inline-PTX rewrite
will not remove instructions; the next serious step is the expert-scoped
cooperative-cluster W13-to-W2 pipeline described below.

## v61-v63 cluster and W13-to-W2 pipeline checkpoints

v61 added a two-CTA Hopper cluster launch and one initial cluster barrier while
leaving arithmetic and scheduling unchanged. Off-board job
`f23ddc3d80b74dd7` passed 5/5 at 7,157.2 us / 1.87901x. Cluster launch/sync was
effectively free: paired tuner medians were equal within 1 us, and the cluster
candidate was selected at M=256 (302 us versus 303 us for ordinary NCW16).
This is the current active candidate, but it has not received a scored run.

v62 then made each pair of W13 units publish one four-chunk W2 group and
interleaved W13/W2 expert segments. Job `8cc75f860f60481e` passed correctness,
but the candidate regressed to 185/262/364/550/642 us. With one resident CTA
per SM, W2 CTAs occupied the machine while waiting and their queue lookahead
held later work: a scheduler dependency inversion, not a math-kernel issue.

v63 restored all-W13-first scheduling and retained only readiness-based tail
overlap. Job `88a86283aa41448d` passed 5/5, but paired results were neutral or
slightly worse: 106/137/186/257/305 us versus 107/138/184/257/304 us. The extra
256-KiB counter allocation and protocol are therefore rejected. Do not retry
global readiness counters without a scheduler that guarantees runnable W13
work or dedicated cluster producer/consumer roles.

## v64 paired-WGMMA result

v64 batches two consecutive K16 WGMMA instructions into each asynchronous
commit group. It uses two rotating register pairs with `wait_group 1`, keeping
four WGMMA operations in flight while halving `wait_group`, accumulator fence,
and `commit_group` control instructions. Packed weights, arithmetic, shared
memory, and output order are unchanged.

Off-board job `9c38c537be5744a1` passed all 5 correctness cases at 7,159.0 us /
1.87853x. The paired tuner showed the mechanism itself winning:

- M=128 NCW16: 253 us versus 256 us (the existing CPB2 path still won at 244).
- M=256 NCW16: 301 us versus 305 us; `WGPAIR` was selected.

The aggregate is statistically tied with v61's 7,157.2-us check, but v64 has
stronger paired evidence and retains v61's cluster candidate as a fallback.
Scored job `09cc04a74eeb4016` then passed 5/5 at 7,151.1 us / 1.88061x and
reproduced a stronger M256 win (297 us versus 302 us). It replaces v57 as the
team's board entry and ranks 387/1,051.

## v65-v66 medium-path structural experiments

v65 tested two independently scheduled 8-warp `gk` CTAs per SM while keeping
sixteen total consumer warps and one total packed-weight pass. Job
`30b54a813ac64627` passed 5/5, but the extra K splits and FP32 partial traffic
regressed to 127 versus 113 us at M=16 and 178 versus 158 us at M=32. The
paired tuner retained v64's one-CTA configurations.

v66 used a two-CTA Hopper cluster as distributed shared memory. The CTAs held
complementary halves of activation/intermediate rows and owned disjoint full-K
output tiles, removing the M=32 W13/W2 K-partial planes and two grid barriers.
Job `79c90c1589754559` passed 5/5, proving the DSM arithmetic and synchronization
correct, but remote shared loads plus per-chunk cluster barriers regressed to
179 versus 111 us at M=16 and 289 versus 157 us at M=32. Its 7,103.6-us
aggregate was run variance; no DSM candidate was selected. Do not submit or
promote v65/v66.

## v67-v72 structural research

v67 attempted two-CTA multicast of weights for adjacent token passes and of
activations for row-split tails. The first off-board check
`e71209b601db4ad7` failed compilation because `crank` was hidden inside the
device-architecture TMA guard. After moving it to host/device scope, the exact
`NCW16/CLUSTER2/MC2` specialization compiled locally for SM90a, including the
multicast PTX. The corrected off-board job `0cf55af34a184057` then remained in
`running` state with no result beyond the 15-minute client timeout, establishing
a runtime synchronization hang. It fails the correctness/robustness gate and
must not be promoted. The design is additionally weak for buckets where most
experts have only one token pass. Retain v67 only as negative/design evidence.

v68-v69 explored two complete four-warpgroup WGMMA CTAs per SM. Exact local
resource accounting showed TPX16 at about 93 KiB/CTA and TPX32 at about
112 KiB/CTA. Local SM90a compilation then separated viable from nominally
resident forms: TPX16 + `WGPAIR` uses 56 registers without a WGMMA
serialization warning, while the single-op TPX16 form and both TPX32 forms are
serialized by ptxas. v71 retains only the viable TPX16 paired candidate.

v70 introduced `WG4`: two decoded-A register banks, four HGMMA instructions per
commit, and `wait_group 1`. This keeps eight operations in flight instead of
v64's four while preserving arithmetic order and all memory layouts. v71 is the
focused form. Local CUDA 12.4 SM90a results are:

- v64 `WGPAIR`: 96 registers, 20-B spill stores / 80-B spill loads.
- v71 `WG4`: 96 registers, 44-B spill stores / 176-B spill loads, no ptxas
  WGMMA-serialization warning.
- SASS shows four consecutive HGMMA instructions per `WARPGROUP.ARRIVE`,
  alternating A-register banks, with `WARPGROUP.DEPBAR.LE ... 1`.

v72 extended the idea to three banks / 12 operations with `wait_group 2`.
Although it compiled at 96 registers without spills, ptxas explicitly
serialized WGMMA for insufficient register resources. Reject v72; eight
operations is the deepest locally validated asynchronous window.

The explicitly authorized v71 scored job `c8cf0df0e3b74ab0` passed 5/5 but
measured 7,202.4 us / 1.86723x, slower than v64's 7,151.1 us / 1.88061x.
Neither new candidate was selected: WG4 tied/trailed WGPAIR, and TPX16 CPB2
lost badly.  v71 also accidentally removed v64's useful M128 TPX32/RPW2/CPB2
fallback.  The result is therefore rejected and `submission.py` remains v64.

## v73 repaired multicast candidate (local validation only)

v73 returns to v67's structural goal: a two-CTA cluster represents one logical
work unit, multicasting weights for adjacent token passes and activations for
an odd final pass whose row blocks are split across ranks.  The v67 timeout had
two concrete protocol bugs:

- The odd tail allocated `nrb` logical units instead of `ceil(nrb/2)`, so half
  the decoded row blocks were out of range.
- Rank 0 could overwrite the one-entry queue mailbox before rank 1 observed an
  exact epoch, causing rank 1 to wait forever on an epoch that had been skipped.

v73 fixes the tail count and adds an acquire/release acknowledgement before
every mailbox reuse.  Exhaustive host enumeration covers every `(token pass,
row block)` exactly once for `np=1..128` and both `nrb=16/32`, with no duplicate
or out-of-range work.  It retains two focused candidates: TPX16 and TPX32,
both `NCW8/RPW2/CPB2/CLUSTER2/MC2`; v64's complete fallback set remains in the
tuner.

Local CUDA 12.4 SM90a validation:

- TPX16: 96 registers, no spills, no WGMMA serialization warning.
- TPX32: 96 registers, 8-B spill stores / 56-B spill loads, identical to its
  v64 control resource footprint and no WGMMA serialization warning.
- SASS contains six `UBLKCP.S.G.MULTICAST` sites and the expected asynchronous
  HGMMA dependency barriers.
- Python parses, CUDA preprocessor nesting is balanced, and the active
  `submission.py` hash is still v64's
  `F0AD063A8A89FC3E40437E8C5BA946C675BFF1BFC0E3866BC1261B50661DB4E6`.

No remote check or submission has been run for v73.  Its synchronization and
resource evidence are substantially stronger than v67's, but performance is
still unmeasured.  No off-board checks remain at the last quota query.

## v74 exact lookup-dequant candidate (local validation only)

v74 attacks the consumer-side bottleneck directly.  It keeps weights in a
lossless four-bit representation but repacks each 32-bit fragment word as four
ordinary signed E2M1 code pairs.  A generic 128 x 256 codebook maps a positive
E4M3 block scale and one signed code pair to the exact BF16x2 value used by the
existing kernel.  The table is model-independent and 128 KiB; it is not an
expanded copy of model weights.

Exhaustive host validation covered all 119 supported positive normal E4M3
scales, all 256 signed E2M1 code pairs, and all four fragment positions.  Every
lookup word matched the original packed decode and BF16 multiply exactly.
The existing arithmetic configurations remain in the autotuner, so every
bucket has a fallback.

Exact CUDA 12.4 SM90a compilation and SASS inspection:

- M1/M8 `gk`: 128 registers, no stack or spills; 128 `HMUL2` and 68 `LOP3`
  instructions removed, replaced by 128 read-only codebook loads; all 32 HMMA
  instructions are preserved.
- M16 `gk`: 128 registers, no stack or spills; 256 `HMUL2` and 136 `LOP3`
  instructions removed, replaced by 256 read-only codebook loads; all 96 HMMA
  instructions are preserved.
- Full M32 lookup was rejected: its NT=4 specialization spilled 68 B of stores
  and 276 B of loads.  The replacement W13-only hybrid selects LUT-layout
  records only for experts with at most 16 routes and retains arithmetic decode
  for NT3/4 and all W2 work.  It uses 128 registers with zero spills and replaces
  128 of 512 static `HMUL2` sites while preserving all 320 HMMA sites.
- M128/M256 `wx`: 96 registers, no stack or spills; all 512 `HMUL2` instructions
  and 192 `LOP3` instructions are removed, all 128 HGMMA instructions and async
  dependency barriers are preserved.
- A 32-KiB sign-factored table and an inline-PTX address variant were both
  rejected locally because they greatly increased bit manipulation or caused
  spills.

Authorized scored job `523e782e654a42d3` passed 5/5 but measured only
7,974.1 us / 1.68652x.  The lookup kernel was not measured: both `gk` and `wx`
wrappers requested `data_ptr<uint32_t>()` from an intentionally `torch.int32`
lookup tensor, so every such tuner candidate failed with `expected scalar type
UInt32 but found Int` and the tuner selected old fallback kernels.  The scored
payload hash was `9e246ff75b5cd940ebfe6ea896bedb5c5e9f49dab659cd993b7360f2782345b0`.

The local candidate now uses an untyped `data_ptr()` cast only inside LUT builds;
baseline builds set the LUT pointer to `nullptr` and never inspect its dtype.
The K extension bridge also passes both original and LUT layouts for the
zero-spill M32 hybrid.  A torch-free contract audit checks Python syntax, CUDA
preprocessor balance, extension argument counts, dtype isolation, active-v64
immutability, and the repaired multicast invariants.

The repaired v73 multicast scheduler is now combined with lookup in two
large-bucket shapes.  NCW16/WGPAIR preserves the v64 M256 consumer and compiles
at 96 registers, zero spills, with no WGMMA serialization warning.  TPX32
preserves the v64 M128 dataflow and compiles at 96 registers, zero spills, but
ptxas warns that WGMMA is serialized; it is retained only as a tuner candidate.
The spill-prone arithmetic TPX32 multicast form is pruned.  TPX16 arithmetic
multicast remains as a spill-free control.  The merged CUDA differs from v73's
repaired scheduler only by lookup decode and comments.

The audit also caught and fixed a second integration defect before launch:
`ctrX` was still 64 integers even though MC2 mailboxes reach beyond index 1,300;
it now matches v73's 2,048-integer allocation.  Exhaustive schedule enumeration
covers `np=1..128` and `nrb in {16,32}` exactly once with no out-of-range unit.
The audited candidate SHA256 was
`7AB543A0B8E4DB3659844794A221D6FC96AFF6FC1551625F1D835509FBBC3BD0`.
Active `submission.py` remained byte-identical to scored v64.

Authorized scored job `aaf8d72c60624cae` subsequently passed all 5 correctness
tests, but measured **7,181.96 us / 1.87254x** (95% CI 1.84445-1.90063x), below
v64's 7,151.14 us / 1.88061x.  The tuner rejected every new candidate:

- M1 LUT: 108 us versus 36 us arithmetic.
- M8 LUT: 292-294 us versus 82 us arithmetic.
- M16 LUT: 411 us versus 114 us arithmetic.
- M32 hybrid LUT: 431 us versus 158 us arithmetic.
- M128 standalone LUT: 835 us versus 246 us TPX32; multicast+LUT: 881-1,494 us.
- M256 standalone LUT: 936 us versus 299 us NCW16/WGPAIR; multicast+LUT:
  985-1,486 us.
- Arithmetic multicast also lost: TPX16 was 310/418 us at M128/M256 and
  NCW16/WGPAIR multicast was 277/332 us versus 252/299 us controls.

Thus the codebook-load hypothesis is rejected: removing `HMUL2`/`LOP3` from
SASS did not predict runtime because the dependent read-only loads introduced a
much larger latency/scoreboard cost.  The job selected unchanged v64 fallbacks
for all six buckets; the small scored difference is run variance.  v64 remains
the best and active implementation.  Job payload content hash:
`ad69abd72675ab03412c066d98429ef6e1608bf9f0d6dac119c65cf153bd4591`.

The six scored v64 latencies total 912.96 us. Reconstructing the reference sum
from the scored table gives 1,714.13 us, so 2.05x requires about 836.16 us: a
76.80-us (8.41%) reduction.  Saved ablations had suggested enough decode
headroom, but v74 demonstrated that replacing arithmetic decode with dependent
table loads cannot realize that bound.

## v75 Hopper dynamic consumer-register allocation (local validation only)

NVIDIA's PTX guidance led to the Hopper `setmaxnreg` mechanism used by
warp-specialized CUTLASS kernels.  The first design padded the lone TMA producer
to a complete warpgroup and attempted producer-to-consumer redistribution.
Exact SM90a compilation rejected the low producer allocation because this fused
kernel's common setup needs the ordinary register floor.  The focused design is
both smaller and stronger: only the already-complete WGMMA consumer warpgroups
execute `setmaxnreg.inc`; the single producer executes no register-reconfiguration
instruction and the original 17-warp CTA geometry, queue order, and fused
norm/combine epilogue remain unchanged.

For the NCW16/WGPAIR M256 path, each of 16 consumer warps grows from the
compiler's 96-register allocation to 120.  Total demand is
`(16 * 120 + 1 * 96) * 32 = 64,512` registers, below H200's 65,536-register SM
file.  Exact CUDA 12.4 `sm_90a` evidence is:

- v64: 96 registers, 24-byte stack, 20-byte spill stores, 80-byte spill loads;
  SASS has 3 `STL` and 12 `LDL` sites.
- v75: base allocation 96 plus `USETMAXREG.TRY_ALLOC.CTAPOOL ... 0x78`, zero
  stack and zero spills; SASS has no `STL`/`LDL`.
- Tensor work is preserved exactly: both cubins contain 128 HGMMA, 512 HMUL2,
  64 `WARPGROUP.ARRIVE`, and 68 `WARPGROUP.DEPBAR` instructions.
- The REGRECONF=0 fallback reproduces v64's 96-register and 20/80-byte spill
  report.

The analogous CPB2 M128 experiment was pruned.  A 112-register consumer target
removed its 8-byte stores / 56-byte loads, but ptxas then serialized WGMMA; 104
registers avoided serialization but did not reduce spills.  v75 therefore adds
only the focused NCW16/WGPAIR tuner candidate.  `audit_v75_contract.py` verifies
the source/ABI/budget invariants and that active `submission.py` remains the
scored v64 hash.  Candidate SHA256:
`BC7424D97F4D54A32DF9F4748C302321F3F522AE26536DE9D84429E064596C20`.
No remote check or scored submission has been run for v75.

The subsequently authorized scored job `9ae81acb80274f40` disproved the pool
availability assumption.  Preparation completed in 148 seconds, but the job
then timed out during autotuning with no correctness records.  Exact SASS shows
`USETMAXREG.TRY_ALLOC.CTAPOOL ... 0x78` in a retry loop and no release
instruction.  Because the consumer allocation could not be satisfied, the
consumers remained in that loop while the producer reached the following CTA
barrier.  v75 is rejected; the physical-register arithmetic above is necessary
but not sufficient evidence for availability in Hopper's per-CTA pool.

## v76 CTA-wide epilogue pointer shadow (scored result)

Line-mapped v64 SASS showed that its 3 `STL` and 12 `LDL` sites are not WGMMA
accumulator spills.  ptxas hoists `g1`/`g2`/`nw` epilogue address calculations
across the persistent consumer loop and stores that long-lived state in a
24-byte per-thread local frame.  v76 adds an opt-in NCW16/WGPAIR candidate that
stores those three base pointers once per CTA in 24 bytes of shared memory and
reloads them only inside the epilogue.  Existing configurations default to the
unchanged code.

Exact CUDA 12.4 `sm_90a` comparison:

- v64/v76 control: 96 registers, 24-byte stack, 20-byte stores, 80-byte loads;
  SASS has 3 `STL`, 12 `LDL`, 286 `LDS`, and 170 `STS` sites.
- v76 EPISHADOW: 96 registers, zero stack/spills; SASS has no `STL`/`LDL`,
  298 `LDS`, and 173 `STS` sites.  The three new shared stores are issued only
  by thread zero, replacing per-thread spill stores.
- Both have 128 HGMMA, 512 HMUL2, 64 `WARPGROUP.ARRIVE`, and 68
  `WARPGROUP.DEPBAR` instructions; v76 contains no `USETMAXREG`.

`audit_v76_contract.py` verifies the active/control v64 hash, opt-in default,
CTA geometry, pointer initialization ordering, and tuner control.  Candidate
SHA256: `69CF44B8D8F801337EAFF876BD7E45A105CDE91332A8180259044BA7E67A7E90`.

Authorized scored job `0b36057067044648` passed 5/5 at 7,178.23 us /
1.873512x (95% CI 1.84541-1.90161x).  This is 27.09 us (0.38%) slower than
v64's best.  The paired tuner did show the intended local effect: EPISHADOW
measured 298 versus 301 us for NCW16 at M=256 and was selected, while M=128
retained CPB2 at 244 us.  The mechanism is real but too small to improve the
aggregate reliably; v64 remains active.

## v77 reciprocal-free SiLU candidate (scored winner)

v77 keeps v64's exact SiLU candidates and adds a focused `POLYSILU` form for
each selected gk bucket, M=128 CPB2, and M=256 EPISHADOW/WGPAIR.  It evaluates
`sigmoid(x)` as a saturated odd polynomial and removes both the exponential
and reciprocal from the W13 epilogue.  The M=256 form stacks this with v76's
validated pointer shadow; CPB2 does not use EPISHADOW because an exact compile
showed its 8-byte stack / 1 STL / 7 LDL footprint was unchanged by the shadow.

Local numerical evidence:

- A compute-isolated CUDA benchmark measured 4.40 ms for polynomial SiLU
  versus 8.34 ms for the exact implementation (47% faster).
- Scalar relative L2 error was 0.94% for normal gates and 0.69% for the
  outlier/skewed distribution.
- Five reduced-dimension end-to-end NVFP4 MoE simulations measured combined
  output error of 0.96-0.99% on normal inputs and 0.72-0.73% on skewed inputs;
  worst-row error stayed below 1.16%, versus the judge's 7.5% row limit.

Every selected CUDA 12.4 `sm_90a` specialization compiles without a WGMMA
serialization warning.  Registers, stack/spills, HMMA/HGMMA counts, and async
barriers match their controls.  Depending on the specialization, POLYSILU
removes 4-60 MUFU sites and 2-30 reciprocal sites in exchange for 6-90 FFMA
sites.  M=256 retains v76's 96-register, zero-stack/zero-spill footprint.
`audit_v77_contract.py` checks syntax, CUDA preprocessor balance, exact
fallbacks, tuner controls, hashes, and deterministic scalar error bounds.
Candidate SHA256:
`71A79837CC7ED8C8E6A7CBF0F650A0FF96583F15AA7F509ECA346740D32CD19B`.

Authorized scored job `c6413be663324360` passed 5/5 at **7,137.57 us /
1.884184x** (95% CI 1.855922-1.912447x).  This improves v64 by 13.57 us
(0.190%).  The paired tuner selected POLYSILU at M=1 on a 36-us tie and at
M=32 by 160 versus 161 us.  It retained exact-SiLU controls for M=8/M=16,
M=128 CPB2 (241 versus 243 us), and M=256 EPISHADOW (293 versus 294 us).
The approximation is therefore a real but narrow win, not the projected
1.91-1.96x structural gain.  v77 is now active; one scored submission remains
and must not be spent without explicit permission.

## Current architecture

### Weight preparation

- Weights remain lossless packed NVFP4.
- Prepare repacks E2M1 codes and E4M3 block scales into MMA/WGMMA fragment
  order; it does not expand weights to BF16/FP8.
- `w1x`/`w2x` are the WGMMA-oriented layout used by the large path.
- Prepare compiles candidate extensions in parallel and tunes the six scored M
  buckets using cold-L2 CUDA graph replay.

### M <= 32: persistent `mma.sync` (`moe_k` / `gk`)

- One persistent CTA per SM, sixteen warps.
- TMA ring streams packed 2,304-byte weight pieces.
- E2M1/E4M3 decoding stays in consumer registers.
- W13, SwiGLU, W2, RMSNorm statistics, normalization, and top-8 combination
  are fused into one kernel launch.
- SW=2 remains useful at M=8. v57 uses barrier-free SW=1 at M=16/32, removing
  the reduction fan-in and making all sixteen warps independent tile groups.

### M=128: two-CTA WGMMA path

- Two persistent CTAs per SM (`CPB=2`).
- Each CTA has eight consumer warps, `RPW=2`, `TPX=32`, two ring slots.
- Each activation tile is reused across two 128-row weight blocks.
- Per-block shared memory at M=257 is 114,440 bytes; two blocks request
  228,880 bytes per SM.
- A runtime `cudaOccupancyMaxActiveBlocksPerMultiprocessor` check is mandatory:
  the monotone grid barriers deadlock if all `2 * SM_count` CTAs are not
  concurrently resident.

### M=256: one-CTA WGMMA path

- One persistent CTA per SM.
- Sixteen consumer warps (`NCW=16`), `RPW=1`, `TPX=64`.
- One producer warp and three TMA ring slots.
- Stable expert route ranks are built once and reused for the fused activation
  gather; there is no separate `xgather` launch.

## Experiments already ruled out

Do not repeat these without a materially different mechanism.

### NCW=24 / six consumer warpgroups (v51)

- Correct and resource-safe, including exact tail masking for 64/32 output
  blocks and variable TMA transaction bytes.
- Scored job: `0462e0cce19241f1`, 5/5, 1.844x / 7,293.2 us.
- Tuner result: M=128 304 us versus NCW16 256 us; M=256 365 us versus 304 us.
- Conclusion: adding WGMMA consumers inside one CTA increases contention and
  does not supply the missing latency hiding.

### Two-CTA `RPW=1, TPX=64`

- Correct and resident, but slower: 268 us at M=128 and 323 us at M=256.
- Conclusion: separate CTAs only win when paired with RPW=2 activation reuse
  and the smaller TPX32 footprint.

### CPB2 WCH=2 / three-stage ring (v53)

- Correct: off-board job `1d7b859dcd8f4397` passed 5/5.
- M=128 was 260 us versus 240 us for WCH4/NSL2; M=256 was 332 us versus
  303 us for NCW16.
- Conclusion: doubling slot/barrier turnover costs more than the deeper ring
  hides.

### Three CTAs per SM / one slot (v54-v55)

- v54 exposed a single-slot protocol deadlock: consumers deferred releasing
  slot N until slot N+1 was full, while the producer needed N released before
  it could fill N+1. Job `3b5d62616e8d4150` timed out.
- v55 added an exact single-slot handoff (`wgmma.wait_group 0`, then release).
  Job `0569248592d344b7` passed 5/5.
- CPB3 measured 260 us at M=128 versus 241 us for CPB2, and 320 us at M=256
  versus 304 us for NCW16. The loss of ring overlap and 75-register/thread
  ceiling outweigh the third CTA.

### Other negative results

- NCW16/RPW2 512-row units: extra live state outweighed activation reuse.
- Additional WGMMA N=24/48 dispatch (`N8STEP`): neutral or slightly slower.
- More TMA ring stages on the small path: no gain.
- Moving dequantization into dedicated producer warps: fewer compute warps and
  extra shared traffic caused a regression.
- Pre-expanded BF16 weights/scales: extra traffic and storage defeat the packed
  NVFP4 design and risk violating the weight-precision rule.
- More K split on the ring kernel: consistently slower at medium/large M.
- Separate activation gather: v50 fusion was better.
- Raising `gk MAXM` directly: impossible for M=128 because that kernel stages
  every token activation in shared memory.

## Recommended next experiments

### 1. Expert-scoped cooperative pipeline

v77 established that reciprocal-free SiLU is sound but wins only one bucket
decisively (M=32 by 1 us); further polynomial tuning is not a credible large
lever.  No check quota remains and no scored run is authorized by this
checkpoint.

v61 proves cluster launch/residency is safe, while v62-v63 prove that global
readiness counters alone do not help. A future pipeline must give cluster CTAs
fixed producer/consumer roles (or a scheduler with guaranteed runnable W13
work), retain W2 accumulators across arriving K groups, and avoid occupying all
SMs with dependency-waiting CTAs. Preserve one weight read per active expert
and the fused norm/reduction epilogue.

## Validation and submission workflow

1. Copy the current active file to a new versioned candidate; never overwrite
   `submission_v50.py` or `submission_v52.py`.
2. Make edits only in the candidate.
3. Parse Python and calculate shared-memory/thread/register ceilings.
4. Preserve `submission.py` as the best validated version until a remote run
   is authorized.
5. Prefer `okc check` after the daily reset. Use a scored attempt only with
   explicit authorization.
6. Read both the scored table and the autotuner line. A better overall score
   caused by run variance is not enough; the new configuration must actually
   be selected and win its paired bucket.
7. After a rejected candidate, restore active `submission.py` to the last
   validated version and retain the candidate as negative evidence.

Useful commands from WSL:

```bash
cd /mnt/host/d/OpenKernelChallenge/muse_moe_w4a16
../bin/okc problems --json
../bin/okc check submission.py
../bin/okc submit submission.py
../bin/okc leaderboard muse_moe_w4a16
```

## Files to preserve

- `submission_v50.py`: fused-gather NCW16 control.
- `submission_v51.py`: rejected NCW24/tail-mask experiment.
- `submission_v52.py`: last scored CPB2/SW2 control.
- `submission_v53.py`: rejected WCH2/three-stage CPB2 experiment.
- `submission_v54.py`: original single-slot CPB3 deadlock reproduction.
- `submission_v55.py`: correct but slower CPB3 experiment.
- `submission_v56.py`: original SW1 illegal-barrier reproduction.
- `submission_v57.py`: scored SW1 control, 7,188.2 us.
- `submission_v58_diag.py`: small-path phase diagnostic.
- `submission_v58w_diag.py`: large-path wait/ablation diagnostic.
- `submission_v59.py`: cleaned M8 SW1 winner and last scored code.
- `submission_v60.py`: rejected doubled-consumer experiment.
- `submission_v61.py`: validated two-CTA cluster checkpoint.
- `submission_v62.py`: correct but rejected interleaved pair pipeline.
- `submission_v63.py`: correct but neutral all-W13-first tail pipeline.
- `submission_v64.py`: prior scored paired-WGMMA winner and exact-SiLU control.
- `submission_v65.py`: correct but rejected two-CTA-per-SM medium kernel.
- `submission_v66.py`: correct but rejected DSM-cluster medium kernel.
- `submission_v67.py`: locally compiled multicast-cluster candidate; remote
  job `0cf55af34a184057` hung, so it is rejected design evidence.
- `submission_v68.py`: initial full-width TPX16 CPB2 experiment.
- `submission_v69.py`: TPX16/TPX32 CPB2 resource exploration.
- `submission_v70.py`: initial two-bank/four-op WGMMA grouping experiment.
- `submission_v71.py`: focused unscored candidate; locally compile/SASS validated.
- `submission_v72.py`: rejected three-bank form; ptxas serialized WGMMA.
- `submission_v73.py`: locally validated repair of the v67 two-CTA multicast
  protocol; not remotely checked or submitted.
- `submission_v74.py`: corrected exact lookup candidate for spill-free M1/M8,
  M16, hybrid M32, and large lookup/multicast candidates; no second remote run.
- `audit_v74_contract.py`: torch-free integration and exhaustive MC2 mapping audit.
- `submission_v75.py`: rejected setmaxnreg candidate; scored job timed out.
- `submission_v76.py`: correct pointer-shadow candidate; scored but 0.38% slower.
- `audit_v76_contract.py`: v76 source and active-hash audit.
- `submission_v77.py`: scored polynomial-SiLU winner; job
  `c6413be663324360`, 7,137.57 us / 1.884184x.
- `audit_v77_contract.py`: v77 source, fallback, hash, and numerical audit.
- `submission.py`: currently identical to `submission_v124_wgs_final_norm_broadcast.py`; v101 remains the public
  leaderboard best at 2.3847x.
- `IMPLEMENTATION_PLAN.md`: full research/design history.
- `OPTIMIZATION_CHECKPOINT.md`: this resumption checkpoint.

## v125 two-CTA / 128-row WGS diagnostic (rejected)

`submission_v125_wgs_2cta128_diag.py` tested whether finer CTA-level parallelism could improve the newer WGS
stream-K path.  It preserved the scored v101 control and compiled an untimed diagnostic specialization with two
resident CTAs per SM, eight consumer warps plus two producer warps per CTA, 128-row/two-warpgroup units, two ring
slots, and `FT_T=32`.  The candidate reused the existing lossless 256-row packed copy by selecting either adjacent
pair of 9-KB warpgroup pieces; `audit_v125_contract.py` exhaustively verifies that packed-byte, output-row, and
phase-A-to-phase-B readiness mappings are unchanged.

CUDA 12.4 `sm_90a` compilation measured 96 registers/thread, a 48-byte stack with 8-byte spill loads/stores, no
`USETMAXREG`, and no WGMMA serialization warning.  Exact dynamic shared memory was 92,416 bytes per CTA, so two
blocks request 184,832 bytes per SM.  The runtime occupancy check requiring two active blocks passed on H200.

Authorized diagnostic job `7998c36fbcea4ed2` passed all five scored-control correctness tests.  The paired untimed
H200 diagnostic was decisively negative:

| shape | v101 control | v125 CPB2 | regression |
|---|---:|---:|---:|
| M=32 | 166.2 us | 193.8 us | 16.6% |
| M=128 | 208.7 us | 257.5 us | 23.4% |
| M=256 | 210.1 us | 273.6 us | 30.2% |

The second TMA/weight pipeline does not compensate for halved work units, doubled CTA scheduling, the reread final
pass, and additional `FT_T=32` job fragmentation.  Do not promote or repeat this two-CTA half-unit design without a
mechanism that removes those costs.  `submission.py` remains unchanged and the leaderboard best remains v101 at
2.3847x.  Local v125 SHA256: `54C67F7842FC297B18EE049AFA6E80F253BE829E41D7512B52DF474B39C2CDB7`.

## v126 independently signaled 18-KB WGS halves (rejected)

`submission_v126_wgs_halfstage_diag.py` tested a finer-grained version of the existing one-CTA/four-warpgroup WGS
pipeline.  The 36,864-byte stage remains in the exact v101 packed layout, but the producer issues it as two adjacent
18,432-byte TMA copies.  Warpgroups q0/q1 and q2/q3 have independent weight-full/weight-empty barriers, while the
activation copy has a separate shared completion barrier.  This was intended to let the first warpgroup pair start
loading weights while the second half was still in flight, without the severe CTA fragmentation of v125.

`audit_v126_contract.py` verifies the immutable v101 control hash, Python/preprocessor balance, the diagnostics-only
macro, the 18-of-24 barrier map, the 1/8/1/16 producer/consumer arrival counts, and exact two-half coverage of the
original packed stage.  CUDA 12.4 `sm_90a` compilation used 96 registers/thread and emitted no WGMMA serialization
warning.  Against an identical local v101 compile, the candidate added 40 bytes of spill stores and 48 bytes of spill
loads (v101 124/244 bytes, v126 164/292 bytes); both retained a 24-byte stack.

Authorized guarded diagnostic job `c6c4ff9b708a4e7b` passed all five scored-control correctness tests and all static
checks.  The candidate was slower in every paired untimed H200 measurement:

| shape | v101 control | v126 half-stage | regression |
|---|---:|---:|---:|
| M=32 | 165.8 us | 167.3 us | 0.9% |
| M=128 | 208.5 us | 211.5 us | 1.4% |
| M=256 | 210.3 us | 213.2 us | 1.4% |

The extra barrier wait, address arithmetic, and spill pressure cost more than the small overlap unlocked by signaling
the first weight half early.  Do not promote this design.  The scored part of the job still ran the unmodified v101
control (2.3517x on that noisy run), `submission.py` remains SHA256
`57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`, and the leaderboard best remains v101 at
2.3847x.  Local v126 SHA256: `D41EBD81C109DA504F3321692516ED7A0E22942AEB53DF1B9008DC9AB5A67707`.

## v127-v130 FT32 spill-removal branch (correct M32 micro-win, not promoted)

An exact production compile of the v101 WGS specialization with the runner's `sm_90a`, optimization, and feature
defines measured 96 registers/thread, a 24-byte stack, 132 bytes of spill stores, and 252 bytes of spill loads.  The
largest accumulator specialization was the source: setting `OKC_FT_T=32` halves the maximum job from eight N tiles
to four.  The same one-CTA/four-warpgroup kernel then compiled at 96 registers/thread with zero stack and zero spills,
and no WGMMA serialization warning.  Existing job splitting already makes this transformation lossless; the Python
scratch allocation was originally sized to cover its larger `JMAX=136` schedule.

`submission_v127_wgs_ft32_diag.py` compiled the FT32 module only for untimed diagnostics and added a hot-expert
adversarial parity case in which every expert must be split across multiple jobs.  `audit_v127_contract.py` verifies
the immutable v101 hash, exhaustive contiguous partitioning for route totals 1..3072, and every scratch bound.  Job
`d49dc5b394dd4c74` passed 5/5 and the hot-split case at relative L2 `1.960e-04`.  FT32 was 0.8% faster at M=32, but
2.3% slower at M=128 and 4.8% slower at M=256, so only M=32 remained eligible.

`submission_v128_wgs_ft32_multiseed.py` then measured that M=32 choice across nine alternating-order seeds with 60
repetitions per module.  Job `a7553c367be340ae` passed the adversarial parity case at relative L2 `2.172e-04`; FT32
won 8/9 seeds with a median candidate/control ratio of `0.99167` (0.83% faster), range `0.97881..1.00903`.  This is
a real, repeatable bucket-level optimization rather than a one-run timing fluctuation.

`submission_v129_wgs_ft32_m32.py` is the clean production candidate: it compiles FT32 beside the normal WGS module
and selects it only for `M == 32`; all other dispatches are byte-for-byte v101 logic.  `audit_v129_contract.py` passes
the immutable-control, exact dispatch, route partition, and scratch checks.  Scored job `ccd49a8f53914da1` passed
5/5 but measured 5,717.7 us / 2.35209x.  A behavior-identical, comment-only rerun artifact,
`submission_v130_wgs_ft32_m32_rerun.py`, also passed 5/5 in job `42fd507f9d684a73` but measured 5,711.4 us /
2.35466x.  Its M=32 median was 122.9 us, consistent with the paired diagnostic, but the optimization touches only
one of six scored shapes and cannot overcome whole-run variance reliably enough to beat the 5,639.5-us public best.

Do not spend more scored attempts on this isolated FT32 change and do not promote it by itself.  Preserve it as a
compatible M=32 micro-optimization that can be combined with a larger independent win.  `submission.py` remains
unchanged at SHA256 `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`; v101 remains the
public best at 5,639.5 us / 2.3847x (rank 122 at the time of this checkpoint).  Candidate hashes are v129
`18C068A1E26FC2DDA078F1385CE62460EE6E00AFFF6B4FB93AA1D40DFC4F12C8` and v130
`465BF3A0AAC308B206A833AD5B3A871C06C517891A52B8C13D6E9C0817255930`.

## v131 intermediate FT-width sweep (correct, rejected for large shapes)

`submission_v131_wgs_ft_sweep.py` tested whether FT40/48/56 could trade FT64's accumulator spills for finer job
parallelism without FT32's large increase in repeated expert-weight passes.  The original switch instantiated every
NTT=5..8 specialization whenever `FT_T > 32`; exact compile guards were required so narrower arrays did not also
instantiate unreachable wider WGMMA templates.  With those guards, local CUDA 12.4 `sm_90a` builds measured:

| width | registers | stack | spill stores / loads |
|---|---:|---:|---:|
| FT40 | 96 | 0 B | 0 / 0 B |
| FT48 | 96 | 0 B | 0 / 0 B |
| FT56 | 96 | 8 B | 4 / 20 B |

No build emitted a WGMMA serialization warning.  `audit_v131_contract.py` proves the immutable v101 control, exact
FT64 switch behavior, exhaustive route partitioning for totals 1..3072, and all job-table/fixup/x/int scratch bounds.
Job `91b6af638ecc4de7` passed 5/5 and adversarial hot-split parity for all widths at relative L2 below `2.3e-04`.
Seven-seed paired medians (candidate/control) were:

| shape | FT40 | FT48 | FT56 |
|---|---:|---:|---:|
| M=128 | 1.01313 | 0.99887 | 0.99840 |
| M=256 | 1.05266 | 1.01916 | 1.00202 |

The spill reduction cannot repay repeated weight passes at M=256.  FT56 is effectively neutral, not a production
gain.  Do not promote or repeat this width sweep.  The scored portion was the unchanged control and measured
5,740.2 us / 2.34285x.  Candidate SHA256:
`A56583771FCA4EED78A6650E0585F4358A0EE3EFB83D4DBCF66944AFDE3AF89D`.

## v132 composed micro-wins with final broadcast (broadcast rejected)

`submission_v132_wgs_composite_diag.py` composed FT32 and snap=1 stream-K boundaries with a final-pass change that
computed each route's normalization factor once and broadcast it through the CTA reduction row.  All features were
default-off so the scored module preprocessed to v101.  `audit_v132_contract.py` verifies the default control paths,
monotone snap mapping, exhaustive FT32 route coverage, and scratch bounds.  Local builds retained 96 registers and
asynchronous WGMMA, but the broadcast raised stack frames to 32 B (FT32 composite) and 64 B (FT64 composite).

Job `6eb38caf25ba4d5a` passed 5/5 and both adversarial parity cases.  Nine-seed medians were 0.99704 at M=32, 1.00230
at M=128, and 1.00515 at M=256.  The large composite lost 8/9 M128 seeds and all 9 M256 seeds: repeated route loads
and `rsqrtf` were cheaper than the broadcast's increased live state/stack traffic.  Reject the broadcast completely.
The unchanged scored control measured 5,687.8 us / 2.36444x.  Candidate SHA256:
`C1A57132F2DB00DBC2ABA26B3C477E0E27659D86362DB44D2EB7455F6B035293`.

## v133 FT32 + snap production composite (correct, no public promotion)

`submission_v133_wgs_ft32_snap.py` removed the rejected broadcast and composed only independently positive pieces:
FT32 plus snap=1 at M=32, FT64 plus snap=1 for M>=64, and the unchanged v101 paths elsewhere.  Local FT32+snap
compiled at 96 registers with zero stack/spills; FT64+snap retained the control's 96 registers, 24-byte stack, and
124/244-byte spill stores/loads.  `audit_v133_contract.py` verifies exact guarded dispatch, absence of the broadcast,
monotone snapped ownership, exhaustive FT32 partitioning, and all scratch bounds.

Final daily-quota job `df47bfde2f624aa3` passed all 5 correctness tests but measured 5,769.7 us / 2.33088x (95% CI
2.29592..2.36585x), so it did not beat the 5,639.5-us v101 public best.  Do not promote it on aggregate-score hope;
retain it as a correct composition checkpoint for future paired retesting after quota reset.  Candidate SHA256:
`4E45D3BA95812D1AE32EB568B4A8BDC92B3053B6AFB9FD9F99CF19DF3841303C`.

At the end of this branch, scored and check quota are both exhausted (10/10 and 20/20).  `submission.py` remains
unchanged at SHA256 `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`; the public best remains
v101 at 5,639.5 us / 2.3847x, rank 124 at the final query.

## v134 paired-job/shared-weight WGS prototype (local proof complete; remote diagnostic queued)

`submission_v134_wgs_pairjob_diag.py` implements the next structural direction behind default-off
`OKC_WGSPAIRJ`.  A normal FT64 job is split inside the existing four consumer warpgroups: q0/q1 own token tiles
0..3 and q2/q3 own tiles 4..7, while q0/q2 share one 9-KB row-weight piece and q1/q3 share the other.  Each logical
row unit is halved, so a stage copies two contiguous pieces (18,432 B) instead of all four (36,864 B); the two token
halves reuse those pieces.  This preserves one physical expert-weight pass rather than FT32's two passes, while no
warpgroup emits an HGMMA wider than N=32.

The doubled logical-unit map is phase A 64 units/job and phase B 32 units/job.  Pair-unit parity selects pieces
0/1 or 2/3 of the original packed stage, `r16` advances by 8, and the second warpgroup pair adds four n-tiles to its
x descriptor and epilogue token index.  Jobs with <=32 tokens leave q2/q3 compute-idle, but those eight warps still
acquire each full stage and contribute their weight-empty/x-empty arrivals.  Phase-B readiness is correspondingly
8 phase-A releases for one token half or 16 for two.  The job-width x ring and fixup strides remain FT64-sized;
only the per-warpgroup accumulator width is capped at four tiles.

`audit_v134_contract.py` passes.  It exhaustively checks counts 1..64, both phases' physical weight-piece and logical
token-half coverage, exact `r16` coverage, doubled stage geometry, 8/16 `doneAs` thresholds, all 16 ring-barrier
arrivals, macro-off scored dispatch, and shared-memory sizing.  CUDA 12.4 `sm_90a` compilation also passes with no
WGMMA serialization warning.  The paired cubin contains only HGMMA N widths 8/16/24/32 (128 static instructions of
each width); no N=40..64 specialization is emitted.

Local resource evidence versus the exact v101-style control compile:

| build | registers | stack | spill stores / loads | weight+x dynamic ring |
|---|---:|---:|---:|---:|
| control | 96 | 24 B | 132 / 252 B | 208,896 B |
| v134 pairj | 96 | 32 B | 56 / 124 B | 153,600 B |

Thus the architecture removes 55,296 B of dynamic shared memory and roughly halves spill traffic, but paired
scheduling state prevents the completely spill-free FT32 profile.  This is still a materially better local resource
shape and is worth one H200 diagnostic; it is not yet a production win.  The embedded reset-ready diagnostic keeps
`kernel_fn` on macro-off v101, checks forced 32/64/256-route expert cases against the control, then measures seven
alternating-order M=256 seeds.  Do not submit until quota resets, and do not promote unless parity passes and the
paired median is clearly below 1.0.

Candidate SHA256: `FD4AC53D2AA1F436311DAC5BECA7453717A007C80E15DDA4C2B3F7357C2C2A63`.
`submission.py` and the immutable public artifacts remain unchanged.

### v134 scored-submission status

The user explicitly authorized a scored v134 submission.  Immediately before submitting, `okc problems --json`
still reported `muse_moe_w4a16` at 10/10 scored runs and 20/20 checks used.  The subsequent command
`okc submit submission_v134_wgs_pairjob_diag.py --json` was rejected before job creation with HTTP 429:
`daily cap of 10 runs on muse_moe_w4a16 reached, counting runs not yet queued`.  The service reported that the cap
resets at 00:00 UTC (6 h 17 min after this attempt).  No job ID was created and no additional quota was charged.

After the reset, the next authorized action is exactly one scored submission of
`submission_v134_wgs_pairjob_diag.py`.  First re-run `audit_v134_contract.py` and confirm candidate SHA256
`FD4AC53D2AA1F436311DAC5BECA7453717A007C80E15DDA4C2B3F7357C2C2A63`; then submit without modifying the file.
The scored dispatch remains macro-off v101 while prepare-time diagnostics validate and time `pairj`.  Read both the
five correctness results and the `[diag]`/`[pair-summary]` log.  Do not promote pairj into production unless forced
32/64/256-route parity passes and the seven-seed M=256 candidate/control median is clearly below 1.0.

## v135-v137 WGS dynamic-register experiment (measured and rejected)

`submission_v135_wgs_regwg_diag.py` tested a complete fifth warpgroup for the two WGS producer warps plus two
padding warps.  All four producer-group warps execute `setmaxnreg.dec 56`; the four consumer warpgroups execute
`setmaxnreg.inc 112`.  Launch size is 640 threads, while named barrier 2 deliberately retains the original 576
live participants.  The static budget is 64,512 of 65,536 registers.  Exact CUDA 12.4 `sm_90a` compilation removed
the control's 24-byte stack and 132/252-byte spill stores/loads completely, retained 96 statically allocated
registers, and preserved all 1,024 HGMMA instructions and their eight N-width counts.  Nevertheless off-board check
`e2599a034a22405c` timed out after compiling in 51 seconds: the 112-register CTAPOOL request never completed.

`submission_v136_wgs_reg104_diag.py` reduced the consumer target to 104 registers, increasing unused pool margin
from 1,024 to 5,120 registers while preserving the zero-stack/zero-spill cubin.  Off-board check
`c26e045c42054c3c` passed all five correctness cases and finished at 5,671.32 us / 2.37132x, with the scored path
still using the untouched v99 control.  Seven alternating-order cold-L2 seeds at each WGS shape rejected both
diagnostic variants:

| variant | M=32 median ratio | M=128 | M=256 | 21-case mean |
|---|---:|---:|---:|---:|
| register GS2 | 1.0058x | 1.0032x | 1.0022x | 1.0034x |
| register GS4 | 1.0129x | 1.0187x | 1.0191x | 1.0173x |

Thus eliminating local spill traffic is real but does not repay the extra 64 launched threads and register-pool
handshake.  Do not promote either path.  `submission_v137_wgs_reg104_noinline_diag.py` explored outlining the two
producer functions after SASS showed high numbered registers in the inlined producer path; local compilation raised
the control stack from 24 to 320 bytes and spill loads from 252 to 400 bytes, so v137 was rejected before a remote
run.  `submission.py` remains unchanged.  v136 SHA256:
`C28DBAADC03853F702D37FD8CB810F92E3D2E5393AD9911419264BF81CDBAA56`.

## v134 H200 paired-job verdict (conclusively rejected)

After quota reset, the frozen v134 candidate was re-audited before any remote work.  `audit_v134_contract.py`
passed and SHA256 remained exactly
`FD4AC53D2AA1F436311DAC5BECA7453717A007C80E15DDA4C2B3F7357C2C2A63`.  Off-board H200 check
`1a7c01fb72a34a7e` then passed all five harness correctness cases and all three forced diagnostic parity cases:

| forced batch | relative L2 | max absolute error |
|---:|---:|---:|
| 32 | 2.279e-4 | 3.906e-3 |
| 64 | 2.972e-4 | 7.812e-3 |
| 256 | 1.597e-4 | 7.812e-3 |

Performance is nevertheless decisively negative.  Across seven alternating-order M=256 seeds, paired-job lost
7/7, with ratios `1.83722, 1.80519, 1.79706, 1.80588, 1.85302, 1.81201, 1.78663`.  The median ratio is
`1.80588` (range `1.78663..1.85302`), meaning v134 is about 80.6% slower than the control.  This is not a noisy or
borderline result: the shared-weight/token-half schedule is rejected and must not receive a scored submission.

The check itself reported 5,740.54 us / 2.34272x because the scored `kernel_fn` deliberately remained on the
macro-off control path; that aggregate number is not evidence for pairj.  The decisive result is the embedded
candidate/control diagnostic above.  `submission.py` remains unchanged at SHA256
`57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`.
This verdict supersedes the earlier post-reset scored-submission authorization in the v134 status section.

### Next gate after v134

Stop the WGS resource-shape branch: v125, v126, v131, v134, v135-v137 collectively show that smaller jobs,
earlier half-stage release, narrower accumulator widths, paired token halves, and spill elimination do not improve
the large-shape critical path.  The next credible experiment must preserve FT64/wide HGMMA execution and one
physical expert-weight pass.  The remaining structural direction is an expert-scoped fixed-role W13-to-W2
pipeline with guaranteed runnable W13 work; do not revive global readiness counters or dependency-waiting CTAs.

Build it diagnostic-only and require: forced parity pass; at least 6/7 paired wins at both M=128 and M=256;
median candidate/control ratio <=0.97 for each shape; and no scored submission until those gates pass.  If a local
resource/schedule proof cannot plausibly meet that bar, retain v101 and stop this optimization branch.

## v138 fixed-role W13-to-W2 pipeline (correct, conclusively rejected)

`submission_v138_wgs_fixed_roles_diag.py` tested the final structural gate above without changing `submission.py`.
The first cohort owns all phase-A/W13 stage ranges and never waits on phase B; the remaining cohort owns all
phase-B/W2 ranges and waits on the existing per-job/per-stage `doneAs` releases.  This guarantees progress and
preserves FT64, wide HGMMA, one physical weight pass, and W2 accumulators across the arriving K stages.  Two splits
were tested on the 132-SM H200: role3 uses 88 W13 / 44 W2 CTAs, while role4 uses 99 / 33.

`audit_v138_contract.py` proves exact phase coverage and physical fixup-slot ownership for grids 2..160 and mixed
job widths.  Exact CUDA 12.4 `sm_90a` builds preserved all eight HGMMA widths and all 1,024 static HGMMA sites with
no serialization warning.  The schedule specialization even reduced local spill traffic:

| build | registers | stack | spill stores / loads |
|---|---:|---:|---:|
| control | 96 | 64 B | 132 / 252 B |
| role3 (2:1) | 96 | 64 B | 88 / 216 B |
| role4 (3:1) | 96 | 48 B | 12 / 68 B |

Off-board H200 job `0c6df4d3a8254c53` passed all five harness cases and all six forced parity checks.  Forced
candidate/control relative L2 error was `1.571e-4..3.368e-4`.  Seven alternating/permuted-order seeds then rejected
both schedules decisively:

| variant | M=128 median ratio | wins | M=256 median ratio | wins |
|---|---:|---:|---:|---:|
| role3 | 1.58513 | 0/7 | 1.64534 | 0/7 |
| role4 | 1.74981 | 0/7 | 1.81661 | 0/7 |

The expected overlap did not materialize: a phase-B stage requires all eight phase-A row contributions, and the
cost-weighted W13 stream distributes those contributions broadly.  Most W2 CTAs therefore remain dependency-waiting
until late in W13, after which each phase runs on only its reduced cohort.  The measured runtime is consequently
close to the sum of the two reduced-SM phase times.  Fewer spills are again irrelevant to the critical path.

Do not score or promote v138.  This closes the fixed-role/global-readiness pipeline branch: making readiness fine
enough would require duplicating/reordering expert work and would abandon the load balance that makes v101 fast.
Candidate SHA256: `26BC89EB6874F0A5213021E50E14CFDF8352AED23C2ABFFB4DC4D18A1FCFD6E2`.
`submission.py` remains unchanged at SHA256
`57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`.

## v139 KPERM3 lossless K re-layout (local proof and SM90a compile complete)

submission_v139_wgs_kperm3.py implements a lossless 4x4 transpose inside every 64-wide K group.  For
old K = block*16 + q*8 + lane2*2 + e, the matched weight and activation layout uses
new K = lane2*16 + q*8 + block*2 + e.  A consumer lane therefore stays on one original K16 block and reuses its
two row-scale bytes across four WGMMA k16 steps.  Codes, E4M3 scale bytes, global scales, and every dequantized value
are unchanged.  The phase-A input gather assembles each permuted 16-byte chunk with four coalesced 32-bit loads;
the phase-A epilogue writes the intermediate directly into the same permuted layout for W2.

audit_v139_kperm3.py proves the 4096-element mapping is bijective and involutive, checks the exact CUDA gather and
weight-lane formulas, proves all 1,024 stage scale bytes are preserved, and verifies randomized dot-product identity
with deliberately distinct scales in each original K16 block.  A direct CPU tensor check also confirmed that the
Python packer performs exactly new(step=j,lane=t) = old(step=t,lane=j) and leaves the scale byte region unchanged.

Exact CUDA 12.4 sm_90a control/candidate builds from compile_v139_static.py preserve all eight HGMMA widths and
all 1,024 HGMMA sites, with no WGMMA serialization warning:

| build | registers | stack | spill stores / loads | PRMT | IMAD | LDG |
|---|---:|---:|---:|---:|---:|---:|
| control | 96 | 64 B | 132 / 252 B | 1,094 | 6,716 | 492 |
| KPERM3 | 96 | 48 B | 44 / 104 B | 326 | 5,966 | 540 |

Thus KPERM3 removes 88 B of spill stores, 148 B of spill loads, 768 static PRMTs, and 750 static IMADs, at the cost
of 48 extra static LDGs in the activation gather.  This is a substantially better resource/instruction shape, but
the gather cost and cold-L2 behavior still require an H200 check before promotion.  No remote check or scored run was
spent during this local experiment.  Candidate SHA256:
A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09.
submission.py remains unchanged at SHA256
57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38.

### v139 remote H200 check and scored submission

The user explicitly authorized one off-board check followed by one scored submission of the frozen v139 hash.
Check job b45b872cd8644c9f passed all 5 correctness cases with no flags or rejections and measured 5,673.48 us /
2.37041x (95% CI 2.33486..2.40597x).  This was 33.98 us slower than the prior 5,639.5-us public best.

The exact same content hash was then submitted as scored job 13cca0e4992b4983.  It again passed all 5 cases with no
flags or rejections and measured 5,627.32 us / 2.38986x (95% CI 2.35401..2.42571x), provisionally improving the
prior public best by 12.18 us (0.216%).  The leaderboard query reported rank 133 of 1,290 at 2.3899x.

The check/run disagreement is 46.16 us (0.82%), much larger than the apparent 12.18-us win, and both intervals
overlap the prior result.  Treat v139 as the provisional board best but not a demonstrated structural speedup yet;
do not replace the active recovery file or spend another attempt without an explicitly requested paired rerun.
After these jobs, 14 checks and 9 scored runs remain for muse_moe_w4a16 today.

## v140-v149 KPERM follow-up verdicts (Nov 1)

The follow-up branch preserved `submission.py` at SHA256
`57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38` and did not spend another scored run.
The frozen v139 file remains SHA256 `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`.

- v140's vector activation transpose retained v139's register/spill shape and reduced static LDGs 540 -> 492, but
  its paired 32-bit scale-load composition raised spills to 188/408 B and was rejected locally.  The clean
  x-shuffle-only v146 later passed 5/5 in job `ed354cdb11cd4c1a` but measured 5,678.1 us; M256 regressed to 212.0 us.
- v141's step-major shared-code layout removed compiler spills (96 registers, 32-B stack) but increased LDS
  1,540 -> 2,308.  Job `dca04f051b1b415f` passed 5/5 at 5,657.2 us; the WGS buckets were effectively neutral.
- v142 changed the WGMMA commit group from 2 to 4.  It removed spills locally, but job `ba1de01ef44f4b46` passed
  at 5,649.8 us with M128/M256 regressions to 178.0/213.6 us.  The GS2 overlap is better.
- v143's four-weight/two-activation split ring fit in shared memory but job `b346ef20f53b4ec8` failed correctness
  after bs4 (NaN/Inf at bs33).  Host shared-memory sizing was correct; the two-deep activation reuse protocol is not.
- v144 composed `OKC_GATHB=4`; job `1c4a784fdb144d32` passed at 5,680.5 us, with M32 at 127.5 us.  Reject.
- v145 composed FT32 at M32 and snap=1 for WGS shapes.  Job `06056a2205b04276` passed at 5,684.8 us; KPERM changes
  the old interaction and both M32 (128.9 us) and M128 (179.3 us) regressed.  Reject.
- v149 ran a seven-seed paired sweep of mild/medium/strong steeper scheduler tables against an exact v139 SASS
  control.  Job `7cfd6c94f1cf4af0` passed at 5,643.0 us.  Best median ratios were 0.99906 at M32, 0.99829 at M128,
  and 1.00644 at M256; stronger curves lost all 7 M256 seeds.  No table is worth promotion.  The earlier v147
  diagnostic setup failure (`a2393b02ff6e446c`) was caused by missing PRMT suffix propagation and is superseded.

These results close the KPERM micro-tuning branch: instruction-count/spill improvements did not translate to the
critical path, and the prior FT/snap/gather/cost knobs do not compose positively.  The live leaderboard now reports
v139 at rank 144, 5,627.3 us / 2.38986x.  Rank 80 is 2.420x, approximately 5,557.2 us: a 70.1-us (1.261%) gap.
The service reports 9/10 scored runs and 4/20 checks remaining today.  Since v139's scored 95% interval reaches
2.42571x, repeated scored runs could cross the cutoff through measurement variance, but require explicit user
authorization; do not spend them automatically.

## v150-v156 structural follow-up (Nov 2)

The producer/readiness profile in v152 showed the WGS producers are consumer-throttled; TMA, gather, ring depth,
and readiness waits are not the next lever.  Whole-unit cyclic assignment in v153 passed correctness but lost all
21 paired cases by 7.6-9.5%.  The corrected isolated dual-edge-fixup v155 passed all five default cases but lost all
21 paired timings: median ratios were 1.0226 at M32, 1.0415 at M128, and 1.0588 at M256.  v156 then exhaustively
audited 217,600 static unit-final bias schedules; every positive bias worsened the modeled maximum load and added
split edges, so no CUDA candidate or H200 run was justified.  These branches are rejected.  Their H200 checks left
6 checks and 5 scored runs available; no score was consumed.

## v157 M32 segmented final pass (local gates complete)

`submission_v157_wgs_finalseg_diag.py` is a default-off scout for the remaining credible M32 lever: use 64 or 128
CTAs instead of 32 for final normalization/output.  Each CTA owns a disjoint column segment, writes one partial per
route into the existing double-buffered ss scratch, joins a race-free second grid barrier, and writes its output
segment.  M != 32 and undersized grids fall back to v139.  No allocation or scratch-size change is required.

Host ownership/barrier audit passed.  Exact SM90a builds show the default-off control is SASS-identical to frozen
v139.  FSEG2/FSEG4 preserve 96 registers and the complete 1,024-site HGMMA mix; their stack remains 48 B and spill
traffic is 4/12 B versus v139's 44/104 B.  Protected v139 and `submission.py` hashes are unchanged.  Candidate hash:
`587AA8F4540764D0D7F6F0203887AF92D267D2877BF7E25193CAC1FDD32A7E6E`.

No H200 check or scored run has been spent.  The next action requires fresh explicit authorization for one isolated
diagnostic check; promotion requires correctness across M32/M33/M128/M256 and M32 median ratio <= 0.985.  A scored
submission remains separately gated.

### v157 H200 verdict (correct, rejected)

The user authorized one check, job `6bc4bb3bfaea45e1`.  It passed all five official correctness cases on H200 with
no flags or rejections, and both embedded diagnostic invocations completed their M32/M33/M128/M256 correctness
comparisons.  Performance rejected both candidates in both invocations:

| sweep | variant | M32 | M33 | M128 | M256 |
|---:|---|---:|---:|---:|---:|
| 1 | FSEG2 | 1.0076x | 1.0044x | 1.0016x | 1.0042x |
| 1 | FSEG4 | 1.0113x | 1.0102x | 1.0018x | 1.0052x |
| 2 | FSEG2 | 1.0111x | 1.0049x | 1.0046x | 1.0048x |
| 2 | FSEG4 | 1.0101x | 1.0057x | 1.0002x | 0.9991x |

The targeted M32 path is consistently 0.76-1.13% slower: the second grid barrier and global partial round trip
outweigh the additional CTA parallelism.  Do not score or promote v157.  The macro-off aggregate check result was
5,645.78 us / 2.38204x and is not evidence for the candidates.  `submission.py` and frozen v139 remain unchanged.
The service now reports 5 checks and 5 scored runs remaining today.

## v158 phase-coupled CTA permutation (host-rejected)

`analyze_v158_phase_permute.py` reconstructed frozen v139's exact cost-weighted A/B stage ranges, phase-A
trailing-partial reorder, split/epilogue costs, and per-stage B readiness.  It evaluated 93 fixed bijections over
3,456 representative M32/M128/M256 schedules, including independent per-shape mapping selection.  A source audit
corrected B readiness to adjacent A units `2*st` and `2*st+1`.  The best shared mapping, `aff25_66`, was neutral at
every median but reached 1.00255x at worst-shape p90 and 1.01682x in the worst case.  Per-shape selection also
failed: best p90 ratios were 1.00000x, 1.00242x, and 1.00168x for M32, M128, and
M256.  The existing A tail is only about 2.5-2.8% of launch time, while B readiness re-couples permuted ranges to
their original A producers.

Reject v158 without a CUDA build or remote run.  No check or score was consumed; quota remains 5/5.  Full evidence
is in `V158_PHASE_PERMUTE_PROGRESS.md`.  Protected v139 and `submission.py` hashes remain unchanged.

## v159 phase-A dependency-pair tail reorder (host-rejected)

The corrected source contract shows v139 already executes the true adjacent dependency pairs in order.  v159
therefore tested only the feasible residual change: when the trailing partial belongs to an odd A unit, move its
complete preceding even mate next to it.  `analyze_v159_pairtail.py` tested partial-first and mate-first orders over
the same 3,456 schedules.  Both preserved exact stage coverage and identical per-CTA A finish times, and both gave
the same result: median and p90 1.00000x at every shape, with worst ratios 1.00425x at M32, 1.00731x at M128, and
1.00781x at M256.  Earlier pair readiness almost never changed the launch-critical CTA and occasionally delayed an
earlier dependency.

Reject v159 without a CUDA build or remote run.  No check or score was consumed; quota remains 5/5.  Full evidence
is in `V159_PAIRTAIL_PROGRESS.md`; protected v139 and `submission.py` remain unchanged.

## v160 paired phase-B norm-weight load (correct, performance-rejected)

`submission_v160_wgs_epinorm_diag.py` losslessly repacks each BF16 post-norm block from `[0..7,8..15]` to adjacent
`(g,g+8)` pairs during untimed preparation.  The phase-B epilogue replaces two `LDG.16` operations with one
`LDG.32` and reconstructs the original BF16 values exactly before the unchanged FP32 `1 + weight` arithmetic.
The extra representation is 8 KB.  The official path remains macro-off; an embedded seven-seed diagnostic enables
the candidate and requires bit-exact output equality.

Host mapping/bit audits pass.  Exact SM90a builds preserve 96 registers, the 48-byte stack, 44/104-byte spills, no
WGMMA serialization, and all 1,024 HGMMA sites.  Candidate SASS reduces LDG 540 -> 492 and IMAD 5,966 -> 5,952,
with unchanged SHFL/PRMT/HMUL2/LDL/STL counts and four additional FADDs.  Two subgroup-broadcast alternatives were
rejected because they increased LDG, shuffle, and integer instruction counts.

The frozen candidate hash is `1911260BA9E70C12F9B5DA44DE97F2A0A9BA8CD2C542E67CC998F9E8DE5960FD`.
Authorized H200 check `3d76440380a84917` passed all five official cases, and all 21 embedded candidate/control
comparisons were bit-exact.  Its paired candidate/control medians were 1.0004x at M32 (3/7 wins), 1.0007x at
M128 (2/7), and 0.9989x at M256 (4/7).  M128 and M256 therefore failed both predeclared promotion requirements:
at least 5/7 wins and median <=0.995.  Reject v160 without a score or promotion.  The official macro-off payload
of 5,642.55 us / 2.38341x is control-only, not candidate evidence.  Quota is now 4 checks and 5 scored runs.
Full evidence is in `V160_EPINORM_PROGRESS.md`; protected v139 and `submission.py` remain unchanged.

## v161/v162 M16 SKT16 follow-up (both diagnostic candidates rejected)

Host analysis over 1,024 M16 seeds found that 33.98% contain a local expert above eight routes, but sixteen-token
jobs reduce mean expert-weight passes by only 3.30% (12.3125 -> 11.9482).  Exact SM90a compilation of the existing
generic SKT16 path nevertheless cleared the resource gate at 148 registers with zero stack/spills, versus 145 for
the active helper-warp MID8 path.  A separate exhaustive bit-layout search proved no four-rotation tiling exists;
the frozen multiply decoder is already at the eight-instruction lower bound for this representation.

Authorized check `60cba60d31014d88` compiled v161 but stopped on its first embedded parity assertion.  SKT16 versus
MID8 had relative L2 `2.049e-7`, but the harness incorrectly required bit-exact equality.  The failure occurred in
prepare before official correctness or any paired timing, so v161 supplies no performance verdict.  Frozen v161
SHA256: `4832D00FCA83D81782528F3B056C2063D80BDB2207CC3BA77F1DE6E19F3AAA51`.

Authorized v162 check `d64205beaa584d68` compiled both variants in 33 seconds.  Its first seed passed the revised
parity gate at relative L2 `2.896e-7`, but SKT16 saved no pass (`jobs8=jobs16=10`) and regressed from `88.2 us` to
`107.0 us` (`1.21291x`).  The second seed reached relative L2 `9.531e-5` and tripped the `1e-5` diagnostic gate,
so setup stopped before official cases or the remaining timings.  A delayed CLI response produced duplicate job
`80e50e6bce204efd`; it did not launch a second evaluation.

Reject v162.  Do not spend another check merely relaxing its assertion: the completed timing already disproves
the unconditional SKT16 dispatch for no-savings histograms.  A successor would need a host profitability gate and
an audit against the task's actual numerical tolerance before off-board evaluation.  No score was consumed; quota
is now two checks and five scored runs.  Full evidence is in `V161_MID_SKT16_PROGRESS.md`; protected v139 and
`submission.py` remain unchanged.

The v163 follow-up modeled the only legal profitability gate: a device-side selector between compiled MID8 and
SKT16 paths, because the task forbids host branching on route values during graph replay.  Across 16,384 modeled
M16 routings, 68.292% saved zero jobs, 30.310% saved one, 1.379% saved two, and 0.018% saved three.  Applying the
measured equal-job penalty (`107.0 / 88.2 = 1.21315x`) leaves only 0.8118% of routings below the `0.995x` promotion
threshold, with a modeled mean gain of just 0.03747% across all M16 calls.  Reject v163 before CUDA implementation:
the coverage is below noise and cannot justify dual-path code, extra second-tile shared memory, or a remote check.
The SKT16 branch is closed; quota remains two checks and five scored runs.

## v164 WGS-only polynomial SiLU (correct, performance-rejected)

A fresh source-attributed v139 disassembly found 216 of 233 MUFU sites per WGS
kernel template in the phase-A exact SiLU epilogue. v164 substitutes the
previously validated saturated polynomial only in that WGS path; M1/M8/M16
remain exact and the diagnostic defaults off. The mixed-grid scalar audit gives
0.267579% relative L2 and 0.053391 maximum absolute error. Earlier v77
end-to-end evidence for the same approximation was below 1% combined-output
error on normal cases and below 1.16% on every row.

Exact SM90a builds preserve 96 registers, the 48-byte stack, 44/104-byte
spills, all 1,024 HGMMA sites, and no WGMMA serialization. The candidate
removes 432 of 466 aggregate MUFU sites across the two templates, while the
macro-off control SASS is byte-identical to freshly compiled frozen v139.
Protected v139 and `submission.py` hashes remain unchanged. Candidate hash:
`3A941F1244B27D2CFAB9EFBF5AEE6B91A65F397046AAA5B4FC7DACD606CFB9C2`.

Authorized H200 check `a85addbaf44340c3` passed all five official cases and all
21 embedded output comparisons. Maximum candidate/control relative L2 was
0.9994% and maximum row error was 1.1305%. Paired candidate/control medians
were 0.9976x at M32 (4/7 wins), 0.9994x at M128 (4/7), and 0.9999x at M256
(5/7); the mean of all 21 ratios was 0.9989x. This fails the predeclared M32
<=0.995x and overall <=0.9975x gates. The 5,701.33 us / 2.35884x aggregate is
the macro-off control and is not candidate evidence.

Reject v164 without a score or promotion. The experiment closes exact-SiLU
special-function latency as a dominant WGS bottleneck: removing 432 MUFU sites
is numerically safe but timing-neutral. Full evidence is in
`V164_WGS_POLYSILU_PROGRESS.md`; protected v139 and `submission.py` remain
unchanged. Quota is now one check and five scored runs.

## Session handoff

The current resume snapshot is `RESUME_NEXT_SESSION.md`. It records the frozen
baseline hashes, remaining quota, v164 H200 verdict, closed branches, artifacts,
and the recommended next critical-path audit. Resume there before making new
changes or consuming remote resources.
