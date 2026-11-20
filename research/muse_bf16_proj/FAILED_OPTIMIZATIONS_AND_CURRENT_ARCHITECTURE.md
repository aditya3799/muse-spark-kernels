# Muse BF16 projection: current architecture and failed optimization ledger

Last consolidated: 2026-11-02, after v146.

This is the single handoff for the production architecture and the optimization
directions that have already been tested. It is intentionally more compact than
the chronological research logs, but it covers every recorded experiment family
and every numbered candidate from the current v23-v146 sequence. When a result
here needs forensic detail, use the linked source notes and the matching file in
`profiles/`.

## Executive status

- Protected production file: `submission_cgemv_threads_v105.py`.
- Protected SHA256:
  `B703EBF929E9958718E0C44ECC17B2825406BBFB9E190012A8C7ACE3782CCCD5`.
- Best scored result: `3089.9113966319087 us / 1.5634105253845498x`, job
  `31fc48f9642b40c5`. This is a fallback-only v145 run; no v145 route enabled,
  so its improvement over v105 is measurement variance rather than a new
  production mechanism.
- Current leaderboard snapshot: rank 146.
- Rank 80 snapshot: `3064.525022276282 us / 1.5764x`; the current best must
  save `25.386374 us` or `0.8216%` to reach it.
- Rank 70 snapshot: `3059.3066207167035 us / 1.5791x`; the current best must
  save `30.604776 us` or `0.9905%` to reach it.
- Daily quota after v146: `20/20` checks used and `10/10` scored attempts used;
  both reset at `00:00 UTC` (`05:30 IST`).
- Do not edit v105 in place. Make append-only or separate candidate files and
  recheck the protected hash before and after experiments.

The score is noisy enough that a fallback-only result is not evidence of a new
optimization. A candidate must select its new route, pass all correctness and
resource gates, and show enough projected whole-workload gain to matter.

## Problem contract

Every timed call computes `out = x @ W.T` and returns a fresh CUDA tensor.
`prepare(weights)` is untimed and may repack the fixed weights. `kernel_fn` is
captured and replayed as a CUDA graph with a cold L2 before every timed replay.

| Layer | Input / weight type | Weight shape `[N,K]` | Output type |
|---|---|---:|---|
| `qkv_gate` | BF16 / BF16 | `[2304,8192]` | BF16 |
| `o` | BF16 / BF16 | `[8192,1024]` | BF16 |
| `pooler` | BF16 / BF16 | `[4096,8192]` | BF16 |
| `unpooler` | BF16 / BF16 | `[8192,4096]` | BF16 |
| `router` | BF16 / FP32 | `[256,8192]` | FP32 |

Scored token counts are `m = 1, 8, 16, 32, 128, 256`. The scored m=1
unpooler output is the first 1024 rows, so its active shape is
`[1,4096] x [1024,4096].T`.

The strict points of the contract are:

- BF16 projection weights must remain BF16 or be compressed losslessly.
- Router FP32 weights may use the approved BF16 high-plus-low residual split,
  with FP32 accumulation.
- Probe rows require exact or very close coefficient recovery; lossy weight
  formats cannot be hidden by acceptable average error.
- `kernel_fn` may branch on shape/configuration, but not on input values, and it
  may not synchronize the host or reuse results derived from earlier inputs.

## Current v105 architecture

### Layered composition

v105 is not one universal GEMM. It is a guarded portfolio assembled in layers:

1. The main implementation prepares raw, transposed, compressed, TMA, WGMMA,
   cuBLASLt, and workspace representations and autotunes legal candidates.
2. An embedded alternate implementation owns a fixed set of shapes that was
   proven faster during the v87 composition work.
3. The v102 wrapper adds the QKV m=128 half-width BF16 DSMEM transport and keeps
   the previous route as its immediate fallback.
4. The v105 wrapper adds m=1 C12 GEMV thread-count tournaments and keeps v102 as
   its immediate fallback.

The important consequence is that a failed outer experiment normally does not
damage production dispatch: it is rejected in `prepare`, and `kernel_fn` falls
through to the protected inner implementation.

### Exact BF16 C12 representation

Small dense weights use a lossless 12-bit main stream plus rare exact escapes:

- one low byte per BF16 coefficient;
- a four-bit code containing the sign bit and a three-bit offset inside an
  eight-exponent window;
- a per-CTA escape table containing the original BF16 bits for values outside
  that window.

The decoder reconstructs the common coefficient from the low byte and code.
An independent helper warp restores the rare exceptions in parallel with the
main work. The final result uses the original BF16 value exactly. This helper
overlap is essential: v135 and v137 showed that serial or compute-warp-owned
correction is slower even when occupancy improves.

For m=1, `cg_compress` produces row-major low-byte/code planes and per-row-pair
escape tables. The stable base configurations are:

| Layer | Rows/CTA | Base compute threads | v105 tournament candidate |
|---|---:|---:|---:|
| QKV | 2 | 256 | 512 |
| O | 2 | 64 | 32 |
| Pooler | 2 | 512 | already the selected production geometry |
| Sliced unpooler | 2 | 128 | 256 |

The QKV 512-thread route repeatedly cleared the 3% adjusted-core gate. The O
and unpooler alternatives repeatedly lost and fall back. Selection is still
guarded at preparation time rather than assumed from a single measurement.

For m=8/16/32, the main SK2 path uses swapped operands, whole-K scheduling with
no cross-CTA reduction, route-specific row/tile geometry, tensor-core MMA, and
balanced QKV row assignment by escape count. The packed stream is arranged in
the exact order consumed by each kernel, avoiding a generic runtime transpose.

### Router representation

- m=1 retains the original FP32 `[256,8192]` matrix and uses a direct FP32
  GEMV-style route.
- Other router routes prepare
  `hi = W.to(BF16)` and `lo = (W - hi.float()).to(BF16)` and accumulate both
  terms into FP32 logits.
- m=8/16/32 use the embedded small-router kernels.
- m=128/256 use the one-wave TMA/WGMMA/DSMEM family, with prepared concatenated
  high/low layouts and an in-kernel FP32 reduction/output.

This is already faster than the currently published vLLM raw-FP32 and BF16x3
router paths for the exact Muse shapes, so those ports are closed.

### Shape dispatch

The table describes ownership rather than promising one immutable kernel name;
the main-library rows can still be selected by the preparation-time tournament.

| Shape bucket | Current owner / dataflow |
|---|---|
| Dense m=1 | Exact C12 `cgemv`; v105 thread tournament for QKV/O/sliced-unpooler; fixed 2x512 pooler |
| QKV m=8/16/32 | Main SK2 exact-C12, swapped operands, whole-K scheduling |
| O m=8 | Embedded alternate route |
| O m=16/32 | Main SK2 route |
| Pooler m=8/16 | Embedded alternate route |
| Pooler m=32 | Main SK2 route |
| Unpooler m=8/16 | Embedded alternate route |
| Unpooler m=32 | Main SK2 route |
| Router m=8/16/32 | Embedded alternate high-plus-low routes |
| QKV m=128 | v102 half-width BF16 DSMEM candidate when its gates pass; otherwise embedded alternate QKV128 |
| O m=128/256 | Embedded alternate wide route |
| Pooler m=128/256 | Main wide portfolio: custom TMA and tuned library plans |
| Unpooler m=128/256 | Main wide portfolio: transposed/non-transposed layouts, custom TMA, and tuned library plans |
| Router m=128/256 | Main one-wave TMA/WGMMA/DSMEM route using BF16 high plus low and FP32 output |
| Untuned or unsupported shapes | Nearest safe bucket only when valid; otherwise the exact PyTorch/library fallback |

The main wide portfolio contains custom GQ/GQX/RK kernels, TMA kernels,
transposed and non-transposed weight copies, cuBLAS/cuBLASLt choices, and exact
fallbacks. The old generic wide-BF16 custom kernel is disabled because it never
beat the library on m=33..256.

### `prepare(weights)` lifecycle

Preparation performs work that would be illegal or too expensive in the timed
call:

- compile the Hopper CUDA extension(s);
- retain raw exact weights and build the required transposes;
- build C12 planes, escape tables, row permutations, TMA maps, router high/low
  layouts, workspaces, and graph-safe counters;
- prepare the embedded alternate state;
- run deterministic correctness checks, exact one-hot coefficient probes,
  finite/FP64 checks, resource/occupancy queries, and cold-L2 paired graph
  tournaments;
- enable a candidate only when its per-route and whole-score gates pass.

The most common gates are zero local memory/spills, valid H200 occupancy,
exact probe recovery, FP64 relative error within the layer limit, a raw timing
win, and an adjusted-core improvement of 2-3% after allowing about 4 us for
graph replay overhead.

### `kernel_fn(data, state)` lifecycle

`kernel_fn` reads only the shape/configuration and the prepared route decision,
allocates the required output, launches the selected graph-safe kernel, and
returns the result. It does not inspect input values and does not mutate the raw
weights. Fallbacks are nested and immediate.

### `__deepcopy__` status

There is no custom `__deepcopy__`. The prepared state is a plain dictionary of
tensors and metadata. Python/Torch deep copy gives copied tensors independent
storage, and the timed kernel does not mutate raw weight tensors. Adding a
custom copier would only affect untimed preparation/copy traffic and has no
credible inference-performance benefit. Do not treat it as an optimization
lead unless a concrete aliasing bug is first demonstrated.

## Result-state legend

- **Retained**: a mechanism became part of a later production ancestor.
- **Superseded**: useful development, but the whole candidate is no longer a
  valid starting point.
- **Closed**: correctness, H200 timing, or score projection rules out the
  mechanism. Do not repeat nearby parameter-only variants.
- **Environment-blocked**: the required legal runtime/compiler is absent from
  the judge image. Do not spend another score probing the same dependency.

## Optimization ledger

### Foundations and imported baselines

| Candidate family | Mechanism | Outcome |
|---|---|---|
| `submission_original.py`, `submission_coldtune.py` | Independent scalar split-K, WMMA, vectorized FP32 router, cp.async/TMA/WGMMA, dense cuBLASLt tuning | Correct but far below the public-kernel lineage. Best fully local branch was about `1.1657x`; scored TMA/WGMMA branch about `1.148x`. **Superseded.** |
| `submission_openkernels_v11.py`, `submission_openkernels_v22_local.py`, `submission_openkernels_latest.py` | Public OpenKernels C12, router, WGMMA, and library tuning | Important ancestor, but slower than Nathen/v50/v87/v105. **Superseded.** |
| `submission_nathen_v18.py` | Nathen Fernandes shape-specialized implementation | `3200.28 us / 1.509493x`; supplied several retained route ideas, but the full file is below v105. **Superseded, not a failure to re-port wholesale.** |
| `submission_wider_n_v23.py` | Wider-N independent WGMMA direction | Did not close the gap to the public implementation; later shape-specialized routes were superior. **Closed as a generic custom-WGMMA direction.** |

Other early or unnumbered artifacts are accounted for as follows:

| Files | What they represent | Status |
|---|---|---|
| `submission_router_2n.py`, `submission_router_2n512.py`, `submission_router_concat.py`, `submission_router_concat_tiny_unpool.py`, `submission_router_kconcat.py`, `submission_router_lt_tuned.py`, `submission_router_m16_m32_v2.py`, `submission_router_triton.py` | Early router layout, concatenation, library, Triton, and small-m experiments | Superseded by the exact raw-FP32 m1 and high-plus-low shape-specialized router lineage. Do not retry these files as-is. |
| `submission_bf16_streaming_v2.py`, `submission_dispatch_v3.py`, `submission_structural_triton.py`, `submission_tinygemm_m8.py` | Early generic streaming/dispatch/Triton/TinyGEMM alternatives | Did not beat the imported C12 plus Hopper-specialized lineage. **Superseded/closed.** |
| `submission_baseline_1p096.py`, `submission_scored_1p0904.py`, `submission_scored_1p1478.py` | Historical score snapshots | Evidence artifacts, not current optimization candidates. |
| `submission_poolerwide_diag_v35a.py`, `submission_pooler_workspace_diag_v36a.py`, `submission_pooler_workspace64_v36b.py`, `submission_pooler_workspace64_v36c.py` | Diagnostic and width/workspace variants of v35-v36 | Covered by the v35-v36 closure below; no separate retry path. |
| `submission_auto_research_dense_v87_repeat.py` | Comment-only repeat of v87 executable behavior | Variance sample, not a separate optimization. |

### v24-v53: hybrid composition and shape-parity work

| Versions / files | Experiment | Outcome |
|---|---|---|
| v24-v26: `submission_router_hybrid_v24.py`, `submission_all_hybrid_v25.py`, `submission_dedicated_router_v26.py` | Mix original/public dense and router paths; isolate dedicated router | Developmental stepping stones. None is a current base. **Superseded.** |
| v27: `submission_integrated_v27.py` | Integrated validated routes | Reached `3254.691 us / 1.484257x`; later parity work beat it. **Superseded.** |
| v28: `submission_final_v28.py` | Faster transposed router-32 plus register-A dense WGMMA | Router-32 improved, but all three dense register-A kernels failed coefficient recovery because the packed lane/register layout was wrong; fallback preserved correctness. `3310.653 us / 1.459168x`. **Dense packing closed; router idea retained later.** |
| v29-v33: `submission_qkv128_v29.py`, `submission_qkv_o32_v30.py`, `submission_qkv_o32_u16_v31.py`, `submission_qkv_o32_u16_u32_v32.py`, `submission_qkv_o32_u16_u32_r32_v33.py` | Port QKV128, O32, unpooler16/32, router32 shape specialists | These are retained milestones, not failures. v33 reached `3187.835967 us / 1.515385x` and became the next production ancestor. |
| v34: `submission_routerwide_v34.py` | Double wide-router split-K | m128 10.432->14.944 us; m256 12.128->15.936 us. **Closed.** |
| v35-v36c | Wider pooler and workspace/counter-reduction variants | Wide pooler failed to beat the incumbent; workspace/counter approach was unsafe or slower and included spill-prone variants. **Closed.** |
| v37-v40: `submission_qkv_o32_u16_u32_r32_v37.py`, `submission_qkv256_split_sweep_v38.py`, `submission_qkv256_independent_tiles_v39.py`, `submission_qkv256_output_tiles_v40.py` | QKV256 dual-M reuse, split/store/pipeline sweep, independent token tiles, independent output tiles | All exact variants were 13-82% slower than the library incumbent. Extra parallelism increased activation traffic, reduction cost, or used a weaker WGMMA orientation. **Entire descriptor-mapping family closed.** |
| v41: `submission_pooler32_v41.py` | Dormant C12 CPSSM/WGMMA pooler32 route | 21.472 vs 20.512 us, ratio 1.0468. **Closed.** |
| v42-v43 | QKV small-m occupancy-cap propagation | QKV16/32 microchanges did not improve aggregate behavior; other shapes regressed. **Closed.** |
| v44-v45 | Direct O128 WGMMA plus BK/stage/pipeline/L2 sweep | Correct, but only timer-quantum changes and worse corrected medians. **Closed.** |
| v46 | Generic M32 QKV route | Correct but 17.440 vs 16.736 us. **Closed.** |
| v47 | Nathen `m32x_q` QKV32 route | Real QKV32 win; later retained in v50 lineage. **Retained mechanism.** |
| v48 | m32x CP depth, reduction mode, launch-bound sweep | Original CP0/RED5/NTMAX128 remained best. **Nearby tuning closed.** |
| v49 | Unpooler32 PF/Q/WK/split/occupancy sweep | Original `(BN32,R64,WK4,PF2,Q2,C12,CAP1)` remained best; some variants spilled. **Closed.** |
| v50 | QKV32 plus guarded exact O16 | QKV32 retained; O16 tied and fell back. v50 became protected at `3186.412227 us / 1.516062x`. **Production ancestor.** |
| v51 | Force-enable near-tie O16 | Expected kernel ran, but corrected median and aggregate regressed to `3236.338 us`. **Closed; never force near-ties.** |
| v52-v53 | Persistent/wide pooler, seven configurations | m128 controls about 26.1 vs 23.6 us; valid m256 routes 42.7-66.0 vs about 28.8 us; one route failed exact one-hot. **Closed.** |

### v54-v78: router, multicast, and wide-shape follow-ups

| Versions / files | Experiment | Outcome |
|---|---|---|
| v54: `submission_router_multicast_v54.py` | N32 router activation multicast across N peers | Correct but m32/m128/m256 adjusted ratios 2.2868/1.869/2.8482. **Closed.** |
| v55: `submission_unpooler128_c12_v55.py` | Lossless C12 wide unpooler128 | 35.136 vs 22.752 us. **Compressed wide-unpooler closed.** |
| v56: `submission_o256_v56.py` | Dedicated O256 WGMMA candidate | 12.640 vs 12.896 us raw, but missed promotion margin. **Closed as too small/noisy.** |
| v57-v60 | No numbered v57, v58, v59, or v60 submission artifacts exist in this workspace | These numbers were not separate recorded candidates; the next preserved numbered experiment is v61. **No missing retry item.** |
| v61: `submission_pooler128_mcast_v61.py` | Pooler128 cluster multicast/split-K | Best 33.152 vs 23.872 us; others about 44-46 us. **Closed.** |
| v62: `submission_qkv128_occupancy_v62.py` | QKV128 occupancy/topology sweep | No production win; one experimental path hit a CUDA failure. Later v93-v95 completed the neighborhood. **Superseded/closed.** |
| v63: `submission_router_weight_mcast_v63.py` | 16-CTA router weight multicast | Correct; adjusted about 38% slower at m128 and 24% slower at m256. **Closed.** |
| v64 | 96/128-CTA wide-router retiles | m128 +37.6% to +64.1%; m256 +25.7% to +45.1%. **Closed.** |
| v65 | No numbered v65 submission artifact exists in this workspace | This was not a separate preserved candidate; the recorded router sequence resumes at v66/v67. **No missing retry item.** |
| v66 | N128 router high/low combine | No target timing improvement. **Closed.** |
| v67 | Tensor-map prefetch | Tiny m128 win around 0.064-0.096 us; m256 neutral. Useful as an ancestor-level microchange, not rank-moving by itself. **No standalone resubmission.** |
| v68/v70 | Diagnostic-only small-router loader floor and wide-router tail split | Intentionally unreachable/invalid diagnostic paths; established that fixed latency and reduction dominate. **Do not submit.** |
| v69 | M128 shared-weight m256 router retile | Legal K64 variants +8.2%; K32 variants failed coefficient recovery. **Closed.** |
| v71-v72 | Rank-local partial experiments | Timed out because receive byte count waited for a nonexistent local async contribution. **Buggy candidates; do not reuse.** |
| v73-v74 | Correct remote-only receive storage, then combine with prefetch | Isolated m128 tail win was non-additive; m256 slower. **Closed as a composition direction.** |
| v75 | Vectorized final router stores | Repeatable m128 improvement of about 0.096 us; m256 no win. Too small for full score. **Micro-win only.** |
| v76 | Deeper WGMMA issue window | Neutral at m128, up to 5.4% slower at m256 due to a true accumulator dependency. **Closed.** |
| v77 | One WGMMA fence per mainloop | Missed margin at m128; m256 49.5% slower. **Closed.** |
| v78 | Remove sender-side CTA barrier | Neutral at m128 and slower at m256; barrier was not critical. **Closed.** |

### v79-v98: skinny kernels, representations, and library/dataflow tests

| Versions / files | Experiment | Outcome |
|---|---|---|
| v79 | Raw-BF16 skinny unpooler | m1 route won locally and moved the table only about 0.11 us; every m8 route lost badly (best adjusted ratio 1.8). Scored result `3188.786 us`, below v50. **m8 closed; m1 insufficient alone.** |
| v80 | Exact C11 loader | Live bytes only 4.875% lower because exceptions rose to 834,382; 13.888 vs 10.912 us, adjusted 1.514x slower. **C11 closed.** |
| v81 | Exact skinny C12 CUDA-core decoder | Exact path remained 9-22% slower; even an intentionally inexact no-escape lower bound was only about 2% faster. **Closed.** |
| v82 | No-escape lower bound for existing C12 kernels | QKV got slower; O/pooler/unpooler ceiling too small for score. **Escape-only QKV/pooler work closed.** |
| v83-v84 | Per-lane ordered escape chains | Only unpooler m8 showed a modest isolated win; full run did not beat v50. **Closed.** |
| v85 | Fused sparse exact-delta epilogue | Shared-memory atomic correction cost more than broadcasts on every target. **Closed.** |
| v86 | True E4M3 FP8 compute plus activation conversion | Failed official accuracy (`3.68e-2` relative L2 on QKV m1) and was slower than exact kernels. **FP8 closed.** |
| v87 | Auto-research exact dense composition | Official checks about 3118-3127 us; scored `3137.2 us / 1.540x`. Became a production ancestor. **Retained.** |
| v88 | Adaptive prepare-time trial allocation | 48.7% fewer untimed tuner replays, no runtime score benefit; result variance changed picks. **Startup-only; do not score.** |
| v89 | Bounded zero-workspace pooler cuBLASLt sweep | Best new plans 16-22% slower; historical cluster schedules also miss the 3% gate. **Library-plan pooler tuning closed.** |
| v90 | Two-kernel pooler partial/reduction | m128 13.1% slower; m256 45.5% slower. PDL cannot erase workspace traffic. **Closed.** |
| v91 | Full-M narrow-N unpooler weight-reuse dataflow | m128 3.1% slower; m256 10.2% slower. **Closed.** |
| v92 | Pinned QKV256 cuBLASLt tile 124 | Valid and 0.73% faster, far below the 3% gate. **Plan neighborhood closed.** |
| v93-v95: `submission_qkv128_tile72_v93.py`, `submission_qkv128_tile80_v94.py`, `submission_qkv128_tile96_v95.py` | QKV128 72/80/96-row cluster topologies | 72/96-row grids failed one-wave residency; 80-row variants were 1.2-5.4% slower. **Nearby cluster/tile topology closed.** |
| v96-v97 | QKV128 bulk and packet-bulk DSMEM | Initial outputs were unwritten because the inherited store guard omitted RED6/RED7. These jobs do not measure the transport. **Bug identified.** |
| v98 | Fixed packet-bulk DSMEM output | Correct, spill-free, but 18.048 vs 17.280 us (4.44% slower). **Packet-bulk transport closed.** |

### v99-v115: production-ancestor refinements and score-wide mechanisms

| Versions / files | Experiment | Outcome |
|---|---|---|
| v99 | Direct global BF16 tensor reduction for QKV128 | 17.024 vs 16.992 us, neutral/slower. **Closed.** |
| v100 | Half-width BF16 DSMEM fragments | Real 2.742% QKV128 win, narrowly below the original 3% gate. **Promising mechanism, advanced as v102.** |
| v101 | Pair adjacent lanes into 128-bit DSMEM stores | Shuffle cost reduced the gain to 0.53%. **Closed.** |
| v102 | Revalidate/promote v100 with strict non-regression plus exhaustive recovery | Scored `3119.532 us / 1.548566x`; route dispatched and became production ancestor. **Retained.** |
| v103 | Retune main versus embedded alternate composition | All ten comparisons valid; none cleared 3%; router32 raw win was only 0.65% adjusted. **Closed.** |
| v104 | C12 m1 ROWS=4 | QKV/O/unpooler adjusted ratios 1.09/1.07/1.09. **Wider row grouping closed.** |
| v105 | C12 m1 K partition/thread count | QKV 512 threads cleared gate; O32 and unpooler256 lost. Scored `3110.414587 us / 1.553105x`. **Current protected production.** |
| v106 | Three aligned C12 pair-planes | QKV tied; O/unpooler gains below gate. **Nearby load-layout changes closed.** |
| v107 | Exact C13 plus no helper warps | Every m8/16/32 route slower; extra 1/16 traffic outweighed helper removal. **Closed.** |
| v108 | Four-CTA activation TMA multicast | O/unpooler m128/m256 2.74-3.10x slower adjusted. **Closed.** |
| v109 | Independent WGMMA accumulator stripes | QKV slower, router128 6.6% slower, router256 tied; extra live registers/reduction work did not expose overlap. Native TF32 router also failed accuracy. **Closed.** |
| v110 | Nonzero-workspace cuBLASLt | QKV256 9.7% and pooler256 14.0% slower adjusted. Block-local C11 also had insufficient byte savings. **Closed.** |
| v111 | Exact dual-C12 router high/low compression | Payload -25%, but every route 2.10-3.32x slower adjusted. **Closed.** |
| v112 | BF16 inter-warp SK2 partial transport | No route met gate; best isolated saving was only 0.288 us. **Closed.** |
| v113 | L2 evict-first weights / evict-last activations | Only unpooler32 cleared route gate; projected whole score 0.155%. **Score-wide cache policy closed.** |
| v114 | Per-CTA C12 bases plus one correction warp | Exception maxima barely changed; every route 4-15% slower adjusted. **Closed.** |
| v115 | Ultra-wide m64n256 exact-C12 WGMMA | QKV/O/pool/unpool 1.31-2.19x slower adjusted. **Closed.** |

### v116-v135: final structural sweep before scored-only work

These candidates all used guarded fallbacks. Their profile logs are the source
of truth because the chronological handoff did not give each one a full section.

| Version / file | Mechanism | Measured conclusion |
|---|---|---|
| v116 `submission_qkv_small_cl2_v116.py` | QKV m8/16/32 cluster-2 alternatives | No candidate accepted; projected saving 0. **Closed.** |
| v117 `submission_qkv_direct_v117.py` | Direct QKV small-m dataflow | m8/16/32 all rejected; projected saving 0. **Closed.** |
| v118 `submission_sk2_rowcorr_v118.py` | Move exact SK2 correction to output-row owners | No eligible route; projected saving 0. **Closed.** |
| v119 `submission_sk2_prodcorr_v119.py` | Producer-side exact correction | No eligible route; projected saving 0. **Closed.** |
| v120 `submission_sk2_localbase4_v120.py` | Four-row local C12 bases | No saving and no enabled route. **Closed with v114 family.** |
| v121 `submission_sk2_halfdecode_v121.py` | Half-width/decomposed SK2 decode | No saving and no enabled route. **Closed.** |
| v122 `submission_sk2_directldg_v122.py` | Direct global loads instead of staged SK2 movement | No saving and no enabled route. **Closed.** |
| v123 `submission_sk2_signsplit_v123.py` | Split sign from exponent/code stream | Only 0.416 us at m32; projected score gain `0.1346%`. **Too small; closed.** |
| v124 `submission_wide_tma_tune_v124.py` | Wide TMA tile/config sweep | Zero eligible saving at m128/m256. **Closed.** |
| v125 `submission_c12_tilebase_v125.py` | Per-tile C12 exponent base | Exact but zero eligible saving. **Closed.** |
| v126 `submission_sk2_slotowner_v126.py` | Assign correction/stream slots to fixed owners | About 0.704 us in one m8 sample; projected gain `0.3178%`, below global gate. **Micro-win only.** |
| v127 `submission_tma_store_v127.py` | TMA output-store epilogue for wide kernels | Zero eligible saving. **Closed.** |
| v128 | m1 C12 no-allocate weights, evict-first; activation evict-last | Pooler sample saved 0.704 us; projected `0.342%`. **No global enable.** |
| v129 | Pooler m1 ROWS 2/4/8 x threads 128/256/512 | Existing ROWS=2,NTH=512 best. **Closed.** |
| v130 | Seven exact m1 cache-policy combinations | Best saved 0.544 us; projected `0.263%`. **Micro-win only.** |
| v131 | Reuse router RK WGMMA for m1, splits 1/2/4/8/16 | Best still 41.8% slower adjusted. **Closed.** |
| v132 | Apply streaming policy to every m1 route | QKV/O neutral; pooler small positive; unpooler/router slower. **Not score-wide.** |
| v133 | External router output clear plus 160 threads | m128 11.4% slower; m256 19.0% slower. **Closed.** |
| v134 | Composite of qkv8/unpool8 slot owner, unpool32 L2, pool1 no-allocate | Only qkv8 cleared its route gate; 0.384 us and `0.173%` projected score gain. **No global enable.** |
| v135 | Remove m1 helper warp and restore escapes serially | Exact/spill-free but QKV/O/pool/unpool 9-17% slower. **Closed.** |

### v136-v142: scored validation after checks were exhausted

| Version / job | Mechanism | Outcome |
|---|---|---|
| v136 `5022c904065e411b` | Persistent m1 C12 grid reusing activation fragments | Every route lost: QKV 13.952 vs 12.320, O 9.056 vs 8.320, pooler 19.744 vs 17.088 us. Score `3129.4 us` was v105 fallback variance. **Closed.** |
| v137 `27545569c6d84e08` | Distribute exact escapes across compute threads, no helper warp | Every route lost by adjusted 8.7-19.0%; score `3130.418 us` fallback-only. **Closed.** |
| v138 `e4b18ec60a074ac5` | Pin validated q102 QKV128 | q102 selected, but QKV m1 missed its separate dynamic gate; `3140.5 us`. **Deterministic promotion closed.** |
| v139 `caae25fa6f9d4db1` | Pin q102 QKV128 and QKV m1 512-thread route | Both selected, but `3116.5 us`, still 6.1 us behind v105 best. **Too small to move rank; closed.** |
| v140 `c877c3be7b614119` | FlashInfer cuDNN BF16 backend on all eight wide dense routes | No route cleared 3%; only unpooler256 raw win was 0.224 us. `3119.7 us` fallback-only. **cuDNN closed.** |
| v141 `35822646d1ae4fac` | FlashInfer cuTile alpha-beta backend | `cuda.tile` imported, but judge FlashInfer lacked `gemm_alpha_beta_cutile`; `3132.064077 us` fallback-only. **Wrapper environment-blocked.** |
| v142 `9aa22516e180487f` | Embedded minimal MIT cuda.tile persistent BF16 kernel | Every route failed compile because the `tileiras` compiler was absent; `3121.763158 us` fallback-only. **Entire cuTile direction environment-blocked.** |

### v143-v144: escape representation and mid-m WGMMA

#### v143: packed radix-leaf escape records

`submission_escape_radix_v143.py` implements the useful sparse form of the
proposed bitmask/radix representation. The incumbent m=1 C12 table uses an
eight-byte `int2` per escape: `(local_row | k, original_bf16)`. Because
`ROWS=2` and `K<=8192`, v143 packs the same information losslessly into one
32-bit leaf:

- bits 0-15: original BF16 coefficient;
- bits 16-28: K coordinate;
- bit 29: local row;
- word 0: unchanged per-CTA escape count.

The encoding passed 2,187 randomized round trips and cut the synthetic table
from 33,792 to 16,896 bytes. All six production/candidate GEMV instantiations
compiled to compute-90 PTX (`149,795` bytes, no diagnostics). It was rejected
before H200 scoring: `cg_compress` already requires fewer than 0.5% escaped
weights, so halving this metadata cannot provide the roughly 9% load-time
saving needed to move the six-load geometric score by 1.5%. A dense K bitmask
is worse for this distribution because it costs 128-1,024 bytes per row before
storing the replacement values. **Packed/radix/bitmask escape indexing is
closed unless the escape density or correction algorithm changes.**

#### v144: TMA/WGMMA migration for m=16/32

`submission_midm_wgmma_v144.py`, job `0c4d19afb0864f6a`, revived the dormant
Hopper TMA/WGMMA skinny pipeline for all eight scored BF16 dense routes at
m=16/32. O, pooler, and unpooler use their existing one-wave geometries (128
CTAs). QKV changes from non-WGMMA `RG=9` to legal `RG=8`, producing the nearest
legal grid, 144 CTAs, because 8,192 requires a power-of-two K split.

All routes were correct: exact coefficient probes passed, FP64 relative L2 was
`1.21e-5` to `7.75e-5`, every kernel had zero local bytes, 64 registers at m16
or 80 at m32, one active CTA/SM, and 150-199 KB dynamic shared memory. Every
route nevertheless lost decisively to SK2:

| Route | WGMMA us | incumbent us | adjusted core ratio |
|---|---:|---:|---:|
| QKV m16 | 24.832 | 13.184 | 2.268x |
| QKV m32 | 28.352 | 14.144 | 2.401x |
| O m16 | 11.456 | 9.120 | 1.456x |
| O m32 | 12.416 | 9.312 | 1.584x |
| Pooler m16 | 23.328 | 17.728 | 1.408x |
| Pooler m32 | 24.224 | 18.848 | 1.362x |
| Unpooler m16 | 22.976 | 17.440 | 1.412x |
| Unpooler m32 | 24.064 | 18.432 | 1.390x |

The projected saving was zero, so the global gate enabled no route. The scored
run therefore used v105 throughout and returned `3118.585044877531 us /
1.5490358385239127x`, fallback variance below the v105 best. **Cross-shape
TMA/WGMMA migration for m16/m32 is closed; m8 cannot use this WGMMA branch
because it does not provide a 16-row WGMMA tile.**

#### Asymmetric router residual precision

INT8/FP8 storage of the router low residual was not implemented because it is
illegal under the weight rule: FP32 router weights may use the explicit BF16
high-plus-low exception, but otherwise may not be stored at a different lossy
precision. Adding another exact residual would make the scheme lossless but
would no longer be the proposed two-term representation, and the exact v111
dual-C12 high/low route was already 2.10-3.32x slower. **Asymmetric lossy
router residual storage is contract-closed.**

#### v145: horizontal half-width DSMEM Pooler port and K-permutation audit

`submission_pooler_bf16_dsmem_v145.py`, job `31fc48f9642b40c5`, transplanted
v102's 8-byte BF16 remote-partial transport onto the exact v52 one-wave wide
Pooler dataflow. It retained the N64 x split-K2 topology, WGMMA mainloop,
owner mapping, and rank-order FP32 accumulation. Local SM90a ptxas validation
reported 114 registers for m128 and 101 for dual-M m256, with zero stack and
zero spills. Both H200 routes were numerically valid and had 66 active
clusters, but neither passed the performance gate:

| Route | candidate us | incumbent us | adjusted core ratio |
|---|---:|---:|---:|
| Pooler m128 | 24.640 | 24.224 | 1.0206x |
| Pooler m256 | 43.008 | 30.528 | 1.4704x |

Both routes stayed disabled. The run passed 9/9 with no flags and scored
`3089.9113966319087 us / 1.5634105253845498x`; because timed dispatch was v105
throughout, this score is fallback variance. **Horizontal v102 half-width
DSMEM transport is now closed for wide Pooler.** Unpooler was not instantiated:
its `[N,K]=[8192,4096]` geometry needs either 256 N64 CTAs or a new N128
mainloop/epilogue, so it is not a one-to-one port of the v102 mechanism.

The same run audited the proposed global K-permutation/raw-tail representation
using the real m1 C12 escape tables. A branchless prefix is exact only for K
columns containing no escape in any output row. The measured escape-column
unions were:

| Layer | escapes | K columns with any escape | minimum exact raw-tail fraction | densest 5% coverage |
|---|---:|---:|---:|---:|
| QKV | 2,606 | 2,248 / 8,192 | 27.44% | 29.47% |
| O | 1,009 | 654 / 1,024 | 63.87% | 17.94% |
| Pooler | 4,549 | 3,482 / 8,192 | 42.50% | 22.14% |
| Unpooler m1 slice | 429 | 403 / 4,096 | 9.84% | 53.85% |

No layer admits the proposed 95% clean prefix and 5% raw tail. Stable global
K permutation would also replace contiguous activation loads with gathers.
The only near case is the already-small unpooler m1 slice, which needs about a
10% raw tail and cannot project enough whole-score gain to justify the final
attempt. **Global K-permuted C12 plus a 5% raw tail is evidence-closed.**

#### v146: Router m32 full preload and QKV256 tile-124 scored diagnostic

`submission_router32_st8_qkv256_lt124_v146.py`, job
`2c333461446f4ebb`, combined two contained experiments on top of the older v92
diagnostic stack:

1. Router m32 changed its exact WGMMA configuration from
   `(32,16,4,1,2,1)` to `(32,16,8,1,2,1)`. The eight stages issue all eight
   K-split slices up front while preserving the same BF16-high-plus-low data,
   FP32 accumulation order, reduction mode, and output.
2. QKV m256 retained v92's exact zero-workspace cuBLASLt algo-66 tile-124,
   custom-0, stage-35, cluster-2 plan, with the original 3% cold-L2 promotion
   gate against the tile-134 incumbent.

The H200 run passed 9/9 with no flags. Router m32 used
`rr_kernel<32,16,8,1,2,1>` and profiled at `4.2 us`, versus `4.1 us` for the
four-stage v145 kernel. Its scored row was `5.03 us` versus v145's `5.04 us`,
which is measurement-level noise and not a route win. **Router m32 full
eight-stage preload is closed.**

The QKV tile-124 plan again passed its FP64 guard (`1.654e-3` relative error),
but measured `22.02 us` against tile-134 at `22.11 us`: only `0.09 us` or
`0.41%` faster. The 3% gate correctly retained tile-134. The earlier v92 run
had measured `21.89 us` versus `22.05 us`, a `0.73%` win, so both official
measurements put the candidate below the noise-aware promotion threshold.
**QKV256 tile-124/cluster-2 is closed unless a new mechanism produces a
materially larger win.**

The complete v146 submission scored `3143.4377634218686 us /
1.5367888164393975x`, `53.526367 us` or `1.73%` behind the current best. That
whole-score delta is not an isolated measurement of Router preload because
v146 deliberately inherited the older v92 composition rather than the full
v145 stack. It did not replace the leaderboard best; rank remains 146.

## Closed directions: do not retry without genuinely new evidence

1. C11, C13/no-helper, FP8, TF32 router, asymmetric INT8/FP8 router residuals,
   or any other lossy weight representation.
2. Skinny raw-BF16/C12 CUDA-core m8, per-lane escapes, sparse atomic deltas,
   serialized correction, distributed compute-thread correction, or producer/
   row-owner correction.
3. Wider C12 row groups, pair-plane rearrangements, per-CTA/tile exponent bases,
   sign-split/half-decode/direct-LDG variants, and nearby decoder instruction
   shaving.
4. Persistent m1 C12 grids, dense bitmasks/radix trees/packed leaves for the
   existing sparse helper-warp table, or forcing the small cache-policy/
   slot-owner micro-wins. Their projection is below the leaderboard gap.
5. QKV256 dual-M reuse, split-only tuning, independent M/N tiles, the exact
   tile-124/custom-0/stage-35/cluster-2 plan, another cuBLASLt
   tile/custom/stage/cluster sweep, or nonzero-workspace Lt plans.
6. QKV128 nearby output-tile/cluster counts, bulk packet transfer, lane-paired
   DSMEM, or another half-width transport microvariant.
7. Pooler wide persistence, DSMEM split-K (including v102 half-width remote
   partials), workspace/counter reduction, multicast, two-kernel partial
   reduction, C12 HALF/H2 two-resident-CTA output decomposition, or more Lt
   plan tuning.
8. Full-M/narrow-N or wide compressed unpooler variants.
9. Router weight/activation multicast, more CTAs, larger split-K/clusters,
   deeper WGMMA issue, m32 full eight-stage upfront preload, fence/barrier
   removal, external clear, direct RK m1, dual-C12, vLLM raw-FP32/BF16x3
   ports, or deterministic pinning of micro-wins.
10. Cross-shape TMA/WGMMA migration for m16/m32; SK2 is 1.36-2.40x faster.
11. cuDNN for the eight wide routes in the current image.
12. cuTile wrappers, embedded cuda.tile kernels, tile changes, or autotuning in
    the current judge image. Do not install, download, execute, or embed a
    compiler toolchain from the submission.
13. A custom `__deepcopy__` as a performance optimization.
14. Producer-side C12 expansion into a full BF16 shared-memory stage, including
    the register-A consumer rung. Four producer warps doing two 16-row
    fragments each measured 2.92x slower than v150 on Pooler m128; SS-WGMMA
    cannot recover the resulting 33 us gap.

## What could justify reopening a closed family

A direction should be reopened only if at least one of these is true:

- the judge image or hardware changes materially;
- a profiler identifies a different bottleneck than the one already measured;
- the algorithm changes dataflow or representation rather than only a tile,
  stage, thread count, cache hint, or ownership permutation;
- the mechanism affects enough weighted shapes to project at least the current
   current `0.92%` rank-70 gap, with margin for score variance;
- exact recovery and graph-safe execution can be demonstrated before spending
  a scored attempt.

For the present board, require at least a 3% adjusted per-route win and a
conservative 0.9% projected whole-score win before using the final scored
attempt. A smaller result is useful research, but not a rank-moving submission.

## Source-of-truth files

- `submission_rs128_faithful_v150.py`: protected/scored implementation (SHA256
  `E405D6464EF5D64FDC896E300DD0F5AC9CD772D4E65F4EFF7FF2338A2D0F7258`).
- `submission_escape_radix_v143.py` and `validate_escape_radix_v143.py`:
  lossless packed escape representation and non-scored validation.
- `submission_midm_wgmma_v144.py` and `validate_midm_wgmma_v144.py`: guarded
  mid-m TMA/WGMMA experiment and compute-90a validation.
- `submission_pooler_bf16_dsmem_v145.py`: guarded horizontal half-width DSMEM
  Pooler port plus the real-weight K-permutation necessary-condition audit.
- `submission_router32_st8_qkv256_lt124_v146.py`: scored diagnostic for Router
  m32 full preload and the guarded QKV256 tile-124 plan; neither route is a
  promotion candidate.
- `submission_rs128_diag_v151.py` through
  `submission_rs128_half_h2_v155.py`: Pooler consumer attribution plus guarded
  raw-weight, WGMMA-overlap, GS2, and HALF/H2 experiments; all are closed.
- `submission_rs128_pdecode_v156.py` and
  `generate_rs128_pdecode_v156.py`: exact producer-side C12 expansion rung;
  rejected at bake after measuring 54.56 us versus v150's 21.28 us.
- `NEXT_OPTIMIZATION_RESEARCH.md`: chronological v79-v142 evidence.
- `TOMORROW_NATHAN_STRUCTURAL_PLAN.md`: v29-v56 structural evidence.
- `ROUTER_ARCHITECTURE_OPTIMIZATION.md`: detailed router dataflow and v63-v64.
- `ROUTER_OPTIMIZATION_SEQUENCE.md`: v67-v78 router sequence.
- `ORIGINAL_OPTIMIZATION.md`: independent-kernel branch and early failures.
- `OPENKERNELS_INTEGRATION.md`: imported public baselines and attribution.
- `RESUME_TOMORROW.md`: v96-v98 transport investigation.
- `profiles/`: job logs, profiles, route gates, and kernel names.

When this file and an old note disagree about the protected baseline, quota, or
leaderboard rank, this file's 2026-11-02 post-v155 status is newer. For kernel
timings and failure mechanics, the original job log remains authoritative.
