# Custom attention optimization report

Date: 2026-10-24

## Safety and methodology

- `submission.py` remains the scored FA3 implementation. Experimental kernels are archived separately.
- No contest-side check was used for this optimization round.
- Relative timings use CUDA events, alternating candidate/baseline order, an L2 flush between samples, synthetic FP8 E4M3 Q/K/V, and randomized page tables.
- Local GPU: NVIDIA GeForce RTX 3050 Laptop GPU (SM86). It can benchmark WMMA but cannot execute Hopper WGMMA.

## Options tested

| Candidate | Main change | Local result | Numerical result | Decision |
|---|---|---:|---:|---|
| `submission_custom_v3.py` | FP8x2 vectorized Q/K/V loads | 1.013x at ctx 1024; 1.025x at ctx 256 vs custom v2 | Exact BF16 output vs v2 | Keep as SIMT cleanup |
| `submission_custom_v4.py` | Branch-based one-exponential online softmax | 0.828x vs v3 | Exact output | Reject: 17% slower |
| `submission_custom_tc_v5.py` | 16-key WMMA fallback plus integrated Hopper WGMMA QK/PV path | 1.48x-1.66x vs v3/v2 on SM86 | Relative L2 6.8e-4 to 8.5e-4 on executable fallback | Best custom design; Hopper path awaits H200 execution |
| `submission_custom_tc32_v7.py` | 32-key tensor-core tile | 0.901x-0.923x vs v5 | Same algorithmic class | Reject: lower occupancy / more staging |
| `submission_custom_tc64_v6.py` | 64-key tensor-core tile | 0.775x-0.802x vs v5 | Same algorithmic class | Reject: shared-memory and occupancy cost |
| `wgmma_fp8_probe.cu` | Hopper FP8 QK plus FP16 PV WGMMA primitives | Cross-compiles to SM90a; cannot execute on SM86 | Logical swizzled staging compiled | Completed its role as the integration probe |

Restricting `exp2f` to lane 0 was also rejected. A warp issues the transcendental instruction as a warp instruction, so lane masking did not provide the expected 32x reduction and the added divergence/shuffles made the kernel much slower.

## Best portable candidate: v5

The kernel processes 16 keys per tile:

1. Gather 16 paged FP8 K/V rows and convert them to FP16 in shared memory.
2. Compute QK with four `m16n16k16` WMMA operations across head dimension 64.
3. Perform one blockwise online-softmax update for the 16 scores instead of a key-by-key state update.
4. Compute probability-times-value with four WMMA output-column tiles.
5. Merge split partials using the existing combine kernel.

### Local relative results at ctx 80,000

| Batch | Custom v2 | Tensor-core v5 | Speedup | Relative L2 |
|---:|---:|---:|---:|---:|
| 16 | 166,216.71 us | 100,462.08 us | 1.655x | 8.121e-4 |
| 32 | 331,414.02 us | 200,114.69 us | 1.656x | 8.506e-4 |

The speedup remained consistent over the tested context range:

| Context | Batch | v3 | v5 | Speedup |
|---:|---:|---:|---:|---:|
| 256 | 16 | 780.29 us | 526.85 us | 1.481x |
| 1,024 | 16 | 2,355.20 us | 1,594.88 us | 1.477x |
| 1,024 | 32 | 4,439.55 us | 2,853.38 us | 1.556x |
| 4,096 | 16 | 8,561.15 us | 5,427.20 us | 1.577x |
| 4,096 | 32 | 16,842.75 us | 10,583.04 us | 1.591x |
| 16,384 | 16 | 33,638.40 us | 20,825.09 us | 1.615x |
| 16,384 | 32 | 66,768.90 us | 41,417.73 us | 1.612x |
| 80,000 | 16 | 164,562.43 us | 100,310.01 us | 1.641x |
| 80,000 | 32 | 327,947.27 us | 200,055.30 us | 1.639x |

The last two rows are an earlier v3-v5 run; the first table is the final paired v2-v5 rerun.

## Relative projection against the scored H200 FA3 submission

Saved H200 measurements:

| Scored shape | FA3 | Custom v2 |
|---|---:|---:|
| Full decode, batch 16, ctx 80,000 | 194.16 us | 1,182.39 us |
| Full decode, batch 32, ctx 80,000 | 378.82 us | 2,351.71 us |

Applying the measured local v2-to-v5 factors only as a rough projection gives:

| Scored shape | Projected v5 | Gap to FA3 |
|---|---:|---:|
| Batch 16 | about 714.44 us | 3.68x slower |
| Batch 32 | about 1,420.11 us | 3.75x slower |

Using the challenge's weighted-load geometric-mean formula and leaving all other saved custom-v2 shapes unchanged produces a projected score of about **4,595 us**, versus custom v2's measured **5,399.94 us** and FA3's measured **3,146.99 us**. This projection is not an H200 measurement: SM86 and SM90 tensor-core scaling can differ materially.

## Integrated WGMMA status

`submission_custom_tc_v5.py` now contains a self-contained Hopper kernel with no CUTLASS runtime/header dependency. One 128-thread warp group owns a `(batch, KV head, split)` work item, gathers 16 keys, evaluates QK directly in FP8, performs a 16-key online-softmax update, evaluates P*V in FP16, and keeps the useful output fragment in registers across tiles. The SM86 build dispatches to the already measured WMMA fallback; devices with compute capability 9.x dispatch to WGMMA.

The complete extension cross-compiled successfully as an SM86+SM90a fat binary. CUDA 12.4 emitted in the integrated SM90a kernel:

- Two `QGMMA.64x16x32.F32.E4M3.E4M3` instructions for the 64-dimensional QK dot product.
- One `HGMMA.64x64x16.F32` instruction for the 16-key probability-times-value product.
- The expected warp-group arrival and dependency-barrier instructions.

The Hopper kernel uses 85 registers per thread and 10,304 bytes of shared memory per block, with zero stack and local-memory use. Its launch bound is two 128-thread blocks per SM.

The hand-coded shared-memory layouts were exhaustively compared on the host with the official CuTe layout definitions:

- FP8 B64 swizzle: zero mismatches.
- FP16 B32 swizzle: zero mismatches after correcting the swizzle selector to bit 7.
- QK and P*V accumulator-register coordinates: matched the CuTe mappings used to select the eight-head result fragments.

The intended Hopper tile is therefore structurally and compile-time validated. It still is not submission-safe because this machine is SM86: the WGMMA branch cannot be executed here, so numerical correctness and actual H200 latency remain unmeasured. Any H200 performance number before that execution would be a projection, not a benchmark.

## Recommendation

Keep FA3 active for submission. Keep v5 as the best locally validated custom fallback and the integrated WGMMA candidate. The next meaningful gate is one real SM90a/H200 execution of this candidate for correctness and latency; do not replace FA3 based only on the current projection and compile-time evidence.
