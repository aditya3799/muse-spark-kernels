# Kernel architectures beyond FlashAttention-3 for this workload

Date: 2026-10-24

## Executive conclusion

There is no generally faster drop-in replacement for FlashAttention-3 (FA3) on Hopper that also directly supports this challenge's exact geometry. The best route is a workload-specific kernel assembled from four complementary ideas:

1. NVIDIA XQA's decode-specific matrix orientation and 8-query-head grouping.
2. FA3/PipeThreader's overlap of data movement, tensor-core work, and softmax.
3. LeanAttention's associative online-softmax reduction and work-equalized split scheduling.
4. HPC-Ops' persistent dynamic scheduling for variable sequence lengths.

The first combined compute-core implementation is in `submission_custom_tc_v5.py` and is identified internally as `v11_xqa64`. It changes the Hopper full-decode path to a 64-key x 8-query-head tile using four fully useful FP8 `m64n8k32` WGMMA instructions per tile. The active `submission.py` remains the known-good scored FA3 kernel.

This is the right architecture to continue, but it is not yet proven faster than FA3: this machine is SM86 and cannot execute WGMMA. The Hopper path has been cross-compiled and inspected, while its arithmetic has been modeled locally. Any H200 latency below is explicitly a target or projection, not a benchmark.

## Exact challenge bottleneck

The important route is full-attention decode with:

- H200 / SM90a
- batch 16 or 32
- one query token
- 16 query heads and 2 KV heads, therefore 8 query heads per KV head
- head dimension 64
- approximately 80,000 keys
- FP8 E4M3 paged KV with page size 1

Saved H200 FA3 measurements are 194.16 us for batch 16 and 378.82 us for batch 32. Their data-movement floors are 63.18 us and 126.35 us, so FA3 takes 3.07x and 3.00x the respective floor. The profiled FA3 attention kernel accounts for 96.7% and 98.1% of those full-decode routes. This leaves meaningful headroom, but only through a decode-specialized dataflow; generic softmax tuning cannot close a threefold floor gap.

## Architecture comparison

The published numbers below come from different models and shapes and are not apples-to-apples challenge predictions.

| Architecture | Reported metric | Best transferable idea | Fit to this challenge | Expected effect here |
|---|---|---|---|---|
| FA3 | Up to 740 FP16 TFLOP/s and about 1.2 PFLOP/s FP8 on H100; 1.5-2.0x FA2 in the reported workloads | WGMMA/TMA, warp specialization, ping-pong scheduling, GEMM-softmax overlap | Baseline already used | Strong general baseline, but its generic full-attention path is about 3x the memory floor on our page-1 decode shapes |
| NVIDIA XQA | TensorRT-LLM reports up to 2.4x H200 Llama-70B throughput at the same latency budget | One QK WGMMA warpgroup, one PV WGMMA warpgroup, four IO warps, 64-token tiles, eight GQA heads, double buffering | Closest architectural match; current public interfaces generally assume page sizes 8 or larger | Strongest compute layout. Page-1 gathering must be implemented manually rather than copied unchanged |
| LeanAttention | Paper reports 2.6x average over FA2 and up to 8.33x at 512K in its broad comparison; its A100 decode data reports up to 2.46x FA2 and 1.06x FlashDecoding | Stream-K work allocation plus associative softmax-state reduction | Very good for 80K and unequal request lengths | Scheduling/reduction improvement, not a replacement tensor-core microkernel |
| PipeThreader | OSDI paper reports H100 FlashDecoding average 1.12x and up to 1.23x over FA3; general FlashAttention average 1.07x and up to 2.18x | Software-defined fine-grained pipeline that overlaps producer/consumer stages | Directly relevant to Hopper latency hiding | Plausible 7-23% class improvement after the compute layout is correct; not independently multiplicative with FA3 overlap |
| Tencent HPC-Ops FP8 decode | Repository reports up to 2x over FA3/FlashInfer/TensorRT-LLM and up to 2.88x for dynamic versus static scheduling | Persistent kernel and dynamic task assignment | Promising, but published benchmark uses head dimension 128 and page size 64 rather than 64/page-1 | Best scheduling model for variable lengths; headline speedups cannot be transferred numerically |
| FlashDecoding++ | Average 1.37x over FlashDecoding; asynchronous softmax component reported at 1.14x and flat-GEMM double buffering up to 52% | Split-KV, double buffering, and reduced softmax synchronization | Split-KV is useful; approximate/recompute-dependent softmax is an unnecessary correctness risk | Retain exact associative softmax and borrow only scheduling ideas |
| FlashMLA | Repository reports up to 3,000 GB/s and 660 TFLOP/s on H800 for its supported MLA shapes | Specialized KV layout and scheduling | Dense path targets MLA dimensions around 512/576; dense FP8 KV and hdim 64 do not match | Not a viable direct kernel for this task |
| FlashAttention-4 on Hopper | SGLang's Hopper tracking reports the SM90 path around 20% slower than FA3 in the cited comparison; Transformer Engine prefers FA3 on Hopper | Newer architecture primarily aimed beyond Hopper | Poor choice for H200 today | Reject for this submission target |

## Combined design

### 1. XQA matrix orientation

For one KV head, all eight associated query heads are processed together.

- QK: `K[64 tokens, 64 dims] * Q[8 heads, 64 dims]^T -> scores[64, 8]`
- PV: `V^T[64 dims, 64 tokens] * P[8 heads, 64 tokens]^T -> output[64, 8]`

Both products map exactly to two `m64n8k32` FP8 WGMMA instructions across K=64. There are no padded query-head columns. The previous 16-key WGMMA design needed four 16-key tiles and issued 12 tensor instructions over the same 64 keys: eight FP8 QK instructions and four FP16 PV instructions. The new tile issues four FP8 instructions total, a 3x reduction in tensor instructions for that interval, while also removing the FP8-to-FP16 V conversion.

### 2. Exact tilewise online softmax

Scores are reduced once per 64-key tile. Each tile produces `(tile_max, tile_sum, tile_output)` and is merged with the running state using the associative online-softmax equations. The probability matrix is scaled to E4M3 range for the PV WGMMA, following the XQA pattern. This replaces key-by-key softmax state updates without using sparse attention, top-k, or key skipping.

### 3. Page-1-aware IO

TMA is excellent for contiguous tiles but cannot directly gather 64 arbitrary physical rows from a page-size-1 page table. The challenge therefore needs dedicated IO warps performing vectorized page-table gathers into swizzled shared memory. TMA should only be used on a separately detected contiguous/page-64 route. Forcing TMA onto the scored page-1 route would add packing work without eliminating the gather.

### 4. Pipeline and scheduling target

The full target block is 384 threads:

- 128-thread QK WGMMA warpgroup
- 128-thread PV WGMMA warpgroup
- four 32-thread IO/softmax warps
- double-buffered K, V, score/probability stages with role-specific named barriers
- persistent work queue assigning `(request, KV head, split)` tasks dynamically
- LeanAttention-style associative combine for split results

That organization overlaps the next page-table/KV gather with QK, softmax, and PV for the current tile. Phase two now implements the 384-thread double-buffered pipeline. Persistent cross-block task assignment remains separate future work; it is less important for the two scored full-decode shapes because both launch 256 equal-sized split tasks.

## Implemented phase-two evidence

The SM90a fat binary contains the intended Hopper instructions:

- separate QK and P*V `QGMMA.64x8x32.F32.E4M3.E4M3` instruction sites in different 128-thread warpgroups
- 128-thread internal barriers plus 256-thread producer/consumer barriers for the two shared buffers
- 48 registers per thread
- 20,096 bytes shared memory per block
- zero stack and local memory in the SM90a kernel

This compares favorably with the first integrated WGMMA kernel's 85 registers and 10,304 bytes of shared memory and the phase-one XQA kernel's 56 registers and 10,816 bytes. Shared memory intentionally increases because K, V, P, and tile statistics are double-buffered. At 48 registers, the 384-thread block remains capable of multiple-block residency on Hopper without spilling.

The barrier order was modeled independently for the IO, QK, and P*V roles. Every producer/consumer rendezvous matched and the schedule drained without deadlock for 0 through 4,096 tiles. The critical ordering releases the next V buffer before acknowledging reuse of the current buffer, avoiding the otherwise possible two-buffer cycle.

The shared-memory swizzles were checked against CuTe's official SM90 layouts with zero mismatches. CuTe's `m64n8k32` accumulator ownership was also checked for all 128 threads.

The FP8 probability model produced these worst errors across three seeds:

| Distribution | Context | Relative L2 | Worst row-relative error |
|---|---:|---:|---:|
| random | 64 | 0.857% | 2.879% |
| random | 2,048 | 0.265% | 1.601% |
| random | 80,000 | 0.099% | 0.428% |
| coherent | 64 | 0.592% | 1.042% |
| coherent | 2,048 | 0.116% | 0.211% |
| coherent | 80,000 | 0.056% | 0.091% |

These arithmetic-model results fit comfortably inside the task's output tolerances, but they do not replace an end-to-end SM90 execution test.

The executable SM86 fallback still passes its local relative comparison against custom v2. After the phase-two edit, a small smoke case measured 239.62 us versus 454.14 us at batch 16/context 32, a 1.895x speedup with relative L2 `5.837e-4`. This number validates only the unchanged fallback path and says nothing about Hopper WGMMA performance.

## Performance expectations

Three different levels must not be confused:

| Level | Batch 16, full 80K | Batch 32, full 80K | Confidence |
|---|---:|---:|---|
| Measured FA3 | 194.16 us | 378.82 us | H200 measurement |
| Physical memory floor | 63.18 us | 126.35 us | Challenge specification |
| Old custom v5 projection | about 714 us | about 1,420 us | SM86-derived projection; not competitive |
| Phase-one XQA64 engineering range | roughly 325-450 us | roughly 645-890 us | Low-confidence projection from reduced tensor work without full overlap |
| Implemented phase-two pipeline target | roughly 90-125 us | roughly 180-250 us | Design target of 1.4-2.0x the floor, not a measured result |

Phase two now contains the producer/QK/softmax/PV overlap needed to challenge FA3. A credible target remains 1.5-2.0x FA3 on the two dominant full-decode shapes, but it cannot be promoted from a target to a result without SM90a execution. Claims above 2x would approach the stated data-movement floor and need exceptional page-1 gather efficiency.

## Decision

- Keep `submission.py` on FA3 until a real SM90a run verifies correctness and latency.
- Continue with `v12_xqa_pipeline`; it is the strongest architecture found for this exact 8:1 GQA, hdim-64 workload.
- The 384-thread double-buffered XQA pipeline is implemented and cross-compiled; its next gate is real SM90a correctness and latency.
- If load imbalance appears on nearby variable-length shapes, add persistent dynamic assignment of `(batch, KV head, split)` work and length-aware split counts.
- Do not spend more time on key-by-key softmax tweaks, FA4-on-Hopper, or adapting FlashMLA's incompatible dense geometry.

## Primary sources

- FlashAttention-3: https://pytorch.org/blog/flashattention-3/ and https://arxiv.org/abs/2407.08608
- NVIDIA XQA overview: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md
- Current XQA SM90 implementation: https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/xqa/mha_sm90.cu
- LeanAttention: https://arxiv.org/abs/2405.10480
- FlashDecoding++: https://arxiv.org/abs/2311.01282
- PipeThreader: https://www.usenix.org/conference/osdi25/presentation/cheng
- HPC-Ops: https://github.com/Tencent/hpc-ops
- HPC-Ops decode benchmark: https://github.com/Tencent/hpc-ops/blob/main/benchmark/attention_decode/README.md
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
- FA4 Hopper comparison: https://github.com/sgl-project/sglang/issues/18265
