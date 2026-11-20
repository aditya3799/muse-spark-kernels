"""Directional local sweep for exact-weight BF16 projection kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def bf16_matmul_kernel(
    x_ptr,
    weight_t_ptr,
    destination_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_split = tl.program_id(2)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    k_per_split: tl.constexpr = K // SPLIT_K
    k_start = pid_split * k_per_split
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, k_per_split, BLOCK_K):
        current_k = k_start + k_offset + offsets_k
        x = tl.load(
            x_ptr + offsets_m[:, None] * K + current_k[None, :],
            mask=offsets_m[:, None] < M,
            other=0.0,
        )
        weight = tl.load(
            weight_t_ptr + current_k[:, None] * N + offsets_n[None, :],
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        accumulator += tl.dot(x, weight, out_dtype=tl.float32)
    if SPLIT_K == 1:
        destination = destination_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    else:
        destination = (
            destination_ptr
            + pid_split * M * N
            + offsets_m[:, None] * N
            + offsets_n[None, :]
        )
    tl.store(
        destination,
        accumulator,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


@triton.jit
def bf16_reduce_kernel(
    partial_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in range(0, SPLIT_K):
        accumulator += tl.load(
            partial_ptr + split * ELEMENTS + offsets, mask=mask, other=0.0
        )
    tl.store(output_ptr + offsets, accumulator.to(tl.bfloat16), mask=mask)


def launch(x, weight_t, output, partial, config):
    rows, n, k = x.shape[0], weight_t.shape[1], x.shape[1]
    bm, bn, bk, split_k, warps, stages = config
    destination = output if split_k == 1 else partial
    bf16_matmul_kernel[(triton.cdiv(rows, bm), triton.cdiv(n, bn), split_k)](
        x,
        weight_t,
        destination,
        M=rows,
        N=n,
        K=k,
        SPLIT_K=split_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    if split_k != 1:
        elements = rows * n
        bf16_reduce_kernel[(triton.cdiv(elements, 256),)](
            partial,
            output,
            ELEMENTS=elements,
            SPLIT_K=split_k,
            BLOCK=256,
            num_warps=4,
        )


shapes = {
    "qkv_gate": (16, 2304, 8192, [
        (16, 64, 32, 4, 4, 3),
        (16, 64, 64, 4, 4, 3),
        (16, 128, 32, 8, 8, 3),
        (16, 128, 64, 8, 8, 3),
    ]),
    "o": (16, 8192, 1024, [
        (16, 64, 32, 1, 4, 3),
        (16, 64, 64, 1, 4, 3),
        (16, 128, 32, 1, 8, 3),
        (16, 128, 64, 1, 8, 3),
    ]),
    "pooler": (16, 4096, 8192, [
        (16, 64, 32, 2, 4, 3),
        (16, 64, 64, 2, 4, 3),
        (16, 128, 32, 4, 8, 3),
        (16, 128, 64, 4, 8, 3),
    ]),
    "unpooler": (16, 8192, 4096, [
        (16, 64, 32, 1, 4, 3),
        (16, 64, 64, 1, 4, 3),
        (16, 128, 32, 1, 8, 3),
        (16, 128, 64, 1, 8, 3),
    ]),
    "qkv_gate_m128": (128, 2304, 8192, [
        (32, 64, 32, 1, 4, 3),
        (32, 128, 32, 1, 8, 3),
        (64, 64, 32, 1, 8, 3),
        (64, 128, 32, 1, 8, 3),
    ]),
    "qkv_gate_m256": (256, 2304, 8192, [
        (32, 64, 32, 1, 4, 3),
        (32, 128, 32, 1, 8, 3),
        (64, 64, 32, 1, 8, 3),
        (64, 128, 32, 1, 8, 3),
    ]),
    "pooler_m128": (128, 4096, 8192, [
        (32, 64, 32, 1, 4, 3),
        (32, 128, 32, 1, 8, 3),
        (64, 64, 32, 1, 8, 3),
        (64, 128, 32, 1, 8, 3),
    ]),
    "pooler_m256": (256, 4096, 8192, [
        (32, 64, 32, 1, 4, 3),
        (32, 128, 32, 1, 8, 3),
        (64, 64, 32, 1, 8, 3),
        (64, 128, 32, 1, 8, 3),
    ]),
    "unpooler_m128": (128, 8192, 4096, [
        (32, 64, 32, 1, 4, 3),
        (32, 128, 32, 1, 8, 3),
        (64, 64, 32, 1, 8, 3),
        (64, 128, 32, 1, 8, 3),
    ]),
    "unpooler_m256": (256, 8192, 4096, [
        (32, 64, 32, 1, 4, 3),
        (32, 128, 32, 1, 8, 3),
        (64, 64, 32, 1, 8, 3),
        (64, 128, 32, 1, 8, 3),
    ]),
}

torch.manual_seed(5)
flush_source = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
flush_destination = torch.empty_like(flush_source)

for name, (rows, n, k, configs) in shapes.items():
    print(name, rows, n, k)
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[0, 123] = 32.0
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight_t = weight.t().contiguous()
    reference = torch.mm(x.float(), weight_t.float()).to(torch.bfloat16)
    torch.mm(x, weight_t)
    torch.cuda.synchronize()
    baseline_samples = []
    for _ in range(12):
        flush_destination.copy_(flush_source)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(x, weight_t)
        stop.record()
        stop.synchronize()
        baseline_samples.append(start.elapsed_time(stop) * 1000.0)
    baseline_samples.sort()
    print(
        f"  torch baseline median={baseline_samples[len(baseline_samples)//2]:.3f}us "
        f"best={baseline_samples[0]:.3f}us"
    )
    for config in configs:
        bm, bn, bk, split_k, warps, stages = config
        output = torch.empty((rows, n), device="cuda", dtype=torch.bfloat16)
        partial = (
            torch.empty((split_k, rows, n), device="cuda", dtype=torch.float32)
            if split_k != 1
            else None
        )
        launch(x, weight_t, output, partial, config)
        torch.cuda.synchronize()
        relative_l2 = torch.linalg.vector_norm(output.float() - reference.float()) / torch.linalg.vector_norm(reference.float())
        probe_l2 = torch.linalg.vector_norm(output[0].float() - reference[0].float()) / torch.linalg.vector_norm(reference[0].float())
        samples = []
        for _ in range(12):
            flush_destination.copy_(flush_source)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(x, weight_t, output, partial, config)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0)
        samples.sort()
        print(
            f"  BM{bm} BN{bn} BK{bk} S{split_k} W{warps}: "
            f"median={samples[len(samples)//2]:.3f}us best={samples[0]:.3f}us "
            f"rel={relative_l2.item():.2e} probe={probe_l2.item():.2e}"
        )
    del x, weight, weight_t, reference
    torch.cuda.empty_cache()
