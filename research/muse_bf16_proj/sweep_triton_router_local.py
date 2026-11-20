"""Directional cold-cache tiling sweep for the fused hi/lo Triton router."""

import ast
from pathlib import Path

import torch
import triton


source_path = Path("submission_router_triton.py")
tree = ast.parse(source_path.read_text())
nodes = []
for node in tree.body:
    if isinstance(node, ast.Import):
        if {alias.name for alias in node.names} & {"torch", "triton", "triton.language"}:
            nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in {
        "_router_hilo_matmul_kernel",
        "_router_split_reduce_kernel",
    }:
        nodes.append(node)
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source_path), "exec"), namespace)
matmul_kernel = namespace["_router_hilo_matmul_kernel"]
reduce_kernel = namespace["_router_split_reduce_kernel"]


def launch(x, hi_t, lo_t, output, partial, config):
    rows = x.shape[0]
    bm, bn, bk, split_k, warps, stages = config
    destination = output if split_k == 1 else partial
    grid = (triton.cdiv(rows, bm), triton.cdiv(256, bn), split_k)
    matmul_kernel[grid](
        x,
        hi_t,
        lo_t,
        destination,
        M=rows,
        N=256,
        K=8192,
        SPLIT_K=split_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    if split_k != 1:
        elements = rows * 256
        reduce_kernel[(triton.cdiv(elements, 256),)](
            partial,
            output,
            ELEMENTS=elements,
            SPLIT_K=split_k,
            BLOCK=256,
            num_warps=4,
        )


configs = {
    16: [
        (16, 16, 64, 8, 4, 3),
        (16, 32, 32, 16, 4, 3),
        (16, 32, 64, 16, 4, 3),
        (16, 32, 128, 16, 4, 3),
        (16, 64, 64, 32, 4, 3),
        (16, 64, 64, 32, 8, 3),
        (32, 32, 64, 16, 4, 3),
    ],
    128: [
        (16, 16, 64, 1, 4, 3),
        (16, 32, 32, 1, 4, 3),
        (16, 32, 64, 1, 4, 3),
        (16, 32, 128, 1, 4, 3),
        (16, 64, 64, 1, 4, 3),
        (32, 32, 64, 1, 4, 3),
        (64, 64, 64, 1, 8, 3),
    ],
    32: [
        (16, 16, 64, 4, 4, 3),
        (16, 32, 32, 8, 4, 3),
        (16, 32, 64, 8, 4, 3),
        (16, 32, 128, 8, 4, 3),
        (16, 64, 32, 16, 4, 3),
        (32, 32, 32, 16, 4, 3),
    ],
    256: [
        (16, 32, 32, 1, 4, 3),
        (16, 32, 64, 1, 4, 3),
        (16, 64, 32, 1, 4, 3),
        (16, 64, 64, 1, 4, 3),
        (32, 32, 32, 1, 4, 3),
        (32, 64, 32, 1, 4, 3),
        (64, 64, 64, 1, 8, 3),
        (32, 64, 32, 2, 4, 3),
    ],
}

torch.manual_seed(4)
weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
hi = weight.to(torch.bfloat16)
lo = (weight - hi.float()).to(torch.bfloat16)
hi_t = hi.t().contiguous()
lo_t = lo.t().contiguous()
flush_source = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
flush_destination = torch.empty_like(flush_source)

for rows, row_configs in configs.items():
    x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(x.float(), hi_t.float()) + torch.mm(x.float(), lo_t.float())
    print(f"m={rows}")
    for config in row_configs:
        bm, bn, bk, split_k, warps, stages = config
        output = torch.empty((rows, 256), device="cuda", dtype=torch.float32)
        partial = (
            torch.empty((split_k, rows, 256), device="cuda", dtype=torch.float32)
            if split_k != 1
            else None
        )
        launch(x, hi_t, lo_t, output, partial, config)
        torch.cuda.synchronize()
        relative_l2 = torch.linalg.vector_norm(output - reference) / torch.linalg.vector_norm(reference)
        samples = []
        for _ in range(12):
            flush_destination.copy_(flush_source)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(x, hi_t, lo_t, output, partial, config)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0)
        samples.sort()
        print(
            f"  BM{bm} BN{bn} BK{bk} S{split_k} W{warps}: "
            f"median={samples[len(samples)//2]:.3f}us best={samples[0]:.3f}us "
            f"rel={relative_l2.item():.2e}"
        )
