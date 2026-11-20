"""Extract and test the Triton router kernels without importing judge-only packages."""

import ast
import sys
from pathlib import Path

import torch


source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("submission_router_triton.py")
tree = ast.parse(source_path.read_text())
nodes = []
for node in tree.body:
    if isinstance(node, ast.Import):
        imported = {alias.name for alias in node.names}
        if imported & {"torch", "triton", "triton.language"}:
            nodes.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
        "_router_hilo_matmul_kernel",
        "_router_split_reduce_kernel",
        "_router_hilo_matmul",
    }:
        nodes.append(node)

namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source_path), "exec"), namespace)
router = namespace["_router_hilo_matmul"]

torch.manual_seed(3)
weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
hi = weight.to(torch.bfloat16)
lo = (weight - hi.float()).to(torch.bfloat16)
hi_t = hi.t().contiguous()
lo_t = lo.t().contiguous()
packed = torch.cat((hi_t, lo_t), dim=1).contiguous()
splits = {16: 16, 32: 8, 128: 1, 256: 1}
flush_source = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
flush_destination = torch.empty_like(flush_source)

for rows, split_k in splits.items():
    x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((rows, 256), device="cuda", dtype=torch.float32)
    partial = (
        torch.empty((split_k, rows, 256), device="cuda", dtype=torch.float32)
        if split_k != 1
        else None
    )
    actual = router(x, hi_t, lo_t, output, partial, split_k)
    expected = torch.mm(x.float(), hi_t.float()) + torch.mm(x.float(), lo_t.float())
    torch.cuda.synchronize()
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    print(
        f"router m={rows} split={split_k}: relative_l2={relative_l2.item():.3e} "
        f"max_abs={(actual - expected).abs().max().item():.3e}"
    )

    router(x, hi_t, lo_t, output, partial, split_k)
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        flush_destination.copy_(flush_source)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        router(x, hi_t, lo_t, output, partial, split_k)
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    samples.sort()
    print(
        f"router m={rows}: cold_median_us={samples[len(samples) // 2]:.3f} "
        f"cold_best_us={samples[0]:.3f}"
    )

    torch.mm(x, packed)
    torch.cuda.synchronize()
    baseline_samples = []
    for _ in range(30):
        flush_destination.copy_(flush_source)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        pair = torch.mm(x, packed)
        pair[:, :256] + pair[:, 256:]
        stop.record()
        stop.synchronize()
        baseline_samples.append(start.elapsed_time(stop) * 1000.0)
    baseline_samples.sort()
    print(
        f"packed BF16 baseline m={rows}: "
        f"cold_median_us={baseline_samples[len(baseline_samples) // 2]:.3f} "
        f"cold_best_us={baseline_samples[0]:.3f}"
    )
