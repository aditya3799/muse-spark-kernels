"""Offline paired benchmark for the custom paged-decode CUDA kernels.

This deliberately does not contact the OpenKernel service.  It extracts the
inline CUDA source from the archived submissions, compiles it for the local
GPU, checks v3 against v2, and reports paired CUDA-event timings.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from statistics import median

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parent
FP8 = torch.float8_e4m3fn


def cuda_source(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "_ATTENTION_CUDA" for target in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"_ATTENTION_CUDA not found in {path}")


def build(name: str, filename: str):
    cuda_flags = ["-O3", "--use_fast_math", "-lineinfo"]
    if os.environ.get("OKC_BUILD_SM90A") == "1":
        cuda_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_source(ROOT / filename),
        extra_cuda_cflags=cuda_flags,
        with_cuda=True,
        verbose=True,
    )


def make_inputs(batch: int, ctx: int, seed: int = 1234):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_pages = batch * ctx
    q = torch.randn((batch, 16, 64), device="cuda", dtype=torch.float16, generator=generator).to(FP8)
    k = torch.randn((total_pages, 1, 2, 64), device="cuda", dtype=torch.float16, generator=generator).to(FP8)
    v = torch.randn((total_pages, 1, 2, 64), device="cuda", dtype=torch.float16, generator=generator).to(FP8)
    page_table = torch.randperm(total_pages, device="cuda", dtype=torch.int64, generator=generator)
    page_table = page_table.to(torch.int32).view(batch, ctx)
    lengths = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    return q, k, v, page_table, lengths


def buffers(batch: int, splits: int):
    partials = torch.empty((batch, 2, splits, 8, 66), device="cuda", dtype=torch.float32)
    output = torch.empty((batch, 16, 64), device="cuda", dtype=torch.bfloat16)
    return partials, output


def invoke(module, tensors, scratch, splits):
    q, k, v, page_table, lengths = tensors
    partials, output = scratch
    return module.paged_full_decode(
        q, k, v, page_table, lengths, partials, output, splits, 1.0 / 8.0
    )


def timed(module, tensors, scratch, splits, flush, warmup: int, repeats: int):
    for _ in range(warmup):
        invoke(module, tensors, scratch, splits)
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        flush.zero_()
        start.record()
        invoke(module, tensors, scratch, splits)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--baseline-file", default="submission_custom_kvshare_v2.py")
    parser.add_argument("--candidate-file", default="submission_custom_v3.py")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} SM{capability[0]}{capability[1]}")
    print("This GPU cannot execute SM90 WGMMA; results cover the portable CUDA path only.")

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".torch_extensions"))
    baseline_stem = Path(args.baseline_file).stem.replace("-", "_")
    candidate_stem = Path(args.candidate_file).stem.replace("-", "_")
    v2 = build(f"okc_local_{baseline_stem}", args.baseline_file)
    v3 = build(f"okc_local_{candidate_stem}", args.candidate_file)
    flush = torch.empty((64 * 1024 * 1024,), device="cuda", dtype=torch.uint8)

    for batch in args.batches:
        splits = 8 if batch == 16 else 4
        tensors = make_inputs(batch, args.ctx, seed=1000 + batch)
        scratch2 = buffers(batch, splits)
        scratch3 = buffers(batch, splits)
        out2 = invoke(v2, tensors, scratch2, splits).float()
        out3 = invoke(v3, tensors, scratch3, splits).float()
        torch.cuda.synchronize()
        relative_l2 = ((out3 - out2).norm() / out2.norm()).item()

        # Alternate ordering to reduce clock and thermal bias.
        t2a = timed(v2, tensors, scratch2, splits, flush, args.warmup, args.repeats)
        t3a = timed(v3, tensors, scratch3, splits, flush, args.warmup, args.repeats)
        t3b = timed(v3, tensors, scratch3, splits, flush, args.warmup, args.repeats)
        t2b = timed(v2, tensors, scratch2, splits, flush, args.warmup, args.repeats)
        t2 = median(t2a + t2b)
        t3 = median(t3a + t3b)
        print(
            f"B={batch:2d} ctx={args.ctx:6d}: baseline={t2:10.2f} us  "
            f"candidate={t3:10.2f} us  speedup={t2 / t3:7.3f}x  rel_l2={relative_l2:.3e}"
        )


if __name__ == "__main__":
    main()
