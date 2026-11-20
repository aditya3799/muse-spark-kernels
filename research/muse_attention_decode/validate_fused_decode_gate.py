"""Local smoke test for the v14 fused decode RMS-normalization and gate."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parent
FP8 = torch.float8_e4m3fn


def cuda_source(filename: str) -> str:
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_ATTENTION_CUDA"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"_ATTENTION_CUDA not found in {filename}")


def build(name: str, filename: str):
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_source(filename),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )


def run_case(v8, v14, batch: int, ctx: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(14000 + batch)
    q = torch.randn((batch, 16, 64), device="cuda", dtype=torch.float16,
                    generator=generator).to(FP8)
    k = torch.randn((batch * ctx, 1, 2, 64), device="cuda", dtype=torch.float16,
                    generator=generator).to(FP8)
    v = torch.randn((batch * ctx, 1, 2, 64), device="cuda", dtype=torch.float16,
                    generator=generator).to(FP8)
    page_table = torch.randperm(batch * ctx, device="cuda", generator=generator)
    page_table = page_table.to(torch.int32).view(batch, ctx)
    lengths = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    qkv = torch.randn((batch, 36 * 64), device="cuda", dtype=torch.bfloat16,
                      generator=generator)
    splits = 64 if batch == 1 else (8 if batch == 16 else 4)

    partials8 = torch.empty((batch, 2, splits, 8, 66), device="cuda")
    partials14 = torch.empty_like(partials8)
    out8 = torch.empty((batch, 16, 64), device="cuda", dtype=torch.bfloat16)
    out14 = torch.empty_like(out8)
    raw = v8.paged_full_decode(
        q, k, v, page_table, lengths, partials8, out8, splits, 1.0 / 8.0
    ).float()
    fused = v14.paged_full_decode(
        q, qkv, k, v, page_table, lengths, partials14, out14, splits, 1.0 / 8.0
    ).float()

    inv_rms = torch.rsqrt(raw.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    gate = qkv[:, 20 * 64:].float().view(batch, 16, 64)
    reference = (raw * inv_rms * torch.sigmoid(gate)).to(torch.bfloat16).float()
    relative_l2 = ((fused - reference).norm() / reference.norm()).item()
    max_abs = (fused - reference).abs().max().item()
    print(
        f"batch={batch:2d} ctx={ctx:4d} relative_l2={relative_l2:.3e} "
        f"max_abs={max_abs:.3e} finite={bool(torch.isfinite(fused).all())}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".torch_extensions"))
    print(f"GPU: {torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    v8 = build("okc_local_warp_private_v8_gate_reference", "submission_warp_private_v8.py")
    v14 = build("okc_local_warp_private_v14_fused_gate", "submission_warp_private_v14.py")
    for batch in (1, 16, 32):
        run_case(v8, v14, batch=batch, ctx=64)


if __name__ == "__main__":
    main()
