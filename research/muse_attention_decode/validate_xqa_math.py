"""Numerical model of the SM90 XQA-style 64-key attention tile.

This does not execute WGMMA.  It isolates the algorithmic error introduced by
mapping per-tile softmax probabilities into E4M3 for the second matrix product.
"""

from __future__ import annotations

import argparse

import torch


FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


def tiled_xqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    heads = q.shape[0]
    running_max = torch.full((heads,), -torch.inf, device=q.device)
    running_sum = torch.zeros((heads,), device=q.device)
    running_out = torch.zeros((heads, q.shape[1]), device=q.device)
    for begin in range(0, k.shape[0], 64):
        kt = k[begin : begin + 64].float()
        vt = v[begin : begin + 64].float()
        scores = q.float() @ kt.T / 8.0
        tile_max = scores.max(dim=1).values
        p = torch.exp(scores - tile_max[:, None])
        p_fp8 = (p * FP8_MAX).to(FP8).float() / FP8_MAX
        tile_sum = p.sum(dim=1)
        tile_out = p_fp8 @ vt
        next_max = torch.maximum(running_max, tile_max)
        old_scale = torch.exp(running_max - next_max)
        tile_scale = torch.exp(tile_max - next_max)
        running_sum = running_sum * old_scale + tile_sum * tile_scale
        running_out = (
            running_out * old_scale[:, None]
            + tile_out * tile_scale[:, None]
        )
        running_max = next_max
    return running_out / running_sum[:, None]


def make_case(ctx: int, seed: int, coherent: bool, device: str):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = rms(torch.randn((8, 64), generator=gen, device=device)) * 5.473
    if coherent:
        direction = rms(torch.randn((1, 64), generator=gen, device=device))
        k = rms(direction + 0.35 * torch.randn((ctx, 64), generator=gen, device=device))
        value_direction = torch.randn((1, 64), generator=gen, device=device)
        v = value_direction + 0.5 * torch.randn((ctx, 64), generator=gen, device=device)
    else:
        k = rms(torch.randn((ctx, 64), generator=gen, device=device))
        v = torch.randn((ctx, 64), generator=gen, device=device)
    return q.to(FP8), k.to(FP8), v.to(FP8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[64, 257, 2048, 80000])
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for local E4M3 conversion")
    device = "cuda"
    for coherent in (False, True):
        label = "coherent" if coherent else "random"
        for ctx in args.contexts:
            errors = []
            row_errors = []
            for seed in range(args.seeds):
                q, k, v = make_case(ctx, 9000 + seed, coherent, device)
                scores = q.float() @ k.float().T / 8.0
                reference = torch.softmax(scores, dim=1) @ v.float()
                candidate = tiled_xqa(q, k, v)
                errors.append(((candidate - reference).norm() / reference.norm()).item())
                per_row = (candidate - reference).norm(dim=1) / reference.norm(dim=1).clamp_min(1e-6)
                row_errors.append(per_row.max().item())
            print(
                f"{label:8s} ctx={ctx:6d} "
                f"rel_l2_max={max(errors):.6e} max_row={max(row_errors):.6e}"
            )


if __name__ == "__main__":
    main()
