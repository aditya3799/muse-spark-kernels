"""Validate the warp-private MMA permutations and FP8-probability model."""

from __future__ import annotations

import argparse

import torch


FP8 = torch.float8_e4m3fn
PROBABILITY_SCALE = 128.0


def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


def qk_dimension_permutation(offset: int) -> list[int]:
    logical_to_physical = []
    for logical in range(32):
        half = logical // 16
        within = logical % 16
        thread = within // 4
        element = within % 4
        logical_to_physical.append(16 * thread + 4 * half + element + offset)
    return logical_to_physical


def pv_key_permutation() -> list[int]:
    logical_to_key = [0] * 32
    for thread in range(4):
        first = [2 * thread, 2 * thread + 1, 8 + 2 * thread, 9 + 2 * thread]
        second = [16 + 2 * thread, 17 + 2 * thread, 24 + 2 * thread, 25 + 2 * thread]
        logical_to_key[4 * thread : 4 * thread + 4] = first
        logical_to_key[16 + 4 * thread : 20 + 4 * thread] = second
    return logical_to_key


def validate_permutations(device: str) -> None:
    gen = torch.Generator(device=device).manual_seed(123)
    q = torch.randn((8, 64), generator=gen, device=device)
    k = torch.randn((32, 64), generator=gen, device=device)
    v = torch.randn((32, 64), generator=gen, device=device)
    p = torch.rand((8, 32), generator=gen, device=device)

    qk = torch.zeros((8, 32), device=device)
    for offset in (0, 8):
        permutation = qk_dimension_permutation(offset)
        qk += q[:, permutation] @ k[:, permutation].T
    qk_reference = q @ k.T

    key_permutation = pv_key_permutation()
    pv = torch.empty((8, 64), device=device)
    for n in range(8):
        logical_p = p[:, key_permutation]
        logical_v = torch.stack(
            [v[key_permutation, 8 * column + n] for column in range(8)], dim=1
        )
        logical_out = logical_p @ logical_v
        for column in range(8):
            pv[:, 8 * column + n] = logical_out[:, column]
    pv_reference = p @ v

    print(f"QK permutation max_abs={(qk - qk_reference).abs().max().item():.3e}")
    print(f"PV permutation max_abs={(pv - pv_reference).abs().max().item():.3e}")


def warp_private_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    heads = q.shape[0]
    warp_states = []
    groups = (k.shape[0] + 31) // 32
    for warp in range(8):
        running_max = torch.full((heads,), -torch.inf, device=q.device)
        running_sum = torch.zeros((heads,), device=q.device)
        running_out = torch.zeros((heads, 64), device=q.device)
        for group in range(warp, groups, 8):
            begin = group * 32
            end = min(k.shape[0], begin + 32)
            scores = q.float() @ k[begin:end].float().T / 8.0
            tile_max = scores.max(dim=1).values
            probability = torch.exp(scores - tile_max[:, None])
            probability_fp8 = (probability * PROBABILITY_SCALE).to(FP8).float()
            tile_sum = probability_fp8.sum(dim=1) / PROBABILITY_SCALE
            tile_out = probability_fp8 @ v[begin:end].float() / PROBABILITY_SCALE
            next_max = torch.maximum(running_max, tile_max)
            old_scale = torch.exp(running_max - next_max)
            tile_scale = torch.exp(tile_max - next_max)
            running_sum = running_sum * old_scale + tile_sum * tile_scale
            running_out = running_out * old_scale[:, None] + tile_out * tile_scale[:, None]
            running_max = next_max
        warp_states.append((running_max, running_sum, running_out))

    global_max = torch.stack([state[0] for state in warp_states]).amax(dim=0)
    merged_sum = torch.zeros((heads,), device=q.device)
    merged_out = torch.zeros((heads, 64), device=q.device)
    for warp_max, warp_sum, warp_out in warp_states:
        factor = torch.exp(warp_max - global_max)
        factor = torch.where(torch.isfinite(factor), factor, torch.zeros_like(factor))
        merged_sum += warp_sum * factor
        merged_out += warp_out * factor[:, None]
    return merged_out / merged_sum[:, None]


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
    parser.add_argument("--contexts", type=int, nargs="+", default=[32, 257, 2048, 80000])
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for E4M3 conversion")
    device = "cuda"
    validate_permutations(device)
    for coherent in (False, True):
        label = "coherent" if coherent else "random"
        for ctx in args.contexts:
            errors = []
            row_errors = []
            for seed in range(args.seeds):
                q, k, v = make_case(ctx, 17000 + seed, coherent, device)
                scores = q.float() @ k.float().T / 8.0
                reference = torch.softmax(scores, dim=1) @ v.float()
                candidate = warp_private_attention(q, k, v)
                errors.append(((candidate - reference).norm() / reference.norm()).item())
                rows = (candidate - reference).norm(dim=1) / reference.norm(dim=1).clamp_min(1e-6)
                row_errors.append(rows.max().item())
            print(
                f"{label:8s} ctx={ctx:6d} rel_l2_max={max(errors):.6e} "
                f"max_row={max(row_errors):.6e}"
            )


if __name__ == "__main__":
    main()
