"""Measure whether row-scaled E4M3 O-weight error has a useful low-rank correction.

This is a local numerical gate only.  The official O weights are IID Gaussian BF16,
so a fresh matrix has the same residual-spectrum distribution without allocating the
unrelated expert weights that precede O in the reference RNG stream.
"""

import math
import time

import torch


N, K = 8192, 1024
RANKS = (16, 32, 64, 128, 192, 256, 384)
Q = max(RANKS)


def sync_time(label, fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    print(f"{label}: {time.perf_counter() - start:.3f}s", flush=True)
    return value


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20261103)
    w = (torch.randn((N, K), device="cuda") / math.sqrt(K)).to(torch.bfloat16)
    wf = w.float()
    scale = wf.abs().amax(1, keepdim=True).clamp_min(1e-30) / 448.0
    wq = (wf / scale).to(torch.float8_e4m3fn).float() * scale
    residual = wf - wq
    total = residual.square().sum().double()
    weight_rel = (total.sqrt() / wf.square().sum().double().sqrt()).item()
    print(f"weight residual relative Frobenius: {weight_rel:.8f}", flush=True)

    # Randomized PCA returns the leading approximate singular triplets.  Two
    # power iterations are enough to decide whether the spectrum is flat.
    _, s, _ = sync_time(
        f"pca_lowrank q={Q}",
        lambda: torch.pca_lowrank(residual, q=Q, center=False, niter=2),
    )
    s2 = s.double().square().cumsum(0)
    for rank in RANKS:
        captured = (s2[rank - 1] / total).item()
        remaining = math.sqrt(max(0.0, 1.0 - captured))
        # Relative FLOPs of the two BF16 correction GEMMs versus the original
        # [T,K] x [N,K]^T projection.  This is an optimistic timing lower bound.
        flop_ratio = rank * (K + N) / (K * N)
        print(
            f"rank {rank:3d}: energy {captured:7.3%}, residual factor {remaining:.5f}, "
            f"correction FLOPs {flop_ratio:7.3%}",
            flush=True,
        )

    # Verify that output error follows the residual-energy prediction.
    rows = 512
    x = torch.randn((rows, K), device="cuda").to(torch.bfloat16)
    ref = x.float() @ wf.t()
    base = x.float() @ wq.t()
    print(f"uncorrected output relative L2: {((base - ref).norm() / ref.norm()).item():.8f}", flush=True)


if __name__ == "__main__":
    main()
