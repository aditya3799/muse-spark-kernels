"""Estimate the denominator error from summing pre-quantized FP8 PV weights."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    # Natural-log score standard deviations.  3.5 approximates the scored full
    # soft/coherent generator after qk scaling; the others cover diffuse and
    # sharper distributions.  Chunk sizes mirror full and sliding split sizes.
    cases = ((80_000, 303), (2_048, 64), (10, 1), (5_000, 64))
    for nkeys, chunk in cases:
        for sigma in (0.0, 1.0, 3.5, 6.0):
            errors = []
            for _ in range(args.trials):
                scores = sigma * torch.randn(nkeys, generator=gen)
                pieces = []
                for start in range(0, nkeys, chunk):
                    x = scores[start : start + chunk]
                    pieces.append(torch.exp(x - x.max()))
                p = torch.cat(pieces)
                p8 = p.to(torch.float8_e4m3fn).float()
                errors.append(float(p.sum() / p8.sum() - 1.0))
            e = torch.tensor(errors)
            print(
                f"n={nkeys:6d} chunk={chunk:3d} sigma={sigma:3.1f}  "
                f"mean={e.mean() * 100:+7.3f}%  "
                f"p01={e.quantile(.01) * 100:+7.3f}%  "
                f"p99={e.quantile(.99) * 100:+7.3f}%  "
                f"worst={e.abs().max() * 100:7.3f}%"
            )


if __name__ == "__main__":
    main()
