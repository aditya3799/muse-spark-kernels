"""Bound denominator-cancellation error for scored coherent full attention."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--nkeys", type=int, default=80_000)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(20261101)
    eps = 1e-5
    print("sigma  tail(mean/min)  output_rms(mean/min)  cancellation_error(mean/max)")
    for sigma in (3.3, 3.5, 3.7, 3.9):
        vals = []
        for _ in range(args.trials):
            scores = sigma * torch.randn(args.nkeys, generator=gen)
            weights = torch.softmax(scores, dim=0)
            top = torch.topk(scores, 2048).indices
            tail = 1.0 - weights[top].sum()
            # Random FP8 values contribute sum(w^2) variance per dimension;
            # the coherent tail adds a unit-RMS shared direction with mass tail.
            output_rms = torch.sqrt(tail.square() + weights.square().sum())
            error = torch.sqrt(1.0 + eps / output_rms.square()) - 1.0
            vals.append((float(tail), float(output_rms), float(error)))
        v = torch.tensor(vals)
        print(
            f"{sigma:3.1f}    {v[:,0].mean():.5f}/{v[:,0].min():.5f}      "
            f"{v[:,1].mean():.5f}/{v[:,1].min():.5f}          "
            f"{100*v[:,2].mean():.4f}%/{100*v[:,2].max():.4f}%"
        )


if __name__ == "__main__":
    main()
