"""Model sparse softmax-denominator estimators for the q_len=8 flat kernel."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def attention_weights(scores: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-quantized and FP8 weights with the kernel's per-chunk offset."""
    n = scores.numel()
    padded = ((n + chunk - 1) // chunk) * chunk
    x = F.pad(scores, (0, padded - n), value=-torch.inf).view(-1, chunk)
    p = torch.exp(x - x.amax(dim=1, keepdim=True)).flatten()[:n]
    return p, p.to(torch.float8_e4m3fn).float()


def patterns(n: int, sigma: float, gen: torch.Generator):
    iid = sigma * torch.randn(n, generator=gen)
    yield "iid", iid

    # Smoothly correlated scores stress fixed-position sampling without making
    # attention artificially sparse (the challenge explicitly spreads it).
    raw = torch.randn(n, generator=gen)
    corr = F.avg_pool1d(raw.view(1, 1, -1), kernel_size=65, stride=1, padding=32).view(-1) * (65.0**0.5)
    yield "ar98", sigma * corr

    phase = torch.rand((), generator=gen) * 6.283185307179586
    x = torch.arange(n, dtype=torch.float32)
    wave = torch.sin(x * (6.283185307179586 / 257.0) + phase)
    yield "wave257", sigma * wave


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    # These are the full-attention correctness/scored lengths affected by the
    # flat q8 specialization.  Chunk 303 is pessimistic versus the thousands
    # of keys usually owned by one flat CTA segment.
    for n in (6_000, 20_000, 80_000, 100_000):
        for sigma in (1.0, 3.5, 6.0):
            by_case: dict[tuple[str, int], list[float]] = {}
            for _ in range(args.trials):
                for name, scores in patterns(n, sigma, gen):
                    pre, post = attention_weights(scores, 303)
                    true = post.sum()
                    for stride in (2, 4, 8):
                        # Average every phase to expose fixed-phase bias as
                        # well as the best/worst phase available in a tile.
                        for phase in range(stride):
                            est = stride * pre[phase::stride].sum()
                            err = float(est / true - 1.0)
                            by_case.setdefault((name, stride), []).append(err)
            for (name, stride), vals in sorted(by_case.items()):
                e = torch.tensor(vals)
                print(
                    f"n={n:6d} sigma={sigma:3.1f} {name:7s} 1/{stride:<2d} "
                    f"mean={e.mean()*100:+7.3f}% p01={e.quantile(.01)*100:+7.3f}% "
                    f"p99={e.quantile(.99)*100:+7.3f}% worst={e.abs().max()*100:7.3f}%"
                )


if __name__ == "__main__":
    main()
