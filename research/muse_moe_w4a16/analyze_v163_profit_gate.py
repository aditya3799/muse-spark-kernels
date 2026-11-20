"""Host model for a legal device-side MID8/SKT16 profitability gate.

The routing generator matches the task's distributional recipe.  CPU and CUDA
generators do not produce identical values for a seed, so this is a coverage
model, not a replacement for paired H200 timing.
"""

from __future__ import annotations

from collections import Counter

import torch


NSEEDS = 16_384
BS = 16
NEXPERTS = 256
NLOCAL = 32
TOPK = 8

# v162 seed 1250: equal job counts, SKT16 / MID8 = 107.0 / 88.2.
OBSERVED_EQUAL_JOB_RATIO = 107.0 / 88.2
PROMOTION_RATIO = 0.995


def route_counts(seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((BS, NEXPERTS), generator=gen)
    z = torch.randn((NEXPERTS,), generator=gen)
    u = torch.rand((NEXPERTS,), generator=gen).clamp(min=1e-12)
    scores.add_(0.03 * z / (-u.log()).sqrt())
    ids = scores.topk(TOPK, dim=-1).indices
    gpu_counts = torch.bincount((ids // NLOCAL).flatten(), minlength=NEXPERTS // NLOCAL)
    local = (ids - NLOCAL * gpu_counts.argmax()) % NEXPERTS
    local = local[(local >= 0) & (local < NLOCAL)]
    return torch.bincount(local, minlength=NLOCAL)


def main() -> None:
    hist: Counter[int] = Counter()
    touched_hist: Counter[int] = Counter()
    ratios: list[float] = []
    gate_hits = 0
    gate_gain = 0.0
    total_jobs8 = 0
    total_jobs16 = 0

    for i in range(NSEEDS):
        counts = route_counts(1234 + 1009 * i + BS)
        touched = int((counts > 0).sum())
        jobs8 = int(((counts + 7) // 8).sum())
        jobs16 = int(((counts + 15) // 16).sum())
        saved = jobs8 - jobs16
        ratio = OBSERVED_EQUAL_JOB_RATIO * jobs16 / jobs8
        hist[saved] += 1
        touched_hist[touched] += 1
        ratios.append(ratio)
        total_jobs8 += jobs8
        total_jobs16 += jobs16
        if ratio <= PROMOTION_RATIO:
            gate_hits += 1
            gate_gain += 1.0 - ratio

    print(f"seeds={NSEEDS}")
    print(f"mean jobs8={total_jobs8 / NSEEDS:.5f} jobs16={total_jobs16 / NSEEDS:.5f} "
          f"reduction={(total_jobs8 - total_jobs16) / total_jobs8:.3%}")
    print("saved-job histogram:")
    for saved in sorted(hist):
        print(f"  {saved}: {hist[saved]:6d} ({hist[saved] / NSEEDS:7.3%})")
    print("gate model: SKT16/MID8 = equal_job_ratio * jobs16/jobs8")
    print(f"  equal_job_ratio={OBSERVED_EQUAL_JOB_RATIO:.6f} promotion<={PROMOTION_RATIO:.3f}")
    print(f"  eligible={gate_hits}/{NSEEDS} ({gate_hits / NSEEDS:.4%})")
    print(f"  modeled mean gain across all M16 calls={gate_gain / NSEEDS:.6%}")
    for need in range(1, 5):
        n = sum(v for saved, v in hist.items() if saved >= need)
        print(f"  saved>={need}: {n}/{NSEEDS} ({n / NSEEDS:.4%})")
    print(f"touched range={min(touched_hist)}..{max(touched_hist)}")


if __name__ == "__main__":
    main()
