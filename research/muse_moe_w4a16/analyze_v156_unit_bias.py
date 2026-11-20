"""Offline balance model for a unit-final cost bias in the v139 WGS scheduler.

The production scheduler assigns every K=256 stage the same cost for a given
job width.  A complete 16-stage output unit also pays an epilogue; split units
pay fixup work at range boundaries.  This model asks whether charging the
unit-final stage one or more extra stage-equivalents can reduce the modeled
CTA tail without changing ownership, ordering, or the CUDA math.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle, islice


GRID = 132
COST_A = (112, 135, 140, 158, 173, 185, 200, 207)
COST_B = (100, 104, 115, 123, 132, 144, 156, 174)


def stage_cost(pb: int, count: int) -> int:
    return (COST_B if pb else COST_A)[(count + 7) // 8 - 1]


def unit_cost(pb: int, count: int, bias: int) -> int:
    return stage_cost(pb, count) * (16 + bias)


def prefixes(counts: tuple[int, ...], pb: int, bias: int) -> tuple[list[int], int]:
    units = 16 if pb else 32
    out = [0]
    for count in counts:
        out.append(out[-1] + units * unit_cost(pb, count, bias))
    return out, out[-1]


def stage_at(counts: tuple[int, ...], pb: int, pre: list[int], x: int, bias: int) -> int:
    lo, hi = 0, len(counts)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pre[mid] <= x:
            lo = mid
        else:
            hi = mid
    c = stage_cost(pb, counts[lo])
    uc = c * (16 + bias)
    within = x - pre[lo]
    unit = within // uc
    stage = min(15, (within - unit * uc) // c)
    return (lo << (8 if pb else 9)) + unit * 16 + stage


def ranges(counts: tuple[int, ...], pb: int, bias: int, grid: int = GRID) -> list[tuple[int, int]]:
    pre, total = prefixes(counts, pb, bias)
    end = len(counts) << (8 if pb else 9)
    out = []
    for cta in range(grid):
        g0 = stage_at(counts, pb, pre, total * cta // grid, bias)
        g1 = end if cta == grid - 1 else stage_at(counts, pb, pre, total * (cta + 1) // grid, bias)
        out.append((g0, g1))
    return out


def true_prefix(counts: tuple[int, ...], pb: int, epi: float) -> list[float]:
    shift = 8 if pb else 9
    end = len(counts) << shift
    out = [0.0]
    for gs in range(end):
        j = gs >> shift
        c = stage_cost(pb, counts[j])
        value = float(c)
        if (gs & 15) == 15:
            value += epi * c
        out.append(out[-1] + value)
    return out


def true_load(counts: tuple[int, ...], pb: int, r: tuple[int, int], pre: list[float], fix: float) -> float:
    shift = 8 if pb else 9
    g0, g1 = r
    total = pre[g1] - pre[g0]
    if g0 < g1:
        if g0 & 15:
            total += fix * stage_cost(pb, counts[g0 >> shift])
        if g1 & 15:
            total += fix * stage_cost(pb, counts[(g1 - 1) >> shift])
    return total


@dataclass(frozen=True)
class Result:
    max_load: float
    spread: float
    split_edges: int


def evaluate(counts: tuple[int, ...], pb: int, bias: int, pre: list[float], fix: float) -> Result:
    rs = ranges(counts, pb, bias)
    end = len(counts) << (8 if pb else 9)
    assert rs[0][0] == 0 and rs[-1][1] == end
    assert all(rs[i][1] == rs[i + 1][0] for i in range(len(rs) - 1))
    loads = [true_load(counts, pb, r, pre, fix) for r in rs]
    nonempty = [v for v, r in zip(loads, rs) if r[0] < r[1]]
    edges = sum(bool(g1 & 15) for _, g1 in rs[:-1])
    return Result(max(loads), max(nonempty) - min(nonempty), edges)


def make_counts(nj: int, pattern: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(islice(cycle(pattern), nj))


def main() -> None:
    observed_jobs = {
        32: (20, 17, 15, 18, 14, 15, 15),
        128: (26, 24, 25, 19, 29, 23, 20),
        256: (24, 29, 25, 27, 24, 26, 20),
    }
    patterns = {
        32: ((1,), (1, 2, 3, 4, 5, 6, 7, 8)),
        128: ((4, 6, 8, 9, 11, 13, 16), (1, 8, 9, 16)),
        256: ((6, 9, 13, 17, 21, 25, 32), (1, 8, 9, 16, 24, 32, 48, 64)),
    }
    # Model a unit epilogue as 0.5..4 stage-equivalents and each partial edge
    # as 0..1 stage-equivalent.  Report the candidate bias that minimizes the
    # worst CTA load over both phases and every observed job-count/pattern case.
    for epi in (0.5, 1.0, 2.0, 4.0):
        for fix in (0.0, 0.5, 1.0):
            worst: dict[int, float] = {b: 1.0 for b in range(5)}
            spread: dict[int, float] = {b: 0.0 for b in range(5)}
            edge_delta: dict[int, int] = {b: 0 for b in range(5)}
            for batch, jobs in observed_jobs.items():
                for nj in jobs:
                    for pattern in patterns[batch]:
                        counts = make_counts(nj, pattern)
                        for pb in (0, 1):
                            load_pre = true_prefix(counts, pb, epi)
                            base = evaluate(counts, pb, 0, load_pre, fix)
                            for bias in range(5):
                                got = evaluate(counts, pb, bias, load_pre, fix)
                                worst[bias] = max(worst[bias], got.max_load / base.max_load)
                                spread[bias] = max(spread[bias], got.spread / base.max_load)
                                edge_delta[bias] = max(edge_delta[bias], got.split_edges - base.split_edges)
            vals = "  ".join(
                f"b{b}:worst={worst[b]:.5f},spread={spread[b]:.4f},edge+={edge_delta[b]:d}" for b in range(5)
            )
            print(f"epi={epi:3.1f} fix={fix:3.1f}  {vals}")

    # Exhaustively prove monotone, gap-free ownership for all supported job and
    # grid bounds under every bounded bias.  Alternating narrow/wide jobs stress
    # the job-prefix inverse and empty-range cases.
    cases = 0
    for grid in range(1, 161):
        for nj in range(1, 137):
            counts = tuple(1 if j & 1 else 64 for j in range(nj))
            for pb in (0, 1):
                for bias in range(5):
                    rs = ranges(counts, pb, bias, grid)
                    end = nj << (8 if pb else 9)
                    assert rs[0][0] == 0 and rs[-1][1] == end
                    assert all(rs[i][1] == rs[i + 1][0] for i in range(grid - 1))
                    assert all(0 <= a <= b <= end for a, b in rs)
                    cases += 1
    print(f"PASS: {cases:,} biased phase schedules are monotone, gap-free, and exact")


if __name__ == "__main__":
    main()
