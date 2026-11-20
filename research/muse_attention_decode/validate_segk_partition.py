"""Stress-test and model the flat kernel's OKC_SEGK partition.

This mirrors submission.py's integer virtual-key partition and vmap logic.  It
does not execute CUDA; it verifies exact coverage and compares tail work under
a configurable per-request-boundary cost.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics


def _cum(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _vmap(v: int, cum: list[int], segk: int) -> int:
    lo_b, hi_b = 0, len(cum) - 2
    while lo_b < hi_b:
        mid = (lo_b + hi_b + 1) >> 1
        if cum[mid] + mid * segk <= v:
            lo_b = mid
        else:
            hi_b = mid - 1
    return min(v - lo_b * segk, cum[lo_b + 1])


def _virtual_ranges(total: int, grid: int, w0: int | None) -> list[tuple[int, int, float]]:
    """Return (start, end, relative_speed) for every resident CTA slot."""
    if w0 is None:
        return [(c * total // grid, (c + 1) * total // grid, 1.0) for c in range(grid)]

    assert grid % 2 == 0
    fast = w0 / (2048 - w0)
    ranges = []
    for pair in range(grid // 2):
        ranges.append(
            (
                total * (2048 * pair) // (1024 * grid),
                total * (2048 * pair + w0) // (1024 * grid),
                fast,
            )
        )
        ranges.append(
            (
                total * (2048 * pair + w0) // (1024 * grid),
                total * (2048 * pair + 2048) // (1024 * grid),
                1.0,
            )
        )
    return ranges


def partition(
    lengths: list[int], grid: int, segk: int, w0: int | None = None
) -> list[tuple[int, int, int, float]]:
    """Return each CTA's (real start, real end, segment count, speed)."""
    cum = _cum(lengths)
    total = cum[-1] + (len(lengths) - 1) * segk
    out = []
    for virtual_start, virtual_end, speed in _virtual_ranges(total, grid, w0):
        start = _vmap(virtual_start, cum, segk) if segk else virtual_start
        end = _vmap(virtual_end, cum, segk) if segk else virtual_end
        nseg = sum(cum[b + 1] > start and cum[b] < end for b in range(len(lengths)))
        out.append((start, end, nseg, speed))
    return out


def verify(lengths: list[int], grid: int, segk: int, w0: int | None = None) -> None:
    parts = partition(lengths, grid, segk, w0)
    nonempty = sorted((start, end) for start, end, _, _ in parts if start < end)
    cursor = 0
    for start, end in nonempty:
        assert start == cursor, (lengths, grid, segk, w0, cursor, start, end)
        assert end > start
        cursor = end
    assert cursor == sum(lengths), (lengths, grid, segk, w0, cursor)
    assert max((nseg for _, _, nseg, _ in parts), default=0) <= 64


def tail_time(lengths: list[int], grid: int, segk: int, boundary_cost: int, w0: int | None) -> float:
    times = []
    for start, end, nseg, speed in partition(lengths, grid, segk, w0):
        if start < end:
            work = end - start + boundary_cost * max(0, nseg - 1)
            times.append(work / speed)
    return max(times, default=0.0)


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--boundary-cost", type=int, default=384)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Scored q8 full shapes, plus small/many/DP-style grids that exercise the
    # same generic mapping.  w0=1135 mirrors the H200 q8 flat resident grid.
    cases = [
        (16, 264, 1135, "scored-b16"),
        (32, 264, 1135, "scored-b32"),
        (2, 33, None, "dp-b2"),
        (4, 264, 1135, "test-b4"),
        (8, 33, None, "dp-b8"),
        (64, 264, 1135, "many-b64"),
    ]
    candidates = (0, 256, 320, 384, 448, 512, 600, 640)
    samples = {name: {segk: [] for segk in candidates} for _, _, _, name in cases}

    for _ in range(args.trials):
        for batch, grid, w0, name in cases:
            # Exact scored spread: uniform around 80K by +/-7.5%.
            lengths = [int(80_000 * (1 + 0.075 * (2 * rng.random() - 1))) + 8 for _ in range(batch)]
            for segk in candidates:
                verify(lengths, grid, segk, w0)
                samples[name][segk].append(tail_time(lengths, grid, segk, args.boundary_cost, w0))

    # Broad correctness stress: tiny lengths expose virtual-gap plateaus and
    # arbitrary grids expose rounding at every integer boundary.
    for _ in range(args.trials):
        batch = rng.randint(2, 64)
        lengths = [rng.randint(1, 400_000) for _ in range(batch)]
        grid = rng.randint(1, 300)
        for segk in (1, 64, 384, 600, 4096):
            verify(lengths, grid, segk)
        if grid % 2 == 0:
            verify(lengths, grid, 384, 1135)

    print(f"coverage: PASS ({args.trials * len(cases) * len(candidates):,} modeled candidate cases + stress)")
    print(f"assumed boundary cost: {args.boundary_cost} key-equivalents")
    for _, _, _, name in cases:
        baseline = statistics.mean(samples[name][600])
        print(name)
        for segk in candidates:
            values = samples[name][segk]
            mean = statistics.mean(values)
            print(
                f"  SEGK={segk:3d}: mean={mean:9.2f}  "
                f"p99={percentile(values, 0.99):9.2f}  vs600={(mean / baseline - 1) * 100:+7.3f}%"
            )


if __name__ == "__main__":
    main()
