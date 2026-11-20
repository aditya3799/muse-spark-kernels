"""Offline model for a non-atomic WGS split-unit fixup protocol.

This mirrors v139's cost-weighted stage partition.  A CTA can own at most two
partial unit segments per phase.  The proposed protocol gives every CTA two
plain-store buffers (range-start and range-end) and lets the designated
finisher gather the contiguous contributor buffers after the existing stage
counter reaches 16.  It preserves the exact v139 stage ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from itertools import product


GRID = 132
COST_A = (112, 135, 140, 158, 173, 185, 200, 207)
COST_B = (100, 104, 115, 123, 132, 144, 156, 174)


def cost(pb: int, count: int) -> int:
    return (COST_B if pb else COST_A)[(count + 7) // 8 - 1]


def prefixes(counts: tuple[int, ...], pb: int) -> tuple[list[int], int]:
    shift = 8 if pb else 9
    out = [0]
    for count in counts:
        out.append(out[-1] + (cost(pb, count) << shift))
    return out, out[-1]


def stage_at(counts: tuple[int, ...], pb: int, pre: list[int], x: int) -> int:
    lo, hi = 0, len(counts)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pre[mid] <= x:
            lo = mid
        else:
            hi = mid
    return (lo << (8 if pb else 9)) + (x - pre[lo]) // cost(pb, counts[lo])


def ranges(counts: tuple[int, ...], pb: int, grid: int = GRID) -> list[tuple[int, int]]:
    pre, total_cost = prefixes(counts, pb)
    end = len(counts) << (8 if pb else 9)
    out = []
    for cta in range(grid):
        g0 = stage_at(counts, pb, pre, total_cost * cta // grid)
        g1 = end if cta == grid - 1 else stage_at(counts, pb, pre, total_cost * (cta + 1) // grid)
        out.append((g0, g1))
    return out


def cta_of(rs: list[tuple[int, int]], stage: int) -> int:
    starts = [g0 for g0, _ in rs]
    cta = bisect_right(starts, stage) - 1
    while cta >= 0 and not (rs[cta][0] <= stage < rs[cta][1]):
        cta -= 1
    assert cta >= 0, (stage, rs)
    return cta


@dataclass
class Stats:
    stages: int = 0
    whole_stages: int = 0
    partial_stages: int = 0
    units: int = 0
    split_units: int = 0
    partial_segments: int = 0
    contributors: int = 0
    max_contributors: int = 0


def inspect(counts: tuple[int, ...], pb: int, grid: int = GRID) -> Stats:
    rs = ranges(counts, pb, grid)
    total_stages = len(counts) << (8 if pb else 9)
    assert rs[0][0] == 0 and rs[-1][1] == total_stages
    assert all(rs[i][1] == rs[i + 1][0] for i in range(grid - 1))

    # Prove that (cta, edge) is unique for all partial segments of a CTA.
    seen_slots: set[tuple[int, int]] = set()
    stats = Stats(stages=total_stages, units=total_stages // 16)
    split_units = {g1 >> 4 for _, g1 in rs[:-1] if g1 & 15}
    for cta, (g0, g1) in enumerate(rs):
        if g0 == g1:
            continue
        # First partial segment.  If the entire range lies in one unit this is
        # also the last segment and it uses the range-end buffer (edge 1).
        first_end = min(g1, (g0 | 15) + 1)
        if (g0 & 15) or first_end - g0 < 16:
            edge = 1 if (g0 & 15) else 0
            key = (cta, edge)
            assert key not in seen_slots, (counts, pb, cta, g0, g1, key)
            seen_slots.add(key)
            stats.partial_stages += first_end - g0
            stats.partial_segments += 1

        # A distinct trailing partial begins on a unit boundary and therefore
        # uses edge 0.  It can coexist only with an edge-1 first partial.
        last_start = g1 & ~15
        if (g1 & 15) and last_start >= first_end:
            key = (cta, 0)
            assert key not in seen_slots, (counts, pb, cta, g0, g1, key)
            seen_slots.add(key)
            stats.partial_stages += g1 - last_start
            stats.partial_segments += 1

    stats.whole_stages = total_stages - stats.partial_stages
    assert stats.whole_stages >= 0 and stats.whole_stages % 16 == 0

    for unit in split_units:
        c0 = cta_of(rs, unit << 4)
        c1 = cta_of(rs, (unit << 4) + 15)
        assert c0 < c1
        covered = 0
        for cta in range(c0, c1 + 1):
            g0, g1 = rs[cta]
            lo, hi = max(g0, unit << 4), min(g1, (unit + 1) << 4)
            assert lo < hi
            edge = 0 if (lo & 15) == 0 else 1
            assert (cta, edge) in seen_slots
            covered += hi - lo
        assert covered == 16
        ncontrib = c1 - c0 + 1
        stats.contributors += ncontrib
        stats.max_contributors = max(stats.max_contributors, ncontrib)
    stats.split_units = len(split_units)
    return stats


def report(label: str, counts: tuple[int, ...]) -> None:
    vals = [inspect(counts, pb) for pb in (0, 1)]
    stages = sum(v.stages for v in vals)
    whole = sum(v.whole_stages for v in vals)
    split_units = sum(v.split_units for v in vals)
    contributors = sum(v.contributors for v in vals)
    print(
        f"{label:24s} jobs={len(counts):2d} stages={stages:5d} "
        f"whole-stage={100.0 * whole / stages:5.1f}% "
        f"split-units={split_units:3d}/{sum(v.units for v in vals):4d} "
        f"avg-contrib={contributors / split_units:4.2f} "
        f"max-contrib={max(v.max_contributors for v in vals):2d}"
    )


def main() -> None:
    # Touched-job counts observed in the seven-seed v152/v153 diagnostics.
    observed = {
        32: (20, 17, 15, 18, 14, 15, 15),
        128: (26, 24, 25, 19, 29, 23, 20),
        256: (24, 29, 25, 27, 24, 26, 20),
    }
    for batch, jobs in observed.items():
        # Uniform n-tile buckets give exact incidence for M=32 and useful bounds
        # for larger batches; the stage count is independent of the bucket.
        ntile = 1 if batch == 32 else 2 if batch == 128 else 3
        for nj in sorted(set(jobs)):
            report(f"M{batch} observed nj={nj}", (8 * ntile,) * nj)

    print("\nAdversarial bucket-pattern audit")
    patterns = ((1,), (8,), (9,), (64,), (1, 64), (8, 9, 16, 64))
    cases = 0
    # Include the H200 grid and representative grids on both sides.  The
    # collision proof above is per range and does not depend on grid size;
    # this sweep stress-tests the stage-at arithmetic and empty-range cases.
    for grid in (1, 2, 3, 7, 16, 31, 64, 127, 132, 160):
        for nj in range(1, 137):
            for pattern in patterns:
                counts = tuple(pattern[i % len(pattern)] for i in range(nj))
                for pb in (0, 1):
                    inspect(counts, pb, grid)
                    cases += 1
    print(f"PASS: {cases:,} phase schedules; two per-CTA edge buffers are collision-free")

    print("\nExhaustive short mixed-bucket audit")
    cases = 0
    for nj in range(1, 6):
        for buckets in product((1, 8, 9, 16, 33, 64), repeat=nj):
            for pb in (0, 1):
                inspect(tuple(buckets), pb, 13)
                cases += 1
    print(f"PASS: {cases:,} phase schedules")


if __name__ == "__main__":
    main()
