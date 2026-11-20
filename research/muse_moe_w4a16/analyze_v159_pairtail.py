"""Host feasibility gate for v159's phase-A trailing dependency-pair reorder.

Phase-B stage st is released by adjacent phase-A units 2*st and 2*st+1.
v139 already executes complete units in that pair-major order.  Its split-edge
optimization can, however, move the first share of a trailing odd unit ahead
while leaving the preceding even mate at the end of the remaining span.  This
model tests the only useful local reorder: pull that mate next to the partial.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from analyze_v158_phase_permute import (
    GRID,
    b_function,
    count_cases,
    ordered_chunks,
    ranges,
    stage_cost,
)


@dataclass(frozen=True)
class PhaseA:
    end: tuple[float, ...]
    ready: dict[tuple[int, int], float]


def split_chunks(a: int, b: int) -> list[tuple[int, int]]:
    chunks = []
    while a < b:
        e = min(b, ((a >> 4) + 1) << 4)
        chunks.append((a, e))
        a = e
    return chunks


def a_chunks(r: tuple[int, int], variant: str) -> list[tuple[int, int]]:
    """Return phase-A unit chunks in execution order for one CTA range."""
    if variant == "baseline":
        return ordered_chunks(0, r)
    g0, g1 = r
    if g0 == g1:
        return []
    ue = ((g1 - 1) >> 4) << 4
    if (g1 & 15) == 0 or ue <= g0:
        return split_chunks(g0, g1)

    # v139: [odd partial], then the complete prefix whose last unit is its mate.
    # Only an odd trailing unit has its preceding even dependency mate in-range.
    odd_tail_with_full_mate = ((ue >> 4) & 1) and ue - 16 >= g0
    if not odd_tail_with_full_mate:
        return split_chunks(ue, g1) + split_chunks(g0, ue)
    if variant == "partial_first":
        return (split_chunks(ue, g1) + split_chunks(ue - 16, ue)
                + split_chunks(g0, ue - 16))
    if variant == "mate_first":
        return split_chunks(ue - 16, g1) + split_chunks(g0, ue - 16)
    raise ValueError(variant)


def simulate_a(counts: tuple[int, ...], epi: float, fix: float, variant: str) -> PhaseA:
    share_done: dict[int, list[float]] = defaultdict(list)
    ends = []
    for r in ranges(counts, 0):
        t = 0.0
        seen = []
        for a, b in a_chunks(r, variant):
            unit = a >> 4
            c = stage_cost(0, counts[unit >> 5])
            t += (b - a) * c
            if b - a != 16:
                t += fix * c
            share_done[unit].append(t)
            seen.extend(range(a, b))
            if (b & 15) == 0:
                t += epi * c
        assert sorted(seen) == list(range(r[0], r[1]))
        ends.append(t)

    unit_ready = {}
    for unit, times in share_done.items():
        c = stage_cost(0, counts[unit >> 5])
        unit_ready[unit] = max(times) + epi * c
    ready = {}
    for j in range(len(counts)):
        for st in range(16):
            ready[j, st] = max(unit_ready[j * 32 + 2 * st],
                               unit_ready[j * 32 + 2 * st + 1])
    return PhaseA(tuple(ends), ready)


def launch_finish(counts: tuple[int, ...], pa: PhaseA,
                  epi: float, fix: float) -> float:
    bf = [b_function(counts, r, pa.ready, epi, fix) for r in ranges(counts, 1)]
    return max(max(pa.end[c] + bf[c][0], bf[c][1]) for c in range(GRID))


def main() -> None:
    cases = count_cases()
    variants = ("partial_first", "mate_first")
    ratios = {v: {m: [] for m in cases} for v in variants}
    changed = {v: 0 for v in variants}
    modeled = 0
    for epi in (0.5, 1.0, 2.0, 4.0):
        for fix in (0.0, 0.5, 1.0):
            for batch, count_list in cases.items():
                for counts in count_list:
                    base = simulate_a(counts, epi, fix, "baseline")
                    base_finish = launch_finish(counts, base, epi, fix)
                    for variant in variants:
                        cand = simulate_a(counts, epi, fix, variant)
                        # Reordering must not alter range ownership or A work.
                        assert cand.end == base.end
                        finish = launch_finish(counts, cand, epi, fix)
                        ratios[variant][batch].append(finish / base_finish)
                        if cand.ready != base.ready:
                            changed[variant] += 1
                    modeled += 1

    print(f"modeled schedules: {modeled:,}")
    for variant in variants:
        print(f"{variant}: readiness changed in {changed[variant]:,}/{modeled:,} schedules")
        gate = 0.0
        worst_all = 0.0
        for batch in (32, 128, 256):
            rs = sorted(ratios[variant][batch])
            med = median(rs)
            p90 = rs[int(0.9 * (len(rs) - 1))]
            worst = rs[-1]
            gate = max(gate, p90)
            worst_all = max(worst_all, worst)
            print(f"  M{batch}: med={med:.5f},p90={p90:.5f},worst={worst:.5f},"
                  f"wins={sum(x < 1 for x in rs)}/{len(rs)},losses={sum(x > 1 for x in rs)}/{len(rs)}")
        verdict = "GO" if gate <= 0.98 and worst_all <= 1.0 else "NO-GO"
        print(f"  {verdict}: gate={gate:.5f}, worst={worst_all:.5f}")


if __name__ == "__main__":
    main()
