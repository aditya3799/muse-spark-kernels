"""Host-only feasibility model for permuting v139's phase-B CTA ranges.

The CUDA candidate would leave both phase partitions byte-for-byte unchanged
and only map physical CTA c to a different logical phase-B range.  This model
keeps v139's exact cost tables, range boundaries, phase-A trailing-partial
reorder, per-stage phase-B readiness, unit epilogues, and split-edge costs.
It tests fixed bijections that need no grid coordination or extra scratch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle, islice
from math import gcd
from statistics import median


GRID = 132
COST_A = (112, 135, 140, 158, 173, 185, 200, 207)
COST_B = (100, 104, 115, 123, 132, 144, 156, 174)


def stage_cost(pb: int, count: int) -> float:
    return float((COST_B if pb else COST_A)[(count + 7) // 8 - 1])


def prefixes(counts: tuple[int, ...], pb: int) -> tuple[list[int], int]:
    units = 16 if pb else 32
    out = [0]
    for count in counts:
        out.append(out[-1] + units * 16 * int(stage_cost(pb, count)))
    return out, out[-1]


def stage_at(counts: tuple[int, ...], pb: int, pre: list[int], x: int) -> int:
    lo, hi = 0, len(counts)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pre[mid] <= x:
            lo = mid
        else:
            hi = mid
    return (lo << (9 - pb)) + (x - pre[lo]) // int(stage_cost(pb, counts[lo]))


def ranges(counts: tuple[int, ...], pb: int, grid: int = GRID) -> list[tuple[int, int]]:
    pre, total = prefixes(counts, pb)
    end = len(counts) << (9 - pb)
    out = []
    for cta in range(grid):
        g0 = stage_at(counts, pb, pre, total * cta // grid)
        g1 = end if cta == grid - 1 else stage_at(counts, pb, pre, total * (cta + 1) // grid)
        out.append((g0, g1))
    assert out[0][0] == 0 and out[-1][1] == end
    assert all(out[i][1] == out[i + 1][0] for i in range(grid - 1))
    return out


def ordered_chunks(pb: int, r: tuple[int, int]) -> list[tuple[int, int]]:
    """Return contiguous [stage, stage) chunks in the kernel's actual order."""
    g0, g1 = r
    if g0 == g1:
        return []
    sp = g0
    if not pb and (g1 & 15):
        ue = ((g1 - 1) >> 4) << 4
        if ue > g0:
            sp = ue
    spans = [(sp, g1)] + ([(g0, sp)] if sp > g0 else [])
    chunks: list[tuple[int, int]] = []
    for a, b in spans:
        while a < b:
            e = min(b, ((a >> 4) + 1) << 4)
            chunks.append((a, e))
            a = e
    return chunks


@dataclass(frozen=True)
class PhaseA:
    end: tuple[float, ...]
    ready: dict[tuple[int, int], float]


def simulate_a(counts: tuple[int, ...], epi: float, fix: float) -> PhaseA:
    share_done: dict[int, list[float]] = defaultdict(list)
    ends: list[float] = []
    for r in ranges(counts, 0):
        t = 0.0
        for a, b in ordered_chunks(0, r):
            unit = a >> 4
            c = stage_cost(0, counts[unit >> 5])
            t += (b - a) * c
            if b - a != 16:
                t += fix * c
            share_done[unit].append(t)
            if (b & 15) == 0:
                t += epi * c
        ends.append(t)

    unit_ready = {}
    for unit, times in share_done.items():
        c = stage_cost(0, counts[unit >> 5])
        unit_ready[unit] = max(times) + epi * c
    ready = {}
    for j in range(len(counts)):
        for st in range(16):
            # A unit ub writes 128 columns and releases doneAs[ub >> 1].
            # B K-stage st consumes the 256 columns from adjacent units 2*st and 2*st+1.
            ready[j, st] = max(unit_ready[j * 32 + 2 * st], unit_ready[j * 32 + 2 * st + 1])
    return PhaseA(tuple(ends), ready)


def b_function(counts: tuple[int, ...], r: tuple[int, int],
               ready: dict[tuple[int, int], float], epi: float,
               fix: float) -> tuple[float, float]:
    """Return (work, intercept) so finish(start) == max(start+work, intercept).

    A serial chain of ``wait-until-ready; do work`` operations is exactly a
    max-plus affine function.  Collapsing it avoids constructing a costly
    132x132 event simulation without changing the modeled result.
    """
    elapsed = 0.0
    waits: list[tuple[float, float]] = []
    for a, b in ordered_chunks(1, r):
        unit = a >> 4
        j = unit >> 4
        c = stage_cost(1, counts[j])
        for gs in range(a, b):
            waits.append((ready[j, gs & 15], elapsed))
            elapsed += c
        if b - a != 16:
            elapsed += fix * c
        if (b & 15) == 0:
            elapsed += epi * c
    intercept = max((ready_at + elapsed - before for ready_at, before in waits), default=0.0)
    return elapsed, intercept


def mappings(grid: int = GRID) -> dict[str, tuple[int, ...]]:
    out = {"identity": tuple(range(grid)), "reverse": tuple(reversed(range(grid)))}
    for k in (1, 2, 4, 8, 11, 16, 22, 33, 44, 55, 66):
        out[f"rot{k}"] = tuple((c + k) % grid for c in range(grid))
    # Affine permutations sample both gentle and strongly mixed ownership without any run-time coordination.
    for a in (5, 7, 13, 17, 19, 25, 31, 35, 37, 41, 43, 47, 53, 59, 61, 65):
        if gcd(a, grid) != 1:
            continue
        for b in (0, 11, 33, 66, 99):
            out[f"aff{a}_{b}"] = tuple((a * c + b) % grid for c in range(grid))
    for name, p in out.items():
        assert sorted(p) == list(range(grid)), name
    return out


def count_cases() -> dict[int, list[tuple[int, ...]]]:
    observed_jobs = {
        32: (20, 17, 15, 18, 14, 15, 15),
        128: (26, 24, 25, 19, 29, 23, 20),
        256: (24, 29, 25, 27, 24, 26, 20),
    }
    patterns = {
        32: ((1, 2, 3, 4, 5, 6, 7, 8), (1, 1, 2, 3, 5, 8)),
        128: ((4, 6, 8, 9, 11, 13, 16), (1, 8, 9, 16)),
        256: ((6, 9, 13, 17, 21, 25, 32), (1, 8, 9, 16, 24, 32, 48, 64)),
    }
    out: dict[int, list[tuple[int, ...]]] = {}
    for batch, jobs in observed_jobs.items():
        cases = set()
        for nj in jobs:
            for pat in patterns[batch]:
                for shift in range(min(len(pat), 4)):
                    rp = pat[shift:] + pat[:shift]
                    vals = tuple(islice(cycle(rp), nj))
                    cases.add(vals)
                    cases.add(tuple(reversed(vals)))
        out[batch] = sorted(cases)
    return out


def main() -> None:
    perms = mappings()
    cases = count_cases()
    ratios: dict[str, dict[int, list[float]]] = {name: {m: [] for m in cases} for name in perms}
    base_tails: dict[int, list[float]] = {m: [] for m in cases}
    modeled = 0
    for epi in (0.5, 1.0, 2.0, 4.0):
        for fix in (0.0, 0.5, 1.0):
            for batch, count_list in cases.items():
                for counts in count_list:
                    pa = simulate_a(counts, epi, fix)
                    br = ranges(counts, 1)
                    # Each logical B range is an exact max-plus timing function.
                    bf = [b_function(counts, r, pa.ready, epi, fix) for r in br]
                    def finish(c: int, logical: int) -> float:
                        work, intercept = bf[logical]
                        return max(pa.end[c] + work, intercept)
                    baseline = max(finish(c, c) for c in range(GRID))
                    nonempty_a = [v for v, r in zip(pa.end, ranges(counts, 0)) if r[0] < r[1]]
                    base_tails[batch].append((max(nonempty_a) - min(nonempty_a)) / baseline)
                    for name, p in perms.items():
                        ratios[name][batch].append(max(finish(c, p[c]) for c in range(GRID)) / baseline)
                    modeled += 1

    def summary(name: str) -> tuple[float, float, str]:
        fields = []
        worst = 0.0
        gate = 0.0
        for batch in (32, 128, 256):
            rs = sorted(ratios[name][batch])
            med = median(rs)
            p90 = rs[int(0.9 * (len(rs) - 1))]
            w = rs[-1]
            worst = max(worst, w)
            gate = max(gate, p90)
            fields.append(f"M{batch}:med={med:.5f},p90={p90:.5f},worst={w:.5f},wins={sum(x < 1 for x in rs)}/{len(rs)}")
        return gate, worst, "  ".join(fields)

    ranked = sorted((summary(name)[0], summary(name)[1], name) for name in perms if name != "identity")
    print(f"modeled schedules: {modeled:,}; fixed bijections: {len(perms):,}")
    for batch in (32, 128, 256):
        print(f"baseline M{batch} A-tail/launch median={median(base_tails[batch]):.5f}")
    print("best fixed phase-B mappings (ranked by worst shape p90):")
    for _, _, name in ranked[:15]:
        gate, worst, fields = summary(name)
        verdict = "PASS" if gate <= 0.98 and worst <= 1.0 else "FAIL"
        print(f"{name:10s} gate={gate:.5f} worst={worst:.5f}  {fields}  {verdict}")

    print("best mapping selected independently for each shape:")
    shape_best = {}
    for batch in (32, 128, 256):
        candidates = []
        for name in perms:
            if name == "identity":
                continue
            rs = sorted(ratios[name][batch])
            med = median(rs)
            p90 = rs[int(0.9 * (len(rs) - 1))]
            candidates.append((p90, rs[-1], med, name, sum(x < 1 for x in rs)))
        p90, worst, med, name, wins = min(candidates)
        shape_best[batch] = (p90, worst, name)
        verdict = "PASS" if p90 <= 0.98 and worst <= 1.0 else "FAIL"
        print(f"M{batch}: {name} med={med:.5f},p90={p90:.5f},worst={worst:.5f},"
              f"wins={wins}/{len(ratios[name][batch])}  {verdict}")

    best_name = ranked[0][2]
    gate, worst, _ = summary(best_name)
    assert all(abs(x - 1.0) < 1e-12 for m in cases for x in ratios["identity"][m])
    if gate <= 0.98 and worst <= 1.0:
        print(f"GO: {best_name} clears the 2% robust tail gate")
    else:
        print(f"NO-GO: best mapping {best_name} has p90 gate={gate:.5f}, worst={worst:.5f}")
    if all(p90 <= 0.98 and worst <= 1.0 for p90, worst, _ in shape_best.values()):
        print("GO: per-shape mapping selection clears every robust tail gate")
    else:
        print("NO-GO: per-shape mapping selection also fails the robust tail gate")


if __name__ == "__main__":
    main()
