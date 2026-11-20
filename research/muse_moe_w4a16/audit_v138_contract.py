"""Static contract audit for the v138 fixed-role W13/W2 diagnostic.

The audit proves phase coverage and cohort ownership without executing CUDA,
checks that W13 progress is independent of W2, and protects the frozen active
submission and macro-off production dispatch.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v138_wgs_fixed_roles_diag.py"
ACTIVE = ROOT / "submission.py"
ACTIVE_SHA256 = "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38"

COST_A = (112, 135, 140, 158, 173, 185, 200, 207)
COST_B = (100, 104, 115, 123, 132, 144, 156, 174)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            values[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return values


def cost(pb: int, count: int) -> int:
    return (COST_B if pb else COST_A)[(count + 7) // 8 - 1]


def prefixes(counts: tuple[int, ...], pb: int) -> tuple[list[int], int]:
    unit_stages = 256 if pb else 512
    pre = [0]
    for count in counts:
        pre.append(pre[-1] + cost(pb, count) * unit_stages)
    return pre, pre[-1]


def stage_at(counts: tuple[int, ...], pb: int, pre: list[int], x: int) -> int:
    lo, hi = 0, len(counts)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pre[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo * (256 if pb else 512) + (x - pre[lo]) // cost(pb, counts[lo])


def ranges(counts: tuple[int, ...], pb: int, workers: int) -> list[tuple[int, int]]:
    pre, total = prefixes(counts, pb)
    end = len(counts) * (256 if pb else 512)
    out = []
    for c in range(workers):
        g0 = stage_at(counts, pb, pre, total * c // workers)
        g1 = end if c == workers - 1 else stage_at(counts, pb, pre, total * (c + 1) // workers)
        out.append((g0, g1))
    return out


def owner(counts: tuple[int, ...], pb: int, workers: int, stage: int) -> int:
    pre, total = prefixes(counts, pb)
    unit_stages = 256 if pb else 512
    j, sj = divmod(stage, unit_stages)
    xe = pre[j] + (sj + 1) * cost(pb, counts[j])
    c = (xe - 1) * workers // total
    c = min(c, workers - 1)
    while c + 1 < workers and total * (c + 1) // workers < xe:
        c += 1
    while c > 0 and total * c // workers >= xe:
        c -= 1
    return c


def check_source() -> None:
    text = CANDIDATE.read_text()
    ast.parse(text)
    required = (
        "#define OKC_WGSROLE 0",
        "OKC_WGSROLE must be 0 or at least 2",
        "return max(1, min(G - 1, ga));",
        "if (c < ga)",
        "S.wgr[2] = S.wgr[3] = 0;",
        "S.wgr[0] = S.wgr[1] = 0;",
        "const int ga = wgs_role_ga(G), off = pb ? ga : 0, RG = pb ? G - ga : ga;",
        "return off + c;",
        '"role3": tuple(MAIN_DEFINES) + ("-DOKC_WGSROLE=3",)',
        '"role4": tuple(MAIN_DEFINES) + ("-DOKC_WGSROLE=4",)',
        "shapes = (128, 256)",
        "[role-parity]",
        "[role-summary]",
        "if DIAG and use_mul:",
    )
    for needle in required:
        assert needle in text, needle
    assert "-DOKC_WGSROLE=" not in text.split("DIAG_VARIANTS", 1)[0]


def check_control_dispatch() -> None:
    active = literal_assignments(ACTIVE)
    cand = literal_assignments(CANDIDATE)
    for name in (
        "MAIN_DEFINES",
        "TINY_DEFINES",
        "TINY_MAX_M",
        "TINY2_DEFINES",
        "TINY2_MAX_M",
        "MID_DEFINES",
        "MID_MAX_M",
        "PDL",
        "PRMT_SCALES",
    ):
        assert cand[name] == active[name], (name, cand[name], active[name])
    assert cand["DIAG"] is True
    assert sha256(ACTIVE) == ACTIVE_SHA256


def check_role_geometry() -> None:
    for grid in range(2, 161):
        for ratio in (3, 4):
            ga = max(1, min(grid - 1, grid * (ratio - 1) // ratio))
            gb = grid - ga
            assert ga > 0 and gb > 0 and ga + gb == grid
    assert 132 * 2 // 3 == 88 and 132 - 88 == 44
    assert 132 * 3 // 4 == 99 and 132 - 99 == 33


def check_coverage() -> None:
    workloads = (
        tuple(range(1, 65)),
        (1, 8, 16, 24, 32, 40, 48, 56, 64) * 4,
        (8,) * 24,
        (32,) * 32,
        (64,) * 48,
    )
    for counts in workloads:
        for grid in (2, 3, 7, 31, 64, 132, 160):
            for ratio in (3, 4):
                ga = max(1, min(grid - 1, grid * (ratio - 1) // ratio))
                for pb, workers in ((0, ga), (1, grid - ga)):
                    rr = ranges(counts, pb, workers)
                    end = len(counts) * (256 if pb else 512)
                    assert rr[0][0] == 0 and rr[-1][1] == end
                    assert all(rr[i][1] == rr[i + 1][0] for i in range(len(rr) - 1))
                    assert all(a <= b for a, b in rr)
                    # Every non-empty worker boundary and every job edge maps
                    # to the same cohort owner used by the CUDA fixup slots.
                    probes = {a for a, b in rr if a < b} | {
                        j * (256 if pb else 512) for j in range(len(counts))
                    }
                    for stage in probes:
                        c = owner(counts, pb, workers, stage)
                        assert rr[c][0] <= stage < rr[c][1], (pb, workers, stage, c, rr[c])


def main() -> None:
    check_source()
    check_control_dispatch()
    check_role_geometry()
    check_coverage()
    print("v138 fixed-role W13/W2 contract audit: PASS")
    print(f"candidate sha256: {sha256(CANDIDATE)}")
    print(f"active sha256:    {sha256(ACTIVE)}")
    print("H200 cohorts: role3 A/B=88/44; role4 A/B=99/33; exact phase coverage verified")


if __name__ == "__main__":
    main()
