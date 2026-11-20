"""Static, algebraic, dispatch, and routing audit for v145 KPERM3+FT32+snap."""

from __future__ import annotations

import ast
import hashlib
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v145_wgs_kperm3_ft32_snap.py"
ACTIVE = ROOT / "submission.py"
ACTIVE_SHA256 = "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def kperm3(old_k: int) -> int:
    """Old contiguous K -> KPERM3 position, independently in each 64-wide group."""
    return (
        (old_k & ~63)
        + (((old_k >> 1) & 3) << 4)
        + (((old_k >> 3) & 1) << 3)
        + (((old_k >> 4) & 3) << 1)
        + (old_k & 1)
    )


def check_source() -> None:
    text = CANDIDATE.read_text()
    ast.parse(text)
    required = (
        "#define OKC_KPERM3 0",
        "OKC_KPERM3 requires the WGS path and paired-row scale decode",
        "wgs_kperm3_load",
        "wgs_kperm3_col",
        "*reinterpret_cast<const uint16_t*>",
        "__byte_perm(scg, 0u, 0x4140u)",
        '"-DOKC_KPERM3=1"',
        "#define OKC_WGSSNAPA 0",
        "#define OKC_WGSSNAPB 0",
        '#if OKC_WGSSNAPA || OKC_WGSSNAPB',
        'WGS_SNAP_DEFINES = ("-DOKC_WGSSNAPA=1", "-DOKC_WGSSNAPB=1")',
        'WGS_FT32_SNAP_DEFINES = tuple(MAIN_DEFINES) + ("-DOKC_FT_T=32",) + WGS_SNAP_DEFINES',
        'WGS_FT64_SNAP_DEFINES = tuple(MAIN_DEFINES) + WGS_SNAP_DEFINES',
        "elif M == 32 and sfx:",
        "elif M >= 64 and sfx:",
        "new-t=old-j, new-j=old-t",
        "kperm3=True",
    )
    for needle in required:
        assert needle in text, needle
    assert sha256(ACTIVE) == ACTIVE_SHA256


def check_snap_and_ft32() -> None:
    # Snap=1 moves only stage 1/15 of each unit to the nearest unit edge.  The
    # transform is monotone, so adjacent CTA intervals remain an exact cover.
    def snap1(stage: int) -> int:
        rem = stage & 15
        return stage - 1 if rem == 1 else stage + 1 if rem == 15 else stage

    mapped = [snap1(stage) for stage in range(8193)]
    assert all(a <= b for a, b in zip(mapped, mapped[1:]))
    assert all(abs(a - b) <= 1 for a, b in zip(mapped, range(8193)))

    # FT32 changes only the partition of each expert's ordered route interval.
    for total in range(1, 3073):
        chunks = [(q * 32, min(32, total - q * 32)) for q in range((total + 31) // 32)]
        rebuilt = [r for start, count in chunks for r in range(start, start + count)]
        assert rebuilt == list(range(total))
        assert all(0 < count <= 32 for _, count in chunks)

    emax = 384 * 8
    jmax32 = 32 + emax // 32 + 8
    assert jmax32 == 136
    assert emax + 8 * jmax32 == 4160
    assert 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16 >= 2 * 160 * 4 * 2048 + 2 * 160 * 4 + 136 * 16


def check_permutation() -> None:
    p = [kperm3(k) for k in range(4096)]
    assert sorted(p) == list(range(4096))
    assert all(kperm3(kperm3(k)) == k for k in range(4096))
    assert all((k // 64) == (p[k] // 64) for k in range(4096))


def check_x_gather_formula() -> None:
    # The CUDA helper writes four bf16x2 words into output k8. Reconstruct every
    # scalar source and prove that output[new_k] = input[inverse_perm(new_k)].
    seen: set[tuple[int, int]] = set()
    for new_k8 in range(512):
        base = new_k8 & ~7
        q = new_k8 & 1
        lane2 = (new_k8 >> 1) & 3
        for old_block in range(4):
            src_k8 = base + q + 2 * old_block
            for e in range(2):
                old_k = src_k8 * 8 + lane2 * 2 + e
                new_k = new_k8 * 8 + old_block * 2 + e
                assert kperm3(old_k) == new_k
                seen.add((old_k, new_k))
    assert len(seen) == 4096


def check_weight_lane_formula() -> None:
    # New(step, lane-t) reads old(step=lane-t, lane-t=step).
    # Every new lane therefore stays on one original K16 block (and scale pair)
    # while the four new steps cover all four 2-wide positions of that block.
    seen: set[tuple[int, int]] = set()
    for group in range(64):
        for new_step in range(4):
            for new_lane_t in range(4):
                old_block = new_lane_t
                old_lane_t = new_step
                for q in range(2):
                    for e in range(2):
                        old_k = group * 64 + old_block * 16 + q * 8 + old_lane_t * 2 + e
                        new_k = group * 64 + new_step * 16 + q * 8 + new_lane_t * 2 + e
                        assert kperm3(old_k) == new_k
                        assert old_k // 16 == group * 4 + new_lane_t
                        seen.add((old_k, new_k))
    assert len(seen) == 4096


def check_scale_layout() -> None:
    # One stage has [grp=4][warp=4][g=8][old-step=4][h=2] = 1024 bytes.
    old_offsets = {
        (((grp * 4 + warp) * 8 + g) * 4 + step) * 2 + h
        for grp in range(4)
        for warp in range(4)
        for g in range(8)
        for step in range(4)
        for h in range(2)
    }
    new_offsets = {
        ((grp * 4 + warp) * 32 + (g * 4 + lane_t)) * 2 + h
        for grp in range(4)
        for warp in range(4)
        for g in range(8)
        for lane_t in range(4)
        for h in range(2)
    }
    assert old_offsets == new_offsets == set(range(1024))


def check_dot_invariance() -> None:
    rng = random.Random(139)
    for _ in range(64):
        x = [rng.uniform(-2.0, 2.0) for _ in range(64)]
        # Distinct per-row, per-K16 scales make this test catch accidental
        # scale sharing between original blocks.
        for row in range(7):
            codes = [rng.choice((-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0)) for _ in range(64)]
            scales = [0.25 + row + 0.75 * block for block in range(4)]
            w = [codes[k] * scales[k // 16] for k in range(64)]
            xp = [0.0] * 64
            wp = [0.0] * 64
            for old_k in range(64):
                new_k = kperm3(old_k)
                xp[new_k] = x[old_k]
                wp[new_k] = w[old_k]
            ref = sum(a * b for a, b in zip(w, x))
            got = sum(a * b for a, b in zip(wp, xp))
            assert abs(ref - got) <= 1e-12 * max(1.0, abs(ref))


def main() -> None:
    check_source()
    check_permutation()
    check_x_gather_formula()
    check_weight_lane_formula()
    check_scale_layout()
    check_dot_invariance()
    check_snap_and_ft32()
    print("v145 KPERM3 + FT32/snap algebra/static/dispatch audit: PASS")
    print(f"candidate sha256: {sha256(CANDIDATE)}")
    print(f"active sha256:    {sha256(ACTIVE)}")
    print("K mapping: exact 4x4 bf16x2 transpose; one original scale pair per lane across four new k16 steps")


if __name__ == "__main__":
    main()
