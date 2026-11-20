"""Static and algebraic audit for the clean v149 KPERM3 cost diagnostic."""

from __future__ import annotations

import ast
import hashlib
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v149_wgs_kperm3_costdiag_clean.py"
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
        "#define OKC_WGSPA wgs_pack8(WGS_CA)",
        "#define OKC_WGSPB wgs_pack8(WGS_CB)",
        '"-DOKC_WGSPA=0xdad2c1b3a28f8970ull"',
        '"-DOKC_WGSPA=0xe4dbc9baa8928c70ull"',
        '"-DOKC_WGSPA=0xf2e8d4c3af968f70ull"',
        'DIAG = True',
        'range(7)',
        'times[name] / times["main"]',
        'if DIAG and use_mul:',
        'tuple(MAIN_DEFINES) + sfx',
        'tuple(defs) + sfx for name, defs in DIAG_VARIANTS.items()',
        "new-t=old-j, new-j=old-t",
        "kperm3=True",
    )
    for needle in required:
        assert needle in text, needle
    assert sha256(ACTIVE) == ACTIVE_SHA256


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


def check_shuffle_network() -> None:
    # Symbolically run the two register/lane butterfly stages for one warp.
    # a[lane][word] denotes the word produced by one contiguous LDG.128.
    a = [[(lane, word) for word in range(4)] for lane in range(32)]
    b: list[list[tuple[int, int]]] = []
    for lane in range(32):
        peer = a[lane ^ 2]
        if (lane & 2) == 0:
            b.append([a[lane][0], peer[0], a[lane][2], peer[2]])
        else:
            b.append([peer[1], a[lane][1], peer[3], a[lane][3]])
    out: list[list[tuple[int, int]]] = []
    for lane in range(32):
        peer = b[lane ^ 4]
        if (lane & 4) == 0:
            out.append([b[lane][0], b[lane][1], peer[0], peer[1]])
        else:
            out.append([peer[2], peer[3], b[lane][2], b[lane][3]])
    for lane in range(32):
        base, q, lane2 = lane & ~7, lane & 1, (lane >> 1) & 3
        expected = [(base + q + 2 * old_block, lane2) for old_block in range(4)]
        assert out[lane] == expected, (lane, out[lane], expected)


def check_paired_scale_extract() -> None:
    scales = [0x5300 + lane for lane in range(32)]
    for lane in range(32):
        lo = scales[lane & ~1]
        hi = scales[(lane & ~1) + 1]
        pair = lo | (hi << 16)
        got = (pair >> ((lane & 1) * 16)) & 0xffff
        assert got == scales[lane]


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
    print("v149 clean KPERM3 cost diagnostic algebra/static audit: PASS")
    print(f"candidate sha256: {sha256(CANDIDATE)}")
    print(f"active sha256:    {sha256(ACTIVE)}")
    print("K mapping: exact 4x4 bf16x2 transpose; one original scale pair per lane across four new k16 steps")


if __name__ == "__main__":
    main()
