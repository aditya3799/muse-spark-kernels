"""Host-only ownership and synchronization-contract audit for the v157 M32 final-pass scout."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v157_wgs_finalseg_diag.py"
PROTECTED = {
    "submission_v139_wgs_kperm3.py": "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    "submission.py": "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}
HD, NT, M, TOPK, EMAX, H200_GRID = 4096, 512, 32, 8, 384 * 8, 132


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit_layout(fseg: int) -> None:
    assert fseg in (2, 4)
    fthreads = HD // (fseg * 4)
    assert fthreads <= NT and fthreads % 32 == 0
    assert H200_GRID >= M * fseg

    owners = {(it // fseg, it % fseg) for it in range(M * fseg)}
    assert owners == {(tk, seg) for tk in range(M) for seg in range(fseg)}

    for tk in range(M):
        cols = []
        for seg in range(fseg):
            cols.extend(seg * (HD // fseg) + tid * 4 + j for tid in range(fthreads) for j in range(4))
        assert sorted(cols) == list(range(HD)), (fseg, tk)

    partials = {((tk * TOPK + route) * fseg) + seg
                for tk in range(M) for route in range(TOPK) for seg in range(fseg)}
    assert partials == set(range(M * TOPK * fseg))
    assert max(partials) < EMAX


def audit_source_contract() -> None:
    src = SOURCE.read_text()
    required = (
        "#define OKC_WGSFSEG 0",
        "if (p.M == 32 && (int)gridDim.x >= 32 * OKC_WGSFSEG)",
        "p.ctr[0] = 0;",
        "while (ld_acquire(p.ctr + 3) == S.gen)",
        "if (i0 != 0) p.ctr[i0] = 0;",
        "atomicAdd(p.ctr, 1);",
        "while (ld_acquire(p.ctr) < (int)gridDim.x)",
        "32 * 8 * FSEG <= EMAX",
        '"fseg2": tuple(MAIN_DEFINES) + ("-DOKC_WGSFSEG=2",)',
        '"fseg4": tuple(MAIN_DEFINES) + ("-DOKC_WGSFSEG=4",)',
    )
    for text in required:
        assert text in src, text

    # The first barrier must publish ctr[0]=0 before publishing the new generation.  Candidate CTAs wait on that
    # generation, and only then can the unconditional (all-CTA) second-barrier atomic execute.
    p_reset = src.index("p.ctr[0] = 0;")
    p_fence = src.index("__threadfence();", p_reset)
    p_gen = src.index("atomicAdd(p.ctr + 3, 1);", p_fence)
    p_wait_gen = src.index("while (ld_acquire(p.ctr + 3) == S.gen)", p_gen)
    p_second = src.index("atomicAdd(p.ctr, 1);", p_wait_gen)
    assert p_reset < p_fence < p_gen < p_wait_gen < p_second

    # The production defines remain macro-off.  Only diagnostic variants opt in.
    main_line = next(line for line in src.splitlines() if line.startswith("MAIN_DEFINES ="))
    assert "WGSFSEG" not in main_line


def audit_barrier_epochs() -> None:
    # Abstract the only globally visible state needed by v157.  The first barrier's last arriver resets ctr0 before
    # advancing gen; a segmented M32 launch consumes ctr0 once and leaves it at G.  The following launch resets it
    # before releasing its first barrier, so alternating segmented/fallback launches cannot inherit a stale count.
    gen, ctr0 = 0, 0
    for segmented in (True, True, False, True, False, False, True):
        ctr0 = 0
        gen += 1
        assert ctr0 == 0
        if segmented:
            ctr0 += H200_GRID
            assert ctr0 == H200_GRID
    assert gen == 7


def main() -> None:
    for name, expected in PROTECTED.items():
        got = sha256(ROOT / name)
        assert got == expected, (name, got, expected)
    for fseg in (2, 4):
        audit_layout(fseg)
    audit_source_contract()
    audit_barrier_epochs()
    print("v157 final-seg ownership/barrier audit: PASS")
    print(f"candidate sha256: {sha256(SOURCE)}")


if __name__ == "__main__":
    main()
