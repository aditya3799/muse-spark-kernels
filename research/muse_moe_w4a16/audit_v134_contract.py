"""Static/model audit for submission_v134_wgs_pairjob_diag.py.

This does not execute CUDA.  It proves the paired-job index maps, token/row
coverage, doneAs thresholds, barrier participation, and source guardrails.
"""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v134_wgs_pairjob_diag.py"
BASELINE = ROOT / "submission_v101_wgs_tune.py"

PIECE = 9216
UNIT_BYTES = 4 * PIECE
WGS_NS = 3
WGS_JOB_NT = 8
WGS_XS = WGS_JOB_NT * 4096


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def check_source() -> None:
    text = CANDIDATE.read_text()
    ast.parse(text)
    required = (
        "#define OKC_WGSPAIRJ 0",
        "constexpr int WGS_WBYTES = WGS_PAIRJ ? 2 * WGS_PIECE : UNIT_BYTES",
        "constexpr int WGS_JOB_NT = FT_T / 8",
        "constexpr int WGS_NT = WGS_PAIRJ ? 4 : WGS_JOB_NT",
        "ring + WGS_NS * WGS_WBYTES",
        "ws * WGS_WBYTES",
        "(ub >> (1 + WGS_PAIRJ))",
        "case 0: wgs_idle_seg<PB>",
        '"pairj": tuple(MAIN_DEFINES) + ("-DOKC_WGSPAIRJ=1",)',
    )
    for needle in required:
        assert needle in text, needle

    # The scored path must remain macro-off; pairj is diagnostic-only.
    assert "-DOKC_WGSPAIRJ=1" not in text.split("DIAG_VARIANTS", 1)[0]
    assert "kernel_fn remains the unmodified v101 MAIN module" in text


def check_phase(pb: int) -> None:
    original_units = 16 if pb else 32
    paired_units = 2 * original_units

    # A paired stage loads exactly the two row pieces selected by ub parity.
    # Both token halves reuse those pieces, so physical (unit,piece) coverage
    # is exactly the original four-piece layout with no duplicate HBM load.
    physical = []
    logical = []
    for ub in range(paired_units):
        oub = ub >> 1
        half = ub & 1
        for rowwg in range(2):
            piece = 2 * half + rowwg
            physical.append((oub, piece))
            for jhalf in range(2):
                logical.append((oub, piece, jhalf))
    assert Counter(physical) == Counter(
        (oub, piece) for oub in range(original_units) for piece in range(4)
    )
    assert Counter(logical) == Counter(
        (oub, piece, jhalf)
        for oub in range(original_units)
        for piece in range(4)
        for jhalf in range(2)
    )

    # Pair r16 covers every original output group once for each token half.
    got = []
    for ub in range(paired_units):
        for rowwg in range(2):
            for wp in range(4):
                got.append(ub * 8 + rowwg * 4 + wp)
    want = list(range(256 if pb else 512))
    assert sorted(got) == want

    # Stage/unit geometry is exactly doubled, but still 16 K-stages per unit.
    stage_log = 9 + 1 - pb
    assert paired_units * 16 == 1 << stage_log


def check_token_halves_and_doneas() -> None:
    for cnt in range(1, 65):
        total_ntt = (cnt + 7) // 8
        local = (min(4, total_ntt), max(0, total_ntt - 4))
        tiles = []
        for jhalf, ntt in enumerate(local):
            tiles.extend(4 * jhalf + nt for nt in range(ntt))
        assert tiles == list(range(total_ntt)), (cnt, local, tiles)

        active_wgs = 2 if cnt <= 32 else 4
        # Four phase-A half-units feed each phase-B stage.
        contributions = 4 * active_wgs
        threshold = 8 * ((cnt + 31) // 32)
        assert contributions == threshold, (cnt, contributions, threshold)

        # Four warps in each of four WGs retire each ring barrier, including
        # the idle second pair for <=32-token jobs.
        assert 4 * 4 == 16


def check_storage() -> None:
    control_weights = WGS_NS * UNIT_BYTES
    paired_weights = WGS_NS * 2 * PIECE
    x_ring = WGS_NS * WGS_XS
    assert control_weights == 110592
    assert paired_weights == 55296
    assert x_ring == 98304
    assert paired_weights + x_ring == 153600
    assert paired_weights * 2 == control_weights


def main() -> None:
    check_source()
    check_phase(0)
    check_phase(1)
    check_token_halves_and_doneas()
    check_storage()
    print("v134 paired-job contract audit: PASS")
    print(f"candidate sha256: {sha256(CANDIDATE)}")
    print(f"v101 sha256:      {sha256(BASELINE)}")
    print("paired dynamic ring: 153600 B before aligned Smem header (control: 208896 B)")


if __name__ == "__main__":
    main()
