"""Static contract audit for submission_v136_wgs_reg104_diag.py.

This does not execute CUDA.  It checks the Hopper register-pool budget, complete
warpgroup participation, barrier counts, control dispatch, and diagnostic gates.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v136_wgs_reg104_diag.py"
BASELINE = ROOT / "submission_v99.py"

NT = 16 * 32
PRODUCER_THREADS = 128
LIVE_PRODUCER_THREADS = 64
CONSUMER_REGS = 104
PRODUCER_REGS = 56
REGISTER_POOL = 65536


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


def check_control_dispatch() -> None:
    base = literal_assignments(BASELINE)
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
        assert cand[name] == base[name], (name, cand[name], base[name])
    assert "-DOKC_WGSREG=1" not in cand["MAIN_DEFINES"]
    assert cand["DIAG"] is True


def check_source_guards() -> None:
    text = CANDIDATE.read_text()
    ast.parse(text)
    required = (
        "#define OKC_WGSREG 0",
        "#define OKC_WGSREGC 104",
        "#define OKC_WGSREGP 56",
        "OKC_WGSREG requires the WGS kernel",
        "constexpr int NTL = NT + (OKC_WGS ? (OKC_WGSREG ? 128 : 64) : OKC_WS ? 32 : 0);",
        "constexpr int NJOIN = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);",
        '"n"(NJOIN)',
        "wgs_reg_dec<OKC_WGSREGP>();",
        "else if (tid < NT + 64) wgs_xproducer",
        "wgs_reg_inc<OKC_WGSREGC>();",
        '"reg104-gs2": tuple(MAIN_DEFINES) + ("-DOKC_WGSREG=1",)',
        '"reg104-gs4": tuple(MAIN_DEFINES) + ("-DOKC_WGSREG=1", "-DOKC_WGSGS=4")',
        "shapes = (32, 128, 256)",
        "mean_ratio <= 0.971",
        "if DIAG and use_mul:",
    )
    for needle in required:
        assert needle in text, needle

    # The producer group must release before any consumer group requests registers.
    dec = text.index("wgs_reg_dec<OKC_WGSREGP>();")
    inc = text.index("wgs_reg_inc<OKC_WGSREGC>();")
    assert dec < inc
    # Padding warps must not enter barrier 2.  NTL launches 640 threads while
    # NJOIN deliberately retains the original 576 live participants.
    assert text.count("#define WS_JOIN()") == 1


def check_register_geometry() -> None:
    assert CONSUMER_REGS % 8 == PRODUCER_REGS % 8 == 0
    assert 24 <= CONSUMER_REGS <= 256
    assert 24 <= PRODUCER_REGS <= 256
    used = NT * CONSUMER_REGS + PRODUCER_THREADS * PRODUCER_REGS
    assert used == 60416
    assert used <= REGISTER_POOL
    assert REGISTER_POOL - used == 5120
    assert NT + PRODUCER_THREADS == 640
    assert NT + LIVE_PRODUCER_THREADS == 576
    assert NT % 128 == PRODUCER_THREADS % 128 == 0


def main() -> None:
    check_control_dispatch()
    check_source_guards()
    check_register_geometry()
    print("v136 WGS register-redistribution contract audit: PASS")
    print(f"candidate sha256: {sha256(CANDIDATE)}")
    print(f"v99 sha256:       {sha256(BASELINE)}")
    print("register pool: 60416 / 65536; launch: 640; barrier-2 participants: 576")


if __name__ == "__main__":
    main()
