"""Offline source/numerical guardrails for v164 WGS-only polynomial SiLU."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v164_wgs_polysilu_diag.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def exact(x: float) -> float:
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return x * ex / (1.0 + ex)


def poly(x: float) -> float:
    z = x * x
    p = (2.82057459e-4 * z - 1.16906086e-2) * z + 2.34261343e-1
    sat = min(1.0, max(0.0, x * p + 0.5))
    return x * sat


def main() -> None:
    text = SOURCE.read_text()
    ast.parse(text)
    for path, digest in EXPECTED.items():
        got = sha256(path)
        assert got == digest, (path.name, got, digest)
    assert text.count("#define OKC_WGSPOLY 0") == 1
    assert text.count("const float v = wgs_silu(gt) * up;") == 1
    assert text.count("-DOKC_WGSPOLY=1") == 1
    assert "2.82057459e-4f" in text and "-1.16906086e-2f" in text and "2.34261343e-1f" in text
    assert '"poly-silu": tuple(MAIN_DEFINES)' in text
    assert "PROMOTE={promote}" in text

    rng = random.Random(164)
    xs = [(-16.0 + 32.0 * i / 200000.0) for i in range(200001)]
    xs.extend(rng.gauss(0.0, 2.5) for _ in range(200000))
    num = den = max_abs = 0.0
    for x in xs:
        y, yp = exact(x), poly(x)
        d = yp - y
        num += d * d
        den += y * y
        max_abs = max(max_abs, abs(d))
    rel_l2 = math.sqrt(num / den)
    assert rel_l2 < 0.012, rel_l2
    print(f"scalar mixed-grid relative L2={rel_l2:.6%}, max_abs={max_abs:.6f}")
    print("protected hashes: PASS")
    print("v164 source/numerical audit: PASS")


if __name__ == "__main__":
    main()
