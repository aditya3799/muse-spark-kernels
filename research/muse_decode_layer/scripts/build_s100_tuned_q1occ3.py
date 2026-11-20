"""Build S100: S099 plus the proven full-context q_len=1 occupancy plan.

The attention change is scheduling-only.  It retains all 128 producer threads,
caps the flat q_len=1 specialization for three resident CTAs per SM, and gives
that plan 3 * SM CTAs.  The same change passed all standalone attention tests
and improved the scored B16/B32 full-context decode cases.
"""

from __future__ import annotations

import base64
import hashlib
import lzma
from pathlib import Path

try:
    from . import build_s098_c12_gqk_m16 as s098
    from . import build_s099_tuned_bundle as s099
except ImportError:
    import build_s098_c12_gqk_m16 as s098
    import build_s099_tuned_bundle as s099


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
OUT = ROOT / "submission_s100_tuned_q1occ3.py"


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one attention anchor, found {count}: {old[:96]!r}")
    return text.replace(old, new, 1)


def _attention_base() -> str:
    text = BASE.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "#define OKC_WSMINB 2\n",
        "#define OKC_WSMINB 2\n#define OKC_WSMINB1 3\n",
    )
    text = _replace_once(
        text,
        "static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);",
        "static constexpr int MINB = WS ? (F2 ? 1 : ((L == 1) ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2);",
    )
    text = _replace_once(
        text,
        "ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target",
        "ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else ((3 * sms) if L == 1 else target)",
    )
    text = _replace_once(text, "muse_att_v298w1pf_", "muse_att_v298w1pf_occ3_")
    return text


def main() -> None:
    got = hashlib.sha256(BASE.read_bytes()).hexdigest()
    expected = s099.EXPECTED[BASE]
    if got != expected:
        raise RuntimeError(f"protected decoder hash mismatch: {got}")

    appendix = s098._c12() + "\n" + s098._m16() + "\n" + s099._m32() + s099.SMALL_ROUTES
    encoded = base64.b85encode(lzma.compress(appendix.encode(), preset=9)).decode("ascii")
    stub = "\nimport base64,lzma\nexec(lzma.decompress(base64.b85decode(" + repr(encoded) + ")))\n"
    candidate = (_attention_base() + stub).encode()
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
