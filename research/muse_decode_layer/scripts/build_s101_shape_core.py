"""Build S101: the minimal shape-aware decoder bundle.

S101 keeps only mechanisms with positive decoder-local evidence:

* exact-C12 T=1 QKV and sliced-unpooler CGEMV;
* the exact standalone-v178 M=16 MoE module on sliding layers only;
* the exact standalone-v178 M=32 MoE module on full layers only.

The dead GQK scout, the decoder-negative small SK2 geometries, and the
decoder-neutral q_len=1 occupancy change are deliberately excluded.
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
OUT = ROOT / "submission_s101_shape_core.py"

EXPECTED = {
    BASE: "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19",
    s098.CGEMV: "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161",
    s099.MOE: "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94",
}


SHAPE_SELECTOR = r'''

# S101: shape-only MoE selection.  Config is fixed before CUDA graph capture.
_S101_SLIDING = False
_s101_parent_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    global _S101_SLIDING
    cfg = data.get("config")
    if cfg is not None:
        _S101_SLIDING = bool(cfg["window"])
    return _s101_parent_after(data, state, o_in, gate, T)

# Bypass S098's unconditional M16 wrapper and select only the two measured
# shape-local wins.  Every other shape uses the protected S081 resident MoE.
_s101_resident_moe = _s098_parent_moe
def _moe(M=None, state=None):
    if state is not None:
        if M == 16 and _S101_SLIDING:
            ext = state.get("s098_v178_m16")
            if ext is not None:
                return ext
        if M == 32 and not _S101_SLIDING:
            ext = state.get("s099_v178_m32")
            if ext is not None:
                return ext
    return _s101_resident_moe(M, state)
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, expected in EXPECTED.items():
        got = _sha(path)
        if got != expected:
            raise RuntimeError(f"protected source hash mismatch for {path.name}: {got}")

    appendix = s098._c12() + "\n" + s098._m16() + "\n" + s099._m32() + SHAPE_SELECTOR
    encoded = base64.b85encode(lzma.compress(appendix.encode(), preset=9)).decode("ascii")
    stub = ("\nimport base64,lzma\n"
            f"exec(lzma.decompress(base64.b85decode({encoded!r})))\n").encode()
    candidate = BASE.read_bytes() + stub
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
