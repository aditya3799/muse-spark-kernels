"""Host contract audit for v160's losslessly paired norm-weight diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent
HASHES = {
    "submission_v139_wgs_kperm3.py": "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    "submission.py": "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    "submission_v160_wgs_epinorm_diag.py": "1911260BA9E70C12F9B5DA44DE97F2A0A9BA8CD2C542E67CC998F9E8DE5960FD",
}


def main() -> None:
    for name, expected in HASHES.items():
        got = hashlib.sha256((ROOT / name).read_bytes()).hexdigest().upper()
        assert got == expected, (name, got, expected)

    path = ROOT / "submission_v160_wgs_epinorm_diag.py"
    text = path.read_text()
    ast.parse(text)
    assert text.startswith("# okc: problem=muse_moe_w4a16")
    assert text.count("#define OKC_WGSEPINORM 0") == 1
    assert text.count('"-DOKC_WGSEPINORM=1"') == 1
    assert "OKC_WGSEPINORM=1" not in text.split("MAIN_DEFINES =", 1)[1].splitlines()[0]
    assert text.count("const uint32_t nwbits = __ldg(p.normwp + r16 * 8 + g);") == 1
    assert text.count("p.normwp = w.size() > 14 ? (const uint32_t*)w[14].data_ptr() : nullptr;") == 1
    assert "assert same and rel == 0.0" in text

    # The prepare layout is exactly [r16, g, (original g, original g+8)].
    original = [[16 * r + c for c in range(16)] for r in range(256)]
    paired = [[[original[r][g], original[r][g + 8]] for g in range(8)] for r in range(256)]
    flat = [x for row in paired for pair in row for x in pair]
    assert len(flat) == 4096 and sorted(flat) == list(range(4096))
    for r in range(256):
        for g in range(8):
            word = paired[r][g]
            assert word == [16 * r + g, 16 * r + g + 8]

    # CUDA reconstructs each BF16 exactly by putting its 16 bits in an FP32 high half.
    for bits in range(1 << 16):
        f = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        roundtrip = struct.unpack("<I", struct.pack("<f", f))[0] >> 16
        # NaN payloads may quiet on some hosts; all finite and infinity encodings are exact.
        if (bits & 0x7F80) != 0x7F80 or (bits & 0x007F) == 0:
            assert roundtrip == bits

    print("v160 host contract audit: PASS")


if __name__ == "__main__":
    main()
