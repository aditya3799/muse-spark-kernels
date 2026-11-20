"""Fail-closed audit for v176's exact-v167 M32 scratch ABI fix."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from make_v172_hybrid_small import literal_assignments
from make_v175_hybrid_m32 import function_source


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
V175 = ROOT / "submission_v175_hybrid_m32.py"
CANDIDATE = ROOT / "submission_v176_hybrid_m32_scratch.py"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    V175: "0BD38ED6639E8C17757A98D8236A282950F5BD717C0CE191709A8170A3D1C9E2",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha(path.read_bytes())
        assert got == expected, (path.name, got, expected)
    v175 = literal_assignments(V175)
    candidate = literal_assignments(CANDIDATE)
    for name in ("CUDA_SRC", "MAIN_DEFINES", "V167_MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES"):
        assert candidate[name] == v175[name], name

    cf = functions(CANDIDATE)
    sentinel_source = function_source(ADVANCED, "_inter_sentinel", "_v167_inter_sentinel").replace("HIDDEN", "4096")
    scratch_source = function_source(ADVANCED, "_make_scratch", "_v167_make_scratch")
    scratch_source = scratch_source.replace("HIDDEN", "4096").replace("_inter_sentinel", "_v167_inter_sentinel")
    expected_sentinel = ast.parse(sentinel_source).body[0]
    expected_scratch = ast.parse(scratch_source).body[0]
    assert ast.dump(cf["_v167_inter_sentinel"]) == ast.dump(expected_sentinel)
    assert ast.dump(cf["_v167_make_scratch"]) == ast.dump(expected_scratch)

    text = CANDIDATE.read_text()
    assert text.count('state["scratch"] = _v167_make_scratch(dev)') == 1
    assert text.count('"scratch": _make_scratch(dev)') == 1
    assert 'torch.zeros(8 + 1024 + EMAX + 4096, dtype=torch.int32' in text
    ast.parse(text)
    print("protected frozen/live/v175 hashes: PASS")
    print("all CUDA sources and define tuples unchanged from v175: PASS")
    print("v167 M32 sentinel and scratch allocator preserved exactly: PASS")
    print("v139 small/large scratch paths remain isolated: PASS")
    print(f"candidate SHA256={sha(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
