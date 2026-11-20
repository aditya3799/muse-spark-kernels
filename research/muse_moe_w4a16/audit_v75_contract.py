"""Static, torch-free contract audit for the unsubmitted v75 candidate."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v75.py"
CONTROL = ROOT / "submission_v64.py"
ACTIVE = ROOT / "submission.py"
V64_SHA256 = "f0ad063a8a89fc3e40437e8c5ba946c675bff1bfc0e3866bc1261b50661db4e6"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_string(text: str, name: str) -> str:
    marker = f'{name} = r"""'
    start = text.index(marker) + len(marker)
    return text[start : text.index('"""', start)]


def audit_preprocessor(name: str, source: str) -> None:
    stack: list[tuple[int, str]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        directive = line.lstrip()
        if re.match(r"#\s*(if|ifdef|ifndef)\b", directive):
            stack.append((lineno, directive))
        elif re.match(r"#\s*endif\b", directive):
            require(bool(stack), f"{name}:{lineno}: unmatched #endif")
            stack.pop()
    require(not stack, f"{name}: unterminated preprocessor blocks: {stack}")


def main() -> None:
    text = CANDIDATE.read_text(encoding="utf-8")
    ast.parse(text)

    require(digest(CONTROL) == V64_SHA256, "v64 control hash changed")
    require(digest(ACTIVE) == V64_SHA256, "active submission is not immutable v64")
    require(text.startswith("# okc: problem=muse_moe_w4a16\n"), "OKC problem header changed")

    cuda_x = raw_string(text, "M_CUDA_X")
    audit_preprocessor("M_CUDA_X", cuda_x)

    for snippet in (
        "#define NTH ((NCW + 1) * 32)",
        "setmaxnreg.inc.sync.aligned.u32 %0;",
        "if (wid < NCW) reg_inc<RCREG_CONS>();",
        "static_assert((NCW & 3) == 0",
        "static_assert((NCW * RCREG_CONS + 128) * CPB <= 2048",
        "if (tid == NCW * 32) u = atomicAdd(p.ctr + C_Q, 1);",
        "for (int it = blockIdx.x * (NTH / 32) + wid;",
    ):
        require(snippet in cuda_x, f"v75 register-allocation contract missing: {snippet}")

    require("setmaxnreg.dec" not in cuda_x, "incomplete producer warp must not execute setmaxnreg")
    require("#define RCREG_CONS 120" in cuda_x, "one-CTA consumer target is not 120 registers")
    require("#define RCREG_CONS 104" in cuda_x, "two-CTA conservative target changed")
    require(cuda_x.index("atomicAdd(p.ctr + C_Q") < cuda_x.index("reg_inc<RCREG_CONS>()"),
            "work queue acquisition moved across common setup")
    require(cuda_x.index("reg_inc<RCREG_CONS>()") < cuda_x.index("const int ntot = sm->ntot"),
            "consumer allocation is not before the WGMMA pipeline")

    candidate = '("-DNCW=16", "-DWGPAIR", "-DREGRECONF=1")'
    require(text.count(candidate) == 1, "focused NCW16/WGPAIR v75 candidate missing or duplicated")
    require('"-DCPB=2", "-DREGRECONF=1"' not in text,
            "serialized two-CTA register-allocation candidate was not pruned")

    prototypes = re.findall(r'cpp_sources="void moe\(([^\"]+)\);"', text)
    counts = [(p.count("torch::Tensor"), p.count("int64_t")) for p in prototypes]
    require((16, 1) in counts, "X extension prototype is not 16 tensors + lo")

    print("PASS: Python AST and CUDA preprocessor balance")
    print(f"PASS: active/control v64 SHA256 {V64_SHA256.upper()}")
    print("PASS: consumer-only setmaxnreg contract and 120-register budget")
    print("PASS: original CTA geometry, queue order, epilogue, and X extension ABI")
    print("PASS: serialized CPB2 register-allocation candidate pruned")
    print(f"candidate SHA256 {digest(CANDIDATE).upper()}")


if __name__ == "__main__":
    main()
