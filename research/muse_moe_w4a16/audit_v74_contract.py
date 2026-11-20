"""Static, torch-free contract audit for the unsubmitted v74 candidate."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v74.py"
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


def audit_mc2_enumeration() -> None:
    """Prove the two-rank logical-unit mapping covers every physical unit once."""
    for npass in range(1, 129):
        for nrb in (16, 32):
            pairs = npass >> 1
            paired_units = nrb * pairs
            logical_units = paired_units + (npass & 1) * ((nrb + 1) >> 1)
            seen: list[tuple[int, int]] = []
            for unit in range(logical_units):
                for cluster_rank in (0, 1):
                    if unit < paired_units:
                        ps = 2 * (unit % pairs) + cluster_rank
                        rb = unit // pairs
                    else:
                        ps = npass - 1
                        rb = 2 * (unit - paired_units) + cluster_rank
                    require(0 <= ps < npass and 0 <= rb < nrb,
                            f"MC2 out-of-range mapping: np={npass}, nrb={nrb}, unit={unit}, rank={cluster_rank}")
                    seen.append((ps, rb))
            expected = [(ps, rb) for ps in range(npass) for rb in range(nrb)]
            require(sorted(seen) == expected, f"MC2 coverage mismatch: np={npass}, nrb={nrb}")


def main() -> None:
    text = CANDIDATE.read_text(encoding="utf-8")
    ast.parse(text)

    require(digest(CONTROL) == V64_SHA256, "v64 control hash changed")
    require(digest(ACTIVE) == V64_SHA256, "active submission is not immutable v64")

    cuda_k = raw_string(text, "M_CUDA_K")
    cuda_x = raw_string(text, "M_CUDA_X")
    audit_preprocessor("M_CUDA_K", cuda_k)
    audit_preprocessor("M_CUDA_X", cuda_x)

    require("data_ptr<uint32_t>()" not in cuda_k + cuda_x, "typed UInt32 LUT access returned")
    require(cuda_k.count("static_cast<const uint32_t*>(lut.data_ptr())") == 2,
            "K LUT/HYBLUT wrapper branches changed")
    require(cuda_x.count("static_cast<const uint32_t*>(lut.data_ptr())") == 1,
            "X LUT wrapper branch changed")
    require("const uint32_t* lutp = nullptr;" in cuda_k and "const uint32_t* lutp = nullptr;" in cuda_x,
            "baseline LUT isolation is missing")

    require('"-DSW=1 -DKS1=2 -DKS2=4 -DMAXM=32 -DHYB13"' in text,
            "zero-spill M32 hybrid is missing")
    require('("-DNCW=16", "-DWGPAIR", "-DLUTDQ")' in text,
            "NCW16 lookup control is missing")
    require('("-DNCW=16", "-DWGPAIR", "-DCLUSTER=2", "-DMC2", "-DLUTDQ")' in text,
            "NCW16 multicast+lookup candidate is missing")
    require('("-DNCW=8", "-DRPW=2", "-DTPX=32", "-DNSL=2", "-DCPB=2", "-DCLUSTER=2", "-DMC2"),' not in text,
            "spill-prone arithmetic TPX32 multicast candidate was not pruned")
    require('("-DNCW=8", "-DRPW=2", "-DTPX=32", "-DNSL=2", "-DCPB=2", "-DCLUSTER=2", "-DMC2", "-DLUTDQ")' in text,
            "zero-spill TPX32 multicast+lookup candidate is missing")

    for snippet in (
        "while (ldacq(qackp) != qseq)",
        "strel(qackp, qseq + 1)",
        "uu += nrb * (np >> 1) + (np & 1) * ((nrb + 1) >> 1)",
        "bulk_mc2(dst + ASB",
        "bulk_mc2(dst + cc * XB",
        "TORCH_CHECK(g_gridX / CLUSTER <= C_MC_CLMAX",
    ):
        require(snippet in cuda_x, f"repaired MC2 contract missing: {snippet}")

    prototypes = re.findall(r'cpp_sources="void moe\(([^\"]+)\);"', text)
    counts = [(p.count("torch::Tensor"), p.count("int64_t")) for p in prototypes]
    require((19, 1) in counts, "K extension prototype is not 19 tensors + lo")
    require((16, 1) in counts, "X extension prototype is not 16 tensors + lo")
    require('M__extK(ex).moe(x, ids, tw, w1, w2, state["w1l"], state["w2l"]' in text,
            "K Python/C++ argument bridge is missing hybrid weight tensors")
    require('"ctrX": torch.zeros(2048, dtype=torch.int32, device=dev)' in text,
            "MC2 mailbox allocation is smaller than the repaired v73 contract")
    require('TORCH_CHECK(ctr.numel() >= 2048, "moe_x: MC2 mailbox tensor is too small")' in cuda_x,
            "MC2 host-side mailbox bound check is missing")
    require(cuda_k.count("lut.scalar_type() == at::kInt") == 2 and
            cuda_x.count("lut.scalar_type() == at::kInt") == 1,
            "LUT dtype/size host guards are missing")

    audit_mc2_enumeration()

    print("PASS: Python AST and CUDA preprocessor balance")
    print(f"PASS: active/control v64 SHA256 {V64_SHA256.upper()}")
    print("PASS: LUT dtype isolation and K/X extension contracts")
    print("PASS: repaired MC2 schedule/acknowledgement invariants")
    print("PASS: exhaustive MC2 mapping for np=1..128 and nrb=16/32")
    print(f"candidate SHA256 {digest(CANDIDATE).upper()}")


if __name__ == "__main__":
    main()
