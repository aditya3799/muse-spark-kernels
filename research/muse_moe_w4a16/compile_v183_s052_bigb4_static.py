"""Compile the decoder S052 large-B4 MoE source exactly for SM90a."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

from compile_v172_hybrid_static import compile_source, compiler_environment


ROOT = Path(__file__).resolve().parent
DECODER = ROOT.parent / "muse_decode_layer" / "submission_s052_large_b4_partial_norm.py"
OUT = ROOT / ".tmp_v183_s052_bigb4_static"
FLAGS = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3",
    "-DOKC_FNST=4", "-DOKC_B4FIN=1", "-DOKC_BIGB4=1",
)


def literal_assignment(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc, cuobjdump = shutil.which("nvcc"), shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    source = literal_assignment(DECODER, "MOE_CUDA_SRC")
    assert "#define OKC_BIGB4 0" in source
    assert "large B4FIN build supports M <= 256" in source
    assert "#if OKC_WGS && OKC_B4FIN" in source
    cu = OUT / "s052_bigb4.cu"
    cubin = OUT / "s052_bigb4.cubin"
    cu.write_text(compile_source(source))
    proc = subprocess.run(
        [nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-cubin", "-Xptxas=-v",
         *FLAGS, str(cu), "-o", str(cubin)],
        text=True, capture_output=True, env=compiler_environment(),
    )
    log = (proc.stdout + proc.stderr).strip()
    (OUT / "s052_bigb4.ptxas.txt").write_text(log)
    print(f"nvcc exit={proc.returncode}")
    for line in log.splitlines():
        if any(key in line.lower() for key in (
            "compiling entry", "used ", "stack frame", "spill", "warning", "error",
        )):
            print(line)
    if proc.returncode:
        raise SystemExit(log or "nvcc failed")
    sass = subprocess.run(
        [cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True
    ).stdout
    resources = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(cubin)], text=True, capture_output=True, check=True
    ).stdout
    (OUT / "s052_bigb4.sass").write_text(sass)
    resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
    hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
    selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
        "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
    )}
    digest = hashlib.sha256(sass.encode()).hexdigest().upper()
    print("resources: " + " | ".join(resource_lines))
    print(f"selected={selected} HGMMA={dict(hgmma)} SHA256={digest}")
    assert all("REG:96" in line and "LOCAL:0" in line for line in resource_lines)
    assert sum(hgmma.values()) == 1024 and len(hgmma) == 8
    assert "serialized due to insufficient register resources" not in log.lower()
    print("v183 S052 BIGB4 preserves 96 registers and all HGMMA sites: PASS")


if __name__ == "__main__":
    main()
