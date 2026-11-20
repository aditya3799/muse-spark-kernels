"""Exact SM90a resource/SASS gate for v181's B4FIN M32 module."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

from compile_v172_hybrid_static import alt_sources, assignments, compile_source, compiler_environment


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v181_hybrid_m32_b4fin.py"
OUT = ROOT / ".tmp_v181_b4fin_static"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    vals = assignments(CANDIDATE)
    alts = alt_sources(CANDIDATE)
    flags = tuple(vals["V167_MAIN_DEFINES"]) + (
        "-DOKC_PRMTSC=1", "-DOKC_SC6W=1", "-DOKC_DQ7=1", "-DOKC_NS=3",
        "-DOKC_NSF=3", "-DOKC_B4FIN=1",
    )
    cu = OUT / "v181_b4fin_m32.cu"
    cubin = OUT / "v181_b4fin_m32.cubin"
    cu.write_text(compile_source(alts["v181_b4fin"]))
    proc = subprocess.run(
        [nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-cubin", "-Xptxas=-v",
         *flags, str(cu), "-o", str(cubin)],
        text=True, capture_output=True, env=compiler_environment(),
    )
    log = (proc.stdout + proc.stderr).strip()
    (OUT / "v181_b4fin_m32.ptxas.txt").write_text(log)
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
    (OUT / "v181_b4fin_m32.sass").write_text(sass)
    resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
    hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
    selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
        "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
    )}
    digest = hashlib.sha256(sass.encode()).hexdigest().upper()
    print("resources: " + " | ".join(resource_lines))
    print(f"selected={selected} HGMMA={dict(hgmma)} SHA256={digest}")
    assert all("REG:96" in line and "LOCAL:0" in line for line in resource_lines)
    assert all("STACK:8" in line for line in resource_lines)
    assert sum(hgmma.values()) == 1024
    assert len(hgmma) == 8 and all(count == 128 for count in hgmma.values())
    assert "serialized due to insufficient register resources" not in log.lower()
    print("v181 B4FIN preserves 96 registers, 8-byte frame, zero local memory: PASS")
    print("v181 B4FIN preserves all 1024 HGMMA sites/eight widths: PASS")
    print("v181 B4FIN static gates: PASS")


if __name__ == "__main__":
    main()
