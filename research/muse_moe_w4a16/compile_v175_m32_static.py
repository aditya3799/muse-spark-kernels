"""Static SM90a resource audit for v175's new exact-v167 M32 module."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

from compile_v172_hybrid_static import alt_sources, assignments, compile_source, compiler_environment


ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "submission_v175_hybrid_m32.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
OUT = ROOT / ".tmp_v175_m32_static"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    env = compiler_environment()
    candidate = assignments(CANDIDATE)
    advanced = assignments(ADVANCED)
    alts = alt_sources(CANDIDATE)
    assert alts["v167_main"] == advanced["CUDA_SRC"]
    assert tuple(candidate["V167_MAIN_DEFINES"]) == tuple(advanced["MAIN_DEFINES"])

    source = compile_source(alts["v167_main"])
    flags = tuple(candidate["V167_MAIN_DEFINES"]) + (
        "-DOKC_PRMTSC=1", "-DOKC_SC6W=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3",
    )
    cu = OUT / "v167_m32.cu"
    cubin = OUT / "v167_m32.cubin"
    cu.write_text(source)
    cmd = [nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-cubin", "-Xptxas=-v",
           *flags, str(cu), "-o", str(cubin)]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    log = (proc.stdout + proc.stderr).strip()
    (OUT / "v167_m32.ptxas.txt").write_text(log)
    print(f"nvcc exit={proc.returncode}")
    for line in log.splitlines():
        if any(key in line.lower() for key in ("compiling entry", "used ", "stack frame", "spill", "warning", "error")):
            print(line)
    if proc.returncode:
        raise SystemExit(log or "nvcc failed")

    sass = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True,
                          capture_output=True, check=True).stdout
    resources = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], text=True,
                               capture_output=True, check=True).stdout
    (OUT / "v167_m32.sass").write_text(sass)
    resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
    selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
        "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
    )}
    hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
    digest = hashlib.sha256(sass.encode()).hexdigest().upper()
    print("resources: " + " | ".join(resource_lines))
    print(f"selected={selected} HGMMA={dict(hgmma)} SHA256={digest}")
    assert all("LOCAL:0" in line for line in resource_lines), resource_lines
    print("exact-v167 M32 source/defines: PASS")
    print("v175 M32 local-memory gate: PASS")


if __name__ == "__main__":
    main()
