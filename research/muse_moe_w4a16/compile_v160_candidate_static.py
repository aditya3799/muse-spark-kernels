"""Compile the generated v160 control and norm-pair variants for SM90a."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v160_wgs_epinorm_diag.py"
EXPECTED = "1911260BA9E70C12F9B5DA44DE97F2A0A9BA8CD2C542E67CC998F9E8DE5960FD"
OUT = Path(tempfile.gettempdir()) / "okc_v160_candidate_static"
BASE = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3",
    "-DOKC_FNST=4", "-DOKC_NSF=3", "-DOKC_KPERM3=1",
)
VARIANTS = {"control": BASE, "normpair": BASE + ("-DOKC_WGSEPINORM=1",)}


def cuda_source() -> str:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper() == EXPECTED
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in node.targets):
            source = ast.literal_eval(node.value)
            break
    else:
        raise RuntimeError("CUDA_SRC assignment not found")
    source = source.split("#include <torch/extension.h>", 1)[0]
    return source + "\n" + "\n".join((
        "template __global__ void okc::moe_fused<false>(okc::Params);",
        "template __global__ void okc::moe_fused<true>(okc::Params);",
    ))


def environment() -> dict[str, str]:
    env = os.environ.copy()
    if shutil.which("cl.exe", path=env.get("PATH")):
        return env
    roots = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
    )
    cls = [p for root in roots if root.exists() for p in root.glob("*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe")]
    if not cls:
        raise RuntimeError("MSVC x64 cl.exe not found")
    env["PATH"] = str(cls[-1].parent) + os.pathsep + env.get("PATH", "")
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = cuda_source()
    nvcc, cuobjdump = shutil.which("nvcc"), shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    env = environment()
    reference_hgmma = None
    control_selected = None
    control_resources = None
    ops = ("LDG", "SHFL", "PRMT", "IMAD", "HMUL2", "FADD", "FMUL", "LDS", "LDL", "STL")
    for name, flags in VARIANTS.items():
        cu, cubin = OUT / f"{name}.cu", OUT / f"{name}.cubin"
        cu.write_text(source)
        proc = subprocess.run([nvcc, "-std=c++17", "-O3", "--use_fast_math", "-arch=sm_90a", "-cubin",
                               "-Xptxas=-v", "-lineinfo", *flags, str(cu), "-o", str(cubin)],
                              text=True, capture_output=True, env=env)
        log = (proc.stdout + proc.stderr).strip()
        print(f"\n[{name}] nvcc exit={proc.returncode}")
        for line in log.splitlines():
            if any(k in line.lower() for k in ("compiling entry", "used ", "stack frame", "spill", "warning", "error")):
                print(line)
        if proc.returncode:
            raise SystemExit(log)
        assert "serialized due to insufficient register resources" not in log.lower()
        sass = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True,
                              capture_output=True, check=True).stdout
        usage = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], text=True,
                               capture_output=True, check=True).stdout
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
        selected = {op: len(re.findall(rf"\b{op}(?:\.|\b)", sass)) for op in ops}
        resources = tuple(x.strip() for x in usage.splitlines() if "REG:" in x)
        print(f"[{name}] resources: {' | '.join(resources)}")
        print(f"[{name}] HGMMA={dict(hgmma)}")
        print(f"[{name}] selected SASS={selected}")
        if reference_hgmma is None:
            reference_hgmma, control_selected, control_resources = hgmma, selected, resources
        else:
            assert hgmma == reference_hgmma
            assert resources == control_resources
            assert selected["LDG"] < control_selected["LDG"]
            assert selected["SHFL"] == control_selected["SHFL"]
            assert selected["LDL"] == control_selected["LDL"] and selected["STL"] == control_selected["STL"]
    print("\nv160 generated-candidate static gates: PASS")


if __name__ == "__main__":
    main()
