"""Extract and compile v136's CUDA kernels without importing PyTorch.

Builds control, register-GS2, and register-GS4 cubins for sm_90a, prints ptxas
resource lines, and dumps SASS for register-pool/HGMMA inspection.
"""

from __future__ import annotations

import ast
from collections import Counter
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v136_wgs_reg104_diag.py"
OUT = Path(tempfile.gettempdir()) / "okc_v136_static"

BASE_FLAGS = (
    "-DOKC_NW=16",
    "-DOKC_FAST1=0",
    "-DOKC_EPI4=0",
    "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1",
    "-DOKC_MULDEC=1",
    "-DOKC_DQ7=1",
    "-DOKC_NS=3",
    "-DOKC_FNST=4",
    "-DOKC_NSF=3",
)
VARIANTS = {
    "control": BASE_FLAGS,
    "reg104_gs2": BASE_FLAGS + ("-DOKC_WGSREG=1",),
    "reg104_gs4": BASE_FLAGS + ("-DOKC_WGSREG=1", "-DOKC_WGSGS=4"),
}


def cuda_source() -> str:
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in node.targets):
            source = ast.literal_eval(node.value)
            break
    else:
        raise RuntimeError("CUDA_SRC assignment not found")
    # The extension wrapper needs PyTorch headers, but all device code is above it.
    source = source.split("#include <torch/extension.h>", 1)[0]
    source += "\n" + "\n".join(
        (
            "template __global__ void okc::moe_fused<false>(okc::Params);",
            "template __global__ void okc::moe_fused<true>(okc::Params);",
        )
    )
    return source


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is not on PATH")
    return path


def compiler_environment() -> dict[str, str]:
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
    # nvcc initializes the matching vcvars64.bat itself once cl.exe is discoverable.
    env["PATH"] = str(cls[-1].parent) + os.pathsep + env.get("PATH", "")
    if not shutil.which("cl.exe", path=env.get("PATH")):
        raise RuntimeError(f"failed to expose {cls[-1]}")
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = cuda_source()
    nvcc = tool("nvcc")
    cuobjdump = tool("cuobjdump")
    env = compiler_environment()
    print(f"build directory: {OUT}")
    reuse = "--reuse" in sys.argv
    hgmma_reference = None
    for name, flags in VARIANTS.items():
        cu = OUT / f"{name}.cu"
        cubin = OUT / f"{name}.cubin"
        sass = OUT / f"{name}.sass"
        cu.write_text(source)
        cmd = [
            nvcc,
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "-arch=sm_90a",
            "-cubin",
            "-Xptxas=-v",
            "-lineinfo",
            *flags,
            str(cu),
            "-o",
            str(cubin),
        ]
        if reuse and cubin.exists() and sass.exists() and cubin.stat().st_mtime >= SOURCE.stat().st_mtime:
            print(f"\n[{name}] reusing {cubin.name}")
        else:
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            log = (proc.stdout + proc.stderr).strip()
            (OUT / f"{name}.ptxas.txt").write_text(log)
            print(f"\n[{name}] nvcc exit={proc.returncode}")
            for line in log.splitlines():
                lower = line.lower()
                if "compiling entry" in lower or "used " in lower or "stack frame" in lower or "spill" in lower or "warning" in lower or "error" in lower:
                    print(line)
            if proc.returncode:
                print("command:", subprocess.list2cmdline(cmd))
                print("stdout:", repr(proc.stdout))
                print("stderr:", repr(proc.stderr))
                raise SystemExit(log or f"nvcc failed without diagnostics ({proc.returncode})")
        dumped = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True).stdout
        sass.write_text(dumped)
        resources = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)], text=True, capture_output=True, check=True
        ).stdout
        lines = dumped.splitlines()
        setmax = [(i + 1, line.strip()) for i, line in enumerate(lines) if "SETMAX" in line.upper()]
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", dumped))
        print(f"[{name}] resources: " + " | ".join(line.strip() for line in resources.splitlines() if "REG:" in line))
        print(f"[{name}] SETMAX instructions={len(setmax)}; HGMMA={dict(hgmma)}")
        for i, line in setmax:
            print(f"  sass:{i}: {line}")
        if name == "control":
            assert not setmax
            assert resources.count("REG:96 STACK:24") == 2, resources
            hgmma_reference = hgmma
        else:
            assert resources.count("REG:96 STACK:0") == 2, resources
            assert sum("TRY_ALLOC.CTAPOOL" in line and "0x68" in line for _, line in setmax) == 2
            assert sum("DEALLOC.CTAPOOL 0x38" in line for _, line in setmax) == 2
            assert dumped.count("ISETP.GT.AND P0, PT, R0, 0x1ff") == 2
            assert hgmma == hgmma_reference, (name, hgmma, hgmma_reference)
    print("\nv136 cubin/SASS gates: PASS")


if __name__ == "__main__":
    main()
