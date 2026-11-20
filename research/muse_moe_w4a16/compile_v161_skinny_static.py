"""Compile the frozen v139 tiny and MID modules for exact SM90a SASS inspection."""

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
SOURCE = ROOT / "submission_v139_wgs_kperm3.py"
EXPECTED = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"
OUT = Path(tempfile.gettempdir()) / "okc_v161_skinny_static"
COMMON = (
    "-DOKC_PRMTSC=1", "-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3",
    "-DOKC_FNST=4", "-DOKC_NSF=3",
)
TINY = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2",
    "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
    "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1",
)
TINY8 = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0",
    "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
    "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1",
)
MID = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
    "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
    "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
    "-DOKC_FINPF=1", "-DOKC_SKSC2=1", "-DOKC_FTSC2=1",
)
MID_SKT16 = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_SKT=16",
    "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4",
    "-DOKC_UWARP=0", "-DOKC_WS=0", "-DOKC_FINPF=1",
    "-DOKC_SKSC2=1", "-DOKC_FTSC2=1",
)
VARIANTS = {
    "tiny1": TINY + COMMON,
    "tiny8": TINY8 + COMMON,
    "mid16": MID + COMMON,
    "mid16_skt16": MID_SKT16 + COMMON,
}


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
    return source + "\ntemplate __global__ void okc::moe_fused<false>(okc::Params);\n"


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
    for name, flags in VARIANTS.items():
        cu, cubin = OUT / f"{name}.cu", OUT / f"{name}.cubin"
        cu.write_text(source)
        proc = subprocess.run(
            [nvcc, "-std=c++17", "-O3", "--use_fast_math", "-arch=sm_90a", "-cubin",
             "-Xptxas=-v", "-lineinfo", *flags, str(cu), "-o", str(cubin)],
            text=True, capture_output=True, env=env,
        )
        log = (proc.stdout + proc.stderr).strip()
        (OUT / f"{name}.ptxas.txt").write_text(log)
        print(f"\n[{name}] nvcc exit={proc.returncode}")
        for line in log.splitlines():
            if any(k in line.lower() for k in ("compiling entry", "used ", "stack frame", "spill", "warning", "error")):
                print(line)
        if proc.returncode:
            raise SystemExit(log)
        sass = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True,
                              capture_output=True, check=True).stdout
        usage = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], text=True,
                               capture_output=True, check=True).stdout
        (OUT / f"{name}.sass").write_text(sass)
        ops = ("HMMA", "LDS", "LDG", "PRMT", "IMAD", "HMUL2", "FADD", "MUFU", "LDL", "STL")
        counts = Counter({op: len(re.findall(rf"\b{op}(?:\.|\b)", sass)) for op in ops})
        resources = tuple(x.strip() for x in usage.splitlines() if "REG:" in x)
        print(f"[{name}] resources: {' | '.join(resources)}")
        print(f"[{name}] SASS={dict(counts)}")
        lines = [line.strip() for line in sass.splitlines() if "/*" in line and "*/" in line]
        hmma = [i for i, line in enumerate(lines) if "HMMA." in line]
        print(f"[{name}] HMMA sites={len(hmma)}")
        for site in hmma[:4]:
            print(f"[{name}] first-HMMA context")
            for line in lines[max(0, site - 14):site + 4]:
                print(line)
    print(f"\nSASS files: {OUT}")


if __name__ == "__main__":
    main()
