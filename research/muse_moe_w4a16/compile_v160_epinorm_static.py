"""Compile v139 against two phase-B norm-weight subgroup-broadcast scouts."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v139_wgs_kperm3.py"
EXPECTED = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"
OUT = Path(tempfile.gettempdir()) / "okc_v160_epinorm_static"
FLAGS = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3",
    "-DOKC_FNST=4", "-DOKC_NSF=3", "-DOKC_KPERM3=1",
)

OLD = """    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;"""

FLOAT_BCAST = """    const float gsc = p.gsB[e] * TWO62;
    float nw0 = 0.f, nw1 = 0.f;
    if (t == 0) {
      nw0 = 1.f + __bfloat162float(__ldg(p.normw + n0));
      nw1 = 1.f + __bfloat162float(__ldg(p.normw + n0 + 8));
    }
    nw0 = __shfl_sync(0xffffffffu, nw0, 0, 4);
    nw1 = __shfl_sync(0xffffffffu, nw1, 0, 4);
    const float w0 = nw0 * gsc, w1 = nw1 * gsc;"""

PACK_BCAST = """    const float gsc = p.gsB[e] * TWO62;
    uint32_t nwbits = 0;
    if (t == 0) {
      const unsigned short* nw16 = reinterpret_cast<const unsigned short*>(p.normw);
      nwbits = (uint32_t)__ldg(nw16 + n0) | ((uint32_t)__ldg(nw16 + n0 + 8) << 16);
    }
    nwbits = __shfl_sync(0xffffffffu, nwbits, 0, 4);
    const float w0 = (1.f + __uint_as_float(nwbits << 16)) * gsc;
    const float w1 = (1.f + __uint_as_float(nwbits & 0xffff0000u)) * gsc;"""

PAIR_LOAD = """    const float gsc = p.gsB[e] * TWO62;
    const uint32_t nwbits = __ldg(reinterpret_cast<const uint32_t*>(p.normw) + r16 * 8 + g);
    const float w0 = (1.f + __uint_as_float(nwbits << 16)) * gsc;
    const float w1 = (1.f + __uint_as_float(nwbits & 0xffff0000u)) * gsc;"""


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
    env["PATH"] = str(cls[-1].parent) + os.pathsep + env.get("PATH", "")
    return env


def main() -> None:
    base = cuda_source()
    assert base.count(OLD) == 1
    sources = {
        "control": base,
        "float_bcast": base.replace(OLD, FLOAT_BCAST),
        "pack_bcast": base.replace(OLD, PACK_BCAST),
        "pair_load": base.replace(OLD, PAIR_LOAD),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    env = compiler_environment()
    reference = None
    selected_ops = ("LDG", "SHFL", "PRMT", "IMAD", "HMUL2", "FFMA", "FADD", "FMUL", "LDS", "LDL", "STL")
    print(f"build directory: {OUT}")
    reuse = "--reuse" in sys.argv
    for name, source in sources.items():
        cu, cubin = OUT / f"{name}.cu", OUT / f"{name}.cubin"
        same_source = cu.exists() and cu.read_text() == source
        cu.write_text(source)
        cmd = [nvcc, "-std=c++17", "-O3", "--use_fast_math", "-arch=sm_90a", "-cubin",
               "-Xptxas=-v", "-lineinfo", *FLAGS, str(cu), "-o", str(cubin)]
        if not (reuse and same_source and cubin.exists()):
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            log = (proc.stdout + proc.stderr).strip()
            (OUT / f"{name}.ptxas.txt").write_text(log)
            code = proc.returncode
        else:
            log = (OUT / f"{name}.ptxas.txt").read_text()
            code = 0
        print(f"\n[{name}] nvcc exit={code}{' (reused)' if reuse and same_source and cubin.exists() else ''}")
        for line in log.splitlines():
            if any(k in line.lower() for k in ("compiling entry", "used ", "stack frame", "spill", "warning", "error")):
                print(line)
        if code:
            raise SystemExit(log or f"nvcc failed ({code})")
        sass = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True,
                              capture_output=True, check=True).stdout
        resources = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], text=True,
                                   capture_output=True, check=True).stdout
        (OUT / f"{name}.sass").write_text(sass)
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
        selected = {op: len(re.findall(rf"\b{op}(?:\.|\b)", sass)) for op in selected_ops}
        print(f"[{name}] resources: " + " | ".join(x.strip() for x in resources.splitlines() if "REG:" in x))
        print(f"[{name}] HGMMA={dict(hgmma)}")
        print(f"[{name}] selected SASS={selected}")
        warnings = log.lower()
        assert "serialized due to insufficient register resources" not in warnings
        if reference is None:
            reference = hgmma
        else:
            assert hgmma == reference, (name, hgmma, reference)
    print("\nv160 phase-B epilogue static gates: PASS")


if __name__ == "__main__":
    main()
