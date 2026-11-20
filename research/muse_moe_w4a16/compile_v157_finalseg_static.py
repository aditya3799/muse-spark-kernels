"""Compile v139 plus v157 control/final-seg variants for exact SM90a evidence."""

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
SOURCE = ROOT / "submission_v157_wgs_finalseg_diag.py"
BASE_SOURCE = ROOT / "submission_v139_wgs_kperm3.py"
OUT = Path(tempfile.gettempdir()) / "okc_v157_finalseg_static"
BASE = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3",
    "-DOKC_FNST=4", "-DOKC_NSF=3",
)
VARIANTS = {
    "v139": BASE + ("-DOKC_KPERM3=1",),
    "control": BASE + ("-DOKC_KPERM3=1",),
    "fseg2": BASE + ("-DOKC_KPERM3=1", "-DOKC_WGSFSEG=2"),
    "fseg4": BASE + ("-DOKC_KPERM3=1", "-DOKC_WGSFSEG=4"),
}


def cuda_source(path: Path) -> str:
    tree = ast.parse(path.read_text())
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
    env["PATH"] = str(cls[-1].parent) + os.pathsep + env.get("PATH", "")
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc, cuobjdump = tool("nvcc"), tool("cuobjdump")
    env = compiler_environment()
    reuse = "--reuse" in sys.argv
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
    old_v139 = OUT / "v139.sass"
    reference = (Counter(re.findall(r"HGMMA\.([^ ]+)", old_v139.read_text()))
                 if only and only != "v139" and old_v139.exists() else None)
    baseline_sass = old_v139.read_text() if only and only != "v139" and old_v139.exists() else None
    baseline_spills = None
    old_v139_log = OUT / "v139.ptxas.txt"
    if only and only != "v139" and old_v139_log.exists():
        old_spills = [tuple(map(int, x)) for x in re.findall(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", old_v139_log.read_text().lower())]
        baseline_spills = tuple(max(x[i] for x in old_spills) for i in range(3)) if old_spills else (0, 0, 0)
    print(f"build directory: {OUT}")
    for name, flags in VARIANTS.items():
        if only and name != only:
            continue
        source_path = BASE_SOURCE if name == "v139" else SOURCE
        source = cuda_source(source_path)
        cu, cubin = OUT / f"{name}.cu", OUT / f"{name}.cubin"
        cu.write_text(source)
        cmd = [nvcc, "-std=c++17", "-O3", "--use_fast_math", "-arch=sm_90a", "-cubin",
               "-Xptxas=-v", "-lineinfo", *flags, str(cu), "-o", str(cubin)]
        if not (reuse and cubin.exists() and cubin.stat().st_mtime >= source_path.stat().st_mtime):
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            log = (proc.stdout + proc.stderr).strip()
            (OUT / f"{name}.ptxas.txt").write_text(log)
            print(f"\n[{name}] nvcc exit={proc.returncode}")
            for line in log.splitlines():
                if any(k in line.lower() for k in ("compiling entry", "used ", "stack frame", "spill", "warning", "error")):
                    print(line)
            if proc.returncode:
                raise SystemExit(log or f"nvcc failed ({proc.returncode})")
        sass = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True).stdout
        resources = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], text=True,
                                   capture_output=True, check=True).stdout
        (OUT / f"{name}.sass").write_text(sass)
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
        selected = {
            op: len(re.findall(rf"\b{op}\b", sass))
            for op in ("PRMT", "IMAD", "HMUL2", "LDS", "LDG", "STG", "RED", "ATOM", "LDL", "STL")
        }
        warnings = (OUT / f"{name}.ptxas.txt").read_text().lower()
        spills = [tuple(map(int, x)) for x in re.findall(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", warnings)]
        spill_max = tuple(max(x[i] for x in spills) for i in range(3)) if spills else (0, 0, 0)
        print(f"[{name}] resources: " + " | ".join(x.strip() for x in resources.splitlines() if "REG:" in x))
        print(f"[{name}] max stack/spill-store/spill-load bytes={spill_max}")
        print(f"[{name}] HGMMA={dict(hgmma)}")
        print(f"[{name}] selected SASS={selected}")
        assert "serialized due to insufficient register resources" not in warnings
        if name == "v139":
            baseline_sass = sass
            baseline_spills = spill_max
        elif name == "control":
            assert sass == baseline_sass, "default-off v157 control SASS differs from frozen v139"
            assert spill_max == baseline_spills, (spill_max, baseline_spills)
        else:
            assert all(a <= b for a, b in zip(spill_max, baseline_spills)), (name, spill_max, baseline_spills)
        if reference is None:
            reference = hgmma
        else:
            assert hgmma == reference, (name, hgmma, reference)
    print("\nv157 segmented-final cubin/SASS gates: PASS")


if __name__ == "__main__":
    main()
