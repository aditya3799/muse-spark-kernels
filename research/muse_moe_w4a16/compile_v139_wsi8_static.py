"""Compile frozen v139, macro-off control, and M8 WSI candidate for SM90a."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
DIAG = ROOT / "submission_v139_wsi8_diag.py"
OUT = ROOT / ".tmp_v139_wsi8_static"
BASE = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0",
    "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
    "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1",
    "-DOKC_PRMTSC=1", "-DOKC_MULDEC=1", "-DOKC_DQ7=1",
    "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3",
)


def cuda_source(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in node.targets):
            source = ast.literal_eval(node.value)
            break
    else:
        raise RuntimeError(f"CUDA_SRC assignment not found in {path.name}")
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
    variants = {
        "frozen": (cuda_source(FROZEN), BASE),
        "control": (cuda_source(DIAG), BASE),
        "wsi": (cuda_source(DIAG), BASE + ("-DOKC_WSI=1",)),
    }
    evidence = {}
    print(f"build directory: {OUT}")
    for name, (source, flags) in variants.items():
        cu, cubin = OUT / f"{name}.cu", OUT / f"{name}.cubin"
        cu.write_text(source)
        cmd = [nvcc, "-std=c++17", "-O3", "--use_fast_math", "-arch=sm_90a", "-cubin",
               "-Xptxas=-v", *flags, str(cu), "-o", str(cubin)]
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
        resource_lines = tuple(x.strip() for x in resources.splitlines() if "REG:" in x)
        selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
            "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
        )}
        digest = hashlib.sha256(sass.encode()).hexdigest().upper()
        evidence[name] = (resource_lines, selected, digest)
        print(f"[{name}] resources: " + " | ".join(resource_lines))
        print(f"[{name}] selected SASS={selected}")
        print(f"[{name}] SASS SHA256={digest}")

    assert evidence["frozen"] == evidence["control"], (evidence["frozen"], evidence["control"])
    print("[control] exact static match to frozen v139: PASS")
    delta = {op: evidence["wsi"][1][op] - evidence["control"][1][op] for op in evidence["control"][1]}
    print(f"[wsi] instruction deltas vs control={delta}")
    print("v139 M8 WSI static gates: PASS")


if __name__ == "__main__":
    main()
