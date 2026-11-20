"""Static SM90a audit for frozen v139 and every v172 dispatch branch."""

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
CANDIDATE = ROOT / "submission_v172_hybrid_small.py"
OUT = ROOT / ".tmp_v172_hybrid_static"


def parsed(path: Path):
    return ast.parse(path.read_text())


def assignments(path: Path) -> dict[str, object]:
    out = {}
    for node in parsed(path).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def alt_sources(path: Path) -> dict[str, str]:
    out = {}
    for node in parsed(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != "ALT_SRCS":
            continue
        slice_node = target.slice.value if isinstance(target.slice, ast.Index) else target.slice
        out[ast.literal_eval(slice_node)] = ast.literal_eval(node.value)
    return out


def compile_source(source: str) -> str:
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
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    env = compiler_environment()
    frozen = assignments(FROZEN)
    candidate = assignments(CANDIDATE)
    alts = alt_sources(CANDIDATE)
    inject_main = ("-DOKC_PRMTSC=1", "-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3")
    inject_small = ("-DOKC_PRMTSC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3")
    strip_alt = lambda defs: tuple(d for d in defs if not d.startswith("@"))
    variants = {
        "frozen_main": (frozen["CUDA_SRC"], tuple(frozen["MAIN_DEFINES"]) + inject_main, True),
        "hybrid_main": (candidate["CUDA_SRC"], tuple(candidate["MAIN_DEFINES"]) + inject_main, True),
        "tiny": (alts["v167_var"], strip_alt(candidate["TINY_DEFINES"]) + inject_small, False),
        "tiny2_wsi": (alts["v167_wsi"], strip_alt(candidate["TINY2_DEFINES"]) + inject_small, False),
        "mid_helper0": (alts["v167_helper0"], strip_alt(candidate["MID_DEFINES"]) + inject_small, False),
    }
    evidence = {}
    print(f"build directory: {OUT}")
    for name, (source, flags, fast_math) in variants.items():
        cu, cubin = OUT / f"{name}.cu", OUT / f"{name}.cubin"
        expected_source = compile_source(source)
        cached = cubin.exists() and cu.exists() and cu.read_text() == expected_source
        if cached:
            log = (OUT / f"{name}.ptxas.txt").read_text()
            print(f"\n[{name}] reusing completed cubin")
        else:
            cu.write_text(expected_source)
            cmd = [nvcc, "-std=c++17", "-O3", *( ["--use_fast_math"] if fast_math else [] ),
                   "-arch=sm_90a", "-cubin", "-Xptxas=-v", *flags, str(cu), "-o", str(cubin)]
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            log = (proc.stdout + proc.stderr).strip()
            (OUT / f"{name}.ptxas.txt").write_text(log)
            print(f"\n[{name}] nvcc exit={proc.returncode}")
            if proc.returncode:
                raise SystemExit(log or f"nvcc failed for {name}")
        for line in log.splitlines():
            if any(key in line.lower() for key in ("compiling entry", "used ", "stack frame", "spill", "warning", "error")):
                print(line)
        sass = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True).stdout
        resources = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], text=True,
                                   capture_output=True, check=True).stdout
        (OUT / f"{name}.sass").write_text(sass)
        resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
        selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
            "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
        )}
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
        digest = hashlib.sha256(sass.encode()).hexdigest().upper()
        evidence[name] = (resource_lines, selected, hgmma, digest)
        print(f"[{name}] resources: " + " | ".join(resource_lines))
        print(f"[{name}] selected={selected} HGMMA={dict(hgmma)} SHA256={digest}")

    assert evidence["frozen_main"] == evidence["hybrid_main"]
    for name in ("tiny", "tiny2_wsi", "mid_helper0"):
        assert all("STACK:0" in line and "LOCAL:0" in line for line in evidence[name][0]), evidence[name][0]
    print("v139 large WGS exact SASS match: PASS")
    print("v167 small branches zero stack/local: PASS")
    print("v172 hybrid static gates: PASS")


if __name__ == "__main__":
    main()
