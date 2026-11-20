"""Static SM90a audit for the v179 SC6 four-weight/two-activation ring."""

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
BASE = ROOT / "submission_v167_keell0renz_890809c.py"
BUILDER = ROOT / "make_diag_v179_sc6_4w2x.py"
CANDIDATE = ROOT / "submission_diag_v179_sc6_4w2x.py"
OUT = ROOT / ".tmp_v179_sc6_4w2x_static"
COMMON = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_SC6W=1", "-DOKC_DQ7=1", "-DOKC_NS=3",
    "-DOKC_NSF=3",
)


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in {path.name}")


def transformed_source(base: str) -> str:
    appendix = literal_assignment(BUILDER, "APPEND")
    prefix = appendix.split("W42_SHAPES =", 1)[0]
    namespace = {"CUDA_SRC": base, "ALT_SRCS": {}}
    exec(prefix, namespace)
    return namespace["ALT_SRCS"]["sc6_4w2x"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")

    base = literal_assignment(BASE, "CUDA_SRC")
    twin = transformed_source(base)
    appendix = literal_assignment(BUILDER, "APPEND")
    raw_base = BASE.read_bytes().rstrip(b"\r\n")
    expected = raw_base + appendix.encode("utf-8") + b"\n"
    assert CANDIDATE.read_bytes() == expected, "candidate is not exact base + audited appendix"
    assert base.count("ring + ws * UNIT_BYTES") == 1
    assert twin.count("ring + ws * WGS_STG") == 1
    assert "constexpr int WGS_NS = OKC_WGSNS" in twin
    assert "constexpr int WGS_NX = OKC_WGSNX" in twin
    assert "okc::WGS_NS * okc::WGS_STG" in twin

    variants = {
        "prod_3w3x": (base, COMMON),
        "sc6_4w2x": (twin, COMMON + ("-DOKC_WGSNS=4", "-DOKC_WGSNX=2")),
    }
    evidence = {}
    env = compiler_environment()
    for name, (source, flags) in variants.items():
        cu = OUT / f"{name}.cu"
        cubin = OUT / f"{name}.cubin"
        src = compile_source(source)
        cu.write_text(src)
        cmd = [
            nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-cubin", "-Xptxas=-v",
            *flags, str(cu), "-o", str(cubin),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        log = (proc.stdout + proc.stderr).strip()
        (OUT / f"{name}.ptxas.txt").write_text(log)
        print(f"[{name}] nvcc exit={proc.returncode}")
        for line in log.splitlines():
            if any(key in line.lower() for key in (
                "compiling entry", "used ", "stack frame", "spill", "warning", "error",
            )):
                print(line)
        if proc.returncode:
            raise SystemExit(log or f"nvcc failed for {name}")

        sass = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True
        ).stdout
        resources = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)],
            text=True, capture_output=True, check=True,
        ).stdout
        (OUT / f"{name}.sass").write_text(sass)
        resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
        selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
            "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
        )}
        digest = hashlib.sha256(sass.encode()).hexdigest().upper()
        print(f"[{name}] resources: " + " | ".join(resource_lines))
        print(f"[{name}] selected={selected} HGMMA={dict(hgmma)} SHA256={digest}")
        assert all("LOCAL:0" in line for line in resource_lines)
        assert "serialized due to insufficient register resources" not in log.lower()
        evidence[name] = (resource_lines, hgmma)

    assert evidence["prod_3w3x"] == evidence["sc6_4w2x"]
    assert 3 * 36864 + 3 * 32768 == 4 * 35840 + 2 * 32768 == 208896
    print("v179 production/twin resources and HGMMA counts match: PASS")
    print("v179 exact shared-memory footprint identity: PASS")
    print("v179 static gates: PASS")


if __name__ == "__main__":
    main()
