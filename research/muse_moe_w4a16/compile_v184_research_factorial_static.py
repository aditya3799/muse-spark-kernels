"""Compile the four research-factorial cells exactly for SM90a."""

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
SUBMISSION = ROOT / "submission_v178_hybrid_m32_retry.py"
OUT = ROOT / ".tmp_v184_research_factorial_static"
COMMON = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3",
)
VARIANTS = {
    "grid_plain": ("-DOKC_MFIN=0", "-DOKC_B4L=0"),
    "mfin_plain": (),
    "grid_sc6": ("-DOKC_SC6W=1", "-DOKC_MFIN=0", "-DOKC_B4L=0"),
    "mfin_sc6": ("-DOKC_SC6W=1",),
}


def literal_dict_item(path: Path, name: str, key: str):
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            continue
        if target.value.id == name and ast.literal_eval(target.slice) == key:
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name}[{key!r}] not found")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc, cuobjdump = shutil.which("nvcc"), shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    source = literal_dict_item(SUBMISSION, "ALT_SRCS", "v167_main")
    assert "#define OKC_YBF 1" in source
    assert "#define OKC_MFIN 1" in source
    assert "OKC_SC6W ? 35840 : 36864" in source
    summaries = {}
    for name, extra in VARIANTS.items():
        cu = OUT / f"{name}.cu"
        cubin = OUT / f"{name}.cubin"
        cu.write_text(compile_source(source))
        proc = subprocess.run(
            [nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-cubin", "-Xptxas=-v",
             *COMMON, *extra, str(cu), "-o", str(cubin)],
            text=True, capture_output=True, env=compiler_environment(),
        )
        log = (proc.stdout + proc.stderr).strip()
        (OUT / f"{name}.ptxas.txt").write_text(log)
        print(f"[{name}] nvcc exit={proc.returncode}")
        for line in log.splitlines():
            if any(key in line.lower() for key in (
                "compiling entry", "used ", "stack frame", "spill", "warning", "error",
            )):
                print(line)
        if proc.returncode:
            raise SystemExit(log or f"{name}: nvcc failed")
        sass = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True
        ).stdout
        resources = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)], text=True, capture_output=True, check=True
        ).stdout
        (OUT / f"{name}.sass").write_text(sass)
        resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
        hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
        selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
            "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
        )}
        digest = hashlib.sha256(sass.encode()).hexdigest().upper()
        summaries[name] = (resource_lines, selected, dict(hgmma), digest)
        print(f"[{name}] resources: " + " | ".join(resource_lines))
        print(f"[{name}] selected={selected} HGMMA={dict(hgmma)} SHA256={digest}")
        assert all("REG:96" in line and "LOCAL:0" in line for line in resource_lines)
        assert sum(hgmma.values()) == 1024 and len(hgmma) == 8
        assert "serialized due to insufficient register resources" not in log.lower()
    assert len({entry[3] for entry in summaries.values()}) == 4
    print("v184 four-cell research factorial preserves 96 registers, LOCAL:0, and all HGMMA sites: PASS")


if __name__ == "__main__":
    main()
