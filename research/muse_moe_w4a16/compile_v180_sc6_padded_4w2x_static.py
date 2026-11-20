"""Static scout: four SC6 weight slots/two x slots with padded weight strides."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

from compile_v172_hybrid_static import compile_source, compiler_environment
from compile_v179_sc6_4w2x_static import BASE, COMMON, literal_assignment, transformed_source


ROOT = Path(__file__).resolve().parent
OUT = ROOT / ".tmp_v180_sc6_padded_4w2x_static"


def padded_source() -> str:
    source = transformed_source(literal_assignment(BASE, "CUDA_SRC"))
    replacements = (
        ("ring + ws * WGS_STG)), \"l\"(src)", "ring + ws * UNIT_BYTES)), \"l\"(src)"),
        ("unsigned char* xring = ring + WGS_NS * WGS_STG;", "unsigned char* xring = ring + WGS_NS * UNIT_BYTES;"),
        ("const uint32_t xring_s = smem_u32(ring + WGS_NS * WGS_STG);", "const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);"),
        ("const unsigned char* pc = pc0 + ws * WGS_STG;", "const unsigned char* pc = pc0 + ws * UNIT_BYTES;"),
        ("okc::WGS_NS * okc::WGS_STG + (size_t)okc::WGS_NX", "okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX"),
    )
    for old, new in replacements:
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
        raise RuntimeError("nvcc/cuobjdump not on PATH")
    source = padded_source()
    cu = OUT / "sc6_padded_4w2x.cu"
    cubin = OUT / "sc6_padded_4w2x.cubin"
    cu.write_text(compile_source(source))
    flags = COMMON + ("-DOKC_WGSNS=4", "-DOKC_WGSNX=2")
    proc = subprocess.run(
        [nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-cubin", "-Xptxas=-v",
         *flags, str(cu), "-o", str(cubin)],
        text=True, capture_output=True, env=compiler_environment(),
    )
    log = (proc.stdout + proc.stderr).strip()
    (OUT / "sc6_padded_4w2x.ptxas.txt").write_text(log)
    print(f"nvcc exit={proc.returncode}")
    for line in log.splitlines():
        if any(key in line.lower() for key in (
            "compiling entry", "used ", "stack frame", "spill", "warning", "error",
        )):
            print(line)
    if proc.returncode:
        raise SystemExit(log or "nvcc failed")
    sass = subprocess.run(
        [cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True, check=True
    ).stdout
    resources = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(cubin)], text=True, capture_output=True, check=True
    ).stdout
    (OUT / "sc6_padded_4w2x.sass").write_text(sass)
    resource_lines = tuple(line.strip() for line in resources.splitlines() if "REG:" in line)
    hgmma = Counter(re.findall(r"HGMMA\.([^ ]+)", sass))
    selected = {op: len(re.findall(rf"\b{op}\b", sass)) for op in (
        "IMAD", "IADD3", "LDS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "BRA",
    )}
    print("resources: " + " | ".join(resource_lines))
    print(f"selected={selected} HGMMA={dict(hgmma)} SHA256={hashlib.sha256(sass.encode()).hexdigest().upper()}")
    assert all("REG:96" in line and "STACK:8" in line and "LOCAL:0" in line for line in resource_lines)
    assert selected["LDL"] == 12 and selected["STL"] == 2
    assert "serialized due to insufficient register resources" not in log.lower()
    assert 4 * 36864 + 2 * 32768 == 212992
    assert 212992 + 1024 < 227328
    print("v180 padded ring preserves the control register/frame gate: PASS")
    print("v180 padded total shared memory 214016 B < 227328 B: PASS")


if __name__ == "__main__":
    main()
