"""Attribute frozen-v139 SM90a SASS instructions to CUDA source lines.

Run ``compile_v139_static.py`` first.  This script is read-only and consumes
the generated KPERM3 cubin in the system temporary directory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import subprocess
import tempfile


BUILD = Path(tempfile.gettempdir()) / "okc_v139_static"
CUBIN = BUILD / "kperm3.cubin"
CUDA = BUILD / "kperm3.cu"

SECTION_RE = re.compile(r"\.section\s+\.text\.([^,]+)")
LINE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
INST_RE = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@[!A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b")


def family(op: str) -> str:
    return op.split(".", 1)[0]


def main() -> None:
    if not CUBIN.exists() or not CUDA.exists():
        raise SystemExit("run compile_v139_static.py first")
    proc = subprocess.run(
        ["nvdisasm", "-c", "-g", str(CUBIN)],
        text=True,
        capture_output=True,
        errors="replace",
        env=os.environ.copy(),
    )
    if not proc.stdout:
        raise SystemExit(proc.stderr or "nvdisasm produced no output")

    source = CUDA.read_text().splitlines()
    current_section = "unknown"
    current_file = ""
    current_line = 0
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    by_line: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)

    for raw in proc.stdout.splitlines():
        m = SECTION_RE.search(raw)
        if m:
            current_section = m.group(1)
            current_file = ""
            current_line = 0
            continue
        m = LINE_RE.search(raw)
        if m:
            current_file = Path(m.group(1).replace("\\\\", "\\")).name
            current_line = int(m.group(2))
            continue
        m = INST_RE.search(raw)
        if not m:
            continue
        op = m.group(1)
        fam = family(op)
        by_section[current_section][fam] += 1
        if current_file == CUDA.name and current_line:
            by_line[(current_section, current_line)][fam] += 1

    focus = ("HGMMA", "IMAD", "PRMT", "HMUL2", "LDG", "LDS", "LDL", "STL", "MUFU", "ATOM", "BAR", "SHFL")
    for section, counts in by_section.items():
        if "moe_fused" not in section:
            continue
        print(f"[{section}]")
        print("  " + " ".join(f"{op}={counts[op]}" for op in focus if counts[op]))
        rows = []
        for (sec, line), ops in by_line.items():
            if sec != section:
                continue
            interesting = sum(ops[op] for op in focus)
            if interesting:
                rows.append((interesting, sum(ops.values()), line, ops))
        print("  top attributed CUDA lines (interesting/total static sites):")
        for interesting, total, line, ops in sorted(rows, reverse=True)[:35]:
            text = source[line - 1].strip() if 0 < line <= len(source) else "?"
            detail = ",".join(f"{op}:{ops[op]}" for op in focus if ops[op])
            print(f"    {line:4d} {interesting:4d}/{total:4d} {detail:35s} | {text}")

    own = Counter()
    for ops in by_line.values():
        own.update(ops)
    print("[all source-attributed instruction families]")
    print("  " + " ".join(f"{op}={n}" for op, n in own.most_common(30)))


if __name__ == "__main__":
    main()
