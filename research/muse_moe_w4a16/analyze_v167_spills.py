"""Attribute every WGS local-memory instruction in the final v166 control build."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".tmp_v166_epiaddr_diag_static"
CUBIN = BUILD / "control.cubin"
CUDA = BUILD / "control.cu"
SECTION_RE = re.compile(r"\.section\s+\.text\.([^,]+)")
LINE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
INST_RE = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@[!A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b(.*?);")


def main() -> None:
    source = CUDA.read_text().splitlines()
    proc = subprocess.run(
        ["nvdisasm", "-c", "-g", str(CUBIN)],
        text=True,
        capture_output=True,
        errors="replace",
        check=True,
    )
    section = "unknown"
    file = ""
    line = 0
    by_line: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    samples: dict[tuple[str, int], list[str]] = defaultdict(list)
    offsets: Counter[str] = Counter()
    for raw in proc.stdout.splitlines():
        match = SECTION_RE.search(raw)
        if match:
            section = match.group(1)
            file = ""
            line = 0
            continue
        match = LINE_RE.search(raw)
        if match:
            file = Path(match.group(1).replace("\\\\", "\\")).name
            line = int(match.group(2))
            continue
        match = INST_RE.search(raw)
        if not match or not match.group(1).startswith(("LDL", "STL")):
            continue
        op = match.group(1).split(".", 1)[0]
        if "moe_fused" not in section or file != CUDA.name or not line:
            continue
        key = (section, line)
        by_line[key][op] += 1
        if len(samples[key]) < 3:
            samples[key].append(match.group(0).strip())
        off = re.search(r"\[R1(?:\+(0x[0-9a-f]+))?\]", match.group(0), re.I)
        offsets[off.group(1) if off and off.group(1) else "0x0/other"] += 1

    for section in sorted({sec for sec, _ in by_line}):
        print(f"[{section}]")
        for (sec, line), counts in sorted(by_line.items(), key=lambda kv: (-sum(kv[1].values()), kv[0][1])):
            if sec != section:
                continue
            code = source[line - 1].strip() if 0 < line <= len(source) else "?"
            print(f"  line {line}: LDL={counts['LDL']} STL={counts['STL']} | {code}")
            for sample in samples[(sec, line)]:
                print(f"    {sample}")
    print("[stack-offset static sites]")
    for offset, count in offsets.most_common():
        print(f"  {offset}: {count}")


if __name__ == "__main__":
    main()
