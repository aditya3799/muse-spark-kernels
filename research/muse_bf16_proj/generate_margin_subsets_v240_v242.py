"""Generate every two-route subset of the v228-v230 margin isolates.

These are staged for use only after the corresponding isolated H200 checks
enable both routes.  The outputs contain literal readable source and preserve
v221 as the fallback beneath every wrapper.
"""

from pathlib import Path
from typing import Tuple


ROOT = Path(__file__).resolve().parent
V221 = ROOT / "submission_plain_full_guard_v221.py"
V228 = ROOT / "submission_plain_pooler256_margin_v228.py"
V229 = ROOT / "submission_plain_unpool256_cudnn_v229.py"
V230 = ROOT / "submission_plain_o256_margin_v230.py"


def tail(path: Path, marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    if source.count(marker) != 1:
        raise RuntimeError(f"marker mismatch in {path.name}: {marker!r}")
    return marker + source.split(marker, 1)[1]


def emit(version: int, label: str, base: Path, blocks: Tuple[str, ...]) -> None:
    source = base.read_text(encoding="utf-8").rstrip()
    source += "".join("\n\n\n" + block.rstrip() for block in blocks) + "\n"
    output = ROOT / f"submission_plain_{label}_v{version}.py"
    compile(source, str(output), "exec")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
    print(output.name, len(source.encode("utf-8")))


def main() -> None:
    block229 = tail(V229, "# v229: test FlashInfer's independent cuDNN BF16 engine stack")
    block230 = tail(V230, "_o256_RL_CUDA = r'''")
    emit(240, "pool256_unpool256_margin", V228, (block229,))
    emit(241, "pool256_o256_margin", V228, (block230,))
    emit(242, "unpool256_o256_margin", V221, (block229, block230))


if __name__ == "__main__":
    main()
