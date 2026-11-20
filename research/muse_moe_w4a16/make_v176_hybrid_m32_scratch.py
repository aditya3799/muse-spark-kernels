"""Fix v175's M32 host ABI with v167's exact scratch layout."""

from __future__ import annotations

from pathlib import Path

from make_v175_hybrid_m32 import function_source


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v175_hybrid_m32.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v176_hybrid_m32_scratch.py"


def main() -> None:
    text = BASE.read_text()
    sentinel = function_source(ADVANCED, "_inter_sentinel", "_v167_inter_sentinel")
    sentinel = sentinel.replace("HIDDEN", "4096")
    scratch = function_source(ADVANCED, "_make_scratch", "_v167_make_scratch")
    scratch = scratch.replace("HIDDEN", "4096").replace("_inter_sentinel", "_v167_inter_sentinel")

    start = text.index("def _prepare_v167_m32_layout")
    end = text.index("\n\ndef prepare(weights):", start)
    block = text[start:end]
    assert block.count('state["scratch"] = _make_scratch(dev)') == 1
    block = block.replace('state["scratch"] = _make_scratch(dev)',
                          'state["scratch"] = _v167_make_scratch(dev)')
    insertion = sentinel + "\n\n" + scratch + "\n\n" + block
    text = text[:start] + insertion + text[end:]
    DESTINATION.write_text(text)
    print(f"wrote {DESTINATION.name} ({len(text)} characters)")


if __name__ == "__main__":
    main()
