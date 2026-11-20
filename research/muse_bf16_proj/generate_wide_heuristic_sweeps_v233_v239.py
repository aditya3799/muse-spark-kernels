"""Generate isolated zero-workspace cuBLASLt heuristic sweeps on v221/v228.

The submitted outputs remain ordinary readable Python/CUDA source.  Each candidate
changes exactly one scored wide route, benchmarks the first twelve viable plans
for both weight layouts, and retains a heuristic only when its adjusted kernel
core beats the best incumbent by at least one percent.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "submission_plain_pooler256_margin_v228.py"

TARGETS = (
    (233, "qkv_gate", 128),
    (234, "pooler", 128),
    (235, "unpooler", 128),
    (236, "qkv_gate", 256),
    (237, "o", 128),
    (238, "o", 256),
    (239, "unpooler", 256),
)


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    old_route = '_V228_ROUTES = (("pooler", 256),)'
    old_log = 'print("v228 Pooler256 LT gate", cands[best][0],'
    if template.count(old_route) != 1:
        raise RuntimeError("unexpected v228 route marker count")
    if template.count(old_log) != 1:
        raise RuntimeError("unexpected v228 log marker count")

    for version, layer, m in TARGETS:
        source = template.replace(
            old_route, f'_V228_ROUTES = (("{layer}", {m}),)', 1
        ).replace(
            old_log,
            f'print("v{version} {layer}{m} LT gate", cands[best][0],',
            1,
        )
        output = ROOT / f"submission_plain_{layer}{m}_heuristic_v{version}.py"
        output.write_text(source, encoding="utf-8")
        print(output.name)


if __name__ == "__main__":
    main()
