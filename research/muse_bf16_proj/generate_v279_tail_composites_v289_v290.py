"""Build v279 plus guarded Pooler/Unpooler m256 tail heuristics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
source_path = ROOT / "generate_m256_deep_heuristics_v252_v254.py"
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    'BASE = ROOT / "submission_plain_full_guard_v221.py"',
    'BASE = ROOT / "submission_plain_complete_pairplanes_v279.py"',
    1,
)
old = '''for candidate_version, candidate_layer in (
        (252, "pooler"), (253, "qkv_gate"), (254, "unpooler")):
    build(candidate_version, candidate_layer)

for candidate_version, candidate_layer in (
        (268, "pooler"), (269, "qkv_gate"), (270, "unpooler")):
    build(candidate_version, candidate_layer, start=48, stop=96,
          request=96, label="tail_heuristics")
'''
new = '''for candidate_version, candidate_layer in (
        (289, "pooler"), (290, "unpooler")):
    build(candidate_version, candidate_layer, start=48, stop=96,
          request=96, label="tail_heuristics")
'''
if source.count(old) != 1:
    raise RuntimeError("tail-loop marker mismatch")
source = source.replace(old, new, 1)
namespace = {"__file__": str(source_path), "__name__": "__main__"}
exec(compile(source, str(source_path), "exec"), namespace)
