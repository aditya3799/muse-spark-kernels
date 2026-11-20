from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_rs128_faithful_v150.py"
OUTPUT = ROOT / "submission_qkv256_lt124_v168.py"


text = SOURCE.read_text()
old = '("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 134, 2, True), ("o", 256, False, 20, 4, False),'
new = '("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 124, 0, False), ("o", 256, False, 20, 4, False),'
if text.count(old) != 1:
    raise RuntimeError("expected exactly one v150 qkv_gate m256 cuBLASLt route")

text = text.replace(old, new)
text = text.replace(
    'RQ_LT = [   # (layer, m, weight pre-transposed, tile id, custom option, pick from heuristic list)',
    'RQ_LT = [   # v168: qkv_gate m256 uses stable algo-66 tile-124/custom-0 (72x64 nvjet).\n'
    '            # (layer, m, weight pre-transposed, tile id, custom option, pick from heuristic list)',
    1,
)
OUTPUT.write_text(text)
print(OUTPUT)
