"""Generate v229: isolate the prior cuDNN Unpooler/m256 micro-win on v221."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_cudnn_wide_v140.py"
OUTPUT = ROOT / "submission_plain_unpool256_cudnn_v229.py"

text = BASE.read_text(encoding="utf-8")
donor = DONOR.read_text(encoding="utf-8")
marker = "# v140: test FlashInfer's independent cuDNN BF16 engine stack"
if donor.count(marker) != 1:
    raise RuntimeError("v140 donor marker mismatch")
block = marker + donor.split(marker, 1)[1]

old_routes = '''_V140_WIDE_ROUTES = tuple(
    (layer, m)
    for m in (128, 256)
    for layer in ('qkv_gate', 'o', 'pooler', 'unpooler')
)
'''
new_routes = "_V140_WIDE_ROUTES = (('unpooler', 256),)\n"
if block.count(old_routes) != 1:
    raise RuntimeError("v140 route marker mismatch")
block = block.replace(old_routes, new_routes, 1)
block = block.replace("_V140_ROUTE_RATIO = 0.97",
                      "_V140_ROUTE_RATIO = 0.99", 1)
block = block.replace("_V140_GLOBAL_GAIN = 0.02",
                      "_V140_GLOBAL_GAIN = 0.0", 1)
block = block.replace("# v140:", "# v229:", 1)
block = block.replace("v140 cuDNN", "v229 cuDNN")

text = text.rstrip() + "\n\n\n" + block.rstrip() + "\n"
OUTPUT.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(OUTPUT), "exec")
print(OUTPUT.name, len(text.encode("utf-8")))
