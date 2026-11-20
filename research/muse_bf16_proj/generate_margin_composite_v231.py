"""Generate v231: guarded composition of the v228-v230 margin isolates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pooler256_margin_v228.py"
V229 = ROOT / "submission_plain_unpool256_cudnn_v229.py"
V230 = ROOT / "submission_plain_o256_margin_v230.py"
OUTPUT = ROOT / "submission_plain_margin_composite_v231.py"

text = BASE.read_text(encoding="utf-8").rstrip()

v229 = V229.read_text(encoding="utf-8")
marker229 = "# v229: test FlashInfer's independent cuDNN BF16 engine stack"
if v229.count(marker229) != 1:
    raise RuntimeError("v229 marker mismatch")
block229 = marker229 + v229.split(marker229, 1)[1]

v230 = V230.read_text(encoding="utf-8")
marker230 = "_o256_RL_CUDA = r'''"
if v230.count(marker230) != 1:
    raise RuntimeError("v230 marker mismatch")
block230 = marker230 + v230.split(marker230, 1)[1]

text += "\n\n\n" + block229.rstrip()
text += "\n\n\n" + block230.rstrip() + "\n"
OUTPUT.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(OUTPUT), "exec")
print(OUTPUT.name, len(text.encode("utf-8")))
