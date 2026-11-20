"""Create corrected O256 composites with a module-scope hashlib import."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAIRS = (
    ("submission_plain_v279_o256_v281.py",
     "submission_plain_v279_o256_hashfix_v285.py"),
    ("submission_plain_v279_pooler256_o256_v283.py",
     "submission_plain_v279_pooler256_o256_hashfix_v286.py"),
)

for source_name, output_name in PAIRS:
    text = (ROOT / source_name).read_text(encoding="utf-8")
    marker = "from typing import TypedDict\n\nimport torch\n"
    if text.count(marker) != 1:
        raise RuntimeError(f"import marker mismatch in {source_name}")
    text = text.replace(
        marker, "from typing import TypedDict\n\nimport hashlib\nimport torch\n", 1)
    output = ROOT / output_name
    compile(text, str(output), "exec")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(output.name, len(text.encode("utf-8")))
