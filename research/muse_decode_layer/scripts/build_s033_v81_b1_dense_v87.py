"""Build S033: S031 with the protected standalone BF16 projection v87."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s031_v81_b1.py"
DENSE_SOURCE = ROOT.parent / "muse_bf16_proj" / "submission_auto_research_dense_v87.py"
DESTINATION = ROOT / "submission_s033b_v81_b1_dense_v87.py"
SOURCE_SHA256 = "f5518e5bb7b78321da571d7479aa931db654af1f468ac01d6196cb6d581530ae"
DENSE_SHA256 = "eced0e32d609100a8946bd3727fb76cdfcbbd15f5c7e5536c6132c145bcbf2fd"
DENSE_MARKER = '\"\"\"Dense projections:'
DECODER_MARKER = '\"\"\"SGLang\'s decoder layer for Muse Spark Open'


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S031 hash: {actual_source} != {SOURCE_SHA256}")
dense_bytes = DENSE_SOURCE.read_bytes()
actual_dense = hashlib.sha256(dense_bytes).hexdigest()
if actual_dense != DENSE_SHA256:
    raise RuntimeError(f"unexpected dense-v87 hash: {actual_dense} != {DENSE_SHA256}")

source = source_bytes.decode("utf-8")
dense = dense_bytes.decode("utf-8").replace("\r\n", "\n").rstrip()
if not dense.startswith("# okc: problem=muse_bf16_proj\n"):
    raise RuntimeError("unexpected dense-v87 problem header")
dense = dense.replace(
    "# okc: problem=muse_bf16_proj\n",
    "# Embedded protected dense-v87 component.\n",
    1,
)
prefix = source[:source.index(DENSE_MARKER)]
decoder = source[source.index(DECODER_MARKER):]
candidate = (
    prefix
    + dense
    + "\n_e002_dense_prepare = prepare\n"
    + "_e002_dense_kernel_fn = kernel_fn\n"
    + "import hashlib\n"
    + decoder
)

required_fragments = (
    "# okc: problem=muse_decode_layer",
    "_e003_attention_kernel_fn = kernel_fn",
    "_V81_CONFIG = {",
    'cfg["batch"] in (1, 16, 32)',
    "Dense projections for Muse Spark Open",
    "_e002_dense_prepare = prepare",
    '"""SGLang\'s decoder layer for Muse Spark Open',
)
for fragment in required_fragments:
    if fragment not in candidate:
        raise RuntimeError(f"missing required S033b fragment: {fragment}")
if candidate.count("_e003_attention_kernel_fn = kernel_fn") != 1:
    raise RuntimeError("unexpected embedded-attention alias count")
if candidate.count("Dense projections for Muse Spark Open") != 1:
    raise RuntimeError("unexpected protected dense-v87 backend count")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError("S033 exceeds the challenge source limit")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
