"""Build S037: exact-budget all-shape routing + B1 v81 + dense v87."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s036_maxbudget_route_shaper.py"
DENSE_SOURCE = ROOT.parent / "muse_bf16_proj" / "submission_auto_research_dense_v87.py"
DESTINATION = ROOT / "submission_s037_maxroute_b1_dense_v87.py"
SOURCE_SHA256 = "38b945997d32c2a597db97e0f9377b80f1aa1efcefe9824028ec2ce8b1355fe0"
DENSE_SHA256 = "eced0e32d609100a8946bd3727fb76cdfcbbd15f5c7e5536c6132c145bcbf2fd"
DECODER_MARKER = '\"\"\"SGLang\'s decoder layer for Muse Spark Open'


source_bytes = SOURCE.read_bytes()
if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("unexpected S036 source hash")
dense_bytes = DENSE_SOURCE.read_bytes()
if hashlib.sha256(dense_bytes).hexdigest() != DENSE_SHA256:
    raise RuntimeError("unexpected dense-v87 source hash")

source = source_bytes.decode("utf-8")
dense = dense_bytes.decode("utf-8").replace("\r\n", "\n").rstrip()
if not dense.startswith("# okc: problem=muse_bf16_proj\n"):
    raise RuntimeError("unexpected dense-v87 problem header")
dense = dense.replace(
    "# okc: problem=muse_bf16_proj\n",
    "# okc: problem=muse_decode_layer\n",
    1,
)
decoder = source[source.index(DECODER_MARKER):]
candidate = dense + "\n_e002_dense_prepare = prepare\n_e002_dense_kernel_fn = kernel_fn\n" + decoder

old = '        and cfg["batch"] in (16, 32)\n'
new = '        and cfg["batch"] in (1, 16, 32)\n'
if candidate.count(old) != 1:
    raise RuntimeError(f"expected one v81 predicate, found {candidate.count(old)}")
candidate = candidate.replace(old, new, 1)

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S037 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
