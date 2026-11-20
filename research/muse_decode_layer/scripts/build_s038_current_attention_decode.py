"""Build S038: S031 with the current validated standalone decode attention.

Only q_len=1 uses this replacement.  The separately validated v81 q_len=8
path remains unchanged.  The standalone source is isolated in a module so its
helper globals cannot collide with the dense, MoE, or decoder backends.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s031_v81_b1.py"
ATTENTION_SOURCE = ROOT.parent / "muse_attention_decode" / "submission.py"
DESTINATION = ROOT / "submission_s038_current_attention_decode.py"
SOURCE_SHA256 = "f5518e5bb7b78321da571d7479aa931db654af1f468ac01d6196cb6d581530ae"
ATTENTION_SHA256 = "e852360748ecc58126f091dd11c1df8014a1db0caf47910fac794ebfcac7ea2f"
OLD_END = "_e003_attention_kernel_fn = kernel_fn\n"


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S031 hash: {actual_source}")
attention_bytes = ATTENTION_SOURCE.read_bytes()
actual_attention = hashlib.sha256(attention_bytes).hexdigest()
if actual_attention != ATTENTION_SHA256:
    raise RuntimeError(f"unexpected standalone attention hash: {actual_attention}")

source = source_bytes.decode("utf-8")
if source.count(OLD_END) != 1:
    raise RuntimeError("unexpected embedded E003 attention boundary")
header = source.splitlines(keepends=True)[0]
tail = source[source.index(OLD_END) + len(OLD_END):]
payload = base64.b64encode(zlib.compress(attention_bytes, level=9)).decode("ascii")
module_block = f'''# Current standalone attention, isolated to avoid global-name collisions.
import base64 as _e004_b64, types as _e004_types, zlib as _e004_zlib
_E004_ATTENTION_SRC = "{payload}"
_e004_attention = _e004_types.ModuleType("_e004_attention")
_e004_attention.__file__ = __file__
exec(compile(_e004_zlib.decompress(_e004_b64.b64decode(_E004_ATTENTION_SRC)).decode(),
             "_e004_attention", "exec"), _e004_attention.__dict__)
_e003_attention_kernel_fn = _e004_attention.kernel_fn
_ROPE = {{}}
'''
candidate = header + module_block + tail

required_fragments = (
    "# okc: problem=muse_decode_layer",
    "_e003_attention_kernel_fn = _e004_attention.kernel_fn",
    "_V81_CONFIG = {",
    'cfg["batch"] in (1, 16, 32)',
    "Dense projections:",
    '"""SGLang\'s decoder layer for Muse Spark Open',
)
for fragment in required_fragments:
    if fragment not in candidate:
        raise RuntimeError(f"missing S038 fragment: {fragment}")
if candidate.count("_e003_attention_kernel_fn =") != 1:
    raise RuntimeError("unexpected attention alias count")
if "paged_full_decode_partials_warp" in candidate[: candidate.index("_V81_CONFIG = {")]:
    raise RuntimeError("old plaintext E003 attention was not removed")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S038 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("attention payload", len(payload), "bytes")
print("sha256", hashlib.sha256(encoded).hexdigest())
