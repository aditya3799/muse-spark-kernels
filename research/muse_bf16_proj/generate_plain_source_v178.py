"""Generate a human-readable v150 composite with no encoded runtime source.

The disqualified v150 family embedded two source payloads:

* a complete alternate Python submission compressed with zlib/base64; and
* the v150 RS128 CUDA translation unit compressed the same way.

This generator replaces the alternate with the pinned, readable v167 upstream
source in a lexical namespace and materializes the RS128 CUDA as a normal raw
source literal.  The generated submission contains no source decoder and no
runtime exec/compile operation.
"""

from __future__ import annotations

import ast
import base64
import textwrap
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
PLAIN_ALT = ROOT / "submission_chenwainuo_full_v167.py"
OUTPUT = ROOT / "submission_plain_source_v178.py"


def assigned_literal(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(f"{name} is not a string literal")
            return value
    raise KeyError(name)


def raw_literal(value: str) -> str:
    """Return an exact, readable multiline literal for native source."""
    for quote in ('"""', "'''"):
        if quote not in value and not value.endswith("\\"):
            return "r" + quote + value + quote
    raise ValueError("native source cannot be represented by one raw literal")


base = BASE.read_text()
alt = PLAIN_ALT.read_text()

# The upstream source uses two module-global extension caches.  Once the whole
# module is placed in a lexical namespace, those writes must target the lexical
# bindings rather than the generated submission's main module.
if alt.count("    global _gvmod") != 1 or alt.count("    global _mod") != 1:
    raise RuntimeError("unexpected v167 global-cache declarations")
alt = alt.replace("    global _gvmod", "    nonlocal _gvmod", 1)
alt = alt.replace("    global _mod", "    nonlocal _mod", 1)

loader_start = "import types as _types, zlib as _zl, base64 as _b64\n"
loader_end = (
    'exec(compile(_zl.decompress(_b64.b64decode(_ALT_SRC)).decode(), '
    '"_alt_proj", "exec"), _alt.__dict__)\n'
)
if base.count(loader_start) != 1 or base.count(loader_end) != 1:
    raise RuntimeError("unexpected v150 alternate-source loader")

start = base.index(loader_start)
end = base.index(loader_end, start) + len(loader_end)
loader_block = base[start:end]
route_line = next(
    line for line in loader_block.splitlines()
    if line.startswith("_ALT_ROUTES = ")
)

plain_loader = (
    "import types as _types\n"
    + route_line + "\n\n"
    + "def _v178_build_alt_namespace():\n"
    + textwrap.indent(alt, "    ")
    + "\n    return _types.SimpleNamespace(**locals())\n\n"
    + "_alt = _v178_build_alt_namespace()\n"
    + "del _v178_build_alt_namespace\n"
)
base = base[:start] + plain_loader + base[end:]
base = base.replace("_ALT_SRC", "readable alternate source")

payload = assigned_literal(base, "_V150_CUDA_PAYLOAD")
cuda = zlib.decompress(base64.b64decode(payload)).decode()

payload_start = base.index('_V150_CUDA_PAYLOAD = "')
payload_end = base.index("\n", payload_start) + 1
base = (
    base[:payload_start]
    + "_V150_CUDA_SOURCE = " + raw_literal(cuda) + "\n"
    + base[payload_end:]
)

base = base.replace("import base64 as _v150_base64\n", "", 1)
base = base.replace("import zlib as _v150_zlib\n", "", 1)
old_decode = (
    "def _v150_cuda_source():\n"
    "    return _v150_zlib.decompress("
    "_v150_base64.b64decode(_V150_CUDA_PAYLOAD)).decode()\n"
)
new_decode = (
    "def _v150_cuda_source():\n"
    "    return _V150_CUDA_SOURCE\n"
)
if base.count(old_decode) != 1:
    raise RuntimeError("unexpected v150 CUDA decoder")
base = base.replace(old_decode, new_decode, 1)

for forbidden in (
    "_ALT_SRC", "_V150_CUDA_PAYLOAD", "b64decode", "decompress(",
    "exec(", "compile(", "_zsrc",
):
    if forbidden in base:
        raise RuntimeError(f"generated submission still contains {forbidden!r}")

if len(base.encode()) > 1_000_000:
    raise RuntimeError(f"generated submission exceeds 1 MB: {len(base.encode())}")

ast.parse(base)
compile(base, str(OUTPUT), "exec")
OUTPUT.write_text(base)
print(OUTPUT.name, len(base.encode()))
