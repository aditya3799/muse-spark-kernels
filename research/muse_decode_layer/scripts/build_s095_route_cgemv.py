"""Bundle S094's T=1 route finalizer with S073's exact-C12 GEMV graft."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
ROUTE_BUILDER = ROOT / "scripts" / "build_s094_route_finalizer.py"
CGEMV_SOURCE = ROOT / "submission_s073_cgemv_m1.py"
OUT = ROOT / "submission_s095_route_cgemv.py"

BASE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"
ROUTE_BUILDER_SHA256 = "986973473663074a225367cbdb0900eeac194f04eb642f273d224da544900811"
CGEMV_SOURCE_SHA256 = "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161"
CGEMV_MARKER = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_appendix() -> str:
    tree = ast.parse(ROUTE_BUILDER.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "APPENDIX" for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                break
            return value
    raise RuntimeError("S094 APPENDIX assignment missing")


def _cgemv_appendix() -> str:
    source = CGEMV_SOURCE.read_text(encoding="utf-8")
    marker_pos = source.index(CGEMV_MARKER)
    appendix_pos = source.rfind("# ---------------------------------------------------------------------------", 0, marker_pos)
    if appendix_pos < 0:
        raise RuntimeError("S073 appendix boundary missing")
    return source[appendix_pos:].rstrip() + "\n"


def main() -> None:
    guards = (
        (BASE, BASE_SHA256),
        (ROUTE_BUILDER, ROUTE_BUILDER_SHA256),
        (CGEMV_SOURCE, CGEMV_SOURCE_SHA256),
    )
    for path, expected in guards:
        got = _sha(path)
        if got != expected:
            raise RuntimeError(f"protected source hash mismatch for {path.name}: {got}")

    # Install the projection graft first. S094 only replaces the T=1 post-attention
    # route path, so the two appendices operate on disjoint dispatch points.
    appendix = _cgemv_appendix() + "\n" + _route_appendix()
    payload = zlib.compress(appendix.encode(), 9)
    encoded = base64.b85encode(payload).decode("ascii")
    stub = (
        "\n\n# S095 compressed C12 GEMV + route-finalizer bundle.\n"
        "import base64 as _s095_b,zlib as _s095_z\n"
        f"exec(compile(_s095_z.decompress(_s095_b.b85decode({encoded!r})).decode(), '<s095_route_cgemv>', 'exec'))\n"
    ).encode()
    candidate = BASE.read_bytes() + stub
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
