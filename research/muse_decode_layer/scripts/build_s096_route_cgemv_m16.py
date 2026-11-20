"""Bundle S094 route fusion, S073 exact-C12 GEMV, and exact v178 M=16 MoE."""

from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import lzma
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
ROUTE_BUILDER = ROOT / "scripts" / "build_s094_route_finalizer.py"
CGEMV_SOURCE = ROOT / "submission_s073_cgemv_m1.py"
MOE_SOURCE = ROOT.parent / "muse_moe_w4a16" / "submission_v178_hybrid_m32_retry.py"
OUT = ROOT / "submission_s096_route_cgemv_m16.py"

BASE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"
ROUTE_BUILDER_SHA256 = "986973473663074a225367cbdb0900eeac194f04eb642f273d224da544900811"
CGEMV_SOURCE_SHA256 = "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161"
MOE_SOURCE_SHA256 = "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94"
CGEMV_MARKER = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _subscript_assignment(path: Path, container: str, key: str) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        if not isinstance(target.value, ast.Name) or target.value.id != container:
            continue
        try:
            target_key = ast.literal_eval(target.slice)
        except Exception:
            continue
        if target_key == key:
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(f"{container}[{key!r}] is not a string")
            return value
    raise RuntimeError(f"{container}[{key!r}] not found in {path}")


def _route_appendix() -> str:
    tree = ast.parse(ROUTE_BUILDER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "APPENDIX" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError("S094 APPENDIX assignment missing")


def _cgemv_appendix() -> str:
    source = CGEMV_SOURCE.read_text(encoding="utf-8")
    marker_pos = source.index(CGEMV_MARKER)
    appendix_pos = source.rfind("# ---------------------------------------------------------------------------", 0, marker_pos)
    if appendix_pos < 0:
        raise RuntimeError("S073 appendix boundary missing")
    return source[appendix_pos:].rstrip() + "\n"


def _m16_appendix() -> str:
    resident = _subscript_assignment(BASE, "_MA_ALT_SRCS", "moe_helper0")
    target = _subscript_assignment(MOE_SOURCE, "ALT_SRCS", "v167_helper0")
    for marker in ("#define OKC_WSH0 0", "#define OKC_DF 1", "void okc_run(std::vector<torch::Tensor> w"):
        if marker not in target:
            raise RuntimeError(f"target ABI marker missing: {marker}")

    a = resident.splitlines(keepends=True)
    b = target.splitlines(keepends=True)
    # Three context lines are the smallest setting whose every hunk is unique in
    # the resident source; the reconstruction guard below proves byte identity.
    groups = difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(3)
    hunks = []
    for group in groups:
        i1, i2 = group[0][1], group[-1][2]
        j1, j2 = group[0][3], group[-1][4]
        hunks.append(("".join(a[i1:i2]), "".join(b[j1:j2])))
    rebuilt = resident
    for old, new in hunks:
        if rebuilt.count(old) != 1:
            raise RuntimeError(f"non-unique source patch hunk: {rebuilt.count(old)} matches")
        rebuilt = rebuilt.replace(old, new, 1)
    if rebuilt != target:
        raise RuntimeError("compact source patch did not reconstruct v178 helper0 exactly")

    # Keep the patch raw here. The whole appendix receives one outer LZMA pass,
    # which is materially smaller than nesting an already encoded patch payload.
    return f'''

# ==============================================================================================
# S096: exact standalone-v178 M=16 MoE dispatch.
_S096_M16_HUNKS = {hunks!r}


def _s096_v178_source():
    src = _MA_ALT_SRCS["moe_helper0"]
    for old, new in _S096_M16_HUNKS:
        if src.count(old) != 1:
            raise RuntimeError(f"S096 source patch expected one match, got {{src.count(old)}}")
        src = src.replace(old, new, 1)
    return src


_MA_ALT_SRCS["s096_v178_m16"] = _s096_v178_source()
_S096_M16_DEFS = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
    "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
    "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
    "-DOKC_FINPF=1", "-DOKC_WSH0=1", "-DOKC_PRMTSC=1", "@s096_v178_m16",
)

_s096_parent_moe = _moe


def _moe(M=None, state=None):
    if state is not None and M == 16:
        ext = state.get("s096_v178_m16")
        if ext is not None:
            return ext
    return _s096_parent_moe(M, state)


_s096_parent_prepare = prepare


def prepare(weights):
    box = {{}}

    def build():
        try:
            ext = _RouterExt(_ma_ext(_S096_M16_DEFS))
            if MOE_HAS_PDL:
                ext.okc_set_pdl(0)
            box["ext"] = ext
        except Exception as ex:
            box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)

    th = threading.Thread(target=build, daemon=True)
    th.start()
    st = _s096_parent_prepare(weights)
    th.join(timeout=25.0)
    if not th.is_alive() and "ext" in box and bool(st.get("ma_sfx")):
        st["s096_v178_m16"] = box["ext"]
        print("[s096] exact v178 M16 dispatch enabled", flush=True)
    else:
        why = "compile still running" if th.is_alive() else box.get("error", "PRMT guard false")
        print(f"[s096] protected M16 fallback: {{why}}", flush=True)
    return st
'''


def main() -> None:
    for path, expected in (
        (BASE, BASE_SHA256),
        (ROUTE_BUILDER, ROUTE_BUILDER_SHA256),
        (CGEMV_SOURCE, CGEMV_SOURCE_SHA256),
        (MOE_SOURCE, MOE_SOURCE_SHA256),
    ):
        got = _sha(path)
        if got != expected:
            raise RuntimeError(f"protected source hash mismatch for {path.name}: {got}")

    appendix = _cgemv_appendix() + "\n" + _route_appendix() + "\n" + _m16_appendix()
    payload = lzma.compress(appendix.encode(), preset=9)
    encoded = base64.b85encode(payload).decode("ascii")
    stub = (
        "\nimport base64,lzma\n"
        f"exec(lzma.decompress(base64.b85decode({encoded!r})))\n"
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
