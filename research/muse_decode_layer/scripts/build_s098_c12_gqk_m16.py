"""Bundle exact-C12, self-gated v177 GQK, and exact v178 M=16 MoE."""

from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import lzma
from pathlib import Path

try:
    from . import build_s097_gqk_probe as gqk_builder
except ImportError:
    import build_s097_gqk_probe as gqk_builder


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
CGEMV = ROOT / "submission_s073_cgemv_m1.py"
MOE = ROOT.parent / "muse_moe_w4a16" / "submission_v178_hybrid_m32_retry.py"
OUT = ROOT / "submission_s098_c12_gqk_m16.py"

EXPECTED = {
    BASE: "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19",
    CGEMV: "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161",
    MOE: "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94",
    gqk_builder.V177: "aaa46a08611f70c2e8a574b136b27b2bdfed397a03927a3fea05c17438affd3a",
    Path(gqk_builder.__file__): "18704a7f53ef4caa7f5582f67000db90ee413c153dd465f01d612d53f0f6fc91",
}
CGEMV_MARKER = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment(path: Path, container: str, key: str) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            continue
        if target.value.id == container and ast.literal_eval(target.slice) == key:
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"{container}[{key!r}] missing from {path.name}")


def _c12() -> str:
    source = CGEMV.read_text(encoding="utf-8")
    pos = source.index(CGEMV_MARKER)
    start = source.rfind("# ---------------------------------------------------------------------------", 0, pos)
    if start < 0:
        raise RuntimeError("S073 appendix boundary missing")
    return source[start:].rstrip() + "\n"


def _m16() -> str:
    resident = _assignment(BASE, "_MA_ALT_SRCS", "moe_helper0")
    target = _assignment(MOE, "ALT_SRCS", "v167_helper0")
    a = resident.splitlines(keepends=True)
    b = target.splitlines(keepends=True)
    offsets = [0]
    for line in a:
        offsets.append(offsets[-1] + len(line))
    patches = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            patches.append((offsets[i1], offsets[i2], "".join(b[j1:j2])))
    rebuilt = resident
    for lo, hi, new in reversed(patches):
        rebuilt = rebuilt[:lo] + new + rebuilt[hi:]
    if rebuilt != target:
        raise RuntimeError("offset patch did not reconstruct v178 helper0 exactly")
    return f'''

# ==============================================================================================
# S098: exact v178 M=16 MoE dispatch, reconstructed by protected-source offsets.
_S098_M16_PATCH = {patches!r}

def _s098_v178_source():
    src = _MA_ALT_SRCS["moe_helper0"]
    for lo, hi, new in reversed(_S098_M16_PATCH):
        src = src[:lo] + new + src[hi:]
    return src

_MA_ALT_SRCS["s098_v178_m16"] = _s098_v178_source()
_S098_M16_DEFS = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
    "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
    "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
    "-DOKC_FINPF=1", "-DOKC_WSH0=1", "-DOKC_PRMTSC=1", "@s098_v178_m16",
)
_s098_parent_moe = _moe
def _moe(M=None, state=None):
    if state is not None and M == 16 and state.get("s098_v178_m16") is not None:
        return state["s098_v178_m16"]
    return _s098_parent_moe(M, state)

_s098_parent_prepare = prepare
def prepare(weights):
    box = {{}}
    def build():
        try:
            ext = _RouterExt(_ma_ext(_S098_M16_DEFS))
            if MOE_HAS_PDL: ext.okc_set_pdl(0)
            box["ext"] = ext
        except Exception as ex:
            box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)
    th = threading.Thread(target=build, daemon=True); th.start()
    st = _s098_parent_prepare(weights); th.join(timeout=25.0)
    if not th.is_alive() and "ext" in box and bool(st.get("ma_sfx")):
        st["s098_v178_m16"] = box["ext"]
        print("[s098] exact v178 M16 dispatch enabled", flush=True)
    else:
        why = "compile still running" if th.is_alive() else box.get("error", "PRMT guard false")
        print(f"[s098] protected M16 fallback: {{why}}", flush=True)
    return st
'''


def main() -> None:
    for path, expected in EXPECTED.items():
        got = _sha(path)
        if got != expected:
            raise RuntimeError(f"protected source hash mismatch for {path.name}: {got}")
    source = gqk_builder._assignment(gqk_builder.V177, "GQX_SRC")
    gqk = "\n_S097_GQX_SRC=" + repr(source) + "\n" + gqk_builder.RUNTIME
    appendix = _c12() + "\n" + gqk + "\n" + _m16()
    payload = lzma.compress(appendix.encode(), preset=9)
    encoded = base64.b85encode(payload).decode("ascii")
    stub = ("\nimport base64,lzma\n"
            f"exec(lzma.decompress(base64.b85decode({encoded!r})))\n").encode()
    candidate = BASE.read_bytes() + stub
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
