"""Build S092: exact standalone-v178 M=16 MoE dispatch only."""

from __future__ import annotations

import ast
import base64
import bz2
import difflib
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
MOE = ROOT.parent / "muse_moe_w4a16" / "submission_v178_hybrid_m32_retry.py"
OUT = ROOT / "submission_s092_v178_m16.py"


def subscript_assignment(path: Path, container: str, key: str) -> str:
    for node in ast.parse(path.read_text()).body:
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


def main() -> None:
    base_bytes = BASE.read_bytes()
    base = base_bytes.decode()
    resident = subscript_assignment(BASE, "_MA_ALT_SRCS", "moe_helper0")
    target = subscript_assignment(MOE, "ALT_SRCS", "v167_helper0")

    for marker in (
        "#define OKC_WSH0 0",
        "#define OKC_DF 1",
        "void okc_run(std::vector<torch::Tensor> w",
    ):
        if marker not in target:
            raise RuntimeError(f"target ABI marker missing: {marker}")

    a = resident.splitlines(keepends=True)
    b = target.splitlines(keepends=True)
    groups = list(difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(4))
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
    patch_data = base64.b85encode(bz2.compress(json.dumps(hunks).encode(), 9)).decode()

    appendix = f'''


# ==============================================================================================
# S092: exact standalone-v178 M=16 MoE dispatch only.
import base64 as _s092_b64
import bz2 as _s092_bz2
import json as _s092_json

_S092_SOURCE_PATCH = {patch_data!r}


def _s092_v178_source():
    src = _MA_ALT_SRCS["moe_helper0"]
    hunks = _s092_json.loads(_s092_bz2.decompress(_s092_b64.b85decode(_S092_SOURCE_PATCH)))
    for old, new in hunks:
        if src.count(old) != 1:
            raise RuntimeError(f"S092 source patch expected one match, got {{src.count(old)}}")
        src = src.replace(old, new, 1)
    return src


_MA_ALT_SRCS["s092_v178_m16"] = _s092_v178_source()
_S092_M16_DEFS = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
    "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
    "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
    "-DOKC_FINPF=1", "-DOKC_WSH0=1", "-DOKC_PRMTSC=1", "@s092_v178_m16",
)

_s092_parent_moe = _moe


def _moe(M=None, state=None):
    if state is not None and M == 16:
        ext = state.get("s092_v178_m16")
        if ext is not None:
            return ext
    return _s092_parent_moe(M, state)


_s092_parent_prepare = prepare


def prepare(weights):
    box = {{}}

    def build():
        try:
            ext = _RouterExt(_ma_ext(_S092_M16_DEFS))
            if MOE_HAS_PDL:
                ext.okc_set_pdl(0)
            box["ext"] = ext
        except Exception as ex:
            box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)

    th = threading.Thread(target=build, daemon=True)
    th.start()
    st = _s092_parent_prepare(weights)
    th.join(timeout=25.0)
    if not th.is_alive() and "ext" in box and bool(st.get("ma_sfx")):
        st["s092_v178_m16"] = box["ext"]
        print("[s092] exact v178 M16 dispatch enabled", flush=True)
    else:
        why = "compile still running" if th.is_alive() else box.get("error", "PRMT guard false")
        print(f"[s092] protected M16 fallback: {{why}}", flush=True)
    return st
'''

    candidate_bytes = base_bytes + appendix.encode()
    if len(candidate_bytes) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate_bytes)} bytes")
    OUT.write_bytes(candidate_bytes)
    compile(candidate_bytes.decode(), str(OUT), "exec")
    print(f"wrote {OUT.name} ({len(candidate_bytes)} bytes; {1_000_000 - len(candidate_bytes)} headroom)")
    print(f"base sha256 {hashlib.sha256(base_bytes).hexdigest()}")
    print(f"candidate sha256 {hashlib.sha256(candidate_bytes).hexdigest()}")


if __name__ == "__main__":
    main()
