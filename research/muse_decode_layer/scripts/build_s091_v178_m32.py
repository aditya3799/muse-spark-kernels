"""Build S091: use the scored MoE incumbent's exact M=32 module in the decoder.

The protected decoder source is copied verbatim.  The appendix embeds the exact
``v167_main`` CUDA source used by standalone MoE v178 and changes only stateful
``_moe(32, state)`` dispatch.  Every other token count falls through to the
decoder's existing implementation.
"""

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
OUT = ROOT / "submission_s091_v178_m32.py"


def subscript_assignment(path: Path, container: str, key: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
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
    source = subscript_assignment(MOE, "ALT_SRCS", "v167_main")

    # Compatibility guards: these are the exact ABI and algorithmic features
    # that let the decoder reuse its existing lossless SC6 weight copies and
    # MFIN-sized scratch buffers.
    required = (
        "#define OKC_MFIN 1",
        "#define OKC_SC6W 0",
        "constexpr int MF_CE = 0",
        "p.mfl = ib + 8 + 1024 + okc::EMAX;",
        "void okc_run(std::vector<torch::Tensor> w",
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"standalone source ABI marker missing: {marker}")
    for marker in ("def _ma_ext(", "def _moe(M=None, state=None):", "def prepare(weights):"):
        if marker not in base:
            raise RuntimeError(f"decoder integration marker missing: {marker}")

    decoder_source = None
    for node in ast.parse(base).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MOE_CUDA_SRC" for target in node.targets
        ):
            decoder_source = ast.literal_eval(node.value)
            break
    if decoder_source is None:
        raise RuntimeError("MOE_CUDA_SRC not found in decoder")

    # Store only a compact, exact line patch from the decoder's already-present
    # MoE source.  A second literal copy exceeds the challenge's 1 MB limit.
    a = decoder_source.splitlines(keepends=True)
    b = source.splitlines(keepends=True)
    groups = list(difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(4))
    hunks = []
    for group in groups:
        i1, i2 = group[0][1], group[-1][2]
        j1, j2 = group[0][3], group[-1][4]
        hunks.append(("".join(a[i1:i2]), "".join(b[j1:j2])))
    rebuilt = decoder_source
    for old, new in hunks:
        if rebuilt.count(old) != 1:
            raise RuntimeError(f"non-unique source patch hunk: {rebuilt.count(old)} matches")
        rebuilt = rebuilt.replace(old, new, 1)
    if rebuilt != source:
        raise RuntimeError("compact source patch did not reconstruct v178 exactly")
    patch_data = base64.b85encode(bz2.compress(json.dumps(hunks).encode(), 9)).decode()

    appendix = f'''


# ==============================================================================================
# S091: exact standalone-v178 M=32 MoE dispatch.
# v178 is the protected standalone MoE incumbent (job 8e29e87068e94f07, 5/5).
import base64 as _s091_b64
import bz2 as _s091_bz2
import json as _s091_json

_S091_SOURCE_PATCH = {patch_data!r}


def _s091_v178_source():
    src = MOE_CUDA_SRC
    hunks = _s091_json.loads(_s091_bz2.decompress(_s091_b64.b85decode(_S091_SOURCE_PATCH)))
    for old, new in hunks:
        if src.count(old) != 1:
            raise RuntimeError(f"S091 source patch expected one match, got {{src.count(old)}}")
        src = src.replace(old, new, 1)
    return src


_MA_ALT_SRCS["s091_v178_m32"] = _s091_v178_source()
_S091_M32_DEFS = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_SC6W=1", "@s091_v178_m32",
)

_s091_parent_moe = _moe


def _moe(M=None, state=None):
    if state is not None and M == 32:
        ext = state.get("s091_v178_m32")
        if ext is not None:
            return ext
    return _s091_parent_moe(M, state)


_s091_parent_prepare = prepare


def prepare(weights):
    # Compile concurrently with the decoder's existing prepare pipeline.  The
    # fixed flags are safe only for the exact PRMT/SC6 scale representation;
    # dispatch remains on the protected implementation if that property or the
    # build is unavailable.
    box = {{}}

    def build():
        try:
            ext = _RouterExt(_ma_ext(_S091_M32_DEFS))
            if MOE_HAS_PDL:
                ext.okc_set_pdl(0)
            box["ext"] = ext
        except Exception as ex:
            box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)

    th = threading.Thread(target=build, daemon=True)
    th.start()
    st = _s091_parent_prepare(weights)
    th.join(timeout=25.0)
    if not th.is_alive() and "ext" in box and bool(st.get("ma_sfx")) and bool(globals().get("_MA_SC6W", False)):
        st["s091_v178_m32"] = box["ext"]
        print("[s091] exact v178 M32 dispatch enabled", flush=True)
    else:
        why = "compile still running" if th.is_alive() else box.get("error", "PRMT/SC6 guard false")
        print(f"[s091] protected M32 fallback: {{why}}", flush=True)
    return st
'''

    candidate_bytes = base_bytes + appendix.encode()
    OUT.write_bytes(candidate_bytes)
    print(f"wrote {OUT.name} ({len(candidate_bytes)} bytes)")
    print(f"base sha256 {hashlib.sha256(base_bytes).hexdigest()}")
    print(f"candidate sha256 {hashlib.sha256(candidate_bytes).hexdigest()}")


if __name__ == "__main__":
    main()
