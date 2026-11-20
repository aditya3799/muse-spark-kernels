"""Build the evidence-backed S099 bundle for the 2.3x decoder target.

The protected S081 decoder is unchanged.  The compressed appendix adds:

* exact-C12 T=1 QKV/sliced-unpooler routes from S073;
* the exact standalone-v178 M=16 MoE module, used on sliding layers only;
* the exact standalone-v178 M=32 MoE module, used on full layers only; and
* the five small-M SK2 geometries measured as winners by bf16_proj v174.

The layer-aware MoE split is important: prior decoder profiles showed v178
winning at M=16 sliding and M=32 full, but losing badly at M=32 sliding.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import lzma
from pathlib import Path

try:
    from . import build_s098_c12_gqk_m16 as s098
except ImportError:
    import build_s098_c12_gqk_m16 as s098


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
MOE = ROOT.parent / "muse_moe_w4a16" / "submission_v178_hybrid_m32_retry.py"
V174 = ROOT.parent / "muse_bf16_proj" / "submission_clean_composite_v174.py"
OUT = ROOT / "submission_s099_tuned_bundle.py"

EXPECTED = {
    BASE: "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19",
    s098.CGEMV: "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161",
    MOE: "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _offset_patch(resident: str, target: str) -> list[tuple[int, int, str]]:
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
        raise RuntimeError("offset patch did not reconstruct the protected source")
    return patches


def _m32() -> str:
    resident = s098.gqk_builder._assignment(BASE, "MOE_CUDA_SRC")
    target = s098._assignment(MOE, "ALT_SRCS", "v167_main")
    patches = _offset_patch(resident, target)
    return f'''

# S099: exact standalone-v178 M=32 main module, reconstructed from protected offsets.
_S099_M32_PATCH = {patches!r}

def _s099_v178_m32_source():
    src = MOE_CUDA_SRC
    for lo, hi, new in reversed(_S099_M32_PATCH):
        src = src[:lo] + new + src[hi:]
    return src

_MA_ALT_SRCS["s099_v178_m32"] = _s099_v178_m32_source()
_S099_M32_DEFS = (
    "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
    "-DOKC_PRMTSC=1", "-DOKC_SC6W=1", "@s099_v178_m32",
)

_s099_parent_prepare = prepare
def prepare(weights):
    box = {{}}
    def build():
        try:
            ext = _RouterExt(_ma_ext(_S099_M32_DEFS))
            if MOE_HAS_PDL: ext.okc_set_pdl(0)
            box["ext"] = ext
        except Exception as ex:
            box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)
    th = threading.Thread(target=build, daemon=True); th.start()
    st = _s099_parent_prepare(weights); th.join(timeout=25.0)
    if (not th.is_alive() and "ext" in box and bool(st.get("ma_sfx"))
            and bool(globals().get("_MA_SC6W", False))):
        st["s099_v178_m32"] = box["ext"]
        print("[s099] exact v178 M32/full dispatch enabled", flush=True)
    else:
        why = "compile still running" if th.is_alive() else box.get("error", "PRMT/SC6 guard false")
        print(f"[s099] protected M32 fallback: {{why}}", flush=True)
    return st
'''


SMALL_ROUTES = r'''

# S099: standalone-v174 small-M winners.  These are geometry-only additions to
# the resident lossless SK2 source; packing, arithmetic and output types stay exact.
_S099_SK2_CONFIGS = [
    ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),
    ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),
    ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),
    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),
]
for _cfg in _S099_SK2_CONFIGS:
    if _cfg not in bfp_SK2_CONFIGS: bfp_SK2_CONFIGS.append(_cfg)
bfp_SK2_PICK.update({
    ('pooler', 16): 'pooler16t_r192',
    ('unpooler', 16): 'unpooler16t_r72',
    ('qkv_gate', 8): 'qkv8t_r144',
    ('qkv_gate', 16): 'qkv16t_r200',
    ('o', 16): 'o16t_r68',
})

# Shape/config-only selection is captured into each CUDA graph.  M16 v178 won
# on sliding and M32 v178 won on full; all other combinations use S081.
_S099_SLIDING = False
_s099_parent_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    global _S099_SLIDING
    cfg = data.get("config")
    if cfg is not None:
        _S099_SLIDING = bool(cfg["window"])
    return _s099_parent_after(data, state, o_in, gate, T)

_s099_base_moe = _s098_parent_moe
def _moe(M=None, state=None):
    if state is not None:
        if M == 16 and _S099_SLIDING:
            ext = state.get("s098_v178_m16")
            if ext is not None: return ext
        if M == 32 and not _S099_SLIDING:
            ext = state.get("s099_v178_m32")
            if ext is not None: return ext
    return _s099_base_moe(M, state)
'''


def main() -> None:
    for path, expected in EXPECTED.items():
        got = _sha(path)
        if got != expected:
            raise RuntimeError(f"protected source hash mismatch for {path.name}: {got}")

    # Guard the exact donor geometry table as well as the file's syntax.
    donor = V174.read_text(encoding="utf-8")
    for marker in (
        "('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1)",
        "('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1)",
        "('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1)",
        "('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1)",
        "('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1)",
    ):
        if marker not in donor:
            raise RuntimeError(f"v174 geometry marker missing: {marker}")

    appendix = s098._c12() + "\n" + s098._m16() + "\n" + _m32() + SMALL_ROUTES
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
