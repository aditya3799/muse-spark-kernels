"""Generate v166: add the upstream-proven HALF RS128 path for O at m=128.

The protected v150 submission is retained byte-for-byte as the candidate prefix.
Its CUDA extension is rebuilt with the protected d2s5hzwP case plus the pinned
upstream h128s8e specialization.  Only O/m128 selects h128s8e (KS=1); the
three protected v150 routes keep d2s5hzwP unchanged.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import tempfile
import urllib.request
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_o_half_v166.py"
UPSTREAM_CACHE = Path(tempfile.gettempdir()) / "chenwainuo_3d418e1_submission.py"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/chenwainuo/okc/3d418e1/"
    "muse_bf16_proj/submission.py"
)
UPSTREAM_SHA256 = "BB50079BE2ED74D65BB7109F1432FBC7A49BBE70913D247C254717CD0F27F712"


def _upstream_text() -> str:
    if UPSTREAM_CACHE.exists():
        data = UPSTREAM_CACHE.read_bytes()
    else:
        with urllib.request.urlopen(UPSTREAM_URL, timeout=30) as response:
            data = response.read()
        UPSTREAM_CACHE.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest().upper()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError(f"upstream SHA256 mismatch: {digest}")
    return data.decode()


def _node_by_name(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    raise KeyError(name)


upstream = _upstream_text()
tree = ast.parse(upstream)
rqcu_node = _node_by_name(tree, "RQCU")
assert isinstance(rqcu_node, ast.Assign)
rqcu = ast.literal_eval(rqcu_node.value)

wg_namespace: dict[str, object] = {}
for name in ("_rqwg", "_rqss"):
    node = _node_by_name(tree, name)
    source = ast.get_source_segment(upstream, node)
    assert source is not None
    exec(compile(source, f"<upstream:{name}>", "exec"), wg_namespace)

cases = "\n".join(
    (
        "    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);",
        "    case 1: return rq_go<128, 8, 1, 0, 0, 0, 52, 1>(p, x, out, (int)KS, (int)info);",
    )
)
cuda_source = rqcu.replace(
    "@@WGMMA@@",
    wg_namespace["_rqwg"](16)
    + wg_namespace["_rqwg"](32)
    + wg_namespace["_rqwg"](128)
    + wg_namespace["_rqwg"](256)
    + wg_namespace["_rqss"](128)
    + wg_namespace["_rqss"](256),
).replace("@@CASES@@", cases)
cuda_source = cuda_source.replace(
    "mbarrier.arrive.relaxed.cta.shared::cta.b64",
    "mbarrier.arrive.shared::cta.b64",
)
cuda_payload = base64.b64encode(zlib.compress(cuda_source.encode(), 9)).decode()

appendix = f'''

# ---------------------------------------------------------------- v166: O/m128 HALF RS128 specialization.
# Apache-2.0 source: github.com/chenwainuo/okc @ 3d418e1.
_V150_CUDA_PAYLOAD = "{cuda_payload}"
_v150_module = [None]
_V150_TARGETS = ("qkv_gate", "o", "pooler", "unpooler")
_V150_KS = {{"qkv_gate": 7, "o": 1, "pooler": 4, "unpooler": 2}}
_V150_TOL = {{1: 5e-3, 2: 5e-3, 4: 6e-3, 7: 8e-3}}


def _v150_variant(plan):
    return 1 if plan["KS"] == 1 else 0


def _v150_call(state, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        _v150_variant(plan), plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v166 rqrun error %d" % rc)
    return out


def _v150_info(state, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        _v150_variant(plan), empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))
'''

candidate = BASE.read_text() + appendix
ast.parse(candidate)
OUT.write_text(candidate)
print(OUT.name, hashlib.sha256(OUT.read_bytes()).hexdigest().upper())
print("CUDA", hashlib.sha256(cuda_source.encode()).hexdigest().upper())
