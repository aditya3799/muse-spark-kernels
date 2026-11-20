"""Graft the corrected BF16 v177 K128 GQK routes into protected S081.

Every m=128/256 route is validated and raced against the decoder's actual
RS128/cuBLASLt path during prepare. Only a >=3% paired cold-L2 winner enables.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
V177 = ROOT.parent / "muse_bf16_proj" / "submission_gqx_k128_v177.py"
OUT = ROOT / "submission_s097_gqk_probe.py"

BASE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"
V177_SHA256 = "aaa46a08611f70c2e8a574b136b27b2bdfed397a03927a3fea05c17438affd3a"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"missing assignment {name} in {path.name}")


RUNTIME = r'''

# ==============================================================================================
# S097: self-gated v177 raw-BF16 K128 GQK routes for m=128/256.
_S097_EXT = None
_S097_CFG = {
    ("qkv_gate", 128): (1, 3, 4), ("qkv_gate", 256): (1, 3, 2),
    ("pooler", 128): (1, 2, 4), ("pooler", 256): (1, 2, 2),
    ("unpooler", 128): (1, 1, 4), ("unpooler", 256): (1, 1, 2),
    ("o", 128): (1, 1, 4), ("o", 256): (1, 1, 2),
}
_S097_NAME = {"qkv": "qkv_gate", "pool": "pooler", "un": "unpooler", "o": "o"}


def _s097_cuda_source():
    common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS)
    defs = bfp_wgmma_defs([8, 16, 32, 64, 128]) + bfp_wgmma_ss_defs([128, 256])
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(_S097_CFG.items()):
        line = (f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}, 2>(W, x, out, M, N, K); return; }}")
        if line not in lines:
            lines.append(line)
    gqk = (_S097_GQX_SRC.replace("@@GQX_DISPATCH@@", "")
           .replace("@@GQK_DISPATCH@@", "\n  ".join(lines)))
    return common + defs + bfp_MAP_SRC + gqk


def _s097_ext():
    global _S097_EXT
    if _S097_EXT is None:
        source = _s097_cuda_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        cpp = ("void gqk(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,"
               "int64_t,int64_t,int64_t,int64_t);\n")
        _S097_EXT = load_inline(
            "s097_gqk_" + tag, "#include <torch/extension.h>\n" + cpp,
            cuda_sources=source, functions=["gqk"], no_implicit_headers=True,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _S097_EXT


def _s097_run(st, name, x, m):
    layer = _S097_NAME[name]
    w = st["bfp"]["raw"][layer]
    n, k = w.shape
    nwg, cl, ns = _S097_CFG[(layer, m)]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    st["s097_ext"].gqk(w, x.contiguous(), out, m, n, k, m, nwg, cl, ns)
    return out


def _s097_base(st, name, x, m):
    out = _s097_parent_rq(st, name, x, m)
    if out is not None:
        return out
    plan = st.get("graft_lt", {}).get((name, m))
    if plan is None:
        return None
    return _graft_lt_run(x, plan)


def _s097_validate(cand, ref):
    cf, rf = cand.float(), ref.float()
    if not bool(torch.isfinite(cf).all()):
        return False, float("inf"), float("inf")
    d = cf - rf
    rel = float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(rf).clamp_min(1e-20))
    mx = float(d.abs().max())
    return rel <= 0.01, rel, mx


_s097_parent_rq = _rq_gemm


def _rq_gemm(state, name, x, T):
    if (name, T) in state.get("s097_use", ()):
        return _s097_run(state, name, x, T)
    return _s097_parent_rq(state, name, x, T)


_s097_parent_prepare = prepare


def prepare(weights):
    box = {}
    def build():
        try: box["ext"] = _s097_ext()
        except Exception as ex: box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)
    th = threading.Thread(target=build, daemon=True)
    th.start()
    st = _s097_parent_prepare(weights)
    th.join(timeout=45.0)
    st["s097_use"] = set()
    if th.is_alive() or "ext" not in box or st.get("bfp") is None:
        why = "compile still running" if th.is_alive() else box.get("error", "bfp unavailable")
        print(f"[s097] protected GQK fallback: {why}", flush=True)
        return st
    st["s097_ext"] = box["ext"]
    gen = torch.Generator(device=weights["o"].device).manual_seed(977177)
    for m in (128, 256):
        for name in ("qkv", "o", "pool", "un"):
            layer = _S097_NAME[name]
            if (layer, m) not in _S097_CFG:
                continue
            k = st["bfp"]["raw"][layer].shape[1]
            x = torch.randn((m, k), generator=gen, device=weights["o"].device).to(torch.bfloat16)
            base = lambda st=st, name=name, x=x, m=m: _s097_base(st, name, x, m)
            cand = lambda st=st, name=name, x=x, m=m: _s097_run(st, name, x, m)
            try:
                ro, co = base(), cand()
                if ro is None:
                    print(f"[s097] {name} m={m} no incumbent route", flush=True)
                    continue
                ok, rel, mx = _s097_validate(co, ro)
                if not ok:
                    print(f"[s097] {name} m={m} invalid rel={rel:.4g} max={mx:.4g}", flush=True)
                    continue
                tb, tc = _compose_times([base, cand], reps=12)
                ratio = tc / tb
                print(f"[s097] {name} m={m} base={tb:.3f}us gqk={tc:.3f}us ratio={ratio:.4f} rel={rel:.4g}", flush=True)
                if ratio <= 0.97:
                    st["s097_use"].add((name, m))
            except Exception as ex:
                print(f"[s097] {name} m={m} rejected: {str(ex).splitlines()[-1][:180]}", flush=True)
    print(f"[s097] enabled {sorted(st['s097_use'])}", flush=True)
    torch.cuda.synchronize()
    return st
'''


def main() -> None:
    for path, expected in ((BASE, BASE_SHA256), (V177, V177_SHA256)):
        got = _sha(path)
        if got != expected:
            raise RuntimeError(f"protected source hash mismatch for {path.name}: {got}")
    source = _assignment(V177, "GQX_SRC")
    configs = _assignment(V177, "GQK_CONFIGS")
    expected_configs = {
        ("qkv_gate", 128): (1, 3, 4), ("qkv_gate", 256): (1, 3, 2),
        ("pooler", 128): (1, 2, 4), ("pooler", 256): (1, 2, 2),
        ("unpooler", 128): (1, 1, 4), ("unpooler", 256): (1, 1, 2),
        ("o", 128): (1, 1, 4), ("o", 256): (1, 1, 2),
    }
    if configs != expected_configs:
        raise RuntimeError(f"v177 GQK configs changed: {configs!r}")
    appendix = "\n_S097_GQX_SRC=" + repr(source) + "\n" + RUNTIME
    payload = zlib.compress(appendix.encode(), 9)
    encoded = base64.b85encode(payload).decode("ascii")
    stub = ("\nimport base64 as _s097_b,zlib as _s097_z\n"
            f"exec(_s097_z.decompress(_s097_b.b85decode({encoded!r})))\n").encode()
    candidate = BASE.read_bytes() + stub
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
