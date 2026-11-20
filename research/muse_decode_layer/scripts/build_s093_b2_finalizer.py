"""Build S093: T=1 sliced-unpooler last-arrival boundary-2 finalizer.

The custom sk2 instantiation keeps the incumbent unpooler math and stores.  A
release/acquire completion counter elects the last producer CTA, which performs
the exact sliced boundary work.  No CTA spins or waits for another CTA.
"""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
OUT = ROOT / "submission_s093_b2_finalizer.py"
BASE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"


APPENDIX = r"""

# ==============================================================================================
# S093: T=1 sliced-unpooler last-arrival boundary-2 finalizer (no grid spin/barrier).
_S093_B2_EXT = None


def _s093_b2_source():
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "slice8_r192")
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    assert (layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL) == ("unpooler", 1024, 8, 1, 1, 8, 2, 2, 1)
    _, nstage = bfp__sk2_geom(cfg, 8)
    assert nstage == 4
    s = bfp_SK2_SRC
    epi_struct = r'''
struct B2Epi {
  const float* carry; const float* tail; const float* beta;
  const __nv_bfloat16* post; const __nv_bfloat16* next;
  float* sout; __nv_bfloat16* hout; int* cnt; int on;
};
static B2Epi g_b2;
'''
    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\n" + epi_struct, 1)
    s = s.replace(
        "           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",
        "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const B2Epi ep) {", 1)

    # Only the consumer warps reach this point.  named_bar(1, NC) is therefore
    # the existing consumer-only barrier; using __syncthreads here would be
    # invalid because the producer and escape warps have already returned.
    finalizer = r'''
  if (ep.on) {
    static_assert(MT == 1 && TOKR == 8 && R == 8 && NW == 8 && CL == 1 && sizeof(OutT) == 2,
                  "S093 finalizer is only for the sliced T<=8 unpooler");
    __shared__ int b2_last;
    named_bar(1, NC);
    if (tid == 0) {
      __threadfence();
      const int old = atomicAdd(ep.cnt, 1);
      b2_last = old == (int)gridDim.x - 1;
      if (b2_last) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (b2_last) {
      // Pass 1: post-FFN norm of the BF16-rounded 1024-column local branch.
      float q = 0.f;
      for (int i = tid; i < 1024; i += NC) {
        const float p = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(out)[i]);
        const float gp = __bfloat162float(ep.post[i]);
        const float f = __bfloat162float(__float2bfloat16_rn(__fmul_rn(p, gp)));
        q = __fmaf_rn(f, f, q);
      }
      for (int d = 16; d; d >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, d));
      if (lane == 0) red[warp] = q;
      named_bar(1, NC);
      if (warp == 0) {
        q = lane < NW ? red[lane] : 0.f;
        for (int d = 16; d; d >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, d));
        if (lane == 0) red[0] = q;
      }
      named_bar(1, NC);
      float r1;
      { float z = __fadd_rn(__fmul_rn(red[0], 1.f / 8192.f), 1.e-8f);
        asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r1) : "f"(z)); }

      // Pass 2: first 1024 residual columns and their contribution to the
      // next pre-attention norm.  The carry tail already covers 1024..8191.
      q = 0.f;
      for (int i = tid; i < 1024; i += NC) {
        const float p = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(out)[i]);
        const float gp = __bfloat162float(ep.post[i]);
        const float f = __bfloat162float(__float2bfloat16_rn(__fmul_rn(p, gp)));
        const float n = __bfloat162float(__float2bfloat16_rn(__fmul_rn(f, r1)));
        const float h = __fmaf_rn(ep.beta[i], n, ep.carry[i]);
        ep.sout[i] = h;
        q = __fmaf_rn(h, h, q);
      }
      for (int d = 16; d; d >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, d));
      if (lane == 0) red[warp] = q;
      named_bar(1, NC);
      if (warp == 0) {
        q = lane < NW ? red[lane] : 0.f;
        for (int d = 16; d; d >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, d));
        if (lane == 0) red[0] = q;
      }
      named_bar(1, NC);
      float r2;
      { float z = __fadd_rn(__fmul_rn(__fadd_rn(ep.tail[0], red[0]), 1.f / 8192.f), 1.e-5f);
        asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r2) : "f"(z)); }

      // Final residual stream and normalized BF16 output.  This is deliberately
      // low-register, sequential work in the elected CTA, not a persistent grid.
      for (int i = tid; i < 8192; i += NC) {
        const float h = i < 1024 ? ep.sout[i] : ep.carry[i];
        if (i >= 1024) ep.sout[i] = h;
        const float gn = __bfloat162float(ep.next[i]);
        ep.hout[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(h, r2), gn));
      }
      named_bar(1, NC);
      if (tid == 0) {
        __threadfence();
        atomicExch(ep.cnt, 0);
      }
    }
    named_bar(1, NC);
  }
'''
    anchor = ('  if (trace && tid == 0) tr[15] = gtime();\n'
              '  if ((dgv & 32) && dbg != nullptr && tid == 0) {')
    assert s.count(anchor) == 1
    s = s.replace(anchor, finalizer + "\n" + anchor, 1)
    s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_b2));", 1)
    s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_b2);", 1)
    dispatch = (f"if (tokr == 8 && M == 1 && N == 1024 && K == 4096) {{ "
                f"launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, {NACC}, 1, __nv_bfloat16>"
                f"(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr); return; }}")
    s = s.replace("@@SK2_DISPATCH@@", dispatch).replace("@@SK2F_DISPATCH@@", "")
    s += r'''
void b2_un(at::Tensor Ws, at::Tensor x, at::Tensor branch, at::Tensor ptab, int64_t pmax, int64_t base4,
           at::Tensor carry, at::Tensor tail, at::Tensor beta, at::Tensor post, at::Tensor next,
           at::Tensor sout, at::Tensor hout, at::Tensor cnt) {
  TORCH_CHECK(x.size(0) == 1 && x.size(1) == 4096 && branch.size(0) == 1 && branch.size(1) == 1024,
              "S093 b2_un shapes");
  TORCH_CHECK(carry.size(0) == 1 && carry.size(1) == 8192 && sout.sizes() == carry.sizes() &&
              hout.size(0) == 1 && hout.size(1) == 8192 && cnt.numel() == 1, "S093 b2 tensors");
  g_b2 = B2Epi{carry.data_ptr<float>(), tail.data_ptr<float>(), beta.data_ptr<float>(),
               reinterpret_cast<const __nv_bfloat16*>(post.data_ptr()),
               reinterpret_cast<const __nv_bfloat16*>(next.data_ptr()), sout.data_ptr<float>(),
               reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), cnt.data_ptr<int>(), 1};
  sk2(Ws, x, branch, ptab, pmax, base4, 1, 1024, 4096, 0, 8, 0, ptab);
  g_b2 = B2Epi{};
}
'''
    shared = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "")
    return shared + bfp_MAP_SRC + s


def _s093_b2_ext():
    global _S093_B2_EXT
    if _S093_B2_EXT is None:
        source = _s093_b2_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace("sk2_kernel", "sk2_kernel_s093" + tag)
                  .replace("launch_sk2", "launch_sk2_s093" + tag)
                  .replace("void sk2(", "void sk2_s093" + tag + "(")
                  .replace("  sk2(Ws,", "  sk2_s093" + tag + "(Ws,")
                  .replace("void sk2f(", "void sk2f_s093" + tag + "("))
        cpp = ("void b2_un(torch::Tensor Ws, torch::Tensor x, torch::Tensor branch, torch::Tensor ptab, int64_t pmax, "
               "int64_t base4, torch::Tensor carry, torch::Tensor tail, torch::Tensor beta, torch::Tensor post, "
               "torch::Tensor next, torch::Tensor sout, torch::Tensor hout, torch::Tensor cnt);\n"
               "void sk2_set_pdl(int64_t n);\n")
        _S093_B2_EXT = load_inline("s093_b2_" + tag, "#include <torch/extension.h>\n" + cpp,
                                   cuda_sources=source, functions=["b2_un", "sk2_set_pdl"],
                                   no_implicit_headers=True,
                                   extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _S093_B2_EXT


def _s093_unpool_b2(m, state, s1, tail, g, s2, out):
    bst = state["bfp"]
    name = bfp__sk2_name("unpooler", 1)
    assert name == "slice8_r192"
    _, stream, tab, pmax, b4, _ = bst["sk2"][name]
    branch = torch.empty((1, 1024), device=m.device, dtype=torch.bfloat16)
    _bfp_pdl(1, "un")
    ext = state["s093_b2_ext"]
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.b2_un(stream, m, branch, tab, pmax, b4, s1, tail, g["beta2"], g["post2"], g["pre2"],
               s2, out, state["s093_b2_cnt"])
    return branch


_s093_prior_gle_after = _gle_after_attention


def _gle_after_attention(data, state, o_in, gate, T):
    if T != 1 or "s093_b2_ext" not in state:
        return _s093_prior_gle_after(data, state, o_in, gate, T)
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    g = state["graft"]
    bst = state["bfp"]
    _, stream, tab, pmax, b4, _ = bst["sk2"][bfp__sk2_name("o", T)]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2 * T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uvf = torch.empty((2 * T, HIDDEN), **f32)
    part = torch.empty((256 * T * 8,), **f32)
    rt = torch.empty((2, T), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    _bfp_pdl(T, "o")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_o(stream, o_in, P, tab, pmax, b4, T, data["stream"], g["alpha1"], g["beta1"], g["alpha2"],
              state["g_preffn"], uv, part, uvf)
    pooled, logits = _gle_frt(uv, uvf, state)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _graft_al(pooled, logits, xm, ids, wts, lids, part, rt)
    _gle_route[(T, SL_TILE_CONFIG[T][1] + 1 + GLE_RC[T])](
        pooled, logits, part, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids, rt,
        P, data["stream"], g["alpha1"], g["beta1"], g["alpha2"], s1, MODE=SL_ROUTE_CONFIG[T][1],
        PARTS=SL_TILE_CONFIG[T][1], NCH=GLE_RC[T], TT=T, num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mx = _moe(T, state)
    if MOE_HAS_PDL:
        mx.okc_set_pdl(1 if _pdl(T) else 0)
    mx.okc_run(state["ma_w"], state["ma_scratch"], xm, lids, wts, m, 0, 32, state["nsm"])
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _s093_unpool_b2(m, state, s1, rt[1], g, s2, out)
    return out, s2, ids


_s093_prior_prepare = prepare


def prepare(weights):
    box = {}
    def build():
        try:
            box["ext"] = _s093_b2_ext()
        except Exception as ex:
            box["error"] = str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)
    th = threading.Thread(target=build, daemon=True)
    th.start()
    st = _s093_prior_prepare(weights)
    th.join(timeout=45.0)
    if not th.is_alive() and "ext" in box and st.get("bfp") is not None:
        st["s093_b2_ext"] = box["ext"]
        st["s093_b2_cnt"] = torch.zeros((1,), dtype=torch.int32, device=st["bfp"]["dummy"].device)
        print("[s093] T1 unpooler+b2 last-arrival finalizer enabled", flush=True)
    else:
        why = "compile still running" if th.is_alive() else box.get("error", "bfp unavailable")
        print(f"[s093] protected boundary fallback: {why}", flush=True)
    return st
"""


def main() -> None:
    base = BASE.read_bytes()
    got = hashlib.sha256(base).hexdigest()
    if got != BASE_SHA256:
        raise RuntimeError(f"protected S081 hash mismatch: {got}")
    payload = zlib.compress(APPENDIX.encode(), 9)
    stub = ("\n\n# S093 compressed implementation appendix.\n"
            "import zlib as _s093_z\n"
            f"exec(compile(_s093_z.decompress({payload!r}).decode(), '<s093_b2_finalizer>', 'exec'))\n").encode()
    candidate = base + stub
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
