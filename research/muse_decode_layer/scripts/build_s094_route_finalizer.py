"""Build S094: T=1 fused pooler/router last-arrival route finalizer."""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
OUT = ROOT / "submission_s094_route_finalizer.py"
BASE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"


APPENDIX = r"""

# ==============================================================================================
# S094: T=1 fused pooler/router last-arrival route finalizer (no spinning CTA).
_S094_ROUTE_EXT = None


def _s094_route_source():
    s = _gle_source()
    old = ("struct GleEpi { const float *s, *a, *b, *a2; const __nv_bfloat16* p; __nv_bfloat16* uv; "
           "float* part; float* uvf; const float *x32, *w32; int fpr, ilv; };")
    new = old[:-3] + r'''
  const float *opart, *stream, *alpha, *beta, *alpha2, *bias;
  const __nv_bfloat16 *po, *gpre;
  int *ids, *lids; float *wts, *rt, *s1; __nv_bfloat16 *xm;
  int *route_cnt; const int *local_start; int route;
};'''
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    helper = r'''
__device__ __forceinline__ float s094_sigmoid(float x) {
  return __fdividef(1.f, __fadd_rn(1.f, __expf(-x)));
}
'''
    s = s.replace("static GleEpi g_gle;", "static GleEpi g_gle;\n" + helper, 1)

    finalizer = r'''
  if constexpr (FUSE) if (ep.route) {
    static_assert(MT == 1 && TOKR == 8 && R == 32 && NW == 8 && CL == 1 && sizeof(OutT) == 2,
                  "S094 route finalizer geometry");
    __shared__ int route_last;
    named_bar(1, NC);
    if (tid == 0) {
      __threadfence();
      const int oldc = atomicAdd(ep.route_cnt, 1);
      route_last = oldc == (int)gridDim.x - 1;
      if (route_last) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (route_last) {
      // Seven O-epilogue reductions: P^2; u^2,uv,v^2; and the carry tail.
      float oq[7];
#pragma unroll
      for (int k = 0; k < 7; ++k) oq[k] = ep.opart[tid * 8 + k];
#pragma unroll
      for (int d = 16; d; d >>= 1)
#pragma unroll
        for (int k = 0; k < 7; ++k) oq[k] = __fadd_rn(oq[k], __shfl_xor_sync(0xffffffffu, oq[k], d));
      if (lane == 0)
#pragma unroll
        for (int k = 0; k < 7; ++k) red[warp * 8 + k] = oq[k];
      named_bar(1, NC);
      if (tid < 7) {
        float z = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) z = __fadd_rn(z, red[w * 8 + tid]);
        red[64 + tid] = z;
      }
      named_bar(1, NC);
      float r1, r2;
      { float z = __fadd_rn(__fmul_rn(red[64], 1.f / 8192.f), 1.e-8f);
        asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r1) : "f"(z)); }
      const float r1sq = __fmul_rn(r1, r1);
      const float ss2 = __fadd_rn(red[65], __fadd_rn(__fmul_rn(2.f * r1, red[66]), __fmul_rn(r1sq, red[67])));
      { float z = __fadd_rn(__fmul_rn(ss2, 1.f / 8192.f), 1.e-5f);
        asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r2) : "f"(z)); }
      if (tid == 0) {
        ep.rt[0] = r1;
        ep.rt[1] = __fadd_rn(red[68], __fadd_rn(__fmul_rn(2.f * r1, red[69]), __fmul_rn(r1sq, red[70])));
      }

      // Boundary-1 carry programs formerly embedded in _gle_route.
      for (int i = tid; i < 8192; i += NC) {
        const float pv = __bfloat162float(ep.po[i]);
        const float bn = __bfloat162float(__float2bfloat16_rn(__fmul_rn(pv, r1)));
        const float ns = __fmaf_rn(ep.beta[i], bn, __fmul_rn(ep.alpha[i], ep.stream[i]));
        ep.s1[i] = __fmul_rn(ep.alpha2[i], ns);
      }

      // Combined two-row pooler, RMSNorm and pre-expert scale.
      float q = 0.f;
      for (int i = tid; i < 4096; i += NC) {
        const float u = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(out)[i]);
        const float v = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(out)[4096 + i]);
        const float p = __fmul_rn(__fadd_rn(u, __fmul_rn(v, r1)), r2);
        q = __fmaf_rn(p, p, q);
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
      float rp;
      { float z = __fadd_rn(__fmul_rn(red[0], 1.f / 4096.f), 1.e-5f);
        asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(rp) : "f"(z)); }
      for (int i = tid; i < 4096; i += NC) {
        const float u = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(out)[i]);
        const float v = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(out)[4096 + i]);
        const float p = __fmul_rn(__fadd_rn(u, __fmul_rn(v, r1)), r2);
        const float g = __fadd_rn(1.f, __bfloat162float(ep.gpre[i]));
        ep.xm[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(p, rp), g));
      }

      // Exact ordered top-8 key: score bits, then lower expert id wins ties.
      const int e = tid;
      const float logit = __fmul_rn(__fadd_rn(outr[e], __fmul_rn(outr[256 + e], r1)), r2);
      const float sc = s094_sigmoid(__fmul_rn(logit, 0.1767766952966369f));
      const float sel = __fadd_rn(sc, ep.bias[e]);
      const uint32_t bits = __float_as_uint(sel);
      const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
      unsigned long long key = ((unsigned long long)ordered << 32) | (unsigned)(255 - e);
      unsigned long long* kr = reinterpret_cast<unsigned long long*>(red + 128);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        unsigned long long wv = key;
        for (int d = 16; d; d >>= 1) {
          const unsigned long long z = __shfl_down_sync(0xffffffffu, wv, d);
          if (z > wv) wv = z;
        }
        if (lane == 0) kr[warp] = wv;
        named_bar(1, NC);
        if (warp == 0) {
          wv = lane < NW ? kr[lane] : 0ull;
          for (int d = 16; d; d >>= 1) {
            const unsigned long long z = __shfl_down_sync(0xffffffffu, wv, d);
            if (z > wv) wv = z;
          }
          if (lane == 0) kr[8] = wv;
        }
        named_bar(1, NC);
        const int win = 255 - (int)(kr[8] & 255ull);
        if (tid == 0) {
          ep.ids[k] = win;
          const int ls = *ep.local_start;
          const int lid = (win - ls + 256) & 255;
          ep.lids[k] = lid < 32 ? lid : -1;
          const float l = __fmul_rn(__fadd_rn(outr[win], __fmul_rn(outr[256 + win], r1)), r2);
          ep.wts[k] = s094_sigmoid(__fmul_rn(l, 0.1767766952966369f));
        }
        if (e == win) key = 0ull;
        named_bar(1, NC);
      }
      if (tid == 0) {
        const float a0 = __fadd_rn(ep.wts[0], ep.wts[1]);
        const float a1 = __fadd_rn(ep.wts[2], ep.wts[3]);
        const float a2 = __fadd_rn(ep.wts[4], ep.wts[5]);
        const float a3 = __fadd_rn(ep.wts[6], ep.wts[7]);
        const float den = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
#pragma unroll
        for (int k = 0; k < 8; ++k) ep.wts[k] = __fdividef(ep.wts[k], __fadd_rn(den, 1.e-15f));
      }
      named_bar(1, NC);
      if (tid == 0) {
        __threadfence();
        atomicExch(ep.route_cnt, 0);
      }
    }
    named_bar(1, NC);
  }
'''
    anchor = ('  if (trace && tid == 0) tr[15] = gtime();\n'
              '  if ((dgv & 32) && dbg != nullptr && tid == 0) {')
    assert s.count(anchor) == 1
    s = s.replace(anchor, finalizer + "\n" + anchor, 1)
    s += r'''
void gle_pr_route(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr,
                  at::Tensor ptab, int64_t pmax, int64_t base4, at::Tensor part, at::Tensor rcnt,
                  at::Tensor x32, at::Tensor w32, int64_t wpf, at::Tensor opart, at::Tensor po,
                  at::Tensor stream, at::Tensor alpha, at::Tensor beta, at::Tensor alpha2,
                  at::Tensor bias, at::Tensor gpre, at::Tensor ids, at::Tensor wts, at::Tensor xm,
                  at::Tensor local_start, at::Tensor lids, at::Tensor rt, at::Tensor s1, at::Tensor route_cnt) {
  TORCH_CHECK(x.size(0) == 2 && out.size(0) == 2 && outr.size(0) == 2 && po.size(0) == 1,
              "S094 T1 two-row route shapes");
  g_gle = GleEpi{};
  g_gle.x32=x32.data_ptr<float>(); g_gle.w32=w32.data_ptr<float>(); g_gle.fpr=1; g_gle.ilv=(int)wpf;
  g_gle.opart=opart.data_ptr<float>(); g_gle.po=(const __nv_bfloat16*)po.data_ptr();
  g_gle.stream=stream.data_ptr<float>(); g_gle.alpha=alpha.data_ptr<float>(); g_gle.beta=beta.data_ptr<float>();
  g_gle.alpha2=alpha2.data_ptr<float>(); g_gle.bias=bias.data_ptr<float>();
  g_gle.gpre=(const __nv_bfloat16*)gpre.data_ptr(); g_gle.ids=ids.data_ptr<int>();
  g_gle.lids=lids.data_ptr<int>(); g_gle.wts=wts.data_ptr<float>(); g_gle.xm=(__nv_bfloat16*)xm.data_ptr();
  g_gle.local_start=local_start.data_ptr<int>(); g_gle.rt=rt.data_ptr<float>(); g_gle.s1=s1.data_ptr<float>();
  g_gle.route_cnt=route_cnt.data_ptr<int>(); g_gle.route=1;
  sk2f(Ws,Wsr,x,out,outr,ptab,pmax,base4,part,rcnt,2,8);
  g_gle = GleEpi{};
}
'''
    return s


def _s094_route_ext():
    global _S094_ROUTE_EXT
    if _S094_ROUTE_EXT is None:
        source = _s094_route_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace("sk2_kernel", "sk2_kernel_s094" + tag)
                  .replace("launch_sk2", "launch_sk2_s094" + tag)
                  .replace("void sk2(", "void sk2_s094" + tag + "(")
                  .replace("  sk2(Ws,", "  sk2_s094" + tag + "(Ws,")
                  .replace("void sk2f(", "void sk2f_s094" + tag + "(")
                  .replace("  sk2f(Ws,", "  sk2f_s094" + tag + "(Ws,"))
        cpp = ("void gle_pr_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
               "int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor,"
               "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
               "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
               "torch::Tensor);\nvoid sk2_set_pdl(int64_t n);\n")
        _S094_ROUTE_EXT = load_inline("s094_route_" + tag, "#include <torch/extension.h>\n" + cpp,
                                      cuda_sources=source, functions=["gle_pr_route", "sk2_set_pdl"],
                                      no_implicit_headers=True,
                                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _S094_ROUTE_EXT


def _s094_pool_route(uv, uvf, state, opart, po, stream, g, ids, wts, xm, local_start, lids, rt, s1):
    bst = state["bfp"]
    _, pstream, ptab, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((2, 4096), device=uv.device, dtype=torch.bfloat16)
    logits = torch.empty((2, 256), device=uv.device, dtype=torch.float32)
    _bfp_pdl(2, "pool")
    ext = state["s094_route_ext"]
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr_route(pstream, bst["sk2"]["router"][1], uv, pooled, logits, ptab, ppmax, pb4,
                     bst["fused_part"], bst["fused_cnt"], uvf, state["router_fp32"], GLE_ILV,
                     opart, po, stream, g["alpha1"], g["beta1"], g["alpha2"], state["bias"],
                     state["g_preexp"], ids, wts, xm, local_start, lids, rt, s1, state["s094_route_cnt"])
    return pooled, logits


_s094_prior_gle_after = _gle_after_attention


def _gle_after_attention(data, state, o_in, gate, T):
    if T != 1 or "s094_route_ext" not in state:
        return _s094_prior_gle_after(data, state, o_in, gate, T)
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    g = state["graft"]
    bst = state["bfp"]
    _, stream, tab, pmax, b4, _ = bst["sk2"][bfp__sk2_name("o", T)]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2*T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uvf = torch.empty((2*T, HIDDEN), **f32)
    part = torch.empty((256*T*8,), **f32)
    rt = torch.empty((2,T), **f32)
    s1 = torch.empty((T,HIDDEN), **f32)
    _bfp_pdl(T,"o")
    ext = _gle_ext(); ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_o(stream,o_in,P,tab,pmax,b4,T,data["stream"],g["alpha1"],g["beta1"],g["alpha2"],
              state["g_preffn"],uv,part,uvf)
    ids=torch.empty((T,8),device=dev,dtype=torch.int32)
    lids=torch.empty((T,8),device=dev,dtype=torch.int32)
    wts=torch.empty((T,8),**f32)
    xm=torch.empty((T,D_EXPERT),device=dev,dtype=torch.bfloat16)
    _s094_pool_route(uv,uvf,state,part,P,data["stream"],g,ids,wts,xm,data["local_start"],lids,rt,s1)
    m=torch.empty((T,D_EXPERT),device=dev,dtype=torch.bfloat16)
    mx=_moe(T,state)
    if MOE_HAS_PDL: mx.okc_set_pdl(1 if _pdl(T) else 0)
    mx.okc_run(state["ma_w"],state["ma_scratch"],xm,lids,wts,m,0,32,state["nsm"])
    U,S=dense(m,state["unpool_slice"],T,state,"un")
    if S != 1 or U.dtype != torch.bfloat16: raise RuntimeError("S094 BF16 sliced unpooler expected")
    s2=torch.empty((T,HIDDEN),**f32)
    out=torch.empty((T,HIDDEN),device=dev,dtype=torch.bfloat16)
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](U,T,s1,rt[1],g["beta2"],g["post2"],g["pre2"],s2,out,
                                    BC=bc,num_warps=nw,launch_pdl=True)
    return out,s2,ids


_s094_prior_prepare=prepare
def prepare(weights):
    box={}
    def build():
        try: box["ext"]=_s094_route_ext()
        except Exception as ex: box["error"]=str(ex).splitlines()[-1][:240] if str(ex) else repr(ex)
    th=threading.Thread(target=build,daemon=True); th.start()
    st=_s094_prior_prepare(weights); th.join(timeout=55.0)
    if not th.is_alive() and "ext" in box and st.get("bfp") is not None:
        st["s094_route_ext"]=box["ext"]
        st["s094_route_cnt"]=torch.zeros((1,),dtype=torch.int32,device=st["bfp"]["dummy"].device)
        print("[s094] T1 pooler/router+route last-arrival finalizer enabled",flush=True)
    else:
        why="compile still running" if th.is_alive() else box.get("error","bfp unavailable")
        print(f"[s094] protected route fallback: {why}",flush=True)
    return st
"""


def main() -> None:
    base = BASE.read_bytes()
    got = hashlib.sha256(base).hexdigest()
    if got != BASE_SHA256:
        raise RuntimeError(f"protected S081 hash mismatch: {got}")
    payload = zlib.compress(APPENDIX.encode(), 9)
    stub = ("\n\n# S094 compressed implementation appendix.\nimport zlib as _s094_z\n"
            f"exec(compile(_s094_z.decompress({payload!r}).decode(), '<s094_route_finalizer>', 'exec'))\n").encode()
    candidate = base + stub
    if len(candidate) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds server limit: {len(candidate)}")
    compile(candidate.decode(), str(OUT), "exec")
    OUT.write_bytes(candidate)
    print(f"wrote {OUT.name}: {len(candidate)} bytes ({1_000_000-len(candidate)} headroom)")
    print(f"sha256 {hashlib.sha256(candidate).hexdigest()}")


if __name__ == "__main__":
    main()
