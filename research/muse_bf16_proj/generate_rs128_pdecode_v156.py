"""Generate v156: producer-side C12 decode rung for v150 RS128.

The protected scored v150 file is retained byte-for-byte as the prefix and as
case 0.  Case 1 keeps its global C12 representation and split-K epilogue, but
the producer warpgroup expands each 128-row x k64 stage into a BF16 fragment
ring in shared memory.  The two math warpgroups only load the resulting 16
register-A words per lane and issue the unchanged RS-WGMMA instructions.

This is deliberately the first diagnostic rung toward an SS-WGMMA design: it
tests whether moving exact decode and escape handling off the consumer critical
path pays before adding the separate SW128 shared layout transformation.
Only pooler m=128 is baked against v150 case 0.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import generate_rs128_faithful_v150 as v150


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_pdecode_v156.py"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return source.replace(old, new)


cuda = v150.cuda_source

# Shared stores for the producer's decoded register-A fragments.  Four uint4
# stores write one lane's 32 BF16 values (64 B) without materialising A[16].
cuda = replace_once(
    cuda,
    """__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile(\"ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n\" : \"=r\"(v.x), \"=r\"(v.y), \"=r\"(v.z), \"=r\"(v.w) : \"r\"(a) : \"memory\"); return v;
}
""",
    """__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile(\"ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n\" : \"=r\"(v.x), \"=r\"(v.y), \"=r\"(v.z), \"=r\"(v.w) : \"r\"(a) : \"memory\"); return v;
}
__device__ __forceinline__ void rq_sts4(uint32_t a, const uint4& v) {
  asm volatile(\"st.shared.v4.u32 [%0], {%1,%2,%3,%4};\\n\" :: \"r\"(a), \"r\"(v.x), \"r\"(v.y), \"r\"(v.z), \"r\"(v.w) : \"memory\");
}
""",
    "decoded shared store helper",
)

# Decode one 16-row fragment and stream its register-A words directly to the
# decoded ring.  Patching the eight high-byte words preserves exact C12 escapes.
cuda = replace_once(
    cuda,
    """__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
""",
    """__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
__device__ __forceinline__ void rq_decstore(const uint4& q0, const uint4& q1, const uint4& q2,
                                             uint32_t Tb, int seq, uint4& E, uint32_t& hs, uint32_t dst) {
  uint32_t H[8];
  rq_hi8(q0, q1, q2, Tb, H);
  rq_fixp8(H, seq, E, hs);
  uint4 v;
  v = make_uint4(rq_prmt(q0.x, H[0], 0x5140), rq_prmt(q0.x, H[0], 0x7362),
                 rq_prmt(q0.y, H[1], 0x5140), rq_prmt(q0.y, H[1], 0x7362)); rq_sts4(dst, v);
  v = make_uint4(rq_prmt(q0.z, H[2], 0x5140), rq_prmt(q0.z, H[2], 0x7362),
                 rq_prmt(q0.w, H[3], 0x5140), rq_prmt(q0.w, H[3], 0x7362)); rq_sts4(dst + 16, v);
  v = make_uint4(rq_prmt(q1.x, H[4], 0x5140), rq_prmt(q1.x, H[4], 0x7362),
                 rq_prmt(q1.y, H[5], 0x5140), rq_prmt(q1.y, H[5], 0x7362)); rq_sts4(dst + 32, v);
  v = make_uint4(rq_prmt(q1.z, H[6], 0x5140), rq_prmt(q1.z, H[6], 0x7362),
                 rq_prmt(q1.w, H[7], 0x5140), rq_prmt(q1.w, H[7], 0x7362)); rq_sts4(dst + 48, v);
}
""",
    "producer decode/store helper",
)

cuda = replace_once(
    cuda,
    """struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB;  // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);  // WK patch table
};
""",
    """struct RqC {
  static constexpr bool PD = (FL & (1 << 27)) != 0;
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB, DB = PD ? 16384 : 0;  // producer-decoded BF16 stage
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS + DB) + 24 * ST + (PD ? 8 * ST : 0) + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);
};
""",
    "producer decoded ring allocation",
)

cuda = replace_once(
    cuda,
    """  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
""",
    """  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS, DB = C::DB;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0, PD = C::PD;
  static_assert(!PD || (NT == 128 && ST == 5 && GS == 1 && PRB == 0 && WM == 0 && MS == 1 && (FL & 16) == 0 && (FL & 32768) == 0),
                \"producer decode is approved only for the faithful full-output m128 path\");
""",
    "producer decode flag",
)

# Keep v150's proven 232/40 dynamic split.  ptxas reserves 168*384 = 64,512
# registers for the CTA; asking producers for 48 would make the math warps wait
# forever in setmaxnreg.inc even though the physical SM register file is larger.
cuda = replace_once(
    cuda,
    """  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
""",
    """  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);
""",
    "producer decode register rebalance",
)

cuda = replace_once(
    cuda,
    """  const uint32_t xring = sbase, wring = sbase + ST * XBS, bars = wring + ST * WBS;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const uint32_t ptab = bars + 24 * ST + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
""",
    """  const uint32_t xring = sbase, wring = sbase + ST * XBS, dring = wring + ST * WBS, bars = dring + ST * DB;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, dfull = bars + 24 * ST;
  const uint32_t zflag = bars + 24 * ST + (PD ? 8 * ST : 0);
  const uint32_t ptab = zflag + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
""",
    "decoded ring shared layout",
)

# The standard (non-EARLY) barrier initialisation is the only live branch for
# the faithful flags.  Producers release wempty after decode; four producer
# warps publish dfull after their fragment stores.
cuda = replace_once(
    cuda,
    """      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\\n\" :: \"r\"(full + 8 * i), \"r\"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\\n\" :: \"r\"(wempty + 8 * i), \"r\"(NW));
      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\\n\" :: \"r\"(xempty + 8 * i), \"r\"(NW));
""",
    """      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\\n\" :: \"r\"(full + 8 * i), \"r\"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\\n\" :: \"r\"(wempty + 8 * i), \"r\"(PD ? 4 : NW));
      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\\n\" :: \"r\"(xempty + 8 * i), \"r\"(NW));
      if constexpr (PD) asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 4;\\n\" :: \"r\"(dfull + 8 * i));
""",
    "decoded ready barrier init",
)

# Producer-side pipeline.  W and x copies retain separate issuing lanes.  The
# four producer warps each decode two of the eight 16-row fragments.  The first
# copies overlap the existing self-zero/poll work in producer warps 10 and 11.
producer = r'''  if constexpr (PD) {
    if (warp >= PW) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
      const int pwg = warp - PW, f0 = pwg, f1 = pwg + 4;
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288;
      uint4 E0 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + f0) * 32 + lane) * 4);
      uint4 E1 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + f1) * 32 + lane) * 4);
      uint32_t hs0 = E0.x >> 17, hs1 = E1.x >> 17;
      uint64_t pol = 0;
      if (HINT && lane == 0 && (warp == PW || warp == PW + 1)) {
        if (warp == PW) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
        else asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      }
      if (ZI && warp >= PW + 2) {
        const int t = tid - (PW + 2) * 32;
        const int r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
        for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
          __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
          asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
        }
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("bar.sync 4, 64;\n" ::: "memory");
        if (t == 0) {
          asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
          uint32_t c = 0;
          for (int it = 0; it < (1 << 22); it++) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
            if ((int)c >= KS) break;
            __nanosleep(128);
          }
          if ((int)c < KS) asm volatile("trap;");
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
          asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
          if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) {
            p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0;
          }
        }
      }
      for (int s = 0; s < KCS; s++) {
        const int slot = s % ST, phase = (s / ST) & 1;
        if (warp == PW && lane == 0) {
          if (s >= ST) rq_wait(wempty + 8 * slot, phase ^ 1);
          rq_expect(full + 8 * slot, WB);
          if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, WB, full + 8 * slot, pol);
          else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, WB, full + 8 * slot);
        }
        if (warp == PW + 1 && lane == 0) {
          if (s >= ST) rq_wait(xempty + 8 * slot, phase ^ 1);
          rq_expect(full + 8 * slot, XB);
          if (HINT) rq_tileh(xring + slot * XBS, &tx, (s0 + s) * 64, 0, full + 8 * slot, pol);
          else rq_tile(xring + slot * XBS, &tx, (s0 + s) * 64, 0, full + 8 * slot);
        }
        rq_wait(full + 8 * slot, phase);
        const uint32_t c0 = wring + slot * WBS + f0 * 1536 + lane * 16;
        uint4 q0 = rq_lds(c0), q1 = rq_lds(c0 + 512), q2 = rq_lds(c0 + 1024);
        rq_decstore(q0, q1, q2, p.Tb, s, E0, hs0,
                    dring + slot * DB + f0 * 2048 + lane * 64);
        const uint32_t c1 = wring + slot * WBS + f1 * 1536 + lane * 16;
        q0 = rq_lds(c1); q1 = rq_lds(c1 + 512); q2 = rq_lds(c1 + 1024);
        rq_decstore(q0, q1, q2, p.Tb, s, E1, hs1,
                    dring + slot * DB + f1 * 2048 + lane * 64);
        __syncwarp();
        if (lane == 0) { rq_arrive(wempty + 8 * slot); rq_arrive(dfull + 8 * slot); }
      }
      return;
    }
  }
'''
cuda = replace_once(
    cuda,
    """  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
""",
    producer + """  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
""",
    "producer decode pipeline",
)

cuda = replace_once(
    cuda,
    """  const uint32_t wl = wring + warp * 1536 + lane * 16;
""",
    """  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint32_t dl = dring + warp * 2048 + lane * 64;
""",
    "consumer decoded lane base",
)

cuda = replace_once(
    cuda,
    """  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
""",
    """  uint4 Q[GS][3], R[GS];
#pragma unroll
  for (int j = 0; j < GS; j++) { Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0); R[j] = make_uint4(0, 0, 0, 0); }
""",
    "consumer fourth decoded load",
)

cuda = replace_once(
    cuda,
    """        if (!RES && (j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
""",
    """        if (!RES && (j % MS) == 0) rq_wait((PD ? dfull : full) + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          if constexpr (PD) {
            const uint32_t a = dl + slot * DB;
            Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 16); Q[j][2] = rq_lds(a + 32); R[j] = rq_lds(a + 48);
          } else {
            const uint32_t a = wl + slot * WBS + (s % MS) * WB;
            Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
          }
        }
""",
    "consumer waits for decoded stage",
)

cuda = replace_once(
    cuda,
    """    if (!RES && !MRG) {
""",
    """    if (!PD && !RES && !MRG) {
""",
    "consumer compressed-ring release",
)

cuda = replace_once(
    cuda,
    """        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
""",
    """        if constexpr (PD) {
          A[j][0] = Q[j][0].x; A[j][1] = Q[j][0].y; A[j][2] = Q[j][0].z; A[j][3] = Q[j][0].w;
          A[j][4] = Q[j][1].x; A[j][5] = Q[j][1].y; A[j][6] = Q[j][1].z; A[j][7] = Q[j][1].w;
          A[j][8] = Q[j][2].x; A[j][9] = Q[j][2].y; A[j][10] = Q[j][2].z; A[j][11] = Q[j][2].w;
          A[j][12] = R[j].x; A[j][13] = R[j].y; A[j][14] = R[j].z; A[j][15] = R[j].w;
        } else if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
""",
    "consumer decoded register copy",
)

PD_FLAGS = v150.variant[7] | (1 << 27)
assert PD_FLAGS == 177217667
pd_case = (
    v150.case
    + f"\n    case 1: return rq_go<128, 5, 1, 0, 0, 0, {PD_FLAGS}, 1>"
      "(p, x, out, (int)KS, (int)info);"
)
cuda = replace_once(cuda, v150.case, pd_case, "producer decode dispatch case")
payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()


append = f'''

# ---------------------------------------------------------------- v156: producer-side exact C12 decode diagnostic.
_V150_CUDA_PAYLOAD = "{payload}"
_V156_CUDA_SHA256 = "{hashlib.sha256(cuda.encode()).hexdigest().upper()}"
_V156_PD_FLAGS = {PD_FLAGS}
_V156_ROUTE_RATIO = 0.97


def _v156_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v156_rs128_pdecode_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
        _v150_module[0] = load_inline(
            tag,
            cpp_sources=_V150_CPP,
            cuda_sources=src,
            functions=["rqrun"],
            extra_cuda_cflags=[
                "-gencode=arch=compute_90a,code=sm_90a",
                "-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "-lineinfo",
            ],
            verbose=False,
        )
    return _v150_module[0]


_v150_build = _v156_build
_v156_v150_prepare = prepare
_v156_v150_kernel_fn = kernel_fn


def _v156_call(state, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        1, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v156 producer-decode rqrun error %d" % rc)
    return out


def _v156_info(state, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        1, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))


def _v156_validate(state, plan, weight):
    m, k, ks = 128, weight.shape[1], plan["KS"]
    cols = {{0, k - 1, k // 3, (2 * k) // 3}}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    plan["cnt"].zero_()
    out = _v156_call(state, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    clean = int(plan["cnt"].abs().sum()) == 0
    if not exact or not clean:
        plan["cnt"].zero_()
        print("v156 pdecode rejected pooler one-hot/counter", exact, clean, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(156128)
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v156_call(state, plan, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    clean = int(plan["cnt"].abs().sum()) == 0
    ok = bool(torch.isfinite(out).all()) and rel <= 6e-3 and worst <= 3e-2 and clean
    if not clean:
        plan["cnt"].zero_()
    print(
        "v156 pdecode exact pooler", "one_hot", exact, "fp64", rel, worst,
        "regs", _v156_info(state, plan, 1),
        "local", _v156_info(state, plan, 2),
        "smem", _v156_info(state, plan, 3), "valid", ok, flush=True,
    )
    return ok


def prepare(weights):
    state = _v156_v150_prepare(weights)
    state["v156_pdecode_enabled"] = False
    try:
        dev = weights["pooler"].device
        props = torch.cuda.get_device_properties(dev)
        vstate = state.get("v150")
        plan = None if vstate is None else vstate.get("plans", {{}}).get("pooler")
        if props.major != 9 or "H200" not in props.name or plan is None:
            return state
        if not _v156_validate(vstate, plan, weights["pooler"]):
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(156129)
        x = torch.randn((128, weights["pooler"].shape[1]), generator=gen, device=dev).to(torch.bfloat16)
        pd_out = torch.empty((128, weights["pooler"].shape[0]), dtype=torch.bfloat16, device=dev)
        c12_out = torch.empty_like(pd_out)
        ratio, pd_us, c12_us = _v150_pair(
            lambda: _v156_call(vstate, plan, x, pd_out),
            lambda: _v150_call(vstate, plan, x, c12_out),
            flush,
        )
        keep = pd_us < c12_us and ratio <= _V156_ROUTE_RATIO
        print(
            "v156 pdecode pooler candidate", pd_us, "v150_c12", c12_us,
            "core_ratio", ratio, "qualified", keep, flush=True,
        )
        if keep:
            state["v156_pdecode_enabled"] = True
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v156 pdecode unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (state.get("v156_pdecode_enabled") and cfg["m"] == 128 and
            cfg["layer"] == "pooler"):
        plan = state["v150"].get("plans", {{}}).get("pooler")
        if plan is not None:
            return _v156_call(state["v150"], plan, data["x"])
    return _v156_v150_kernel_fn(data, state)
'''


base = BASE.read_text()
candidate = base + append
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("v150_sha256", hashlib.sha256(BASE.read_bytes()).hexdigest().upper())
print("pdecode_flags", PD_FLAGS, hex(PD_FLAGS))
print("pdecode_cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
print("bytes", len(candidate.encode()))
