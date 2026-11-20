"""Generate v159: one-wave four-consumer-WG exact-C12 m=256 diagnostic.

The protected v150 file is read but never modified.  Its faithful RS128 CUDA
translation unit is specialized with a second entry point: four m64n128
consumer warpgroups cover the two 128-token halves of m=256 while sharing one
compressed weight ring.  A synchronous single-buffer consumer loop keeps the
640-thread CTA within Hopper's register budget; prepare-time gates retain v150
unless the new route is exact, spill-free, at least 3% faster per route, and
projects to at least a 0.9% whole-score gain.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs256_fourwg_v159.py"


def _assignment(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node
    raise KeyError(name)


base_bytes = BASE.read_bytes()
base = base_bytes.decode()
tree = ast.parse(base)
payload_node = _assignment(tree, "_V150_CUDA_PAYLOAD")
cuda = zlib.decompress(base64.b64decode(ast.literal_eval(payload_node.value))).decode()


def replace_once(old: str, new: str) -> None:
    global cuda
    count = cuda.count(old)
    if count != 1:
        raise RuntimeError(f"expected one CUDA match, found {count}: {old[:100]!r}")
    cuda = cuda.replace(old, new, 1)


# Compile-time geometry.  Bit 26 is private to this diagnostic.
replace_once(
    "  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA\n",
    "  static constexpr bool FOUR = (FL & 67108864) != 0;\n"
    "  static constexpr int CT = FOUR ? 128 : NT;\n"
    "  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = CT / 2;  // FOUR: m256 as two n128 token halves\n",
)
replace_once(
    "__global__ void __launch_bounds__((FL & 1024) ? 256 : 384, (FL & 3072) ? 2 : 1) rq",
    "__global__ void __launch_bounds__((FL & 67108864) ? 640 : ((FL & 1024) ? 256 : 384), (FL & 3072) ? 2 : 1) rq",
)
replace_once(
    "  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;\n",
    "  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, CT = C::CT, XBS = C::XBS, WBS = C::WBS;\n",
)
replace_once(
    "  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)\n",
    "  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)\n"
    "  static_assert(!FOUR || (NT == 256 && ST == 5 && GS == 1 && PRB == 0 && WM == 0 && INS == 0), \"FOUR geometry\");\n",
)
replace_once(
    "  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;\n",
    "  constexpr bool FOUR = (FL & 67108864) != 0;  // v159: four consumer WGs, two m128 token halves\n"
    "  constexpr int NW = FOUR ? 16 : (HALF ? 4 : 8), CW = HALF ? 64 : 128;\n",
)
replace_once(
    "  constexpr int PW = H2 ? 4 : 8;                       // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)\n",
    "  constexpr int PW = FOUR ? 16 : (H2 ? 4 : 8);        // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)\n",
)
replace_once(
    "  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)\n",
    "  constexpr int MREG = FOUR ? 112 : (H2 ? 128 : (F2 ? 96 : 232));  // 512*112 + 128*40 = 62464 registers\n",
)
replace_once(
    "  if constexpr (MREG == 232) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 232;\\n\");\n"
    "  else if constexpr (MREG == 128) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 128;\\n\");\n"
    "  else asm volatile(\"setmaxnreg.inc.sync.aligned.u32 96;\\n\");\n",
    "  if constexpr (!FOUR) {\n"
    "    if constexpr (MREG == 232) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 232;\\n\");\n"
    "    else if constexpr (MREG == 128) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 128;\\n\");\n"
    "    else if constexpr (MREG == 112) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 112;\\n\");\n"
    "    else asm volatile(\"setmaxnreg.inc.sync.aligned.u32 96;\\n\");\n"
    "  }\n",
)
replace_once(
    "  const int g = warp >> 2, w = warp & 3;\n",
    "  const int g = warp >> 2, w = warp & 3, og = FOUR ? (g & 1) : g, th = FOUR ? (g >> 1) : 0;\n",
)
replace_once(
    "  const uint32_t wl = wring + warp * 1536 + lane * 16;\n",
    "  const uint32_t wl = wring + (warp & 7) * 1536 + lane * 16;  // token-half WG pairs share the same W rows\n",
)
replace_once(
    "    asm volatile(\"setmaxnreg.dec.sync.aligned.u32 40;\\n\");\n",
    "    asm volatile(\"setmaxnreg.dec.sync.aligned.u32 40;\\n\");\n",
)

# Keep the four k16 operations in a single PTX block.  Four separate C++ calls
# let NVCC sink PRMT definitions between asynchronous HGMMAs, which makes PTXAS
# surround every HGMMA with ARRIVE/DEPBAR (C7515).
_d = ", ".join(f"%{i}" for i in range(64))
_a = [", ".join(f"%{64 + 4*q + i}" for i in range(4)) for q in range(4)]
_mma = ""
for _q in range(4):
    _mma += (
        f"wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        f"{{{_d}}}, {{{_a[_q]}}}, d{_q}, p{0 if _q == 0 else 1}, 1, 1, 0;\\n"
    )
_outs = ", ".join(f'\"+f\"(D[{i}])' for i in range(64))
_ins = ", ".join([*(f'\"r\"(a[{i}])' for i in range(16)), '\"l\"(desc)', '\"r\"(scd)'])
_helper = (
    "__device__ __forceinline__ void rq_wgmma4_128(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {\n"
    "  asm volatile(\"{\\n.reg .pred p0, p1;\\n.reg .b64 d0, d1, d2, d3;\\n"
    "mov.b64 d0, %80;\\nadd.u64 d1, %80, 2;\\nadd.u64 d2, %80, 4;\\nadd.u64 d3, %80, 6;\\n"
    "setp.ne.b32 p0, %81, 0;\\nsetp.eq.b32 p1, %81, %81;\\n"
    + _mma
    + "}\\n\"\n    : " + _outs + "\n    : " + _ins + ");\n}\n"
)
_insert = cuda.index("template <> __device__ __forceinline__ void rq_wgmma<256>")
cuda = cuda[:_insert] + _helper + cuda[_insert:]

# FOUR reuses the otherwise-idle lanes of the two TMA producer warps for the
# small output-zero loop after they finish issuing transfers.  Removing the two
# dedicated zero warps reduces the CTA to 576 threads, which raises the static
# register cap enough to keep D[64] and A[16] live together.
replace_once(
    "  if (tid == (PW + 2) * 32 && XON) asm volatile(\"prefetch.tensormap [%0];\\n\" :: \"l\"((unsigned long long)&tx) : \"memory\");\n",
    "  if (tid == (PW + 2) * 32 && XON) asm volatile(\"prefetch.tensormap [%0];\\n\" :: \"l\"((unsigned long long)&tx) : \"memory\");\n",
)
_producer_tail = """    } else if (WK && warp == PW + 3) {
      walk();
    }
    return;
  }
"""
_four_zero_tail = """    } else if (WK && warp == PW + 3) {
      walk();
    }
    if constexpr (FOUR) {
      if (ZI) {
        const int t = (warp - PW) * 32 + lane;
        const int r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
        for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
          __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
          asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\\n" :: "l"(d), "r"(0) : "memory");
        }
        asm volatile("fence.proxy.async.global;\\n" ::: "memory");
        asm volatile("bar.sync 4, 64;\\n" ::: "memory");
        if (t == 0) {
          asm volatile("red.release.gpu.global.add.u32 [%0], 1;\\n" :: "l"(p.cnt + blockIdx.x) : "memory");
          uint32_t c = 0;
          for (int it = 0; it < (1 << 22); it++) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
            if ((int)c >= KS) break;
            __nanosleep(128);
          }
          if ((int)c < KS) asm volatile("trap;");
          asm volatile("fence.proxy.async.global;\\n" ::: "memory");
          asm volatile("st.release.cta.shared.u32 [%0], 1;\\n" :: "r"(zflag) : "memory");
          if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) {
            p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0;
          }
        }
      }
    }
    return;
  }
"""
replace_once(_producer_tail, _producer_tail)

# Replace the generic double/triple-buffered consumer body with a compile-time
# v159 branch that needs only D[64] + A[16] + three uint4 load registers.
body_start = cuda.index("  uint32_t A0[GS][16], A1[GS][16];\n")
body_end_marker = "    if (WM != 1) xrel(NGT - 1);\n  }\n"
body_end = cuda.index(body_end_marker, body_start) + len(body_end_marker)
old_body = cuda[body_start:body_end]

common_lines = (
    "  uint32_t sink = 0;\n"
    "  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;\n"
    "#define RQ_T0() do { if (INS) tq = clock64(); } while (0)\n"
    "#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)\n"
    "  if (INS) { tq0 = clock64(); tq = tq0; }\n"
    "  const unsigned long long gtl0 = INS ? rq_gt() : 0;\n"
    "  unsigned long long gtw0 = 0;\n"
    "  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;\n"
)
for line in (
    "  uint32_t sink = 0;\n",
    "  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;\n",
    "#define RQ_T0() do { if (INS) tq = clock64(); } while (0)\n",
    "#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)\n",
    "  if (INS) { tq0 = clock64(); tq = tq0; }\n",
    "  const unsigned long long gtl0 = INS ? rq_gt() : 0;\n",
    "  unsigned long long gtw0 = 0;\n",
    "  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds\n",
):
    if old_body.count(line) != 1:
        raise RuntimeError(f"missing consumer-body line: {line!r}")
    old_body = old_body.replace(line, "", 1)

four_body = r'''  if constexpr (FOUR) {
    // Four independent consumer WGs.  WG pairs (0,2) and (1,3) decode the
    // same 64 weight rows but multiply token halves 0 and 1.  A wait-0 loop
    // intentionally trades intra-WG overlap for a spill-free 112-register cap;
    // four WGs provide the latency-level parallelism at the SM level.
    uint32_t A[16];
    rq_facc<ND>(D);
    for (int s = 0; s < KCS; ++s) {
      const int slot = s % ST;
      rq_wait(full + 8 * slot, (s / ST) & 1);
      const uint32_t wa = wl + slot * WBS;
      const uint4 q0 = rq_lds(wa), q1 = rq_lds(wa + 512), q2 = rq_lds(wa + 1024);
      __syncwarp();
      if (lane == 0) rq_arrive(wempty + 8 * slot);
      uint32_t H[8];
      rq_hi8(q0, q1, q2, Tb, H);
      rq_fixp8(H, s, E4, hs);
      rq_prmtA(q0, q1, H, A);
      // Keep all four k16 A fragments simultaneously live.  Without this
      // pre-fence operand barrier, NVCC sinks each PRMT between HGMMAs and
      // PTXAS serializes the group (C7515).
      rq_fa(A);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint64_t d = xd0 + (uint64_t)((slot * XB + th * 16384) >> 4);
      rq_wgmma4_128(D, A, d, s != 0);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      rq_fa(A);
      __syncwarp();
      if (lane == 0) rq_arrive(xempty + 8 * slot);
    }
    rq_facc<ND>(D);
  } else {
'''
cuda = cuda[:body_start] + common_lines + four_body + old_body + "  }\n" + cuda[body_end:]

# The epilogue has one 128-token slab per consumer WG.  For v159, g selects
# both the output-row half (og) and token half (th).
cuda = cuda.replace("const uint32_t slab = xring + g * XB;", "const uint32_t slab = xring + g * (CT * 128);")
cuda = cuda.replace("cb + 64 * g", "cb + 64 * og")
cuda = cuda.replace("for (int pp = 0; pp < NT / 32; pp++)", "for (int pp = 0; pp < CT / 32; pp++)")
cuda = cuda.replace("for (int pp = NT / 32; pp < NT / 16; pp++)", "for (int pp = CT / 32; pp < CT / 16; pp++)")
cuda = cuda.replace("for (int pp = 0; pp < NT / 16; pp++)", "for (int pp = 0; pp < CT / 16; pp++)")
cuda = cuda.replace("issue(NT / 2, slab + (NT / 2) * 128);", "issue(CT / 2, slab + (CT / 2) * 128);")
old_issue = ":: \"l\"((unsigned long long)&to), \"r\"(cb + 64 * og), \"r\"(tok0), \"r\"(src) : \"memory\");"
new_issue = ":: \"l\"((unsigned long long)&to), \"r\"(cb + 64 * og), \"r\"(tok0 + th * CT), \"r\"(src) : \"memory\");"
if cuda.count(old_issue) != 2:
    raise RuntimeError("unexpected EH issue-call count")
cuda = cuda.replace(old_issue, new_issue)
# The non-EH epilogue emits its TMA directly rather than through issue().
cuda = cuda.replace(':: "l"((unsigned long long)&to), "r"(cb + 64 * og), "r"(0), "r"(slab) : "memory");',
                    ':: "l"((unsigned long long)&to), "r"(cb + 64 * og), "r"(th * CT), "r"(slab) : "memory");')

# Host geometry, tensor-map box, and the second dispatch case.
cuda = cuda.replace("(FL & 1024) ? 256 : 384", "(FL & 67108864) ? 640 : ((FL & 1024) ? 256 : 384)")
replace_once(
    "  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);\n",
    "  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 67108864) ? 128 : ((FL & 65536) ? NT / 2 : NT));\n",
)
replace_once(
    "    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);\n",
    "    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);\n"
    "    case 1: return rq_go<256, 5, 1, 0, 0, 0, 110108803, 1>(p, x, out, (int)KS, (int)info);\n",
)

new_payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
payload_line = base.splitlines()[payload_node.lineno - 1]
if not payload_line.startswith("_V150_CUDA_PAYLOAD = "):
    raise RuntimeError("unexpected payload assignment layout")
candidate = base.replace(payload_line, f'_V150_CUDA_PAYLOAD = "{new_payload}"', 1)

integration = r'''

# ---------------------------------------------------------------- v159: four-consumer-WG m256 exact-C12 diagnostic.
_V159_TARGETS = ("qkv_gate", "o", "pooler", "unpooler")
_V159_KS = {"qkv_gate": 7, "o": 2, "pooler": 4, "unpooler": 2}
_V159_ROUTE_RATIO = 0.97
_V159_SCORE_GATE = 0.009


def _v159_call(vstate, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = vstate["ext"].rqrun(
        1, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v159 rqrun error %d" % rc)
    return out


def _v159_info(vstate, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(vstate["ext"].rqrun(
        1, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], kind,
    ))


def _v159_validate(vstate, plan, weight, layer):
    m, k = 256, weight.shape[1]
    cols = sorted({0, k - 1, k // 3, (2 * k) // 3} | {
        max(0, k * split // plan["KS"] - 1)
        for split in range(1, plan["KS"])
    } | {
        min(k - 1, k * split // plan["KS"])
        for split in range(1, plan["KS"])
    })
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    out = _v159_call(vstate, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    if not exact or int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_()
        print("v159 rejected", layer, "one-hot/counter", exact, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(1590 + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v159_call(vstate, plan, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    tol = _V150_TOL[plan["KS"]]
    regs = _v159_info(vstate, plan, 1)
    local = _v159_info(vstate, plan, 2)
    smem = _v159_info(vstate, plan, 3)
    ok = (bool(torch.isfinite(out).all()) and rel <= tol and worst <= 5 * tol
          and local <= 64 and regs <= 112 and smem <= 227328)
    if int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_(); ok = False
    print("v159 exact", layer, "one_hot", exact, "fp64", rel, worst,
          "regs", regs, "local", local, "smem", smem,
          "esc2_max", plan["esc2_max"], "valid", ok, flush=True)
    return ok


_v159_base_prepare = prepare
_v159_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v159_base_prepare(weights)
    state["v159_enabled"] = set()
    state["v159_projected_gain"] = 0.0
    vstate = state.get("v150")
    if vstate is None:
        return state
    dev = weights["pooler"].device
    try:
        plans = {}
        for layer in _V159_TARGETS:
            if layer in vstate["plans"] and _V159_KS[layer] == vstate["plans"][layer]["KS"]:
                plans[layer] = dict(vstate["plans"][layer])
            else:
                weight = weights[layer]
                packed = _v150_pack(weight)
                if packed is None:
                    continue
                lay, tab, esc6 = packed
                if not torch.equal(_v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
                                   weight.view(torch.int16).int() & 0xffff):
                    continue
                ks = _V159_KS[layer]
                esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
                esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
                if esc2 is None:
                    continue
                plans[layer] = {
                    "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                    "esc2_max": esc2_max,
                    "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                    "KS": ks, "n": weight.shape[0],
                }
            if not _v159_validate(vstate, plans[layer], weights[layer], layer):
                del plans[layer]
        state["v159_plans"] = plans
        if not plans:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(159256)
        qualified = set()
        inc_total = cand_total = 0.0
        for layer in _V159_TARGETS:
            weight = weights[layer]
            x = torch.randn((256, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 256, "n": weight.shape[0], "k": weight.shape[1]}
            plan = plans.get(layer)
            if plan is None:
                us = _v150_time(lambda x=x, cfg=cfg: _v159_base_kernel_fn({"x": x, "config": cfg}, state), flush)
                core = max(us - _V150_REPLAY_US, 0.05)
                inc_total += core; cand_total += core
                continue
            out = torch.empty((256, weight.shape[0]), dtype=torch.bfloat16, device=dev)
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda x=x, out=out, plan=plan: _v159_call(vstate, plan, x, out),
                lambda x=x, cfg=cfg: _v159_base_kernel_fn({"x": x, "config": cfg}, state),
                flush,
            )
            keep = candidate_us < incumbent_us and ratio <= _V159_ROUTE_RATIO
            if keep:
                qualified.add((256, layer))
            inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
            cand_core = max(candidate_us - _V150_REPLAY_US, 0.05) if keep else inc_core
            inc_total += inc_core; cand_total += cand_core
            print("v159 rs256", layer, "candidate", candidate_us,
                  "incumbent", incumbent_us, "core_ratio", ratio,
                  "qualified", keep, flush=True)

        rw = weights["router"]
        rx = torch.randn((256, rw.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
        rcfg = {"layer": "router", "m": 256, "n": rw.shape[0], "k": rw.shape[1]}
        router_us = _v150_time(lambda: _v159_base_kernel_fn({"x": rx, "config": rcfg}, state), flush)
        router_core = max(router_us - _V150_REPLAY_US, 0.05)
        inc_total += router_core; cand_total += router_core
        projected = (inc_total / cand_total) ** (1.0 / 6.0) - 1.0
        state["v159_projected_gain"] = projected
        if projected >= _V159_SCORE_GATE:
            state["v159_enabled"] = qualified
        print("v159 load256 incumbent", inc_total, "candidate", cand_total,
              "projected_score_gain", projected,
              "enabled", sorted(state["v159_enabled"]), flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state["v159_enabled"] = set()
        print("v159 unavailable", type(exc).__name__, str(exc).splitlines()[-1][:240], flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v159_enabled", ()):
        plan = state.get("v159_plans", {}).get(cfg["layer"])
        vstate = state.get("v150")
        if plan is not None and vstate is not None:
            return _v159_call(vstate, plan, data["x"])
    return _v159_base_kernel_fn(data, state)
'''

candidate += integration
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("protected_sha256", hashlib.sha256(base_bytes).hexdigest().upper())
print("candidate_sha256", hashlib.sha256(candidate.encode()).hexdigest().upper())
print("cuda_bytes", len(cuda.encode()), "candidate_bytes", len(candidate.encode()))
