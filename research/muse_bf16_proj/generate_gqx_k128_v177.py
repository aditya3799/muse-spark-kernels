"""Generate v177: add separately autotuned K128 GQX mainloops.

The proven GQX path moves one K64 W/X tile pair per pipeline stage.  This
candidate keeps the same total ring footprint but moves two adjacent K64 pairs
per stage and issues their eight WGMMA operations in original K order.  It is
never selected statically: H200 preparation validates and races it against the
existing K64 and cuBLAS/Lt routes for the official m=128 and m=256 shapes.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_clean_composite_v174.py"
OUTPUT = ROOT / "submission_gqx_k128_v177.py"


text = BASE.read_text()


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


# Export a distinct entry point so the production K64 path remains available.
replace_once(
    '''void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk''',
    '''void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void gqk(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk''',
)

# Restrict the mechanical CUDA edits to GQX_SRC.
head, marker, tail = text.partition('GQX_SRC = r"""')
if not marker:
    raise RuntimeError("GQX_SRC marker not found")
gqx_body, marker2, rest = tail.partition('"""\n\nGQX_CONFIGS =')
if not marker2:
    raise RuntimeError("GQX_SRC end marker not found")


def k128_wgmma_def(bn: int) -> str:
    """Emit eight K16 HGMMAs in one inline-PTX operand scope.

    Separate wgmma_ss() calls each expose the accumulator array as a C++
    read/write operand.  At eight calls PTXAS treats those intervening operand
    definitions as hazards and inserts a DEPBAR after every HGMMA.  Keeping all
    eight instructions in one asm block exposes the accumulator bank only once.
    """
    nacc = bn // 2
    regs = ", ".join(f"%{i}" for i in range(nacc))
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
    a0, b0, a1, b1, pred = range(nacc, nacc + 5)
    lines = [
        f"template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type",
        "wgmma_ss_k128(float* d, uint64_t da0i, uint64_t db0i, uint64_t da1i, uint64_t db1i) {",
        '  asm volatile("{\\n"',
        '      ".reg .pred p;\\n"',
        '      ".reg .b64 da0, db0, da1, db1;\\n"',
        f'      "mov.b64 da0, %{a0};\\n"',
        f'      "mov.b64 db0, %{b0};\\n"',
        f'      "mov.b64 da1, %{a1};\\n"',
        f'      "mov.b64 db1, %{b1};\\n"',
        f'      "setp.ne.b32 p, %{pred}, 0;\\n"',
    ]
    for unit in range(2):
        for j in range(4):
            lines.append(
                f'      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 '
                f'{{{regs}}}, da{unit}, db{unit}, p, 1, 1, 0, 0;\\n"'
            )
            if j != 3:
                lines.append(f'      "add.u64 da{unit}, da{unit}, 2;\\n"')
                lines.append(f'      "add.u64 db{unit}, db{unit}, 2;\\n"')
    lines.extend([
        '      "}\\n"',
        f"      : {cons}",
        '      : "l"(da0i), "l"(db0i), "l"(da1i), "l"(db1i), "r"(1));',
        "}",
    ])
    return "\n".join(lines)


gqx_body = "\n\n".join(k128_wgmma_def(bn) for bn in (128, 256)) + "\n\n" + gqx_body


def gqx_replace(old: str, new: str, expected: int = 1) -> None:
    global gqx_body
    count = gqx_body.count(old)
    if count != expected:
        raise RuntimeError(f"GQX expected {expected} matches, found {count}: {old[:100]!r}")
    gqx_body = gqx_body.replace(old, new, expected)


gqx_replace(
    'template <int BN, int NWG, int NSTAGE, int CL>\n__global__ void',
    'template <int BN, int NWG, int NSTAGE, int CL, int KU>\n__global__ void',
)
gqx_replace(
    '''  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;''',
    '''  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int KST = A_BYTES + B_BYTES;
  constexpr int ST = KU * KST;
  static_assert(KU == 1 || KU == 2, "gqx: KU must be one or two K64 units");''',
)
gqx_replace(
    '''  const int nkb = u1 - u0;
  const int xrows = min(M, BN);''',
    '''  const int nkb = u1 - u0;
  const int nks = (nkb + KU - 1) / KU;
  const int xrows = min(M, BN);''',
)
gqx_replace(
    '''    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }''',
    '''    for (int i = 0; i < NSTAGE && i < nks; ++i) {
      const int units = min(KU, nkb - i * KU);
      mbar_expect_tx(full0 + 8 * i, units * (A_BYTES + xrows * 128));
#pragma unroll
      for (int ku = 0; ku < KU; ++ku) {
        const int kb = i * KU + ku;
        if (kb < nkb) {
          const int kg = (u0 + kb) * 64;
          const uint32_t dst = sbase + i * ST + ku * KST;
          tma_load_2d_ef(dst, &mapW, full0 + 8 * i, kg, rb * BM);
          tma_load_2d_el(dst + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
        }
      }
    }''',
)
gqx_replace(
    '''      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }''',
    '''      for (int i = NSTAGE; i < nks; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        const int units = min(KU, nkb - i * KU);
        mbar_expect_tx(full0 + 8 * slot, units * (A_BYTES + xrows * 128));
#pragma unroll
        for (int ku = 0; ku < KU; ++ku) {
          const int kb = i * KU + ku;
          if (kb < nkb) {
            const int kg = (u0 + kb) * 64;
            const uint32_t dst = sbase + slot * ST + ku * KST;
            tma_load_2d_ef(dst, &mapW, full0 + 8 * slot, kg, rb * BM);
            tma_load_2d_el(dst + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
          }
        }
      }''',
)
gqx_replace(
    '''  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }''',
    '''  for (int i = 0; i < nks; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
    if constexpr (KU == 2) {
      const uint32_t kst0 = sbase + slot * ST;
      const uint32_t a0 = kst0 + wg * (64 * 128);
      const uint32_t b0 = kst0 + A_BYTES;
      if (i * KU + 1 < nkb) {
        const uint32_t kst1 = kst0 + KST;
        wgmma_ss_k128<BN>(acc, make_desc_sw128(a0), make_desc_sw128(b0),
                          make_desc_sw128(kst1 + wg * (64 * 128)), make_desc_sw128(kst1 + A_BYTES));
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wgmma_ss<BN>(acc, make_desc_sw128(a0 + j * 32), make_desc_sw128(b0 + j * 32));
      }
    } else {
      const uint32_t kst = sbase + slot * ST;
      const uint32_t a_st = kst + wg * (64 * 128);
      const uint32_t b_st = kst + A_BYTES;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    }
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }''',
)
gqx_replace(
    'template <int BN, int NWG, int NSTAGE, int CL>\nstatic void launch_gqx',
    'template <int BN, int NWG, int NSTAGE, int CL, int KU>\nstatic void launch_gqx',
)
gqx_replace(
    '  constexpr int ST = BM * 128 + BN * 128;\n',
    '  constexpr int ST = KU * (BM * 128 + BN * 128);\n',
)
gqx_replace(
    '  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;',
    '  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL, KU>;',
)
gqx_replace(
    '''void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}''',
    '''void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}

void gqk(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQK_DISPATCH@@
  TORCH_CHECK(false, "gqk: no config");
}''',
)

text = head + marker + gqx_body + marker2 + rest

# K128 keeps the K64 ring footprint: four stages at BN128 and two at BN256.
replace_once(
    '''GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}
USE_GQX = True''',
    '''GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}
GQK_CONFIGS = {
    ("qkv_gate", 128): (1, 3, 4), ("qkv_gate", 256): (1, 3, 2),
    ("pooler", 128): (1, 2, 4), ("pooler", 256): (1, 2, 2),
    ("unpooler", 128): (1, 1, 4), ("unpooler", 256): (1, 1, 2),
    ("o", 128): (1, 1, 4), ("o", 256): (1, 1, 2),
}
USE_GQX = True
USE_GQK = True''',
)

replace_once(
    '''def _gqx_src(sm90):
    if not sm90:
        return ('void gqx(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gqx needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQX_SRC.replace("@@GQX_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))''',
    '''def _gqx_src(sm90):
    if not sm90:
        return ('void gqx(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gqx needs sm_90"); }\\n'
                'void gqk(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gqk needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}, 1>(W, x, out, M, N, K); return; }}")
    klines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQK_CONFIGS.items()):
        klines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                      f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}, 2>(W, x, out, M, N, K); return; }}")
    return (GQX_SRC.replace("@@GQX_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))
            .replace("@@GQK_DISPATCH@@", "\\n  ".join(dict.fromkeys(klines))))''',
)

replace_once(
    '''def _run_gqx(x, layer, m, n, k, state):
    nwg, cl, ns = GQX_CONFIGS[(layer, _gqx_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gqx(state["raw"][layer], x, out, m, n, k, _gqx_bn(m), nwg, cl, ns)
    return out''',
    '''def _run_gqx(x, layer, m, n, k, state):
    nwg, cl, ns = GQX_CONFIGS[(layer, _gqx_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gqx(state["raw"][layer], x, out, m, n, k, _gqx_bn(m), nwg, cl, ns)
    return out


def _run_gqk(x, layer, m, n, k, state):
    nwg, cl, ns = GQK_CONFIGS[(layer, _gqx_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gqk(state["raw"][layer], x, out, m, n, k, _gqx_bn(m), nwg, cl, ns)
    return out''',
)

replace_once(
    '''    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
    if kind == "cgemv":''',
    '''    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
    if kind == "gqk":
        return _run_gqk(x, layer, m, n, k, state)
    if kind == "cgemv":''',
)

replace_once(
    '''    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
    if layer == "router"''',
    '''    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
    if (USE_GQK and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQK_CONFIGS
            and _route_key(layer, m, "gqk") not in state["disabled"]):
        c.insert(0, ("gqk",))
    if layer == "router"''',
)

replace_once(
    '''    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    return ch''',
    '''    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    if ch[0] == "gqk" and (layer, _gqx_bn(m)) not in GQK_CONFIGS:
        return None
    return ch''',
)

# Both extension builders exist in the historical composite; change both lists
# and names so no old cache entry can mask this CUDA source.
if text.count('"gq", "gqx", "rgemv"') != 1:
    raise RuntimeError("first extension function list changed")
text = text.replace('"gq", "gqx", "rgemv"', '"gq", "gqx", "gqk", "rgemv"', 1)
if text.count('"gqx", "rgemv", "sgemv"') != 1:
    raise RuntimeError("production extension function list changed")
text = text.replace('"gqx", "rgemv", "sgemv"', '"gqx", "gqk", "rgemv", "sgemv"', 1)
text = text.replace('okc_bf16proj_v141_', 'okc_bf16proj_v177_', 1)
text = text.replace('okc_bf16proj_v105_cg_', 'okc_bf16proj_v177_cg_', 1)

OUTPUT.write_text(text)
print(OUTPUT)
