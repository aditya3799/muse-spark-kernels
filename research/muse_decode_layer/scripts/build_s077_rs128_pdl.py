"""Build S077: test end-to-end PDL around the v150 RS128 T=128 path.

S076 proved that the exact v150 projection kernels are valid for QKV, pooler,
and unpooler, but its weighted verify-16 result was effectively neutral.  S052
does not enable PDL at T=128 (PDL_MAX_T is 32), so this candidate deliberately
tests a new scheduling regime.  It gives the copied RS128 kernel the same
programmatic-dependent-launch contract used by the decoder's small-T SK2
kernels:

* issue two dependency-free weight stages before griddepcontrol.wait;
* wait on every thread before reading the producer's activation;
* release dependent kernels before the output epilogue; and
* opt the launch into programmatic stream serialization.

The existing prepare-time validation calls keep PDL disabled. Runtime T=128
calls enable it, and PDL_MAX_T is raised to 128 so that the surrounding
attention, boundary, router, routing, and MoE launches participate in a
consistent producer/consumer chain. This is an experiment, not a promotion;
only a whole-layer H200 check can establish whether the new overlap is a win.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_s076_rs128_pool_unpool.py"
OUT = ROOT / "submission_s077_rs128_pdl.py"

BASE_SHA256 = "b6ae3bee687027949cd2670b2f0f0ec61d2094d092115fedb4b5a00df27d5c91"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("validated S076 hash mismatch; refusing to extend")

base = BASE.read_text(encoding="utf-8")
tree = ast.parse(base)
payload_node = None
payload = None
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    if any(isinstance(target, ast.Name) and target.id == "_V150_CUDA_PAYLOAD"
           for target in node.targets):
        payload_node = node
        payload = ast.literal_eval(node.value)
        break
if payload_node is None or payload is None:
    raise RuntimeError("could not locate the v150 CUDA payload")

cuda = zlib.decompress(base64.b64decode(payload)).decode()

cuda = _replace_once(
    cuda,
    "  int m, N, K, KC, NB, walias, ncnt;\n",
    "  int m, N, K, KC, NB, walias, ncnt, pdl;\n",
    "RqP pdl field",
)

cuda = _replace_once(
    cuda,
    '''    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
    if (EARLY) {
''',
    '''    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
    // PDL prologue: weights are immutable prepare-time data, so two stages can
    // stream while the preceding kernel finishes.  X is its output and is not
    // touched until every thread has executed griddepcontrol.wait below.
    if (p.pdl && !EARLY && WON) {
      const int P = KCS < 2 ? KCS : 2;
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n" : "=l"(pw));
      for (int s = 0; s < P; ++s) {
        rq_expect(full + 8 * s, WB);
        if (HINT) rq_bulkh(wring + s * WBS, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WBS, src + (size_t)s * 12288, WB, full + 8 * s);
      }
    }
    if (EARLY) {
''',
    "dependency-free weight prologue",
)

cuda = _replace_once(
    cuda,
    '''  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
''',
    '''  __syncthreads();
  if (p.pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
''',
    "all-thread dependency wait",
)

producer_start = '''      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
'''
if cuda.count(producer_start) != 2:
    raise RuntimeError(f"producer start: expected two matches, found {cuda.count(producer_start)}")
cuda = cuda.replace(
    producer_start,
    '''      const int se = p.pdl ? (KCS < 2 ? KCS : 2) : (EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0);
      int slot = (se / MS) % ST, r = (se / MS) / ST;
''',
    1,
)

cuda = _replace_once(
    cuda,
    '''  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
''',
    '''  if (p.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
''',
    "dependent release",
)

cuda = _replace_once(
    cuda,
    '''  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
''',
    '''  if (p.pdl) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(p.NB * ((FL & 16) ? 2 : 1), KS);
    lc.blockDim = dim3((FL & 1024) ? 256 : 384);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = &attr;
    lc.numAttrs = 1;
    const cudaError_t le = cudaLaunchKernelEx(&lc, k, p, tx, to);
    if (le != cudaSuccess) return (int)le;
  } else {
    k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  }
  const cudaError_t e = cudaGetLastError();
''',
    "PDL launcher",
)

cuda = _replace_once(
    cuda,
    '''int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info) {
  RqP p = {};
''',
    '''int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info, int64_t pdl) {
  RqP p = {};
  p.pdl = pdl ? 1 : 0;
''',
    "rqrun pdl argument",
)

new_payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
start = payload_node.value.col_offset
line_start = sum(len(line) + 1 for line in base.splitlines()[:payload_node.value.lineno - 1])
value_start = line_start + start
value_end = line_start + payload_node.value.end_col_offset
base = base[:value_start] + repr(new_payload) + base[value_end:]

base = _replace_once(
    base,
    '_V150_CPP = "int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);"',
    '_V150_CPP = "int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t);"',
    "C++ binding declaration",
)
base = _replace_once(base, "PDL_MAX_T = 32", "PDL_MAX_T = 128", "T=128 PDL chain")
base = _replace_once(
    base,
    "def _v150_call(state, plan, x, out=None, info=0):",
    "def _v150_call(state, plan, x, out=None, info=0, pdl=False):",
    "Python call signature",
)
base = _replace_once(
    base,
    '''        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
''',
    '''        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info, int(bool(pdl)),
''',
    "Python runtime pdl argument",
)
base = _replace_once(
    base,
    '''        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
''',
    '''        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind, 0,
''',
    "Python info pdl argument",
)
base = _replace_once(
    base,
    '''    return _v150_call(vstate, vstate["plan"], x)
''',
    '''    return _v150_call(vstate, vstate["plan"], x, pdl=True)
''',
    "QKV runtime PDL",
)
base = _replace_once(
    base,
    '''    return _v150_call(vstate, plan, x)
''',
    '''    return _v150_call(vstate, plan, x, pdl=True)
''',
    "pool/unpool runtime PDL",
)

banner = '''
# ---------------------------------------------------------------------------
# S077: experimental v150 RS128 programmatic dependent launch at T=128.
# Two immutable weight stages issue before the dependency wait; all activation
# reads remain ordered, and prepare-time numerical validation runs without PDL.
# ---------------------------------------------------------------------------
'''
base = base.rstrip() + "\n" + banner

compile(base, str(OUT), "exec")
OUT.write_bytes(base.encode())
print(OUT)
print(hashlib.sha256(base.encode()).hexdigest())
print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest())
