"""Generate v193: RK ring-4 for m<=128 and protected ring-6 for m=256."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
OUTPUT = ROOT / "submission_plain_router_hybrid_ring_v193.py"

text = BASE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    if text.count(old) != 1:
        raise RuntimeError(f"unexpected anchor count for {old[:80]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "__global__ void __launch_bounds__(192, 1) rk_kernel(",
    "template <int QS>\n__global__ void __launch_bounds__(192, 1) rk_kernel(",
)
replace_once(
    "  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;",
    "  constexpr int BM = 64, BN = 64, NJ = 4;",
)

old_launch = """  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
"""
new_launch = """  static bool attr4 = false, attr6 = false;
  if (M <= 128) {
    constexpr int QS = 4;
    const int smem = 1024 + QS * 24576 + 8 * (2 * QS + 1) + 64;
    if (!attr4) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel<QS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr4 = true; }
    rk_kernel<QS><<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  } else {
    constexpr int QS = 6;
    const int smem = 1024 + QS * 24576 + 8 * (2 * QS + 1) + 64;
    if (!attr6) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel<QS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr6 = true; }
    rk_kernel<QS><<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  }
"""
replace_once(old_launch, new_launch)

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
