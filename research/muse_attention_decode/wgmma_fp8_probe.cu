// SM90a cross-compilation probe for the decode attention QK tile.
// This is intentionally standalone: it validates the exact dense FP8
// WGMMA instruction shape needed by the Hopper-only path.
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

extern "C" __global__ __launch_bounds__(128)
void wgmma_fp8_qk_probe(
    cutlass::float_e4m3_t const* q,
    cutlass::float_e4m3_t const* k,
    float* scores) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using Element = cutlass::float_e4m3_t;
  constexpr int M = 64;
  constexpr int N = 16;
  constexpr int K = 64;

  __shared__ alignas(128) uint8_t storage[(M + N) * K];
  Element* q_smem = reinterpret_cast<Element*>(storage);
  Element* k_smem = q_smem + M * K;

  auto q_layout = tile_to_shape(
      GMMA::Layout_K_SW64_Atom<Element>{}, make_shape(Int<M>{}, Int<K>{}));
  auto k_layout = tile_to_shape(
      GMMA::Layout_K_SW64_Atom<Element>{}, make_shape(Int<N>{}, Int<K>{}));
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), q_layout);
  Tensor sK = make_tensor(make_smem_ptr(k_smem), k_layout);

  // Populate the logical matrices through their swizzled tensor views. The
  // input arrays are ordinary row-major MxK and NxK matrices.
  for (int i = int(threadIdx.x); i < M * K; i += int(blockDim.x)) {
    sQ(i / K, i % K) = q[i];
  }
  for (int i = int(threadIdx.x); i < N * K; i += int(blockDim.x)) {
    sK(i / K, i % K) = k[i];
  }
  __syncthreads();

  auto mma = make_tiled_mma(SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_SS_TN<>{});
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tQsQ = thr_mma.partition_A(sQ);
  Tensor tKsK = thr_mma.partition_B(sK);
  Tensor gScores = make_tensor(make_gmem_ptr(scores),
                               make_shape(Int<M>{}, Int<N>{}),
                               make_stride(Int<1>{}, Int<M>{}));
  Tensor tSgS = thr_mma.partition_C(gScores);
  Tensor accum = thr_mma.make_fragment_C(tSgS);
  clear(accum);
  Tensor q_desc = thr_mma.make_fragment_A(tQsQ);
  Tensor k_desc = thr_mma.make_fragment_B(tKsK);

  warpgroup_arrive();
  gemm(mma, q_desc, k_desc, accum);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  copy(accum, tSgS);
#else
  (void)q;
  (void)k;
  (void)scores;
#endif
}

// The second attention GEMM: FP16 probabilities (64x16, only 8 useful rows)
// times an FP16 value tile (64x16 interpreted as N x K), accumulating FP32.
extern "C" __global__ __launch_bounds__(128)
void wgmma_f16_pv_probe(
    cutlass::half_t const* probabilities,
    cutlass::half_t const* values,
    float* output) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using Element = cutlass::half_t;
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 16;

  __shared__ alignas(128) uint8_t storage[(M + N) * K * sizeof(Element)];
  Element* p_smem = reinterpret_cast<Element*>(storage);
  Element* v_smem = p_smem + M * K;

  auto p_layout = tile_to_shape(
      GMMA::Layout_K_SW32_Atom<Element>{}, make_shape(Int<M>{}, Int<K>{}));
  auto v_layout = tile_to_shape(
      GMMA::Layout_K_SW32_Atom<Element>{}, make_shape(Int<N>{}, Int<K>{}));
  Tensor sP = make_tensor(make_smem_ptr(p_smem), p_layout);
  Tensor sV = make_tensor(make_smem_ptr(v_smem), v_layout);
  for (int i = int(threadIdx.x); i < M * K; i += int(blockDim.x)) {
    sP(i / K, i % K) = probabilities[i];
  }
  for (int i = int(threadIdx.x); i < N * K; i += int(blockDim.x)) {
    sV(i / K, i % K) = values[i];
  }
  __syncthreads();

  auto mma = make_tiled_mma(
      SM90::GMMA::MMA_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tPsP = thr_mma.partition_A(sP);
  Tensor tVsV = thr_mma.partition_B(sV);
  Tensor gOutput = make_tensor(make_gmem_ptr(output),
                               make_shape(Int<M>{}, Int<N>{}),
                               make_stride(Int<1>{}, Int<M>{}));
  Tensor tOgO = thr_mma.partition_C(gOutput);
  Tensor accum = thr_mma.make_fragment_C(tOgO);
  clear(accum);
  Tensor p_desc = thr_mma.make_fragment_A(tPsP);
  Tensor v_desc = thr_mma.make_fragment_B(tVsV);

  warpgroup_arrive();
  gemm(mma, p_desc, v_desc, accum);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  copy(accum, tOgO);
#else
  (void)probabilities;
  (void)values;
  (void)output;
#endif
}
