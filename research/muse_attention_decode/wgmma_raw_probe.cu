#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t make_wgmma_desc_b64(const void* ptr) {
  const uint64_t start =
      (static_cast<uint64_t>(__cvta_generic_to_shared(ptr)) >> 4) & 0x3fff;
  return start | (uint64_t(1) << 16) | (uint64_t(32) << 32) |
         (uint64_t(2) << 62);
}

__device__ __forceinline__ uint64_t make_wgmma_desc_b32(const void* ptr) {
  const uint64_t start =
      (static_cast<uint64_t>(__cvta_generic_to_shared(ptr)) >> 4) & 0x3fff;
  return start | (uint64_t(1) << 16) | (uint64_t(16) << 32) |
         (uint64_t(3) << 62);
}

__device__ __forceinline__ void wgmma_arrive() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

__device__ __forceinline__ void wgmma_qk_f32_e4m3(
    uint64_t desc_a, uint64_t desc_b, float (&d)[8], int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(desc_a), "l"(desc_b), "r"(accumulate));
}

__device__ __forceinline__ void wgmma_pv_f32_f16(
    uint64_t desc_a, uint64_t desc_b, float (&d)[32]) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%32,%33,p,1,1,0,0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(1));
}

extern "C" __global__ __launch_bounds__(128)
void wgmma_raw_probe(float* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ __align__(512) uint8_t q[64 * 64];
  __shared__ __align__(512) uint8_t k[16 * 64];
  __shared__ __align__(256) __half p[64 * 16];
  __shared__ __align__(256) __half v[64 * 16];
  float scores[8] = {};
  float values[32] = {};
  const auto q_desc = make_wgmma_desc_b64(q);
  const auto k_desc = make_wgmma_desc_b64(k);
  const auto p_desc = make_wgmma_desc_b32(p);
  const auto v_desc = make_wgmma_desc_b32(v);
  wgmma_arrive();
  wgmma_qk_f32_e4m3(q_desc, k_desc, scores, 0);
  wgmma_qk_f32_e4m3(q_desc + 2, k_desc + 2, scores, 1);
  wgmma_commit();
  wgmma_wait();
  wgmma_arrive();
  wgmma_pv_f32_f16(p_desc, v_desc, values);
  wgmma_commit();
  wgmma_wait();
  if (threadIdx.x < 8) output[threadIdx.x] = scores[threadIdx.x];
  if (threadIdx.x < 32) output[8 + threadIdx.x] = values[threadIdx.x];
#else
  (void)output;
#endif
}
