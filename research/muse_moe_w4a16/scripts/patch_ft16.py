with open("submission.py") as f:
    text = f.read()

idx_main = text.find("ALT_SRCS['v167_main'] = '")
idx_defines = text.find("V167_MAIN_DEFINES", idx_main)
v167_main = text[idx_main:idx_defines]

# 1. WG<3> and WG<4>
old_wg34 = """template <> struct WG<3> {\\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\\n    asm volatile(\\n        "{\\\\n .reg .pred p;\\\\n setp.ne.b32 p, 1, 0;\\\\n"\\n        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "\\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\\\n}\\\\n"\\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])\\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\\n        : "l"(desc));\\n  }\\n};\\ntemplate <> struct WG<4> {\\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\\n    asm volatile(\\n        "{\\\\n .reg .pred p;\\\\n setp.ne.b32 p, 1, 0;\\\\n"\\n        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\\\n}\\\\n"\\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])\\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\\n        : "l"(desc));\\n  }\\n};\\n#if (OKC_FT_T / 8) >= 5"""

new_wg34 = """#if (OKC_FT_T / 8) >= 3\\ntemplate <> struct WG<3> {\\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\\n    asm volatile(\\n        "{\\\\n .reg .pred p;\\\\n setp.ne.b32 p, 1, 0;\\\\n"\\n        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "\\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\\\n}\\\\n"\\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])\\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\\n        : "l"(desc));\\n  }\\n};\\n#endif\\n#if (OKC_FT_T / 8) >= 4\\ntemplate <> struct WG<4> {\\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\\n    asm volatile(\\n        "{\\\\n .reg .pred p;\\\\n setp.ne.b32 p, 1, 0;\\\\n"\\n        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\\\n}\\\\n"\\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])\\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\\n        : "l"(desc));\\n  }\\n};\\n#endif\\n#if (OKC_FT_T / 8) >= 5"""

assert old_wg34 in v167_main, "old_wg34 not found in v167_main"
v167_main = v167_main.replace(old_wg34, new_wg34, 1)

# 2. fat_compute_wg switch
old_fcwg = """case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\\n          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\\n#if (OKC_FT_T / 8) >= 5"""
new_fcwg = """#if (OKC_FT_T / 8) >= 3\\n          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\\n#endif\\n#if (OKC_FT_T / 8) >= 4\\n          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\\n#endif\\n#if (OKC_FT_T / 8) >= 5"""

assert old_fcwg in v167_main, "old_fcwg not found in v167_main"
v167_main = v167_main.replace(old_fcwg, new_fcwg, 1)

# 3. fat_compute switch
old_fc = """case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;\\n          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;\\n#if (OKC_FT_T / 8) >= 5"""
new_fc = """#if (OKC_FT_T / 8) >= 3\\n          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;\\n#endif\\n#if (OKC_FT_T / 8) >= 4\\n          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;\\n#endif\\n#if (OKC_FT_T / 8) >= 5"""

assert old_fc in v167_main, "old_fc not found in v167_main"
v167_main = v167_main.replace(old_fc, new_fc, 1)

# 4. WGSMMA<3> and WGSMMA<4>
old_wgmma34 = """template <> struct WGSMMA<3> {\\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\\\\n"\\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\\\\n"\\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n};\\ntemplate <> struct WGSMMA<4> {\\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\\\\n"\\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\\\\n"\\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n};\\n#if (OKC_FT_T / 8) >= 5"""

new_wgmma34 = """#if (OKC_FT_T / 8) >= 3\\ntemplate <> struct WGSMMA<3> {\\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\\\\n"\\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\\\\n"\\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n};\\n#endif\\n#if (OKC_FT_T / 8) >= 4\\ntemplate <> struct WGSMMA<4> {\\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\\\\n"\\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\\\\n"\\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\\n  }\\n};\\n#endif\\n#if (OKC_FT_T / 8) >= 5"""

assert old_wgmma34 in v167_main, "old_wgmma34 not found in v167_main"
v167_main = v167_main.replace(old_wgmma34, new_wgmma34, 1)

# 5. wgs_seg switch
old_wgs_switch = """case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;\\n      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;\\n#if (OKC_FT_T / 8) >= 5"""
new_wgs_switch = """#if (OKC_FT_T / 8) >= 3\\n      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;\\n#endif\\n#if (OKC_FT_T / 8) >= 4\\n      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;\\n#endif\\n#if (OKC_FT_T / 8) >= 5"""

assert old_wgs_switch in v167_main, "old_wgs_switch not found in v167_main"
v167_main = v167_main.replace(old_wgs_switch, new_wgs_switch, 1)

# Assemble patched text
text = text[:idx_main] + v167_main + text[idx_defines:]

# 6. Change V167_M32_DEFINES to -DOKC_FT_T=16
old_m32 = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=32', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1')"
new_m32 = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=16', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1')"

assert old_m32 in text, "old_m32 not found in text"
text = text.replace(old_m32, new_m32, 1)

# Write to submission.py
with open("submission.py", "w") as f:
    f.write(text)

import ast
ast.parse(text)
print("Successfully patched submission.py for FT_T=16! Syntax OK!")
