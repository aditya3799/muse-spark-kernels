#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cutlass/numeric_types.h>

#include <cstdio>

using namespace cute;

int main() {
  bool ok = true;
  {
    using Element = cutlass::float_e4m3_t;
    auto layout = tile_to_shape(
        GMMA::Layout_K_SW64_Atom<Element>{}, make_shape(_64{}, _64{}));
    for (int row = 0; row < 64; ++row) {
      for (int col = 0; col < 64; ++col) {
        const int linear = row * 64 + col;
        const int expected = linear ^ ((linear & 0x180) >> 3);
        const int actual = int(layout(row, col));
        if (actual != expected) {
          std::printf("FP8 mismatch row=%d col=%d actual=%d expected=%d\n",
                      row, col, actual, expected);
          ok = false;
          goto fp8_done;
        }
      }
    }
  fp8_done:;
  }
  {
    using Element = cutlass::float_e4m3_t;
    auto layout = tile_to_shape(
        GMMA::Layout_K_SW64_Atom<Element>{}, make_shape(_8{}, _64{}));
    int mismatches = 0;
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 64; ++col) {
        const int linear = row * 64 + col;
        const int expected = linear ^ ((linear & 0x180) >> 3);
        mismatches += int(layout(row, col)) != expected;
      }
    }
    std::printf("FP8 8x64 swizzle: %d mismatches\n", mismatches);
    if (mismatches != 0) ok = false;
  }
  {
    using Element = cutlass::half_t;
    auto layout = tile_to_shape(
        GMMA::Layout_K_SW32_Atom<Element>{}, make_shape(_64{}, _16{}));
    int mismatches_bit3 = 0;
    int mismatches_bit6 = 0;
    int mismatches_bit7 = 0;
    for (int row = 0; row < 64; ++row) {
      for (int col = 0; col < 16; ++col) {
        const int linear = row * 16 + col;
        const int actual = int(layout(row, col));
        mismatches_bit3 += actual != (linear ^ ((linear & 0x8) >> 3));
        mismatches_bit6 += actual != (linear ^ ((linear & 0x40) >> 3));
        mismatches_bit7 += actual != (linear ^ ((linear & 0x80) >> 3));
      }
    }
    std::printf("F16 candidates: bit3=%d bit6=%d bit7=%d mismatches\n",
                mismatches_bit3, mismatches_bit6, mismatches_bit7);
    if (mismatches_bit7 != 0) ok = false;
  }

  std::printf("swizzle_formulas=%s\n", ok ? "OK" : "FAILED");

  // Print the useful row (0..7) coordinates owned by warp 0. These layouts
  // verify the register-to-(row,column) formulas used by the raw PTX kernel.
  {
    auto mma = make_tiled_mma(
        SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_SS_TN<>{});
    auto identity = make_identity_tensor(make_shape(_64{}, _8{}));
    for (int tid = 0; tid < 128; ++tid) {
      auto fragment = mma.get_thread_slice(tid).partition_C(identity);
      std::printf("XQA tid=%d:", tid);
      for (int i = 0; i < size(fragment); ++i) {
        auto coord = fragment(i);
        std::printf(" r%d=(%d,%d)", i, int(get<0>(coord)), int(get<1>(coord)));
      }
      std::printf("\n");
    }
  }
  {
    auto mma = make_tiled_mma(
        SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_SS_TN<>{});
    auto identity = make_identity_tensor(make_shape(_64{}, _16{}));
    for (int tid = 0; tid < 32; ++tid) {
      auto fragment = mma.get_thread_slice(tid).partition_C(identity);
      std::printf("QK tid=%d:", tid);
      for (int i = 0; i < size(fragment); ++i) {
        auto coord = fragment(i);
        std::printf(" r%d=(%d,%d)", i, int(get<0>(coord)), int(get<1>(coord)));
      }
      std::printf("\n");
    }
  }
  {
    auto mma = make_tiled_mma(
        SM90::GMMA::MMA_64x64x16_F32F16F16_SS<
            GMMA::Major::K, GMMA::Major::K>{});
    auto identity = make_identity_tensor(make_shape(_64{}, _64{}));
    for (int tid = 0; tid < 32; ++tid) {
      auto fragment = mma.get_thread_slice(tid).partition_C(identity);
      std::printf("PV tid=%d:", tid);
      for (int i = 0; i < size(fragment); ++i) {
        auto coord = fragment(i);
        std::printf(" r%d=(%d,%d)", i, int(get<0>(coord)), int(get<1>(coord)));
      }
      std::printf("\n");
    }
  }
  {
    // Register-A coordinates for the FP8 m64n64k32 VP instruction.  Four
    // uint32 registers hold these 16 e4m3 values per thread in printed order.
    auto mma = make_tiled_mma(
        SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_RS_TN<>{});
    auto identity = make_identity_tensor(make_shape(_64{}, _32{}));
    for (int tid = 0; tid < 32; ++tid) {
      auto fragment = mma.get_thread_slice(tid).partition_A(identity);
      std::printf("PVA tid=%d:", tid);
      for (int i = 0; i < size(fragment); ++i) {
        auto coord = fragment(i);
        std::printf(" a%d=(%d,%d)", i, int(get<0>(coord)), int(get<1>(coord)));
      }
      std::printf("\n");
    }
  }
  return ok ? 0 : 1;
}
