#include <array>
#include <intrin.h>
#include <iostream>

bool is_sse2_supported() {
  std::array<int, 4> cpui;
  __cpuid(cpui.data(), 1);
  return (cpui[3] & (1 << 26)) != 0;
}

bool is_avx_supported() {
  std::array<int, 4> cpui;
  __cpuid(cpui.data(), 1);
  bool osUsesXSAVE_XRSTORE = (cpui[2] & (1 << 27)) != 0;
  bool cpuAVXSupport       = (cpui[2] & (1 << 28)) != 0;
  if (osUsesXSAVE_XRSTORE && cpuAVXSupport) {
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0x6) == 0x6;
  }
  return false;
}

bool is_avx2_supported() {
  std::array<int, 4> cpui;
  __cpuid(cpui.data(), 7);
  return (cpui[1] & (1 << 5)) != 0;
}

bool is_avx512_supported() {
  std::array<int, 4> cpui;
  __cpuidex(cpui.data(), 7, 0);
  return (cpui[1] & (1 << 16)) != 0;
}

int main() {
  if (is_avx512_supported()) {
    std::cout << "AVX512";
  } else if (is_avx2_supported()) {
    std::cout << "AVX2";
  } else if (is_avx_supported()) {
    std::cout << "AVX";
  } else if (is_sse2_supported()) {
    std::cout << "SSE2";
  } else {
    std::cout << "NONE";
  }
  return 0;
}
