#include <array>
#include <intrin.h>
#include <iostream>

bool is_sse_supported() {
  std::array<int, 4> cpui;
  __cpuid(cpui.data(), 1);
  return (cpui[3] & (1 << 25)) != 0;
}

bool is_avx2_supported() {
  std::array<int, 4> cpui;
  __cpuid(cpui.data(), 1);
  return (cpui[2] & (1 << 5)) != 0;
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
  } else if (is_sse_supported()) {
    std::cout << "SSE";
  } else {
    std::cout << "NONE";
  }
  return 0;
}