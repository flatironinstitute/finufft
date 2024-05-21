//
// Created by mbarbone on 5/17/24.
//
#include <xsimd/xsimd.hpp>
#include <cstdint>
#include <type_traits>
#include <iostream>

template<class T, uint16_t N, uint16_t K = N>
static constexpr auto BestSIMDHelper();

template<class T, uint16_t N>
static constexpr auto GetPaddedSIMDSize();

template<class T>
static uint16_t get_padding(uint16_t ns);

template<class T, uint16_t ns>
static constexpr auto get_padding();

template<class T, uint16_t N, uint16_t K = N>
using BestSIMD = typename decltype(BestSIMDHelper<T, N, K>())::type;


template<class T, uint16_t N, uint16_t K>
static constexpr auto BestSIMDHelper() {
  if constexpr (K==20 && !std::is_void<xsimd::make_sized_batch_t<T, 8>>::value && std::is_same_v<T, double>) {
    return xsimd::make_sized_batch<T, 8>{};
  } else if constexpr (K == 0) {
    return xsimd::make_sized_batch<T, 0>{}; // or whatever base case value is appropriate
  } else if constexpr (!std::is_void<xsimd::make_sized_batch_t<T, K>>::value && N % K == 0) {
    return xsimd::make_sized_batch<T, K>{};
  } else {
    return BestSIMDHelper<T, N, K - 1>();
  }
}

// below there is some trickery to obtain the padded SIMD type to vectorize
// the given number of elements.
// improper use will cause the compiler to either throw an error on the recursion depth
// or on older ones... "compiler internal error please report"
// you have been warned.
//template<class T, uint8_t N, typename = void>
//struct GetPaddedSIMDSize {
//  static constexpr uint8_t value = GetPaddedSIMDSize<T, N + 1>::value;
//};
//
//
//template<class T, uint8_t N>
//struct GetPaddedSIMDSize<T, N, std::enable_if_t<!std::is_void<BestSIMD_t<T, N>>::value>> {
//static constexpr uint8_t value = BestSIMD_t<T, N>::size;
//};



template<class T, uint16_t N>
static constexpr auto GetPaddedSIMDSize() {
  static_assert(N < 128);
  if constexpr (!std::is_void<BestSIMD<T, N>>::value) {
    return BestSIMD<T, N>::size;
  } else {
    return GetPaddedSIMDSize<T, N + 1>();
  }
}

//

// This is a templated version of the above function
// Much cleaner and easier to understand

template<class T, uint16_t ns>
constexpr auto get_padding() {
  constexpr uint16_t width = GetPaddedSIMDSize<T, ns>();
  return ns % width == 0 ? 0 : width - (ns % width);
}

// FIXME: all of this can be templated properly to avoid the switch statement
//        It requires major changes to the codebase though
//        Since 2 <= ns <= 16, It is fine for now

template<class T, uint16_t ns>
static constexpr auto get_padding_helper(uint16_t runtime_ns) {
  if constexpr (ns < 2) {
    return 0;
  } else {
    if (runtime_ns == ns) {
      return get_padding<T, ns>();
    } else {
      return get_padding_helper<T, ns - 1>(runtime_ns);
    }
  }
}

template<class T>
static uint16_t get_padding(uint16_t ns) {
  return get_padding_helper<T, 32>(ns);
}

int main(int argc, char *argv[]) {

  std::cout << "Best SIMD single precision" << std::endl;
  std::cout << "SIMD for " <<  4 << " is " << uint64_t(BestSIMD<float,  4>::size) << std::endl;
  std::cout << "SIMD for " <<  8 << " is " << uint64_t(BestSIMD<float,  8>::size) << std::endl;
  std::cout << "SIMD for " << 12 << " is " << uint64_t(BestSIMD<float, 12>::size) << std::endl;
  std::cout << "SIMD for " << 16 << " is " << uint64_t(BestSIMD<float, 16>::size) << std::endl;
  std::cout << "SIMD for " << 20 << " is " << uint64_t(BestSIMD<float, 20>::size) << std::endl;
  std::cout << "SIMD for " << 24 << " is " << uint64_t(BestSIMD<float, 24>::size) << std::endl;
  std::cout << "SIMD for " << 28 << " is " << uint64_t(BestSIMD<float, 28>::size) << std::endl;
  std::cout << "SIMD for " << 32 << " is " << uint64_t(BestSIMD<float, 32>::size) << std::endl;

  std::cout << "Best SIMD double precision" << std::endl;
  std::cout << "SIMD for " <<  4 << " is " << uint64_t(BestSIMD<double,  4>::size)  << std::endl;
  std::cout << "SIMD for " <<  8 << " is " << uint64_t(BestSIMD<double,  8>::size)  << std::endl;
  std::cout << "SIMD for " << 12 << " is " << uint64_t(BestSIMD<double, 12>::size)  << std::endl;
  std::cout << "SIMD for " << 16 << " is " << uint64_t(BestSIMD<double, 16>::size)  << std::endl;
  std::cout << "SIMD for " << 20 << " is " << uint64_t(BestSIMD<double, 20>::size)  << std::endl;
  std::cout << "SIMD for " << 24 << " is " << uint64_t(BestSIMD<double, 24>::size)  << std::endl;
  std::cout << "SIMD for " << 28 << " is " << uint64_t(BestSIMD<double, 28>::size)  << std::endl;
  std::cout << "SIMD for " << 32 << " is " << uint64_t(BestSIMD<double, 32>::size)  << std::endl;

  std::cout << "Padded SIMD single precision" << std::endl;
  std::cout << "Padded SIMD for " <<  6 << " is " << uint64_t(GetPaddedSIMDSize<float,  6 >()) << std::endl;
  std::cout << "Padded SIMD for " << 10 << " is " << uint64_t(GetPaddedSIMDSize<float, 10>()) << std::endl;
  std::cout << "Padded SIMD for " << 15 << " is " << uint64_t(GetPaddedSIMDSize<float, 15>()) << std::endl;
  std::cout << "Padded SIMD for " << 18 << " is " << uint64_t(GetPaddedSIMDSize<float, 18>()) << std::endl;
  std::cout << "Padded SIMD for " << 22 << " is " << uint64_t(GetPaddedSIMDSize<float, 22>()) << std::endl;
  std::cout << "Padded SIMD for " << 26 << " is " << uint64_t(GetPaddedSIMDSize<float, 26>()) << std::endl;
  std::cout << "Padded SIMD for " << 30 << " is " << uint64_t(GetPaddedSIMDSize<float, 30>()) << std::endl;
  std::cout << "Padded SIMD for " << 32 << " is " << uint64_t(GetPaddedSIMDSize<float, 32>()) << std::endl;

  std::cout << "Padded SIMD double precision" << std::endl;
  std::cout << "Padded SIMD for " <<  6 << " is " << uint64_t(GetPaddedSIMDSize<float,  6>())  << std::endl;
  std::cout << "Padded SIMD for " << 10 << " is " << uint64_t(GetPaddedSIMDSize<float, 10>())  << std::endl;
  std::cout << "Padded SIMD for " << 15 << " is " << uint64_t(GetPaddedSIMDSize<float, 15>())  << std::endl;
  std::cout << "Padded SIMD for " << 18 << " is " << uint64_t(GetPaddedSIMDSize<float, 18>())  << std::endl;
  std::cout << "Padded SIMD for " << 22 << " is " << uint64_t(GetPaddedSIMDSize<float, 22>())  << std::endl;
  std::cout << "Padded SIMD for " << 26 << " is " << uint64_t(GetPaddedSIMDSize<float, 26>())  << std::endl;
  std::cout << "Padded SIMD for " << 30 << " is " << uint64_t(GetPaddedSIMDSize<float, 30>())  << std::endl;
  std::cout << "Padded SIMD for " << 32 << " is " << uint64_t(GetPaddedSIMDSize<float, 32>())  << std::endl;

  std::cout << "single precision" << std::endl;
  for(auto i = 2; i < 16; i++){
    std::cout << "Padding for " << i*2 << " is " << uint64_t(get_padding<float>(i*2)) << std::endl;
  }

  std::cout << "double precision" << std::endl;
  for(auto i = 2; i < 16; i++){
    std::cout << "Padding for " << i*2 << " is " << uint64_t(get_padding<double>(i*2)) << std::endl;
  }

  std::cout << "single precision" << std::endl;
  std::cout << "Padding for " <<  3 * 2 << " is " << uint64_t(get_padding<float,  3 * 2>()) << std::endl;
  std::cout << "Padding for " <<  5 * 2 << " is " << uint64_t(get_padding<float,  5 * 2>()) << std::endl;
  std::cout << "Padding for " <<  9 * 2 << " is " << uint64_t(get_padding<float,  9 * 2>()) << std::endl;
  std::cout << "Padding for " << 11 * 2 << " is " << uint64_t(get_padding<float, 11 * 2>()) << std::endl;
  std::cout << "Padding for " << 13 * 2 << " is " << uint64_t(get_padding<float, 13 * 2>()) << std::endl;
  std::cout << "Padding for " << 15 * 2 << " is " << uint64_t(get_padding<float, 15 * 2>()) << std::endl;
  std::cout << "double precision" << std::endl;
  std::cout << "Padding for " <<  3*2 << " is " << uint64_t(get_padding<double,  3 * 2>()) << std::endl;
  std::cout << "Padding for " <<  5*2 << " is " << uint64_t(get_padding<double,  5 * 2>()) << std::endl;
  std::cout << "Padding for " <<  7*2 << " is " << uint64_t(get_padding<double,  7 * 2>()) << std::endl;
  std::cout << "Padding for " <<  9*2 << " is " << uint64_t(get_padding<double,  9 * 2>()) << std::endl;
  std::cout << "Padding for " << 11*2 << " is " << uint64_t(get_padding<double, 11 * 2>()) << std::endl;
  std::cout << "Padding for " << 13*2 << " is " << uint64_t(get_padding<double, 13 * 2>()) << std::endl;
  std::cout << "Padding for " << 15*2 << " is " << uint64_t(get_padding<double, 15 * 2>()) << std::endl;

  return 0;
}