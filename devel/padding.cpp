//
// Created by mbarbone on 5/17/24.
//
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <xsimd/xsimd.hpp>

template<class T, uint16_t N, uint16_t K = N> static constexpr auto BestSIMDHelper();

template<class T, uint16_t N> static constexpr auto GetPaddedSIMDWidth();

template<class T> static uint16_t get_padding(uint16_t ns);

template<class T, uint16_t ns> static constexpr auto get_padding();

template<class T, uint16_t N>
using BestSIMD = typename decltype(BestSIMDHelper<T, N, xsimd::batch<T>::size>())::type;

template<class T, uint16_t N = 1> static constexpr uint16_t min_simd_width();

template<class T, uint16_t N = min_simd_width<T>()> constexpr uint16_t max_simd_width();

template<class T, uint16_t N> static constexpr auto find_optimal_simd_width();

// below there is some trickery to obtain the padded SIMD type to vectorize
// the given number of elements.
// improper use will cause the compiler to either throw an error on the recursion depth
// or on older ones... "compiler internal error please report"
// you have been warned.

template<class T, uint16_t N, uint16_t K> static constexpr auto BestSIMDHelper() {
  if constexpr (N % K == 0) { // returns void in the worst case
    return xsimd::make_sized_batch<T, K>{};
  } else {
    return BestSIMDHelper<T, N, (K >> 1)>();
  }
}

template<class T, uint16_t N> constexpr uint16_t min_simd_width() {
  if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
    return min_simd_width<T, N * 2>();
  } else {
    return N;
  }
};

template<class T, uint16_t N> constexpr uint16_t max_simd_width() {
  if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, N * 2>>) {
    return max_simd_width<T, N * 2>();
  } else {
    return N;
  }
};

template<class T, uint16_t N> static constexpr auto find_optimal_simd_width() {
  uint16_t min_iterations     = N;
  uint16_t optimal_batch_size = 1;
  for (uint16_t batch_size = min_simd_width<T>(); batch_size <= xsimd::batch<T>::size;
       batch_size *= 2) {
    uint16_t iterations = (N + batch_size - 1) / batch_size;
    if (iterations < min_iterations) {
      min_iterations     = iterations;
      optimal_batch_size = batch_size;
    }
  }
  return optimal_batch_size;
}

template<class T, uint16_t N> static constexpr auto GetPaddedSIMDWidth() {
  static_assert(N < 128);
  return xsimd::make_sized_batch<T, find_optimal_simd_width<T, N>()>::type::size;
}

template<class T, uint16_t ns> static constexpr auto get_padding() {
  constexpr uint16_t width = GetPaddedSIMDWidth<T, ns>();
  return ((ns + width - 1) & (-width)) - ns;
}

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

template<class T> static uint16_t get_padding(uint16_t ns) {
  return get_padding_helper<T, 32>(ns);
}

template<class T> std::ostream &print(T arg) {
  typename T::value_type sum = 0;
  for (const auto &elem : arg) {
    std::cout << elem << " ";
    sum += elem;
  }
  std::cout << "sum is " << sum;
  return std::cout;
}

template<uint16_t low, uint16_t high> constexpr uint16_t po2_in_between() {
  std::uint16_t result = 0;
  for (auto i = low; i <= high; i <<= 1) {
    result++;
  }
  return result;
}

template<class T, uint16_t N> constexpr auto mixed_vectors() {
  constexpr auto min_batch = min_simd_width<T>();
  constexpr auto max_batch = max_simd_width<T>();
  // compute all the power of 2 between min_batch and max_batch

  std::array<uint16_t, po2_in_between<min_batch, max_batch>() + 1> batch_sizes{1};
  for (uint16_t i = 1; i < batch_sizes.size(); i++) {
    batch_sizes[i] = min_batch << (i - 1);
  }
  print(batch_sizes);
  std::array<uint16_t, N + 1> chosen_batch_sizes{0}, dp{N + 1};
  dp[0] = 0; // 0 amount requires 0 coins

  for (uint16_t i = 0; i < N + 1; ++i) {
    for (const auto batch_size : batch_sizes) {
      if (batch_size <= i && dp[i - batch_size] + 1 < dp[i]) {
        dp[i]                 = dp[i - batch_size] + 1;
        chosen_batch_sizes[i] = batch_size;
      }
    }
  }
  // Build the sequence of coins that fit in N
  std::array<uint16_t, N> sequence{0};
  auto index = 0;
  for (int i = N; i > 0; i -= chosen_batch_sizes[i]) {
    sequence[index++] = chosen_batch_sizes[i];
  }
  // return the not zero elements in the sequence
  return sequence;
}

int main(int argc, char *argv[]) {
  std::cout << "Min batch size for single precision is "
            << uint64_t(min_simd_width<float>()) << std::endl;
  std::cout << "Max batch size for single precision is "
            << uint64_t(max_simd_width<float>()) << std::endl;
  std::cout << "Min batch size for double precision is "
            << uint64_t(min_simd_width<double>()) << std::endl;
  std::cout << "Max batch size for double precision is "
            << uint64_t(max_simd_width<double>()) << std::endl;

  std::cout << "Best SIMD single precision" << std::endl;
  std::cout << "SIMD for " << 4 << " is " << uint64_t(BestSIMD<float, 4>::size)
            << std::endl;
  std::cout << "SIMD for " << 8 << " is " << uint64_t(BestSIMD<float, 8>::size)
            << std::endl;
  std::cout << "SIMD for " << 12 << " is " << uint64_t(BestSIMD<float, 12>::size)
            << std::endl;
  std::cout << "SIMD for " << 16 << " is " << uint64_t(BestSIMD<float, 16>::size)
            << std::endl;
  std::cout << "SIMD for " << 20 << " is " << uint64_t(BestSIMD<float, 20>::size)
            << std::endl;
  std::cout << "SIMD for " << 24 << " is " << uint64_t(BestSIMD<float, 24>::size)
            << std::endl;
  std::cout << "SIMD for " << 28 << " is " << uint64_t(BestSIMD<float, 28>::size)
            << std::endl;
  std::cout << "SIMD for " << 32 << " is " << uint64_t(BestSIMD<float, 32>::size)
            << std::endl;

  std::cout << "Best SIMD double precision" << std::endl;
  std::cout << "SIMD for " << 4 << " is " << uint64_t(BestSIMD<double, 4>::size)
            << std::endl;
  std::cout << "SIMD for " << 8 << " is " << uint64_t(BestSIMD<double, 8>::size)
            << std::endl;
  std::cout << "SIMD for " << 12 << " is " << uint64_t(BestSIMD<double, 12>::size)
            << std::endl;
  std::cout << "SIMD for " << 16 << " is " << uint64_t(BestSIMD<double, 16>::size)
            << std::endl;
  std::cout << "SIMD for " << 20 << " is " << uint64_t(BestSIMD<double, 20>::size)
            << std::endl;
  std::cout << "SIMD for " << 24 << " is " << uint64_t(BestSIMD<double, 24>::size)
            << std::endl;
  std::cout << "SIMD for " << 28 << " is " << uint64_t(BestSIMD<double, 28>::size)
            << std::endl;
  std::cout << "SIMD for " << 32 << " is " << uint64_t(BestSIMD<double, 32>::size)
            << std::endl;

  std::cout << "Padded SIMD single precision" << std::endl;
  std::cout << "Padded SIMD for " << 4 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 4>()) << std::endl;
  std::cout << "Padded SIMD for " << 6 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 6>()) << std::endl;
  std::cout << "Padded SIMD for " << 10 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 10>()) << std::endl;
  std::cout << "Padded SIMD for " << 12 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 12>()) << std::endl;
  std::cout << "Padded SIMD for " << 15 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 15>()) << std::endl;
  std::cout << "Padded SIMD for " << 18 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 18>()) << std::endl;
  std::cout << "Padded SIMD for " << 22 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 22>()) << std::endl;
  std::cout << "Padded SIMD for " << 26 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 26>()) << std::endl;
  std::cout << "Padded SIMD for " << 30 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 30>()) << std::endl;
  std::cout << "Padded SIMD for " << 32 << " is "
            << uint64_t(GetPaddedSIMDWidth<float, 32>()) << std::endl;

  std::cout << "Padded SIMD double precision" << std::endl;
  std::cout << "Padded SIMD for " << 4 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 4>()) << std::endl;
  std::cout << "Padded SIMD for " << 6 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 6>()) << std::endl;
  std::cout << "Padded SIMD for " << 10 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 10>()) << std::endl;
  std::cout << "Padded SIMD for " << 12 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 12>()) << std::endl;
  std::cout << "Padded SIMD for " << 15 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 15>()) << std::endl;
  std::cout << "Padded SIMD for " << 18 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 18>()) << std::endl;
  std::cout << "Padded SIMD for " << 22 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 22>()) << std::endl;
  std::cout << "Padded SIMD for " << 26 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 26>()) << std::endl;
  std::cout << "Padded SIMD for " << 30 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 30>()) << std::endl;
  std::cout << "Padded SIMD for " << 32 << " is "
            << uint64_t(GetPaddedSIMDWidth<double, 32>()) << std::endl;

  std::cout << "single precision" << std::endl;
  for (auto i = 2; i < 16; i++) {
    std::cout << "Padding for " << i * 2 << " is " << uint64_t(get_padding<float>(i * 2))
              << std::endl;
  }

  std::cout << "double precision" << std::endl;
  for (auto i = 2; i < 16; i++) {
    std::cout << "Padding for " << i * 2 << " is " << uint64_t(get_padding<double>(i * 2))
              << std::endl;
  }

  std::cout << "single precision" << std::endl;
  std::cout << "Padding for " << 3 * 2 << " is " << uint64_t(get_padding<float, 3 * 2>())
            << std::endl;
  std::cout << "Padding for " << 5 * 2 << " is " << uint64_t(get_padding<float, 5 * 2>())
            << std::endl;
  std::cout << "Padding for " << 9 * 2 << " is " << uint64_t(get_padding<float, 9 * 2>())
            << std::endl;
  std::cout << "Padding for " << 11 * 2 << " is "
            << uint64_t(get_padding<float, 11 * 2>()) << std::endl;
  std::cout << "Padding for " << 13 * 2 << " is "
            << uint64_t(get_padding<float, 13 * 2>()) << std::endl;
  std::cout << "Padding for " << 15 * 2 << " is "
            << uint64_t(get_padding<float, 15 * 2>()) << std::endl;
  std::cout << "double precision" << std::endl;
  std::cout << "Padding for " << 3 * 2 << " is " << uint64_t(get_padding<double, 3 * 2>())
            << std::endl;
  std::cout << "Padding for " << 5 * 2 << " is " << uint64_t(get_padding<double, 5 * 2>())
            << std::endl;
  std::cout << "Padding for " << 7 * 2 << " is " << uint64_t(get_padding<double, 7 * 2>())
            << std::endl;
  std::cout << "Padding for " << 9 * 2 << " is " << uint64_t(get_padding<double, 9 * 2>())
            << std::endl;
  std::cout << "Padding for " << 11 * 2 << " is "
            << uint64_t(get_padding<double, 11 * 2>()) << std::endl;
  std::cout << "Padding for " << 13 * 2 << " is "
            << uint64_t(get_padding<double, 13 * 2>()) << std::endl;
  std::cout << "Padding for " << 15 * 2 << " is "
            << uint64_t(get_padding<double, 15 * 2>()) << std::endl;

  return 0;
}