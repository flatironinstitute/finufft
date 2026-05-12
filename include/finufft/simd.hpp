#pragma once

#include <finufft_common/defines.h>
#include <finufft_common/kernel.h>

// xsimd configuration: ensure unsupported architectures fall back to emulated.
#include <xsimd/config/xsimd_config.hpp>
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#undef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define XSIMD_WITH_EMULATED 1
#define XSIMD_DEFAULT_ARCH  emulated<128>
#endif
#include <xsimd/xsimd.hpp>

#include <array>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

// SIMD width utilities (moved from utils.hpp to keep utils.hpp SIMD-free).
#if defined(XSIMD_VERSION_MAJOR) && (XSIMD_VERSION_MAJOR >= 14)
namespace finufft::utils {

template<class T, uint8_t N = 1> constexpr uint8_t min_simd_width() {
  // finds the smallest simd width that can handle N elements
  // simd size is batch size the SIMD width in xsimd terminology
  if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
    return min_simd_width<T, N * 2>();
  } else {
    return N;
  }
};

template<class T, uint8_t N> constexpr std::size_t find_optimal_simd_width() {
  // finds the smallest simd width that minimizes the number of iterations
  // NOTE: might be suboptimal for some cases 2^N+1 for example
  // in the future we might want to implement a more sophisticated algorithm

  uint8_t optimal_simd_width = min_simd_width<T>();
  uint8_t min_iterations     = (N + optimal_simd_width - 1) / optimal_simd_width;
  for (uint8_t simd_width = optimal_simd_width;
       simd_width <= xsimd::batch<T, xsimd::best_arch>::size; simd_width *= 2) {
    uint8_t iterations = (N + simd_width - 1) / simd_width;
    if (iterations < min_iterations) {
      min_iterations     = iterations;
      optimal_simd_width = simd_width;
    }
  }
  return static_cast<std::size_t>(optimal_simd_width);
}

template<class T, uint8_t N> constexpr std::size_t get_padded_simd_width() {
  // helper function to get the SIMD width with padding for the given number of elements
  // that minimizes the number of iterations

  return xsimd::make_sized_batch<T, find_optimal_simd_width<T, N>()>::type::size;
}
template<class T, uint8_t ns>
constexpr std::size_t get_simd_width_helper(uint8_t runtime_ns) {
  if constexpr (ns < finufft::common::MIN_NSPREAD) {
    return static_cast<std::size_t>(0);
  } else {
    if (runtime_ns == ns) {
      return get_padded_simd_width<T, ns>();
    } else {
      return get_simd_width_helper<T, ns - 1>(runtime_ns);
    }
  }
}
template<class T> constexpr std::size_t get_padded_simd_width(int runtime_ns) {
  return get_simd_width_helper<T, 2 * ::finufft::common::MAX_NSPREAD<T>>(runtime_ns);
}

} // namespace finufft::utils
#endif // XSIMD_VERSION_MAJOR >=14

namespace finufft::spreadinterp {

using finufft::common::INV_2PI;
using finufft::common::MAX_NSPREAD;
using finufft::common::MIN_NSPREAD;
using finufft::utils::find_optimal_simd_width;
using finufft::utils::get_padded_simd_width;

struct zip_low {
  // helper struct to get the lower half of a SIMD register and zip it with itself
  // it returns index 0, 0, 1, 1, ... N/2, N/2
  static constexpr unsigned get(unsigned index, unsigned /*size*/) { return index / 2; }
};

struct zip_hi {
  // helper struct to get the upper half of a SIMD register and zip it with itself
  // it returns index N/2, N/2, N/2+1, N/2+1, ... N, N
  static constexpr unsigned get(unsigned index, unsigned size) {
    return (size + index) / 2;
  }
};

template<unsigned cap> struct reverse_index {
  static constexpr unsigned get(unsigned index, const unsigned /*size*/) {
    return index < cap ? (cap - 1 - index) : index;
  }
};

template<unsigned cap> struct shuffle_index {
  static constexpr unsigned get(unsigned index, const unsigned size) {
    return index < cap ? (cap - 1 - index) : size + size + cap - 1 - index;
  }
};

struct [[maybe_unused]] select_even {
  static constexpr unsigned get(unsigned index, unsigned /*size*/) { return index * 2; }
};

struct [[maybe_unused]] select_odd {
  static constexpr unsigned get(unsigned index, unsigned /*size*/) {
    return index * 2 + 1;
  }
};

// this finds the largest SIMD instruction set that can handle N elements
// void otherwise -> compile error
template<class T, uint8_t N, uint8_t K = N> constexpr auto best_simd_helper() {
  if constexpr (N % K == 0) {
    // returns void in the worst case
    return xsimd::make_sized_batch<T, K>{};
  } else {
    return best_simd_helper<T, N, (K >> 1)>();
  }
}

template<class T, uint8_t N>
using PaddedSIMD =
    typename xsimd::make_sized_batch<T, get_padded_simd_width<T, N>()>::type;

template<class T, uint8_t ns> constexpr auto get_padding() {
  constexpr uint8_t width = get_padded_simd_width<T, ns>();
  return ((ns + width - 1) & (-width)) - ns;
}

template<class T, uint8_t ns> constexpr auto get_padding_helper(uint8_t runtime_ns) {
  if constexpr (ns < MIN_NSPREAD) {
    return 0;
  } else {
    if (runtime_ns == ns) {
      return get_padding<T, ns>();
    } else {
      return get_padding_helper<T, ns - 1>(runtime_ns);
    }
  }
}

template<class T> uint8_t get_padding(uint8_t ns) {
  return get_padding_helper<T, 2 * MAX_NSPREAD<T>>(ns);
}

// Single source of truth for the kernel-evaluation buffer layout.
//
// `evaluate_kernel_vector<NS, NC, T, simd_type>` writes one direction's
// worth of kernel values into a buffer; spread/interp call sites allocate
// `K * stride` bytes for K=1..3 directions. Today every CPU caller uses
// `simd_type = PaddedSIMD<T, 2*NS>` (the same SIMD width the inner
// complex-pair loops use). This trait bundles that simd_type, the
// per-direction stride, and the buffer alignment derived from the
// *same* simd_type, so the buffer cannot drift from the writer.
template<class T, uint8_t NS> struct KernelBufferLayout {
  using simd_type                        = PaddedSIMD<T, 2 * NS>;
  using arch_t                           = typename simd_type::arch_type;
  static constexpr std::size_t simd_size = simd_type::size;
  static constexpr std::size_t stride =
      (std::size_t(NS) + simd_size - 1) & ~(simd_size - 1);
  static constexpr std::size_t alignment = arch_t::alignment();

  static_assert(stride >= NS, "kernel buffer stride must hold NS elements");
  static_assert(stride % simd_size == 0,
                "kernel buffer stride must be a multiple of SIMD write width");
};

namespace detail {
template<class T, std::size_t... I>
constexpr std::size_t fold_max_layout_stride(std::index_sequence<I...>) {
  std::size_t m = 0;
  ((m = KernelBufferLayout<T, uint8_t(MIN_NSPREAD + I)>::stride > m
            ? KernelBufferLayout<T, uint8_t(MIN_NSPREAD + I)>::stride
            : m),
   ...);
  return m;
}
} // namespace detail

// Worst-case stride across every NS that may be instantiated for T.
// Consumed by plan.hpp's horner_coeffs (type-erased over NS).
template<class T>
inline constexpr std::size_t max_kernel_buffer_stride = detail::fold_max_layout_stride<T>(
    std::make_index_sequence<MAX_NSPREAD<T> - MIN_NSPREAD + 1>{});

// Runtime mirror of KernelBufferLayout<T, NS>::stride. Same formula, runtime ns.
// Pre: ns in [MIN_NSPREAD, MAX_NSPREAD<T>]. Post: when ns == NS,
//   kernel_buffer_stride_runtime<T>(ns) == KernelBufferLayout<T, NS>::stride.
template<class T> inline std::size_t kernel_buffer_stride_runtime(int ns) {
  const std::size_t w = get_padded_simd_width<T>(2 * ns);
  return (std::size_t(ns) + w - 1) & ~(w - 1);
}

// plan.hpp sizes horner_coeffs as MAX_NSPREAD<double>*MAX_NC (a loose bound that
// keeps plan.hpp xsimd-free for test headers). Pin that bound here so it is
// provably >= the actual stride for both precisions.
static_assert(MAX_NSPREAD<double> >= max_kernel_buffer_stride<float>,
              "horner_coeffs sizing in plan.hpp must cover float kernel buffer stride");
static_assert(MAX_NSPREAD<double> >= max_kernel_buffer_stride<double>,
              "horner_coeffs sizing in plan.hpp must cover double kernel buffer stride");

template<class T, uint8_t N>
using BestSIMD = typename decltype(best_simd_helper<T, N, xsimd::batch<T>::size>())::type;

template<class T, class V, size_t... Is>
constexpr T generate_sequence_impl(V a, V b, std::index_sequence<Is...>) noexcept {
  // utility function to generate a sequence of a, b interleaved as function arguments
  return T(((Is % 2 == 0) ? a : b)...);
}

template<class T, class V = typename T::value_type, std::size_t N = T::size>
constexpr auto initialize_complex_register(V a, V b) noexcept {
  // populates a SIMD register with a and b interleaved
  // for example:
  // +-------------------------------+
  // | a | b | a | b | a | b | a | b |
  // +-------------------------------+
  // it uses index_sequence to generate the sequence of a, b at compile time
  return generate_sequence_impl<T>(a, b, std::make_index_sequence<N>{});
}

template<class arch_t, typename T>
constexpr auto zip_low_index =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, zip_low, arch_t>();
template<class arch_t, typename T>
constexpr auto zip_hi_index =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, zip_hi, arch_t>();
// template<class arch_t, typename T>
// constexpr auto select_even_mask =
//     xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, select_even, arch_t>();
// template<class arch_t, typename T>
// constexpr auto select_odd_mask =
//     xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, select_odd, arch_t>();
template<typename T, std::size_t N, std::size_t M, std::size_t PaddedM>
constexpr std::array<std::array<T, PaddedM>, N> pad_2D_array_with_zeros(
    const std::array<std::array<T, M>, N> &input) noexcept {
  constexpr auto pad_with_zeros = [](const auto &input) constexpr noexcept {
    std::array<T, PaddedM> padded{0};
    for (size_t i = 0; i < input.size(); ++i) {
      padded[i] = input[i];
    }
    return padded;
  };
  std::array<std::array<T, PaddedM>, N> output{};
  for (std::size_t i = 0; i < N; ++i) {
    output[i] = pad_with_zeros(input[i]);
  }
  return output;
}

template<typename T> FINUFFT_ALWAYS_INLINE auto xsimd_to_array(const T &vec) noexcept {
  constexpr auto alignment = T::arch_type::alignment();
  alignas(alignment) std::array<typename T::value_type, T::size> array{};
  vec.store_aligned(array.data());
  return array;
}

// Forward declarations (defined in src/utils.cpp):
FINUFFT_NEVER_INLINE void print_subgrid_info(
    int ndims, BIGINT offset1, BIGINT offset2, BIGINT offset3, UBIGINT padded_size1,
    UBIGINT size1, UBIGINT size2, UBIGINT size3, UBIGINT M0);
// Helper for runtime diagnostic when dispatch picks invalid kernel params.
// Defined noinline to avoid code bloat on the valid path.
FINUFFT_NEVER_INLINE int report_invalid_kernel_params(int ns, int nc);

constexpr uint8_t ndims_from_Ns(const UBIGINT /*N1*/, const UBIGINT N2, const UBIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
  return 1 + (N2 > 1) + (N3 > 1);
}

/* local NU coord fold+rescale macro. Folds x into [-pi,pi) by addition of some integer
   multiple of 2pi, then linearly maps [-pi,pi) to [0,N). This is done in precision T
   (float or double).
   Note: folding larger x will cause a larger roundoff error.
   Martin Reinecke, 8.5.2024 used floor to speedup the function and removed the range
   limitation Marco Barbone, 8.5.2024 Changed it from a macro to an inline function
*/
template<typename T>
FINUFFT_ALWAYS_INLINE T fold_rescale(const T x, const UBIGINT N) noexcept {
  // using namespace to make the code compatible with both std and xsimd functions without qualification
  using namespace std;
  using namespace xsimd;
  const T result = fma(x, T(INV_2PI), T(0.5)); // x/(2pi) + 0.5
  return (result - floor(result)) * T(N);
}

template<int ns, int nc, class T, class simd_type, typename... V>
FINUFFT_ALWAYS_INLINE auto evaluate_kernel_vector(
    T *FINUFFT_RESTRICT ker, const T *horner_coeffs_ptr, const V... elems) noexcept {
  using KBL = KernelBufferLayout<T, uint8_t(ns)>;
  static_assert(std::is_same_v<simd_type, typename KBL::simd_type>,
                "evaluate_kernel_vector simd_type must equal "
                "KernelBufferLayout<T, NS>::simd_type — any other simd_type "
                "would mis-size the caller's buffer");
  /* Main SIMD-accelerated 1D kernel evaluator, using precomputed Horner coeffs to
     evaluate kernel on a grid of ns ordinates lying in kernel support, which is
     here scaled to [-ns/2,ns/2].
     Inputs are:
     ns = kernel width (spread points)
     T = (single or double precision) type of the kernel
     simd_type = xsimd::batch for Horner
     vectorization (default is the optimal simd size)
     elems = one or more leftmost ordinates x, which should be in [-ns/2,ns/2+1].
             For each such a

    Example usages:
    evaluate_kernel_vector<ns,nc,T,simd_type>(ker, coeffs, x)
    evaluate_kernel_vector<ns,nc,T,simd_type>(ker, coeffs, x, y) // for 3D
    etc

    See: evaluate_kernel_runtime() for a simplified code which does the same thing.

    Barbone (Dec/25): Removed unused opts parameter, kerevalmeth and inlined horner code
    since it is only used here now.
    *** to document this better.
   */
  const std::array inputs{elems...};
  // Only Horner piecewise-polynomial evaluation is used now. Call Horner
  // evaluator for each input value (compile-time loop over elements).
  for (size_t i = 0; i < sizeof...(elems); ++i) {
    // Inline SIMD Horner evaluation previously in eval_kernel_vec_Horner
    // Parameters: ns (w), nc (NC), simd_type
    const T x                         = inputs[i];
    const T z                         = std::fma(T(2.0), x, T(ns - 1));
    using arch_t                      = typename KBL::arch_t;
    static constexpr auto simd_size   = KBL::simd_size;
    static constexpr auto padded_ns   = KBL::stride;
    static constexpr auto use_ker_sym = (simd_size < ns);
    static constexpr auto stride      = padded_ns;

    T *KER = ker + (i * KBL::stride);

    if constexpr (use_ker_sym) {
      static constexpr uint8_t tail          = ns % simd_size;
      static constexpr uint8_t if_odd_degree = ((nc + 1) % 2);
      static constexpr uint8_t offset_start  = tail ? ns - tail : ns - simd_size;
      static constexpr uint8_t end_idx       = (ns + (tail > 0)) / 2;
      const simd_type zv{z};
      const auto z2v                      = zv * zv;
      static constexpr auto shuffle_batch = []() constexpr noexcept {
        if constexpr (tail) {
          return xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>,
                                            shuffle_index<tail>, arch_t>();
        } else {
          return xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>,
                                            reverse_index<simd_size>, arch_t>();
        }
      }();

      simd_type k_prev, k_sym{0};
      for (uint8_t ii{0}, offset = offset_start; ii < end_idx;
           ii += simd_size, offset -= simd_size) {
        auto k_odd = [ii, horner_coeffs_ptr]() constexpr noexcept {
          if constexpr (if_odd_degree) {
            return simd_type::load_aligned(horner_coeffs_ptr + ii);
          } else {
            return simd_type{0};
          }
        }();
        auto k_even =
            simd_type::load_aligned(horner_coeffs_ptr + if_odd_degree * stride + ii);
        for (uint8_t j{1 + if_odd_degree}; j < nc; j += 2) {
          const auto cji_odd =
              simd_type::load_aligned(horner_coeffs_ptr + j * stride + ii);
          const auto cji_even =
              simd_type::load_aligned(horner_coeffs_ptr + (j + 1) * stride + ii);
          k_odd  = xsimd::fma(k_odd, z2v, cji_odd);
          k_even = xsimd::fma(k_even, z2v, cji_even);
        }
        xsimd::fma(k_odd, zv, k_even).store_aligned(KER + ii);
        if (offset >= end_idx) {
          if constexpr (tail) {
            k_prev = k_sym;
            k_sym  = xsimd::fnma(k_odd, zv, k_even);
            xsimd::shuffle(k_sym, k_prev, shuffle_batch).store_aligned(KER + offset);
          } else {
            xsimd::swizzle(xsimd::fnma(k_odd, zv, k_even), shuffle_batch)
                .store_aligned(KER + offset);
          }
        }
      }
    } else {
      const simd_type zv(z);
      for (uint8_t ii = 0; ii < ns; ii += simd_size) {
        auto k = simd_type::load_aligned(horner_coeffs_ptr + ii);
        for (uint8_t j = 1; j < nc; ++j) {
          const auto cji = simd_type::load_aligned(horner_coeffs_ptr + j * stride + ii);
          k              = xsimd::fma(k, zv, cji);
        }
        k.store_aligned(KER + ii);
      }
    }
  }
  return ker;
}

} // namespace finufft::spreadinterp
