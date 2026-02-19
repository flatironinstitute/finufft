/* Benchmark for bin_sort_singlethread strategies.
 *
 * Compares:
 * 1. Scalar baseline (compute bins twice, no SIMD)
 * 2. Current SIMD approach (SIMD bin compute + has_duplicates + scatter/gather fallback)
 * 3. Precomputed bins (SIMD compute all bins once, scalar count+place)
 * 4. SIMD compute + always-scalar accumulate (no scatter/gather, no has_duplicates)
 *
 * Build: from build dir, cmake -DFINUFFT_BUILD_TESTS=ON .. && make binsort_bench
 * Usage: ./binsort_bench [M [N1 [dims]]]
 *   M    = number of NU points (default 1e7)
 *   N1   = grid size per dim (default 256)
 *   dims = 1, 2, or 3 (default 3)
 */

#include <finufft/test_defs.h>

#include <finufft/xsimd.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace std;

// ---- fold_rescale (same as in spreadinterp.cpp) ----
template<typename T> static inline T fold_rescale(const T x, const UBIGINT N) noexcept {
  constexpr double INV_2PI = 0.159154943091895336;
  const T result           = x * T(INV_2PI) + T(0.5);
  return (result - floor(result)) * T(N);
}

// ---- Approach 1: Scalar baseline (master's bin_sort_singlethread) ----
template<typename T>
static void bin_sort_scalar(vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky,
                            const T *kz, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                            double bin_size_x, double bin_size_y, double bin_size_z) {
  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1         = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2         = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3         = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins          = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x = T(1.0 / bin_size_x);
  const auto inv_bin_size_y = T(1.0 / bin_size_y);
  const auto inv_bin_size_z = T(1.0 / bin_size_z);

  vector<BIGINT> counts(nbins, 0);
  for (UBIGINT i = 0; i < M; i++) {
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  BIGINT current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    BIGINT tmp = counts[i];
    counts[i]  = current_offset;
    current_offset += tmp;
  }

  for (UBIGINT i = 0; i < M; i++) {
    const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i);
    ++counts[bin];
  }
}

// ---- Approach 2: Current SIMD with has_duplicates + scatter/gather ----
template<typename simd_type, size_t... Is>
static inline simd_type make_incremented_vectors(index_sequence<Is...>) {
  return simd_type{static_cast<typename simd_type::value_type>(Is)...};
}

template<size_t N, typename simd_type>
static inline bool has_duplicates_impl(const simd_type &vec) noexcept {
  if constexpr (N == simd_type::size) {
    return false;
  } else {
    if (xsimd::any((xsimd::rotate_right<N>(vec) == vec))) return true;
    return has_duplicates_impl<N + 1>(vec);
  }
}

template<typename simd_type>
static inline bool has_duplicates(const simd_type &vec) noexcept {
  return has_duplicates_impl<1, simd_type>(vec);
}

template<typename T>
static void bin_sort_simd_scatter(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using int_simd_type             = xsimd::batch<xsimd::as_integer_t<T>>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();

  static constexpr auto to_array = [](const auto &vec) constexpr noexcept {
    using VT = decltype(decay_t<decltype(vec)>());
    alignas(alignment) array<typename VT::value_type, VT::size> arr{};
    vec.store_aligned(arr.data());
    return arr;
  };

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));
  const auto increment        = xsimd::to_int(
      make_incremented_vectors<simd_type>(make_index_sequence<simd_size>{}));

  vector<xsimd::as_integer_t<T>> counts(nbins + simd_size, 0);
  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};

  // counting pass
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    if (has_duplicates(bin)) {
      const auto bin_array = to_array(bin);
      for (int j = 0; j < simd_size; j++) ++counts[bin_array[j]];
    } else {
      const auto bins = int_simd_type::gather(counts.data(), bin);
      xsimd::incr(bins).scatter(counts.data(), bin);
    }
  }
  for (; i < M; i++) {
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  // prefix sum
  BIGINT current_offset = 0;
  for (UBIGINT ii = 0; ii < UBIGINT(nbins); ii++) {
    auto tmp   = counts[ii];
    counts[ii] = current_offset;
    current_offset += tmp;
  }

  // placement pass
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    if (has_duplicates(bin)) {
      const auto bin_array = to_array(xsimd::to_int(bin));
      for (int j = 0; j < simd_size; j++) {
        ret[counts[bin_array[j]]] = j + i;
        counts[bin_array[j]]++;
      }
    } else {
      const auto bins      = decltype(bin)::gather(counts.data(), bin);
      const auto incr_bins = xsimd::incr(bins);
      incr_bins.scatter(counts.data(), bin);
      const auto result = increment + int_simd_type(i);
      result.scatter(ret.data(), bins);
    }
  }
  for (; i < M; i++) {
    const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i);
    ++counts[bin];
  }
}

// ---- Approach 3: Precomputed bins (SIMD compute once, scalar count+place) ----
template<typename T>
static void bin_sort_precomputed(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using int_simd_type             = xsimd::batch<xsimd::as_integer_t<T>>;
  static constexpr auto simd_size = simd_type::size;

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  // Phase 1: SIMD-vectorized bin index computation
  // Use the same integer type as xsimd produces for to_int
  using bin_int_t = xsimd::as_integer_t<T>;
  vector<bin_int_t> bins(M);

  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    bin.store_unaligned(bins.data() + i);
  }
  for (; i < M; i++) {
    const auto i1 = bin_int_t(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2 = isky ? bin_int_t(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3 = iskz ? bin_int_t(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    bins[i]       = i1 + nbins1 * (i2 + nbins2 * i3);
  }

  // Phase 2: scalar counting
  vector<BIGINT> counts(nbins, 0);
  for (UBIGINT i = 0; i < M; i++) ++counts[bins[i]];

  // prefix sum
  BIGINT current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    BIGINT tmp = counts[i];
    counts[i]  = current_offset;
    current_offset += tmp;
  }

  // Phase 3: scalar placement
  for (UBIGINT i = 0; i < M; i++) {
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ---- Approach 4: SIMD compute + always-scalar accumulate (no scatter, no
// has_duplicates) ----
template<typename T>
static void bin_sort_simd_noscatter(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();

  static constexpr auto to_array = [](const auto &vec) constexpr noexcept {
    using VT = decltype(decay_t<decltype(vec)>());
    alignas(alignment) array<typename VT::value_type, VT::size> arr{};
    vec.store_aligned(arr.data());
    return arr;
  };

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  vector<BIGINT> counts(nbins, 0);
  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};

  // counting pass: SIMD compute, scalar accumulate
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin       = i1 + nbins1 * (i2 + nbins2 * i3);
    const auto bin_array = to_array(bin);
    for (int j = 0; j < simd_size; j++) ++counts[bin_array[j]];
  }
  for (; i < M; i++) {
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  // prefix sum
  BIGINT current_offset = 0;
  for (BIGINT ii = 0; ii < nbins; ii++) {
    BIGINT tmp = counts[ii];
    counts[ii] = current_offset;
    current_offset += tmp;
  }

  // placement pass: SIMD compute, scalar accumulate
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin       = i1 + nbins1 * (i2 + nbins2 * i3);
    const auto bin_array = to_array(bin);
    for (int j = 0; j < simd_size; j++) {
      ret[counts[bin_array[j]]] = BIGINT(j + i);
      ++counts[bin_array[j]];
    }
  }
  for (; i < M; i++) {
    const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i);
    ++counts[bin];
  }
}

// ---- Approach 5: SIMD noscatter + int32 counts (halved cache footprint) ----
template<typename T>
static void bin_sort_noscatter_i32(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();

  static constexpr auto to_array = [](const auto &vec) constexpr noexcept {
    using VT = decltype(decay_t<decltype(vec)>());
    alignas(alignment) array<typename VT::value_type, VT::size> arr{};
    vec.store_aligned(arr.data());
    return arr;
  };

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  // Use int32 counts to halve cache footprint
  vector<int32_t> counts(nbins, 0);
  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};

  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin       = i1 + nbins1 * (i2 + nbins2 * i3);
    const auto bin_array = to_array(bin);
    for (int j = 0; j < simd_size; j++) ++counts[bin_array[j]];
  }
  for (; i < M; i++) {
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  int32_t current_offset = 0;
  for (BIGINT ii = 0; ii < nbins; ii++) {
    int32_t tmp = counts[ii];
    counts[ii]  = current_offset;
    current_offset += tmp;
  }

  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin       = i1 + nbins1 * (i2 + nbins2 * i3);
    const auto bin_array = to_array(bin);
    for (int j = 0; j < simd_size; j++) {
      ret[counts[bin_array[j]]] = BIGINT(j + i);
      ++counts[bin_array[j]];
    }
  }
  for (; i < M; i++) {
    const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i);
    ++counts[bin];
  }
}

// ---- Approach 6: SIMD noscatter + software prefetching ----
template<typename T>
static void bin_sort_noscatter_prefetch(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();

  static constexpr auto to_array = [](const auto &vec) constexpr noexcept {
    using VT = decltype(decay_t<decltype(vec)>());
    alignas(alignment) array<typename VT::value_type, VT::size> arr{};
    vec.store_aligned(arr.data());
    return arr;
  };

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  vector<BIGINT> counts(nbins, 0);
  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};

  // Prefetch distance in SIMD iterations (tunable)
  constexpr int PREFETCH_DIST = 8;

  // counting pass with prefetching
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin       = i1 + nbins1 * (i2 + nbins2 * i3);
    const auto bin_array = to_array(bin);
    // Prefetch counts entries for upcoming iterations
    if (i + PREFETCH_DIST * simd_size < simd_M) {
      const auto pf_i1 = xsimd::to_int(
          fold_rescale(simd_type::load_unaligned(kx + i + PREFETCH_DIST * simd_size),
                       N1) *
          inv_bin_size_x_v);
      const auto pf_i2 =
          isky ? xsimd::to_int(fold_rescale(simd_type::load_unaligned(
                                                ky + i + PREFETCH_DIST * simd_size),
                                            N2) *
                               inv_bin_size_y_v)
               : zero;
      const auto pf_i3 =
          iskz ? xsimd::to_int(fold_rescale(simd_type::load_unaligned(
                                                kz + i + PREFETCH_DIST * simd_size),
                                            N3) *
                               inv_bin_size_z_v)
               : zero;
      const auto pf_bin       = pf_i1 + nbins1 * (pf_i2 + nbins2 * pf_i3);
      const auto pf_bin_array = to_array(pf_bin);
      for (int j = 0; j < simd_size; j++)
        __builtin_prefetch(&counts[pf_bin_array[j]], 1, 1);
    }
    for (int j = 0; j < simd_size; j++) ++counts[bin_array[j]];
  }
  for (; i < M; i++) {
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  BIGINT current_offset = 0;
  for (BIGINT ii = 0; ii < nbins; ii++) {
    BIGINT tmp = counts[ii];
    counts[ii] = current_offset;
    current_offset += tmp;
  }

  // placement pass with prefetching
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin       = i1 + nbins1 * (i2 + nbins2 * i3);
    const auto bin_array = to_array(bin);
    if (i + PREFETCH_DIST * simd_size < simd_M) {
      const auto pf_i1 = xsimd::to_int(
          fold_rescale(simd_type::load_unaligned(kx + i + PREFETCH_DIST * simd_size),
                       N1) *
          inv_bin_size_x_v);
      const auto pf_i2 =
          isky ? xsimd::to_int(fold_rescale(simd_type::load_unaligned(
                                                ky + i + PREFETCH_DIST * simd_size),
                                            N2) *
                               inv_bin_size_y_v)
               : zero;
      const auto pf_i3 =
          iskz ? xsimd::to_int(fold_rescale(simd_type::load_unaligned(
                                                kz + i + PREFETCH_DIST * simd_size),
                                            N3) *
                               inv_bin_size_z_v)
               : zero;
      const auto pf_bin       = pf_i1 + nbins1 * (pf_i2 + nbins2 * pf_i3);
      const auto pf_bin_array = to_array(pf_bin);
      for (int j = 0; j < simd_size; j++) {
        __builtin_prefetch(&counts[pf_bin_array[j]], 1, 1);
        __builtin_prefetch(&ret[counts[pf_bin_array[j]]], 1, 0);
      }
    }
    for (int j = 0; j < simd_size; j++) {
      ret[counts[bin_array[j]]] = BIGINT(j + i);
      ++counts[bin_array[j]];
    }
  }
  for (; i < M; i++) {
    const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i);
    ++counts[bin];
  }
}

// ---- Approach 7: Precomputed bins + int32 counts (best of both) ----
template<typename T>
static void bin_sort_precomp_i32(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using int_simd_type             = xsimd::batch<xsimd::as_integer_t<T>>;
  static constexpr auto simd_size = simd_type::size;

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  // Precompute bins with SIMD, store as int32
  using bin_int_t = xsimd::as_integer_t<T>;
  vector<bin_int_t> bins(M);
  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    bin.store_unaligned(bins.data() + i);
  }
  for (; i < M; i++) {
    const auto i1 = bin_int_t(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2 = isky ? bin_int_t(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3 = iskz ? bin_int_t(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    bins[i]       = i1 + nbins1 * (i2 + nbins2 * i3);
  }

  // int32 counts
  vector<int32_t> counts(nbins, 0);
  for (UBIGINT i = 0; i < M; i++) ++counts[bins[i]];

  int32_t current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    int32_t tmp = counts[i];
    counts[i]   = current_offset;
    current_offset += tmp;
  }

  for (UBIGINT i = 0; i < M; i++) {
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ---- Approach 8: Precomputed + int32 + prefetch ----
template<typename T>
static void bin_sort_precomp_i32_prefetch(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto isky = (N2 > 1), iskz = (N3 > 1);
  const auto nbins1           = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2           = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3           = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins            = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  using bin_int_t = xsimd::as_integer_t<T>;
  vector<bin_int_t> bins(M);
  const auto simd_M = M & UBIGINT(-simd_size);
  UBIGINT i{};
  for (i = 0; i < simd_M; i += simd_size) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + i), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(ky + i), N2) * inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(
                   fold_rescale(simd_type::load_unaligned(kz + i), N3) * inv_bin_size_z_v)
             : zero;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    bin.store_unaligned(bins.data() + i);
  }
  for (; i < M; i++) {
    const auto i1 = bin_int_t(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2 = isky ? bin_int_t(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3 = iskz ? bin_int_t(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    bins[i]       = i1 + nbins1 * (i2 + nbins2 * i3);
  }

  // Counting with prefetch
  constexpr UBIGINT PF_DIST = 32;
  vector<int32_t> counts(nbins, 0);
  for (UBIGINT i = 0; i < M; i++) {
    if (i + PF_DIST < M) __builtin_prefetch(&counts[bins[i + PF_DIST]], 1, 1);
    ++counts[bins[i]];
  }

  int32_t current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    int32_t tmp = counts[i];
    counts[i]   = current_offset;
    current_offset += tmp;
  }

  // Placement with prefetch
  for (UBIGINT i = 0; i < M; i++) {
    if (i + PF_DIST < M) {
      __builtin_prefetch(&counts[bins[i + PF_DIST]], 1, 1);
      __builtin_prefetch(&ret[counts[bins[i + PF_DIST]]], 1, 0);
    }
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ---- Timing helper ----
template<typename F> double bench(F &&f, int nruns) {
  // warmup
  f();
  double best = 1e30;
  for (int r = 0; r < nruns; r++) {
    auto t0 = chrono::high_resolution_clock::now();
    f();
    auto t1  = chrono::high_resolution_clock::now();
    double s = chrono::duration<double>(t1 - t0).count();
    best     = min(best, s);
  }
  return best;
}

// ---- Correctness check ----
static bool check_equal(const vector<BIGINT> &a, const vector<BIGINT> &b,
                        const char *name) {
  if (a.size() != b.size()) {
    printf("  %s: SIZE MISMATCH %zu vs %zu\n", name, a.size(), b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      printf("  %s: MISMATCH at i=%zu: %lld vs %lld\n", name, i, (long long)a[i],
             (long long)b[i]);
      return false;
    }
  }
  printf("  %s: PASS\n", name);
  return true;
}

int main(int argc, char *argv[]) {
  UBIGINT M  = UBIGINT(1e7);
  UBIGINT N1 = 256;
  int dims   = 3;
  int nruns  = 5;

  if (argc > 1) M = UBIGINT(atof(argv[1]));
  if (argc > 2) N1 = UBIGINT(atof(argv[2]));
  if (argc > 3) dims = atoi(argv[3]);
  if (argc > 4) nruns = atoi(argv[4]);

  UBIGINT N2 = (dims >= 2) ? N1 : 1;
  UBIGINT N3 = (dims >= 3) ? N1 : 1;

  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;

  printf("bin_sort benchmark: M=%llu, N1=%llu, N2=%llu, N3=%llu, dims=%d, nruns=%d\n",
         (unsigned long long)M, (unsigned long long)N1, (unsigned long long)N2,
         (unsigned long long)N3, dims, nruns);
  printf("  SIMD width: %zu doubles, %zu floats\n", xsimd::batch<double>::size,
         xsimd::batch<float>::size);

  BIGINT nbins1 = BIGINT(FLT(N1) / bin_size_x + 1);
  BIGINT nbins2 = (N2 > 1) ? BIGINT(FLT(N2) / bin_size_y + 1) : 1;
  BIGINT nbins3 = (N3 > 1) ? BIGINT(FLT(N3) / bin_size_z + 1) : 1;
  printf("  nbins = %lld x %lld x %lld = %lld\n", (long long)nbins1, (long long)nbins2,
         (long long)nbins3, (long long)(nbins1 * nbins2 * nbins3));
  printf("  ~pts/bin = %.1f\n", double(M) / (nbins1 * nbins2 * nbins3));

  // Generate random NU points in [-pi, pi)
  mt19937 rng(42);
  uniform_real_distribution<FLT> dist(-M_PI, M_PI);

  vector<FLT> kx(M), ky(M), kz(M);
  for (UBIGINT i = 0; i < M; i++) {
    kx[i] = dist(rng);
    if (dims >= 2) ky[i] = dist(rng);
    if (dims >= 3) kz[i] = dist(rng);
  }

  vector<BIGINT> ret_scalar(M), ret_simd_scatter(M), ret_precomp(M),
      ret_simd_noscatter(M);
  vector<BIGINT> ret_noscatter_i32(M), ret_noscatter_pf(M), ret_precomp_i32(M),
      ret_precomp_i32_pf(M);

  // Correctness check first
  printf("\nCorrectness check:\n");
  bin_sort_scalar(ret_scalar, M, kx.data(), ky.data(), kz.data(), N1, N2, N3, bin_size_x,
                  bin_size_y, bin_size_z);
  bin_sort_simd_scatter(ret_simd_scatter, M, kx.data(), ky.data(), kz.data(), N1, N2, N3,
                        bin_size_x, bin_size_y, bin_size_z);
  bin_sort_precomputed(ret_precomp, M, kx.data(), ky.data(), kz.data(), N1, N2, N3,
                       bin_size_x, bin_size_y, bin_size_z);
  bin_sort_simd_noscatter(ret_simd_noscatter, M, kx.data(), ky.data(), kz.data(), N1, N2,
                          N3, bin_size_x, bin_size_y, bin_size_z);
  bin_sort_noscatter_i32(ret_noscatter_i32, M, kx.data(), ky.data(), kz.data(), N1, N2,
                         N3, bin_size_x, bin_size_y, bin_size_z);
  bin_sort_noscatter_prefetch(ret_noscatter_pf, M, kx.data(), ky.data(), kz.data(), N1,
                              N2, N3, bin_size_x, bin_size_y, bin_size_z);
  bin_sort_precomp_i32(ret_precomp_i32, M, kx.data(), ky.data(), kz.data(), N1, N2, N3,
                       bin_size_x, bin_size_y, bin_size_z);
  bin_sort_precomp_i32_prefetch(ret_precomp_i32_pf, M, kx.data(), ky.data(), kz.data(),
                                N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);

  check_equal(ret_scalar, ret_simd_scatter, "simd_scatter");
  check_equal(ret_scalar, ret_precomp, "precomputed");
  check_equal(ret_scalar, ret_simd_noscatter, "simd_noscatter");
  check_equal(ret_scalar, ret_noscatter_i32, "noscatter_i32");
  check_equal(ret_scalar, ret_noscatter_pf, "noscatter_pf");
  check_equal(ret_scalar, ret_precomp_i32, "precomp_i32");
  check_equal(ret_scalar, ret_precomp_i32_pf, "precomp_i32_pf");

  // Benchmark
  printf("\nTiming (best of %d runs):\n", nruns);

  double t_scalar = bench(
      [&]() {
        bin_sort_scalar(ret_scalar, M, kx.data(), ky.data(), kz.data(), N1, N2, N3,
                        bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  scalar:          %.4f s  (%.1f Mpts/s)\n", t_scalar, M / t_scalar / 1e6);

  double t_simd_scatter = bench(
      [&]() {
        bin_sort_simd_scatter(ret_simd_scatter, M, kx.data(), ky.data(), kz.data(), N1,
                              N2, N3, bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  simd+scatter:    %.4f s  (%.1f Mpts/s)  %.2fx\n", t_simd_scatter,
         M / t_simd_scatter / 1e6, t_scalar / t_simd_scatter);

  double t_precomp = bench(
      [&]() {
        bin_sort_precomputed(ret_precomp, M, kx.data(), ky.data(), kz.data(), N1, N2, N3,
                             bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  precomputed:     %.4f s  (%.1f Mpts/s)  %.2fx\n", t_precomp,
         M / t_precomp / 1e6, t_scalar / t_precomp);

  double t_noscatter = bench(
      [&]() {
        bin_sort_simd_noscatter(ret_simd_noscatter, M, kx.data(), ky.data(), kz.data(),
                                N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  simd_noscatter:  %.4f s  (%.1f Mpts/s)  %.2fx\n", t_noscatter,
         M / t_noscatter / 1e6, t_scalar / t_noscatter);

  double t_noscatter_i32 = bench(
      [&]() {
        bin_sort_noscatter_i32(ret_noscatter_i32, M, kx.data(), ky.data(), kz.data(), N1,
                               N2, N3, bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  noscatter_i32:   %.4f s  (%.1f Mpts/s)  %.2fx\n", t_noscatter_i32,
         M / t_noscatter_i32 / 1e6, t_scalar / t_noscatter_i32);

  double t_noscatter_pf = bench(
      [&]() {
        bin_sort_noscatter_prefetch(ret_noscatter_pf, M, kx.data(), ky.data(), kz.data(),
                                    N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  noscatter_pf:    %.4f s  (%.1f Mpts/s)  %.2fx\n", t_noscatter_pf,
         M / t_noscatter_pf / 1e6, t_scalar / t_noscatter_pf);

  double t_precomp_i32 = bench(
      [&]() {
        bin_sort_precomp_i32(ret_precomp_i32, M, kx.data(), ky.data(), kz.data(), N1, N2,
                             N3, bin_size_x, bin_size_y, bin_size_z);
      },
      nruns);
  printf("  precomp_i32:     %.4f s  (%.1f Mpts/s)  %.2fx\n", t_precomp_i32,
         M / t_precomp_i32 / 1e6, t_scalar / t_precomp_i32);

  double t_precomp_i32_pf = bench(
      [&]() {
        bin_sort_precomp_i32_prefetch(ret_precomp_i32_pf, M, kx.data(), ky.data(),
                                      kz.data(), N1, N2, N3, bin_size_x, bin_size_y,
                                      bin_size_z);
      },
      nruns);
  printf("  precomp_i32_pf:  %.4f s  (%.1f Mpts/s)  %.2fx\n", t_precomp_i32_pf,
         M / t_precomp_i32_pf / 1e6, t_scalar / t_precomp_i32_pf);

  printf("\nMemory footprint (approx):\n");
  double base_MB = (3.0 * M * sizeof(FLT) + M * sizeof(BIGINT)) / 1e6;
  printf("  Input arrays (kx,ky,kz,ret): %.0f MB\n", base_MB);
  printf("  Extra for precomputed bins:   %.0f MB\n", M * sizeof(int32_t) / 1e6);

  return 0;
}
