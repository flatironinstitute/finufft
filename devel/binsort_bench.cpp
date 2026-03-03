/* Benchmark for bin_sort_singlethread strategies using Google Benchmark.
 *
 * Compares:
 * 1. Scalar baseline (compute bins twice, no SIMD)
 * 2. Current SIMD approach (SIMD bin compute + has_duplicates + scatter/gather)
 * 3. Precomputed bins (SIMD compute all bins once, scalar count+place)
 * 4. SIMD compute + always-scalar accumulate (simd_noscatter)
 * 5. SIMD noscatter + int32 counts (halved cache footprint)
 * 6. SIMD noscatter + software prefetching
 * 7. Precomputed bins + int32 counts
 * 8. Precomputed bins + int32 counts + prefetching
 *
 * Build: from build dir, cmake -DFINUFFT_BUILD_DEVEL=ON .. && make binsort_bench
 * Usage: ./binsort_bench [--benchmark_filter=<regex>] [--benchmark_repetitions=N]
 */

#include <finufft/test_defs.h>

#include <finufft/xsimd.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
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

// ---- SIMD helpers ----
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

// ---- Common SIMD to_array helper ----
template<typename arch_t> struct ToArray {
  static constexpr auto alignment = arch_t::alignment();
  template<typename V> static inline auto call(const V &vec) noexcept {
    alignas(alignment) array<typename V::value_type, V::size> arr{};
    vec.store_aligned(arr.data());
    return arr;
  }
};

// ---- Common setup: compute bin parameters ----
struct BinParams {
  BIGINT nbins1, nbins2, nbins3, nbins;
  bool isky, iskz;
};

static BinParams compute_bin_params(UBIGINT N1, UBIGINT N2, UBIGINT N3, double bin_size_x,
                                    double bin_size_y, double bin_size_z) {
  BinParams p;
  p.isky   = (N2 > 1);
  p.iskz   = (N3 > 1);
  p.nbins1 = BIGINT(double(N1) / bin_size_x + 1);
  p.nbins2 = p.isky ? BIGINT(double(N2) / bin_size_y + 1) : 1;
  p.nbins3 = p.iskz ? BIGINT(double(N3) / bin_size_z + 1) : 1;
  p.nbins  = p.nbins1 * p.nbins2 * p.nbins3;
  return p;
}

// ========================================================================
// Approach 1: Scalar baseline
// ========================================================================
template<typename T>
static void bin_sort_scalar(vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky,
                            const T *kz, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                            double bin_size_x, double bin_size_y, double bin_size_z) {
  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

// ========================================================================
// Approach 2: SIMD with has_duplicates + scatter/gather
// ========================================================================
template<typename T>
static void bin_sort_simd_scatter(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using int_simd_type             = xsimd::batch<xsimd::as_integer_t<T>>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  using ta                        = ToArray<arch_t>;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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
      const auto bin_array = ta::call(bin);
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

  BIGINT current_offset = 0;
  for (UBIGINT ii = 0; ii < UBIGINT(nbins); ii++) {
    auto tmp   = counts[ii];
    counts[ii] = current_offset;
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
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    if (has_duplicates(bin)) {
      const auto bin_array = ta::call(xsimd::to_int(bin));
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

// ========================================================================
// Approach 3: Precomputed bins (SIMD compute once, scalar count+place)
// ========================================================================
template<typename T>
static void bin_sort_precomputed(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

  vector<BIGINT> counts(nbins, 0);
  for (UBIGINT i = 0; i < M; i++) ++counts[bins[i]];

  BIGINT current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    BIGINT tmp = counts[i];
    counts[i]  = current_offset;
    current_offset += tmp;
  }

  for (UBIGINT i = 0; i < M; i++) {
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ========================================================================
// Approach 4: SIMD compute + always-scalar accumulate (no scatter)
// ========================================================================
template<typename T>
static void bin_sort_simd_noscatter(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  using ta                        = ToArray<arch_t>;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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
    const auto bin_array = ta::call(bin);
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
    const auto bin_array = ta::call(bin);
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

// ========================================================================
// Approach 5: SIMD noscatter + int32 counts (halved cache footprint)
// ========================================================================
template<typename T>
static void bin_sort_noscatter_i32(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  using ta                        = ToArray<arch_t>;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

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
    const auto bin_array = ta::call(bin);
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
    const auto bin_array = ta::call(bin);
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

// ========================================================================
// Approach 6: SIMD noscatter + software prefetching
// ========================================================================
template<typename T>
static void bin_sort_noscatter_prefetch(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  using ta                        = ToArray<arch_t>;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

  constexpr int PREFETCH_DIST = 8;

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
    const auto bin_array = ta::call(bin);
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
      const auto pf_bin_array = ta::call(pf_bin);
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
    const auto bin_array = ta::call(bin);
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
      const auto pf_bin_array = ta::call(pf_bin);
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

// ========================================================================
// Approach 7: Precomputed bins + int32 counts
// ========================================================================
template<typename T>
static void bin_sort_precomp_i32(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

// ========================================================================
// Approach 8: Precomputed + int32 + prefetch
// ========================================================================
template<typename T>
static void bin_sort_precomp_i32_prefetch(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

  for (UBIGINT i = 0; i < M; i++) {
    if (i + PF_DIST < M) {
      __builtin_prefetch(&counts[bins[i + PF_DIST]], 1, 1);
      __builtin_prefetch(&ret[counts[bins[i + PF_DIST]]], 1, 0);
    }
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ========================================================================
// Approach 9: Precomputed + uint32_t counts + interleaved histograms (2x)
// Reduces store-to-load forwarding stalls by alternating between two
// independent counts arrays. When consecutive points hit nearby bins,
// the CPU pipeline doesn't stall waiting for the previous store.
// ========================================================================
template<typename T>
static void bin_sort_precomp_u32_interleaved(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  // Precompute bins
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

  // Two interleaved histograms to reduce store-forwarding stalls
  vector<uint32_t> counts0(nbins, 0);
  vector<uint32_t> counts1(nbins, 0);
  const auto even_M = M & ~UBIGINT(1);
  for (UBIGINT i = 0; i < even_M; i += 2) {
    ++counts0[bins[i]];
    ++counts1[bins[i + 1]];
  }
  if (M & 1) ++counts0[bins[M - 1]];

  // Merge and prefix sum
  vector<uint32_t> counts(nbins);
  uint32_t current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    counts[i] = current_offset;
    current_offset += counts0[i] + counts1[i];
  }

  // Placement (single pass, single counts)
  for (UBIGINT i = 0; i < M; i++) {
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ========================================================================
// Approach 10: SIMD noscatter + uint32_t + 2x unrolled
// Process 2 SIMD vectors per iteration, interleave scalar updates.
// ========================================================================
template<typename T>
static void bin_sort_noscatter_u32_unrolled(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto simd_size = simd_type::size;
  using ta                        = ToArray<arch_t>;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
  const auto inv_bin_size_x_v = simd_type(1.0 / bin_size_x);
  const auto inv_bin_size_y_v = simd_type(1.0 / bin_size_y);
  const auto inv_bin_size_z_v = simd_type(1.0 / bin_size_z);
  const auto inv_bin_size_x   = T(1.0 / bin_size_x);
  const auto inv_bin_size_y   = T(1.0 / bin_size_y);
  const auto inv_bin_size_z   = T(1.0 / bin_size_z);
  const auto zero             = xsimd::to_int(simd_type(0));

  auto compute_bins = [&](UBIGINT offset) {
    const auto i1 = xsimd::to_int(
        fold_rescale(simd_type::load_unaligned(kx + offset), N1) * inv_bin_size_x_v);
    const auto i2 =
        isky ? xsimd::to_int(fold_rescale(simd_type::load_unaligned(ky + offset), N2) *
                             inv_bin_size_y_v)
             : zero;
    const auto i3 =
        iskz ? xsimd::to_int(fold_rescale(simd_type::load_unaligned(kz + offset), N3) *
                             inv_bin_size_z_v)
             : zero;
    return i1 + nbins1 * (i2 + nbins2 * i3);
  };

  vector<uint32_t> counts(nbins, 0);
  const auto simd_M  = M & UBIGINT(-simd_size);
  const auto simd_M2 = simd_M & UBIGINT(-2 * simd_size); // 2x unrolled limit
  UBIGINT i{};

  // Counting pass: 2x unrolled
  for (i = 0; i < simd_M2; i += 2 * simd_size) {
    const auto bin_a = compute_bins(i);
    const auto bin_b = compute_bins(i + simd_size);
    const auto arr_a = ta::call(bin_a);
    const auto arr_b = ta::call(bin_b);
    // Interleave updates from two vectors to reduce dependency chains
    for (int j = 0; j < simd_size; j++) {
      ++counts[arr_a[j]];
      ++counts[arr_b[j]];
    }
  }
  for (; i < simd_M; i += simd_size) {
    const auto bin       = compute_bins(i);
    const auto bin_array = ta::call(bin);
    for (int j = 0; j < simd_size; j++) ++counts[bin_array[j]];
  }
  for (; i < M; i++) {
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  uint32_t current_offset = 0;
  for (BIGINT ii = 0; ii < nbins; ii++) {
    uint32_t tmp = counts[ii];
    counts[ii]   = current_offset;
    current_offset += tmp;
  }

  // Placement pass: 2x unrolled
  for (i = 0; i < simd_M2; i += 2 * simd_size) {
    const auto bin_a = compute_bins(i);
    const auto bin_b = compute_bins(i + simd_size);
    const auto arr_a = ta::call(bin_a);
    const auto arr_b = ta::call(bin_b);
    for (int j = 0; j < simd_size; j++) {
      ret[counts[arr_a[j]]] = BIGINT(j + i);
      ++counts[arr_a[j]];
      ret[counts[arr_b[j]]] = BIGINT(j + i + simd_size);
      ++counts[arr_b[j]];
    }
  }
  for (; i < simd_M; i += simd_size) {
    const auto bin       = compute_bins(i);
    const auto bin_array = ta::call(bin);
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

// ========================================================================
// Approach 11: Precomputed + uint32_t + interleaved + prefetch (kitchen sink)
// ========================================================================
template<typename T>
static void bin_sort_precomp_u32_interleaved_pf(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

  // Interleaved counting with prefetch
  constexpr UBIGINT PF_DIST = 32;
  vector<uint32_t> counts0(nbins, 0);
  vector<uint32_t> counts1(nbins, 0);
  const auto even_M = M & ~UBIGINT(1);
  for (UBIGINT i = 0; i < even_M; i += 2) {
    if (i + PF_DIST < M) {
      __builtin_prefetch(&counts0[bins[i + PF_DIST]], 1, 1);
      __builtin_prefetch(&counts1[bins[i + PF_DIST + 1]], 1, 1);
    }
    ++counts0[bins[i]];
    ++counts1[bins[i + 1]];
  }
  if (M & 1) ++counts0[bins[M - 1]];

  // Merge and prefix sum
  vector<uint32_t> counts(nbins);
  uint32_t current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    counts[i] = current_offset;
    current_offset += counts0[i] + counts1[i];
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

// ========================================================================
// Approach 12: Precomputed + uint32_t (simple, no tricks)
// Just like approach 7 but with uint32_t instead of int32_t
// ========================================================================
template<typename T>
static void bin_sort_precomp_u32(
    vector<BIGINT> &ret, UBIGINT M, const T *kx, const T *ky, const T *kz, UBIGINT N1,
    UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z) {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;

  const auto [nbins1, nbins2, nbins3, nbins, isky, iskz] =
      compute_bin_params(N1, N2, N3, bin_size_x, bin_size_y, bin_size_z);
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

  vector<uint32_t> counts(nbins, 0);
  for (UBIGINT i = 0; i < M; i++) ++counts[bins[i]];

  uint32_t current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    uint32_t tmp = counts[i];
    counts[i]    = current_offset;
    current_offset += tmp;
  }

  for (UBIGINT i = 0; i < M; i++) {
    ret[counts[bins[i]]] = BIGINT(i);
    ++counts[bins[i]];
  }
}

// ========================================================================
// Google Benchmark fixture
// ========================================================================

// Global test data (allocated once, shared across benchmarks)
struct TestData {
  UBIGINT M, N1, N2, N3;
  double bin_size_x, bin_size_y, bin_size_z;
  vector<FLT> kx, ky, kz;
  vector<BIGINT> ret;

  void init(UBIGINT M_, UBIGINT N1_, int dims) {
    M  = M_;
    N1 = N1_;
    N2 = (dims >= 2) ? N1 : 1;
    N3 = (dims >= 3) ? N1 : 1;

    bin_size_x = 16;
    bin_size_y = 4;
    bin_size_z = 4;

    mt19937 rng(42);
    uniform_real_distribution<FLT> dist(-M_PI, M_PI);
    kx.resize(M);
    ky.resize(M);
    kz.resize(M);
    ret.resize(M);
    for (UBIGINT i = 0; i < M; i++) {
      kx[i] = dist(rng);
      if (dims >= 2) ky[i] = dist(rng);
      if (dims >= 3) kz[i] = dist(rng);
    }
  }
};

static TestData g_data;

// Macro to define a benchmark for each approach
#define DEFINE_BM(name, func)                                                          \
  static void BM_##name(benchmark::State &state) {                                     \
    for (auto _ : state) {                                                             \
      func(g_data.ret, g_data.M, g_data.kx.data(), g_data.ky.data(), g_data.kz.data(), \
           g_data.N1, g_data.N2, g_data.N3, g_data.bin_size_x, g_data.bin_size_y,      \
           g_data.bin_size_z);                                                         \
    }                                                                                  \
    state.SetItemsProcessed(state.iterations() * g_data.M);                            \
    state.SetBytesProcessed(                                                           \
        state.iterations() * g_data.M * (3 * sizeof(FLT) + sizeof(BIGINT)));           \
  }                                                                                    \
  BENCHMARK(BM_##name)->Unit(benchmark::kMillisecond)->MinTime(1.0)

DEFINE_BM(scalar, bin_sort_scalar<FLT>);
DEFINE_BM(simd_scatter, bin_sort_simd_scatter<FLT>);
DEFINE_BM(precomputed, bin_sort_precomputed<FLT>);
DEFINE_BM(simd_noscatter, bin_sort_simd_noscatter<FLT>);
DEFINE_BM(noscatter_i32, bin_sort_noscatter_i32<FLT>);
DEFINE_BM(noscatter_prefetch, bin_sort_noscatter_prefetch<FLT>);
DEFINE_BM(precomp_i32, bin_sort_precomp_i32<FLT>);
DEFINE_BM(precomp_i32_prefetch, bin_sort_precomp_i32_prefetch<FLT>);
DEFINE_BM(precomp_u32_interleaved, bin_sort_precomp_u32_interleaved<FLT>);
DEFINE_BM(noscatter_u32_unrolled, bin_sort_noscatter_u32_unrolled<FLT>);
DEFINE_BM(precomp_u32_interleaved_pf, bin_sort_precomp_u32_interleaved_pf<FLT>);
DEFINE_BM(precomp_u32, bin_sort_precomp_u32<FLT>);

int main(int argc, char **argv) {
  // Parse custom args before Google Benchmark consumes them
  UBIGINT M  = UBIGINT(1e7);
  UBIGINT N1 = 256;
  int dims   = 3;

  // Look for --M=, --N1=, --dims= before the standard benchmark args
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--M=", 4) == 0)
      M = UBIGINT(atof(argv[i] + 4));
    else if (strncmp(argv[i], "--N1=", 5) == 0)
      N1 = UBIGINT(atof(argv[i] + 5));
    else if (strncmp(argv[i], "--dims=", 7) == 0)
      dims = atoi(argv[i] + 7);
  }

  g_data.init(M, N1, dims);

  auto bp = compute_bin_params(g_data.N1, g_data.N2, g_data.N3, g_data.bin_size_x,
                               g_data.bin_size_y, g_data.bin_size_z);
  printf("bin_sort benchmark: M=%llu, N1=%llu, dims=%d\n", (unsigned long long)M,
         (unsigned long long)N1, dims);
  printf("  SIMD: %zu doubles, %zu floats. nbins=%lld, ~pts/bin=%.1f\n",
         xsimd::batch<double>::size, xsimd::batch<float>::size, (long long)bp.nbins,
         double(M) / bp.nbins);
  printf("  precision: %s\n", sizeof(FLT) == 4 ? "single" : "double");

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
