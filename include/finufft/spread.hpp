#pragma once

#include <finufft/plan.hpp>
#include <finufft/simd.hpp>
#include <finufft/utils.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>

#include <poet/poet.hpp> // poet::dispatch / inclusive_range / dispatch_param

namespace finufft::spreadinterp {
// SIMD min/max over a contiguous array. Same contract as utils::arrayrange but
// vectorized: the scalar form there stores through *lo/*hi each iteration, which
// GCC cannot prove non-aliasing with a[], and its min/max reduction won't
// auto-vectorize without -ffinite-math-only (unsafe: INFINITY is the n==0
// sentinel). Used by the hot get_subgrid bounding box (was ~8% of spread time).
template<typename T>
FINUFFT_ALWAYS_INLINE void simd_arrayrange(int64_t n, const T *a, T *lo, T *hi) noexcept {
  using batch = xsimd::batch<T>;
  static constexpr int64_t W = batch::size;
  if (n < W) { // includes n==0 -> lo=+inf, hi=-inf (arrayrange contract)
    T l = INFINITY, h = -INFINITY;
    for (int64_t m = 0; m < n; ++m) {
      l = std::min(l, a[m]);
      h = std::max(h, a[m]);
    }
    *lo = l;
    *hi = h;
    return;
  }
  batch vlo = batch::load_unaligned(a), vhi = vlo;
  int64_t m = W;
  for (; m + W <= n; m += W) {
    const batch v = batch::load_unaligned(a + m);
    vlo = xsimd::min(vlo, v);
    vhi = xsimd::max(vhi, v);
  }
  if (m < n) { // last full window ending at n; min/max idempotent on the overlap
    const batch v = batch::load_unaligned(a + n - W);
    vlo = xsimd::min(vlo, v);
    vhi = xsimd::max(vhi, v);
  }
  *lo = xsimd::reduce_min(vlo);
  *hi = xsimd::reduce_max(vhi);
}
} // namespace finufft::spreadinterp

// ---------- FINUFFT_PLAN_T spread_subproblem_*_kernel method definitions ----------
// FINUFFT_PLAN_T is already defined via the transitive include chain:
//   simd.hpp -> finufft/plan.hpp

template<typename TF>
template<int NS, int NC>
FINUFFT_NEVER_INLINE void FINUFFT_PLAN_T<TF>::spread_subproblem_1d_kernel(
    const BIGINT off1, TF *FINUFFT_RESTRICT du, const UBIGINT M, const TF *const kx,
    const TF *const dd) const noexcept {
  /* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
     Inputs:
     off1 - integer offset of left end of du subgrid from that of overall fine
            periodized output grid {0,1,...N-1}.
     size1 - integer length of output subgrid du
     M - number of NU pts in subproblem
     kx (length M) - are rescaled NU source locations, should lie in
                     [off1+ns/2,off1+size1-1-ns/2] so as kernels stay in bounds
     dd (length M complex, interleaved) - source strengths
     Outputs:
     du (length size1 complex, interleaved) - preallocated uniform subgrid array,
          zero on entry: the kernel accumulates into it (see spreadSorted)

     The reason periodic wrapping is avoided in subproblems is speed: avoids
     conditionals, indirection (pointers), and integer mod. Originally 2017.
     Kernel eval mods by Ludvig al Klinteberg.
     Fixed so rounding to integer grid consistent w/ get_subgrid, prevents
     chance of segfault when epsmach*N1>O(1), assuming max() and ceil() commute.
     This needed off1 as extra arg. AHB 11/30/20.
     Vectorized using xsimd by M. Barbone 06/24.
     Previous arg horner_coeffs_ptr is now read from plan member horner_coeffs.data().
     Converted to class member, Barbone 2/24/26.
  */
  using namespace finufft::spreadinterp;
  using T                         = TF;
  using KBL                       = KernelBufferLayout<T, NS>;
  using simd_type                 = typename KBL::simd_type;
  using arch_t                    = typename KBL::arch_t;
  static constexpr auto padding   = get_padding<T, 2 * NS>();
  static constexpr auto simd_size = KBL::simd_size;
  static constexpr auto ns2       = NS * T(0.5); // half spread width
  const T *horner_coeffs_ptr      = m.horner_coeffs.data();
  // something weird here. Reversing ker{0} and std fill causes ker
  // to be zeroed inside the loop GCC uses AVX, clang AVX2
  alignas(KBL::alignment) std::array<T, KBL::stride> ker{0};
  // no padding needed if MAX_NSPREAD is 16
  // the largest read is 16 floats with avx512
  // if larger instructions will be available or half precision is used, this should be
  // padded
  for (uint64_t i{0}; i < M; i++) {
    // loop over NU pts
    // initializes a dd_pt that is const
    // should not make a difference in performance
    // but is a hint to the compiler that after the lambda
    // dd_pt is not modified and can be kept as is in a register
    // given (re, im) in this case dd[i*2] and dd[i*2+1]
    // this function returns a simd register of size simd_size
    // initialized as follows:
    // +-----------------------+
    // |re|im|re|im|re|im|re|im|
    // +-----------------------+
    const auto dd_pt = initialize_complex_register<simd_type>(dd[i * 2], dd[i * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = BIGINT(std::ceil(kx[i] - ns2)); // fine grid start index
    // T(i1) has different semantics and results an extra cast
    const auto x1 = [i, kx]() constexpr noexcept {
      auto x1 = std::ceil(kx[i] - ns2) - kx[i]; // x1 in [-w/2,-w/2+1], up to rounding
      // However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
      // kernel evaluation will fall outside their designed domains, >>1 errors.
      // This can only happen if the overall error would be O(1) anyway. Clip x1??
      if (x1 < -ns2) x1 = -ns2;
      if (x1 > -ns2 + 1) x1 = -ns2 + 1;
      return x1;
    }();
    // Libin improvement: pass ker as a parameter and allocate it outside the loop
    // gcc13 + 10% speedup (relative to const auto ker = evaluate_kernel_vec...etc).
    evaluate_kernel_vector<NS, NC, T, simd_type>(ker.data(), horner_coeffs_ptr, x1);
    const auto j = i1 - off1; // offset rel to subgrid, starts the output indices
    auto *FINUFFT_RESTRICT trg = du + 2 * j; // restrict helps compiler to vectorize
    // du is padded, so we can use SIMD even if we write more than ns values in du
    // ker is also padded.
    // regular_part, source Agner Fog
    // [VCL](https://www.agner.org/optimize/vcl_manual.pdf)
    // Given 2*ns+padding=L so that L = M*simd_size
    // if M is even then regular_part == M else regular_part == (M-1) * simd_size
    // this means that the elements from regular_part to L are a special case that
    // needs a different handling. These last elements are not computed in the loop but,
    // the if constexpr block at the end of the loop takes care of them.
    // This allows to save one load at each loop iteration.
    // The special case, allows to minimize padding otherwise out of bounds access.
    // See below for the details.
    static constexpr auto regular_part =
        finufft::utils::round_down<2 * simd_size>(std::size_t(2 * NS + padding));
    // this loop increment is 2*simd_size by design
    // it allows to save one load this way at each iteration

    // This does for each element e of the subgrid, x1 defined above and pt the NU point
    // the following: e += scaled_kernel(2*x1/n_s)*pt, where "scaled_kernel" is defined
    // on [-1,1].
    // Using uint8_t in loops to favor unrolling.
    // Most compilers limit the unrolling to 255, uint8_t is at most 255
    for (uint8_t dx{0}; dx < regular_part; dx += 2 * simd_size) {
      // read ker_v which is simd_size wide from ker
      // ker_v looks like this:
      // +-----------------------+
      // |y0|y1|y2|y3|y4|y5|y6|y7|
      // +-----------------------+
      const auto ker_v = simd_type::load_aligned(ker.data() + dx / 2);
      // read 2*SIMD vectors from the subproblem grid
      const auto du_pt0 = simd_type::load_unaligned(trg + dx);
      const auto du_pt1 = simd_type::load_unaligned(trg + dx + simd_size);
      // swizzle is faster than zip_lo(ker_v, ker_v) and zip_hi(ker_v, ker_v)
      // swizzle in this case is equivalent to zip_lo and zip_hi respectively
      const auto ker0low = xsimd::swizzle(ker_v, zip_low_index<arch_t, T>);
      // ker 0 looks like this now:
      // +-----------------------+
      // |y0|y0|y1|y1|y2|y2|y3|y3|
      // +-----------------------+
      const auto ker0hi = xsimd::swizzle(ker_v, zip_hi_index<arch_t, T>);
      // ker 1 looks like this now:
      // +-----------------------+
      // |y4|y4|y5|y5|y6|y6|y7|y7|
      // +-----------------------+
      // same as before each element of the subproblem grid is multiplied by the
      // corresponding element of the kernel since dd_pt is re|im interleaves res0 is also
      // correctly re|im interleaved
      // doing this for two SIMD vectors at once allows to fully utilize ker_v instead of
      // wasting the higher half
      const auto res0 = xsimd::fma(ker0low, dd_pt, du_pt0);
      const auto res1 = xsimd::fma(ker0hi, dd_pt, du_pt1);
      res0.store_unaligned(trg + dx);
      res1.store_unaligned(trg + dx + simd_size);
    }
    // sanity check at compile time that all the elements are computed
    static_assert(regular_part + simd_size >= 2 * NS);
    // case where the 2*ns is not a multiple of 2*simd_size
    // checking 2*ns instead of 2*ns+padding as we do not need to compute useless zeros...
    if constexpr (regular_part < 2 * NS) {
      // here we need to load the last kernel values,
      // but we can avoid computing extra padding
      // also this padding will result in out-of-bounds access to trg
      // The difference between this and the loop is that ker0hi is not computed and
      // the corresponding memory is not accessed
      const auto ker0    = simd_type::load_unaligned(ker.data() + (regular_part / 2));
      const auto du_pt   = simd_type::load_unaligned(trg + regular_part);
      const auto ker0low = xsimd::swizzle(ker0, zip_low_index<arch_t, T>);
      const auto res     = xsimd::fma(ker0low, dd_pt, du_pt);
      res.store_unaligned(trg + regular_part);
    }
  }
}

template<typename TF>
template<int NS, int NC>
FINUFFT_NEVER_INLINE void FINUFFT_PLAN_T<TF>::spread_subproblem_2d_kernel(
    const BIGINT off1, const BIGINT off2, const UBIGINT size1, TF *FINUFFT_RESTRICT du,
    const UBIGINT M, const TF *kx, const TF *ky, const TF *dd) const noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array, zero on entry
   For algoritmic details see spread_subproblem_1d_kernel.
   Previous arg horner_coeffs_ptr is now read from plan member horner_coeffs.data().
   Converted to class member, Barbone 2/24/26.
*/
{
  using namespace finufft::spreadinterp;
  using T                         = TF;
  using KBL                       = KernelBufferLayout<T, NS>;
  using simd_type                 = typename KBL::simd_type;
  using arch_t                    = typename KBL::arch_t;
  static constexpr auto padding   = get_padding<T, 2 * NS>();
  static constexpr auto simd_size = KBL::simd_size;
  const T *horner_coeffs_ptr      = m.horner_coeffs.data();
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  static constexpr auto ns2 = NS * T(0.5);  // half spread width
  alignas(KBL::alignment) std::array<T, 2 * KBL::stride> kernel_values{0};
  for (uint64_t pt = 0; pt < M; pt++) {
    // loop over NU pts
    const auto dd_pt = initialize_complex_register<simd_type>(dd[pt * 2], dd[pt * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT)std::ceil(kx[pt] - ns2); // fine grid start indices
    const auto i2 = (BIGINT)std::ceil(ky[pt] - ns2);
    const auto x1 = (T)std::ceil(kx[pt] - ns2) - kx[pt];
    const auto x2 = (T)std::ceil(ky[pt] - ns2) - ky[pt];
    evaluate_kernel_vector<NS, NC, T, simd_type>(kernel_values.data(), horner_coeffs_ptr,
                                                 x1, x2);
    const auto *ker1 = kernel_values.data();
    const auto *ker2 = kernel_values.data() + KBL::stride;
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    static constexpr uint8_t kerval_vectors = (2 * NS + padding) / simd_size;
    static_assert(kerval_vectors > 0, "kerval_vectors must be greater than 0");
    // wrapping this in a lambda gives an extra 10% speedup (gcc13)
    // the compiler realizes the values are constant after the lambda
    // Guess: it realizes what is the invariant and moves some operations outside the loop
    //        it might also realize that some variables are not needed anymore and can
    //        re-use the registers with other data.
    const auto ker1val_v = [ker1, dd_pt]() constexpr noexcept {
      // array of simd_registers that will store the kernel values
      std::array<simd_type, kerval_vectors> ker1val_v{};
      // similar to the 1D case, we compute the kernel values in advance
      // and store them in simd_registers.
      // Compared to the 1D case the difference is that here ker values are stored in
      // an array of simd_registers.
      // This is a hint to the compiler to keep the values in registers, instead of
      // pushing them to the stack.
      // Same as the 1D case, the loop is structured in a way to half the number of loads
      // This cause an issue with the last elements, but this is handled in the
      // `if constexpr`.
      // For more details please read the 1D case. The difference is that
      // here the loop is on the number of simd vectors In the 1D case the loop is on the
      // number of elements in the kernel
      for (uint8_t i = 0; i < (kerval_vectors & ~1); // NOLINT(*-too-small-loop-variable)
           i += 2) {
        const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>);
        const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t, T>);
        // this initializes the entire vector registers with the same value
        // the ker1val_v[i] looks like this:
        // +-----------------------+
        // |y0|y0|y0|y0|y0|y0|y0|y0|
        // +-----------------------+
        ker1val_v[i]     = ker1low * dd_pt;
        ker1val_v[i + 1] = ker1hi * dd_pt; // same as above
      }
      if constexpr (kerval_vectors % 2) {
        const auto ker1_v =
            simd_type::load_unaligned(ker1 + (kerval_vectors - 1) * simd_size / 2);
        const auto res = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>) * dd_pt;
        ker1val_v[kerval_vectors - 1] = res;
      }
      return ker1val_v;
    }();

    // critical inner loop:
    for (auto dy = 0; dy < NS; ++dy) {
      const auto j = size1 * (i2 - off2 + dy) + i1 - off1; // should be in subgrid
      auto *FINUFFT_RESTRICT trg = du + 2 * j;
      const simd_type kerval_v(ker2[dy]);
      for (uint8_t i = 0; i < kerval_vectors; ++i) {
        const auto trg_v  = simd_type::load_unaligned(trg + i * simd_size);
        const auto result = xsimd::fma(kerval_v, ker1val_v[i], trg_v);
        result.store_unaligned(trg + i * simd_size);
      }
    }
  }
}

template<typename TF>
template<int NS, int NC>
FINUFFT_NEVER_INLINE void FINUFFT_PLAN_T<TF>::spread_subproblem_3d_kernel(
    const BIGINT off1, const BIGINT off2, const BIGINT off3, const UBIGINT size1,
    const UBIGINT size2, TF *FINUFFT_RESTRICT du, const UBIGINT M, const TF *kx,
    const TF *ky, const TF *kz, const TF *dd) const noexcept
// 3D version of spread_subproblem_1d_kernel. du is zero on entry.
// Previous arg horner_coeffs_ptr is now read from plan member horner_coeffs.data().
// Converted to class member, Barbone 2/24/26.
{
  using namespace finufft::spreadinterp;
  using T                         = TF;
  using KBL                       = KernelBufferLayout<T, NS>;
  using simd_type                 = typename KBL::simd_type;
  using arch_t                    = typename KBL::arch_t;
  static constexpr auto padding   = get_padding<T, 2 * NS>();
  static constexpr auto simd_size = KBL::simd_size;
  const T *horner_coeffs_ptr      = m.horner_coeffs.data();

  static constexpr auto ns2 = NS * T(0.5); // half spread width
  alignas(KBL::alignment) std::array<T, 3 * KBL::stride> kernel_values{0};

  for (uint64_t pt = 0; pt < M; pt++) {
    // loop over NU pts
    const auto dd_pt = initialize_complex_register<simd_type>(dd[pt * 2], dd[pt * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT)std::ceil(kx[pt] - ns2); // fine grid start indices
    const auto i2 = (BIGINT)std::ceil(ky[pt] - ns2);
    const auto i3 = (BIGINT)std::ceil(kz[pt] - ns2);
    const auto x1 = std::ceil(kx[pt] - ns2) - kx[pt];
    const auto x2 = std::ceil(ky[pt] - ns2) - ky[pt];
    const auto x3 = std::ceil(kz[pt] - ns2) - kz[pt];

    evaluate_kernel_vector<NS, NC, T, simd_type>(kernel_values.data(), horner_coeffs_ptr,
                                                 x1, x2, x3);
    const auto *ker1 = kernel_values.data();
    const auto *ker2 = kernel_values.data() + KBL::stride;
    const auto *ker3 = kernel_values.data() + 2 * KBL::stride;
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    // kerval_vectors is the number of SIMD iterations needed to compute all the elements
    static constexpr uint8_t kerval_vectors = (2 * NS + padding) / simd_size;
    static_assert(kerval_vectors > 0, "kerval_vectors must be greater than 0");
    const auto ker1val_v = [ker1, dd_pt]() constexpr noexcept {
      std::array<simd_type, kerval_vectors> ker1val_v{};
      // Iterate over kerval_vectors but in case the number of kerval_vectors is odd
      // we need to handle the last batch separately
      // to the & ~1 is to ensure that we do not iterate over the last batch if it is odd
      // as it sets the last bit to 0
      for (uint8_t i = 0; i < (kerval_vectors & ~1); // NOLINT(*-too-small-loop-variable
           i += 2) {
        const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>);
        const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t, T>);
        ker1val_v[i]       = ker1low * dd_pt;
        ker1val_v[i + 1]   = ker1hi * dd_pt;
      }
      // (at compile time) check if the number of kerval_vectors is odd
      // if it is we need to handle the last batch separately
      if constexpr (kerval_vectors % 2) {
        const auto ker1_v =
            simd_type::load_unaligned(ker1 + (kerval_vectors - 1) * simd_size / 2);
        const auto res = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>) * dd_pt;
        ker1val_v[kerval_vectors - 1] = res;
      }
      return ker1val_v;
    }();
    // critical inner loop:
    for (uint8_t dz{0}; dz < NS; ++dz) {
      const auto oz = size1 * size2 * (i3 - off3 + dz); // offset due to z
      for (uint8_t dy{0}; dy < NS; ++dy) {
        const auto j = oz + size1 * (i2 - off2 + dy) + i1 - off1; // should be in subgrid
        auto *FINUFFT_RESTRICT trg = du + 2 * j;
        const simd_type kerval_v(ker2[dy] * ker3[dz]);
        for (uint8_t i{0}; i < kerval_vectors; ++i) {
          const auto trg_v  = simd_type::load_unaligned(trg + i * simd_size);
          const auto result = xsimd::fma(kerval_v, ker1val_v[i], trg_v);
          result.store_unaligned(trg + i * simd_size);
        }
      }
    }
  }
}

template<typename TF>
template<typename OnRun>
void FINUFFT_PLAN_T<TF>::walk_wrapped_subgrid(const Subgrid &sub, OnRun &&on_run) const
/* Walk the contiguous runs the subgrid shares with the fine grid.
   Calls on_run(gi, si, n) with the grid offset, the subgrid offset, and the run length
   (all counted in real elements). Wraps periodically onto the N1,N2,N3 box.
   Works in all dimensions. A run that falls entirely outside the box has n == 0, so
   on_run must accept n == 0.
*/
{
  const BIGINT N1 = m.nfdim[0], N2 = m.nfdim[1], N3 = m.nfdim[2];
  // A subgrid overhangs by less than one period on either side, so one shift wraps
  // any index. A subgrid wider than the box overhangs both edges; the runs below
  // then cover the box once and the wrapped pieces add onto it again.
  const auto wrap = [](BIGINT i, BIGINT N) {
    if (i < 0) return i + N;
    if (i >= N) return i - N;
    return i;
  };
  // How much of a row hangs off each x edge of the box.
  const BIGINT below  = std::max(BIGINT(0), -sub.off1);
  const BIGINT above  = std::max(BIGINT(0), sub.off1 + sub.size1 - N1);
  const BIGINT inside = sub.size1 - below - above;
  for (BIGINT dz = 0; dz < sub.size3; dz++) {
    const BIGINT oz = N1 * N2 * wrap(sub.off3 + dz, N3);              // 0 below 3D
    for (BIGINT dy = 0; dy < sub.size2; dy++) {
      const BIGINT oy = N1 * wrap(sub.off2 + dy, N2) + oz;            // 0 in 1D
      const BIGINT si = 2 * sub.padded_size1 * (dy + sub.size2 * dz); // subgrid row
      // the three runs of one row: what wraps below the x edge, what lies inside the box,
      // and what wraps above it
      on_run(2 * (oy + sub.off1 + N1), si, 2 * below);
      on_run(2 * (oy + sub.off1) + 2 * below, si + 2 * below, 2 * inside);
      on_run(2 * (oy + sub.off1 - N1) + 2 * (below + inside), si + 2 * (below + inside),
             2 * above);
    }
  }
}

template<typename TF>
template<bool thread_safe>
void FINUFFT_PLAN_T<TF>::drain_wrapped_subgrid(
    const Subgrid &sub, TF *FINUFFT_RESTRICT data_uniform, TF *du0) const
/* Add a large subgrid (du0) to output grid (data_uniform), with periodic wrapping, and
   zero du0 in the same pass. The thread_safe variant adds atomically, so any number of
   subproblems may write one grid at once; the plain variant is for a writer that owns the
   grid alone.
   The zeroing rides along because the runs cover every cell of the box exactly once. The
   cells no run reaches, the anti-alias gap and the tail, stay zero on their own: the
   store past a row's end carries zeroed kernel lanes, so it writes back what it read.
   Atomic writes: R Blackwell, Nov 2020; the two variants merged into one function,
   M. Barbone 06/24.
   Previous args (N1, N2, N3) are now read from plan member nfdim[0..2].
   Converted to class member, Barbone 2/24/26.
   The add back carries the zeroing, M. Barbone 8/26/26.
*/
{
  walk_wrapped_subgrid(sub, [&](BIGINT gi, BIGINT si, BIGINT n) {
    for (BIGINT j = 0; j < n; ++j) {
      if constexpr (thread_safe) { // NOLINT(*-branch-clone)
#pragma omp atomic
        data_uniform[gi + j] += du0[si + j];
      } else {
        data_uniform[gi + j] += du0[si + j];
      }
      du0[si + j] = 0;
    }
  });
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::copy_wrapped_subgrid(const Subgrid &sub, const TF *data_uniform,
                                              TF *FINUFFT_RESTRICT du0) const
/* Read the subgrid (du0) back out of the input grid (data_uniform), the transpose of
   drain_wrapped_subgrid: same box, same wrapping, values copied instead of added. Every
   point of a subproblem then interpolates out of one cache-sized block rather than out of
   the whole fine grid. The overread at the end of a row falls in the next row, or in the
   buffer tail on the last row; the interp kernels weight it by the zeroed kernel lanes,
   so its value never reaches the output (see interp_line) (M. Barbone 8/25/26).
*/
{
  walk_wrapped_subgrid(sub, [&](BIGINT gi, BIGINT si, BIGINT n) {
    std::copy_n(data_uniform + gi, n, du0 + si);
  });
}

// ---------- Tiled spreading policy ----------
// Tile size and the subproblem cut: pure functions of the tile layout, the point count
// and the thread geometry, so a test can check them on a synthetic layout.

// Subproblems one thread draws in spread_schedule, so a thread that takes a long one
// still ends within 1/K of the ideal makespan.
constexpr UBIGINT spread_subproblems_per_thread = 4;

// The fraction of one core's L2 the point budget below may fill, as its reciprocal.
constexpr UBIGINT spread_l2_share               = 4;
// Bytes one point is counted as. A fixed count, not sizeof: fp32 does better on the
// smaller tile the same count gives it.
constexpr UBIGINT spread_bytes_per_point        = 16;

// Points whose strengths fill that fraction of L2: the budget the tile sizer aims at and
// the cap a single subproblem may hold.
inline UBIGINT spread_point_budget() noexcept {
  return UBIGINT(finufft::utils::getL2CacheSize()) /
         (spread_l2_share * spread_bytes_per_point);
}

// x to the power of the dimension.
inline double spread_pow_ndims(double x, int ndims) noexcept {
  double v = 1;
  for (int d = 0; d < ndims; ++d) v *= x;
  return v;
}

inline SpreadSchedule spread_schedule(const SpreadTileData &tiles, UBIGINT M,
                                      UBIGINT grid_cells, int nthr, int batchSize) {
  // Equal pieces of the point list, one piece at a time out of each tile: the pieces of a
  // tile share that tile's box, so a tile over the cap becomes that many subproblems of
  // the same box rather than one that holds a thread for several tiles' worth of work.
  const auto cut = [](const std::vector<BIGINT> &starts, UBIGINT cap) {
    SpreadSchedule sched;
    sched.points_per_subproblem = cap;
    // one bound per tile plus one per split, so the whole cut takes one allocation
    sched.bounds.reserve(starts.size() + UBIGINT(starts.back()) / cap + 1);
    sched.bounds.push_back(0);
    for (size_t t = 1; t < starts.size(); ++t) {
      const UBIGINT lo = sched.bounds.back(), hi = UBIGINT(starts[t]);
      if (hi <= lo) continue; // empty tile
      const UBIGINT pieces = 1 + (hi - lo - 1) / cap;
      for (UBIGINT c = 1; c < pieces; ++c)
        sched.bounds.push_back(lo + (hi - lo) * c / pieces);
      sched.bounds.push_back(hi);
    }
    return sched;
  };
  // The threads want K subproblems each; one vector of the batch per thread already fills
  // the machine, so the count divides by the threads sharing a vector.
  const UBIGINT threads_per_vector = (UBIGINT(nthr) + batchSize - 1) / batchSize;
  // An unsorted point list is one tile spanning the whole fine grid. A subproblem there
  // pays one pass over its whole box to drain it and saves the gather of its own points,
  // so it holds max(point budget, grid cells) and never more than a thread's share.
  if (tiles.starts.size() < 2) {
    const UBIGINT cap = std::min((M + threads_per_vector - 1) / threads_per_vector,
                                 std::max(spread_point_budget(), grid_cells));
    return cut({0, BIGINT(M)}, std::max(cap, UBIGINT(1)));
  }

  // One subproblem per non-empty cache tile, read straight off the tile offsets. A cache
  // tile's padded subgrid fits L2 by construction, so two ceilings cap the points one
  // subproblem holds: twice what an average filled tile holds, and the point budget.
  const auto &starts   = tiles.starts;
  UBIGINT filled_tiles = 0;                          // tiles holding at least one point
  for (size_t t = 1; t < starts.size(); ++t) filled_tiles += starts[t] > starts[t - 1];
  filled_tiles = std::max(filled_tiles, UBIGINT(1)); // an empty layout must not divide by
                                                     // 0
  UBIGINT cap  = std::min(2 * M / filled_tiles, spread_point_budget());
  // Lower the cap only when the filled tiles cannot supply the K subproblems a thread
  // wants, since a cap near the occupancy a tile typically has would split half the tiles
  // for nothing.
  const UBIGINT wanted = spread_subproblems_per_thread * threads_per_vector;
  if (filled_tiles < wanted) cap = std::min(cap, 1 + (M - 1) / wanted);
  return cut(starts, std::max(cap, UBIGINT(1)));
}

// Doublings of the fine cell per spread tile edge; zero keeps the tile at one cell. One
// core's L2 sets the size: the strengths want a quarter of it, the padded subgrid (the
// tile grown by the kernel width) all of it, so the ceiling is on the padded edge.

// TODO: the tile pays for its halo whatever the density, and empty tiles cost nothing,
// so what is left to win is a tile edge that grows with the halo it has to pay for.
template<class TF>
inline int spread_tile_doublings(int cell, int ndims, int nspread,
                                 double density) noexcept {
  // What the subproblem allocates, not the bare padded tile: set_row_layout may add an
  // anti-alias line to the row stride, and cells() one tail past the last row. The line
  // is counted always, since which sizes take it is not known until the points are in.
  constexpr double line = 64.0 / double(2 * sizeof(TF)); // complex cells per cache line
  const double tail     = double(finufft::spreadinterp::get_padding<TF>(2 * nspread) / 2);
  const auto padded_fits_l2 = [=](double edge) {
    const double rows = spread_pow_ndims(edge + nspread, ndims - 1);
    // all of L2, in the fixed spread_bytes_per_point unit, so fp32 gets no wider a tile
    return (edge + nspread + line) * rows + tail <=
           double(finufft::utils::getL2CacheSize() / spread_bytes_per_point);
  };
  // Grow while the next doubling keeps the padded subgrid in L2 and the tile's strengths
  // in a quarter of it. The point budget only caps growth: a tile over it is split into
  // subproblems of the cap instead.
  int doublings = 0;
  for (double edge = 2.0 * cell;
       padded_fits_l2(edge) &&
       density * spread_pow_ndims(edge, ndims) <= double(spread_point_budget());
       edge *= 2)
    ++doublings;
  return doublings;
}

// SIMD-vectorized bin sort helpers, templated on ndims to eliminate branching.
// Called by the FINUFFT_PLAN_T methods below via runtime ndims dispatch.
namespace {

// Two-level count-sort key: the tile index high, the fine cell inside that tile low. A
// run of cells_per_tile consecutive bins is one cuboid tile of the fine grid, so
// spreadSorted and interpSorted take their subproblems straight off the tile offsets.
// cell_bits==0 makes a tile one cell and leaves the plain row-major cell index.
template<int ndims> struct TileKey {
  BIGINT cell_bits, cell_mask; // log2 of the cells per tile edge, and that minus one
  BIGINT tile_bits;            // cell_bits*ndims: bits the tile index moves up by
  BIGINT nt1, nt2, nt3;        // tiles along x, y and z
  BIGINT cells_per_tile;       // fine cells one tile holds
  BIGINT ntiles, nbins;        // nbins == ntiles*cells_per_tile

  TileKey(int cell_bits_, BIGINT nb1, BIGINT nb2, BIGINT nb3)
      : cell_bits(cell_bits_), cell_mask((BIGINT(1) << cell_bits_) - 1),
        tile_bits(cell_bits * ndims) {
    const auto tiles_along = [&](BIGINT nb) {
      return (nb + cell_mask) >> cell_bits;
    };
    nt1            = tiles_along(nb1);
    nt2            = ndims > 1 ? tiles_along(nb2) : 1;
    nt3            = ndims > 2 ? tiles_along(nb3) : 1;
    cells_per_tile = BIGINT(1) << tile_bits;
    ntiles         = nt1 * nt2 * nt3;
    nbins          = ntiles * cells_per_tile;
  }
  // The tile layout the schedule reads back, alongside the offsets.
  void geometry(SpreadTileData &td, int cell, UBIGINT n1, UBIGINT n2, UBIGINT n3) const {
    td.edge  = cell << cell_bits;
    td.nt    = {nt1, nt2, nt3};
    td.ngrid = {BIGINT(n1), BIGINT(n2), BIGINT(n3)};
  }
  // c1,c2,c3 are fine-cell indices, batches or scalars alike; unused dims pass 0
  template<typename I> I operator()(I c1, I c2, I c3) const {
    I tile = c1 >> cell_bits, cell = c1 & cell_mask;
    if constexpr (ndims > 1) {
      tile = tile + I(nt1) * (c2 >> cell_bits);
      cell = cell | ((c2 & cell_mask) << cell_bits);
    }
    if constexpr (ndims > 2) {
      tile = tile + I(nt1 * nt2) * (c3 >> cell_bits);
      cell = cell | ((c3 & cell_mask) << (2 * cell_bits));
    }
    return (tile << tile_bits) | cell;
  }
};

// The bin a point falls in, one SIMD batch or one point at a time. Both sorts ask the
// same question, so the coords, the box and the key live here.
template<typename T, int ndims> struct BinIndexer {
  using simd_type                 = xsimd::batch<T>;
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = simd_type::arch_type::alignment();

  const T *kx, *ky, *kz;
  UBIGINT N1, N2, N3;
  T inv; // reciprocal cell size, one cubic cell for every axis
  TileKey<ndims> key;

  BinIndexer(const T *kx_, const T *ky_, const T *kz_, UBIGINT n1, UBIGINT n2, UBIGINT n3,
             int cell, int cell_bits)
      : kx(kx_), ky(ky_), kz(kz_), N1(n1), N2(n2), N3(n3), inv(T(1.0 / cell)),
        // the +1 leaves room for round-off giving i1 = N1/cell where exact arithmetic
        // gives 0..N1-1, for kx near +pi; round-off near -pi stably rounds negative to 0
        key(cell_bits, BIGINT(T(n1) * inv + 1), ndims > 1 ? BIGINT(T(n2) * inv + 1) : 1,
            ndims > 2 ? BIGINT(T(n3) * inv + 1) : 1) {}

  // The bins of the simd_size points at offset, as a plain array to walk. A scatter into
  // the histogram was rejected: duplicate-bin conflicts dominate.
  auto batch(UBIGINT offset) const noexcept {
    const auto cell = [&](const T *k, UBIGINT N) {
      return xsimd::to_int(
          finufft::spreadinterp::fold_rescale(simd_type::load_unaligned(k + offset), N) *
          simd_type(inv));
    };
    const auto c1 = cell(kx, N1);
    auto c2 = decltype(c1)(0), c3 = decltype(c1)(0);
    if constexpr (ndims > 1) c2 = cell(ky, N2);
    if constexpr (ndims > 2) c3 = cell(kz, N3);
    const auto bins = key(c1, c2, c3);
    alignas(alignment) std::array<typename decltype(bins)::value_type, simd_size> arr{};
    bins.store_aligned(arr.data());
    return arr;
  }
  // The bin of one point, for the tail the batch loop leaves.
  BIGINT at(UBIGINT i) const noexcept {
    const auto cell = [&](const T *k, UBIGINT N) {
      return BIGINT(finufft::spreadinterp::fold_rescale<T>(k[i], N) * inv);
    };
    return key(cell(kx, N1), ndims > 1 ? cell(ky, N2) : BIGINT(0),
               ndims > 2 ? cell(kz, N3) : BIGINT(0));
  }
};

// FIXME: bin_sort_singlethread_impl can be changed to take XYZ and nfdim directly
// instead of separate kx, ky, kz and N1, N2, N3 arguments.
template<typename T, int ndims>
inline void bin_sort_singlethread_impl(std::vector<BIGINT> &ret, UBIGINT M, const T *kx,
                                       const T *ky, const T *kz, UBIGINT N1, UBIGINT N2,
                                       UBIGINT N3, int cell, int cell_bits,
                                       SpreadTileData &tile_data_out)
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D.
 *
 * This is achieved by binning into cuboids (of given bin_size within the
 * overall box domain), then reading out the indices within these bins in a
 * Cartesian cuboid ordering (x fastest, y med, z slowest). Finally the
 * permutation is inverted, so that the good ordering is: the NU pt of index
 * ret[0], the NU pt of index ret[1], ..., NU pt of index ret[M-1].
 *
 * Inputs: M points kx,ky,kz in [-pi, pi) (folded in), box N1,N2,N3 (trailing dims 1),
 *         cubic bins of `cell` fine grid points, 2^cell_bits cells per tile edge.
 * Output: ret (preallocated to M) gets the permutation; tile_data_out the tile offsets.
 *
 * Notes: I compared RAM usage against declaring an internal vector and passing
 * back; the latter used more RAM and was slower.
 * Avoided the bins array, as in JFM's spreader of 2016,
 * tidied up, early 2017, Barnett.
 * Timings (2017): 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
 * Simplified by Martin Reinecke, 6/19/23 (no apparent effect on speed).
 *
 * Implementation: SIMD-vectorized bin index computation (xsimd::batch<T>),
 * scalar histogram accumulation (scatter/gather rejected: duplicate-bin
 * conflicts dominate). uint32_t counts halve cache footprint vs BIGINT.
 * Templated on ndims to eliminate isky/iskz branching in inner loops.
 * SIMD vectorization, uint32_t counts, ndims dispatch: Barbone 2/2026.
 */
{
  using namespace finufft::spreadinterp;
  static_assert(ndims >= 1 && ndims <= 3, "ndims must be 1, 2, or 3");
  const BinIndexer<T, ndims> bins(kx, ky, kz, N1, N2, N3, cell, cell_bits);
  const auto &key                 = bins.key;
  static constexpr auto simd_size = decltype(bins)::simd_size;

  // uint32_t counts halves cache footprint vs BIGINT (int64_t)
  std::vector<uint32_t> counts(key.nbins, 0);
  const auto simd_M = finufft::utils::round_down<simd_size>(M);
  UBIGINT i{};

  // counting pass: SIMD bin compute, scalar accumulate
  for (i = 0; i < simd_M; i += simd_size) {
    const auto batch = bins.batch(i);
    for (std::size_t j = 0; j < simd_size; ++j) ++counts[batch[j]];
  }
  for (; i < M; i++) ++counts[bins.at(i)];

  // compute the offsets directly in the counts array (Reinecke's trick)
  std::exclusive_scan(counts.begin(), counts.end(), counts.begin(), uint32_t{0});

  // placement pass: SIMD bin compute, scalar placement
  for (i = 0; i < simd_M; i += simd_size) {
    const auto batch = bins.batch(i);
    for (std::size_t j = 0; j < simd_size; ++j) {
      ret[counts[batch[j]]] = BIGINT(j + i);
      ++counts[batch[j]];
    }
  }
  for (; i < M; i++) {
    const auto bin   = bins.at(i);
    ret[counts[bin]] = BIGINT(i);
    ++counts[bin];
  }
  // after placement counts[b] is the end of bin b, so the last bin of tile t-1
  // ends exactly where tile t starts
  tile_data_out.starts.resize(key.ntiles + 1);
  tile_data_out.starts[0] = 0;
  for (BIGINT t = 1; t <= key.ntiles; ++t)
    tile_data_out.starts[t] = BIGINT(counts[t * key.cells_per_tile - 1]);
  key.geometry(tile_data_out, cell, N1, N2, N3);
}

// FIXME: same as bin_sort_singlethread_impl - can take XYZ and nfdim instead of the
// separate per-dimension arguments.
template<typename T, int ndims>
inline void bin_sort_multithread_impl(std::vector<BIGINT> &ret, UBIGINT M, const T *kx,
                                      const T *ky, const T *kz, UBIGINT N1, UBIGINT N2,
                                      UBIGINT N3, int cell, int nthr, int cell_bits,
                                      SpreadTileData &tile_data_out)
/* Mostly-OpenMP'ed version of bin_sort, SIMD-vectorized per thread.
   Templated on ndims to eliminate branching in inner loops.
   For documentation see: bin_sort_singlethread_impl.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Originally by Barnett 2/8/18
   Explicit #threads control argument 7/20/20.
   Improved by Martin Reinecke, 6/19/23 (up to 50% faster at 1 thr/core).
   SIMD vectorization with uint32_t counts, Barbone 2/2026.
   Todo: if debug, print timing breakdowns.
 */
{
  using namespace finufft::spreadinterp;
  static_assert(ndims >= 1 && ndims <= 3, "ndims must be 1, 2, or 3");
  const BinIndexer<T, ndims> bins(kx, ky, kz, N1, N2, N3, cell, cell_bits);
  const auto &key                 = bins.key;
  const auto nbins                = key.nbins;
  static constexpr auto simd_size = decltype(bins)::simd_size;

  int nt = std::min(M, UBIGINT(nthr));
  std::vector<UBIGINT> brk(nt + 1);

  for (int t = 0; t <= nt; ++t) brk[t] = (UBIGINT)(0.5 + M * t / (double)nt);

  std::vector<std::vector<uint32_t>> counts(nt);
  std::vector<uint32_t> bin_offset(nbins);
  std::vector<uint32_t> thread_totals(nt);

#pragma omp parallel num_threads(nt)
  {
    const int t            = MY_OMP_GET_THREAD_NUM();
    const auto chunk_start = brk[t];
    const auto chunk_end   = brk[t + 1];
    const auto chunk_simd =
        chunk_start + finufft::utils::round_down<simd_size>(chunk_end - chunk_start);

    // each thread allocates its own histogram
    counts[t].resize(nbins, 0);
    auto &my_counts = counts[t];

    // counting pass: SIMD bin compute, scalar accumulate
    UBIGINT i;
    for (i = chunk_start; i < chunk_simd; i += simd_size) {
      const auto batch = bins.batch(i);
      for (std::size_t j = 0; j < simd_size; ++j) ++my_counts[batch[j]];
    }
    for (; i < chunk_end; i++) ++my_counts[bins.at(i)];

    // ensure all threads have finished counting before computing offsets
#pragma omp barrier

    // Phase 1+2a (parallel): per-bin totals and local exclusive prefix sum.
    // Each thread owns a static chunk of bins; stores its running total.
    const BIGINT bin_chunk = (nbins + nt - 1) / nt;
    const BIGINT bin_start = t * bin_chunk;
    const BIGINT bin_end   = std::min(bin_start + bin_chunk, nbins);
    uint32_t running       = 0;
    for (BIGINT b = bin_start; b < bin_end; ++b) {
      uint32_t total = 0;
      for (int tt = 0; tt < nt; ++tt) total += counts[tt][b];
      bin_offset[b] = running;
      running += total;
    }
    thread_totals[t] = running;

#pragma omp barrier

    // Phase 2b: every thread sums the totals before its own, O(nt) each, no sync
    uint32_t thread_prefix = 0;
    for (int tt = 0; tt < t; ++tt) thread_prefix += thread_totals[tt];

    // Phase 3 (parallel): finalize global offsets and per-thread offsets
    for (BIGINT b = bin_start; b < bin_end; ++b) {
      uint32_t off = bin_offset[b] + thread_prefix;
      for (int tt = 0; tt < nt; ++tt) {
        uint32_t tmp  = counts[tt][b];
        counts[tt][b] = off;
        off += tmp;
      }
    }

#pragma omp barrier

    // thread 0's slot of a bin holds that bin's start in the sorted output, and
    // this reads the slots before the placement pass advances them
#pragma omp single
    {
      tile_data_out.starts.resize(key.ntiles + 1);
      for (BIGINT t = 0; t < key.ntiles; ++t)
        tile_data_out.starts[t] = BIGINT(counts[0][t * key.cells_per_tile]);
      tile_data_out.starts[key.ntiles] = BIGINT(M);
      key.geometry(tile_data_out, cell, N1, N2, N3);
    }

    // placement pass: SIMD bin compute, scalar placement
    for (i = chunk_start; i < chunk_simd; i += simd_size) {
      const auto batch = bins.batch(i);
      for (std::size_t j = 0; j < simd_size; ++j) {
        ret[my_counts[batch[j]]] = BIGINT(j + i);
        ++my_counts[batch[j]];
      }
    }
    for (; i < chunk_end; i++) {
      const auto bin      = bins.at(i);
      ret[my_counts[bin]] = BIGINT(i);
      ++my_counts[bin];
    }
  }
}

} // anonymous namespace

template<typename TF>
void FINUFFT_PLAN_T<TF>::bin_sort_singlethread(int cell, int cell_bits,
                                               SpreadTileData &tile_data_out) {
  using namespace finufft::spreadinterp;
  const UBIGINT N1 = m.nfdim[0], N2 = m.nfdim[1], N3 = m.nfdim[2];
  const int ndims = ndims_from_Ns(N1, N2, N3);
  if (ndims == 1)
    bin_sort_singlethread_impl<TF, 1>(m.sortIndices, m.nj, m.XYZ[0], m.XYZ[1], m.XYZ[2],
                                      N1, N2, N3, cell, cell_bits, tile_data_out);
  else if (ndims == 2)
    bin_sort_singlethread_impl<TF, 2>(m.sortIndices, m.nj, m.XYZ[0], m.XYZ[1], m.XYZ[2],
                                      N1, N2, N3, cell, cell_bits, tile_data_out);
  else
    bin_sort_singlethread_impl<TF, 3>(m.sortIndices, m.nj, m.XYZ[0], m.XYZ[1], m.XYZ[2],
                                      N1, N2, N3, cell, cell_bits, tile_data_out);
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::bin_sort_multithread(int cell, int nthr, int cell_bits,
                                              SpreadTileData &tile_data_out) {
  using namespace finufft::spreadinterp;
  const UBIGINT N1 = m.nfdim[0], N2 = m.nfdim[1], N3 = m.nfdim[2];
  const int ndims = ndims_from_Ns(N1, N2, N3);
  if (ndims == 1)
    bin_sort_multithread_impl<TF, 1>(m.sortIndices, m.nj, m.XYZ[0], m.XYZ[1], m.XYZ[2],
                                     N1, N2, N3, cell, nthr, cell_bits, tile_data_out);
  else if (ndims == 2)
    bin_sort_multithread_impl<TF, 2>(m.sortIndices, m.nj, m.XYZ[0], m.XYZ[1], m.XYZ[2],
                                     N1, N2, N3, cell, nthr, cell_bits, tile_data_out);
  else
    bin_sort_multithread_impl<TF, 3>(m.sortIndices, m.nj, m.XYZ[0], m.XYZ[1], m.XYZ[2],
                                     N1, N2, N3, cell, nthr, cell_bits, tile_data_out);
}

template<typename TF>
Subgrid FINUFFT_PLAN_T<TF>::get_subgrid(UBIGINT M, const TF *kx, const TF *ky,
                                        const TF *kz) const
/* Returns the smallest subgrid enclosing the kernel support of all M points, with
   non-periodic padding of half the kernel width to each side in every dimension. The
   points are assumed to lie in [0,Nj] for dimension j. Unused dimensions get offset 0 and
   size 1, which the calling code requires.

 Example: ndims=1, M=2, kx[0]=0.2, kx[1]=4.9, ns=3 gives off1=-1, since kx[0] spreads to
 {-1,0,1}, and size1=8, since kx[1] spreads to {4,5,6} so the subgrid is {-1,..,6}. The
 right-most index of an axis is thus off+size-1.

 The rounding of the coords to the grid must match the rounding in
 spread_subproblem_{1,2,3}d_kernel: the ceil of the coord minus ns/2 gives the left-most
 index. A mismatch segfaults the subproblem spread. This assumes max() and ceil() commute
 in the floating point implementation.

 Costs O(M) reads to find the bounds of the coord arrays, which is almost negligible in
 tests. Originally by J Magland, 2017. AHB realised the rounding issue in 6/16/17, but
 only fixed a rounding bug causing segfault in (highly inaccurate) single-precision with
 N1>>1e7 on 11/30/20. Previous args (ns, ndims) are now read from plan members
 spopts.nspread and dim. Converted to class member, Barbone 2/24/26.
*/
{
  using namespace finufft::spreadinterp;
  const int ns      = m.spopts.nspread;
  const TF half     = TF(ns) / TF(2);
  // The support of one axis: its lowest index touched by a kernel, and how many it spans.
  const auto extent = [&](const TF *k) {
    TF lo, hi;
    simd_arrayrange(int64_t(M), k, &lo, &hi);
    const BIGINT off = BIGINT(std::ceil(lo - half)); // int(ceil) first!
    return std::pair{off, BIGINT(std::ceil(hi - half)) - off + ns};
  };
  Subgrid sub;
  const auto [o1, n1] = extent(kx);
  sub.off1            = o1;
  sub.set_row_layout<TF>(n1, BIGINT(get_padding<TF>(2 * ns) / 2));
  if (dim > 1) std::tie(sub.off2, sub.size2) = extent(ky);
  if (dim > 2) std::tie(sub.off3, sub.size3) = extent(kz);
  return sub;
}

// ---------- FINUFFT_PLAN_T spread-subproblem nested caller definitions ----------
// Out-of-class definitions of the nested types declared in plan.hpp.
// Member function templates are not allowed in local classes (GCC restriction),
// so these must be proper nested class definitions of FINUFFT_PLAN_T<TF>.

template<typename TF> struct FINUFFT_PLAN_T<TF>::SpreadSubproblem1dCaller {
  const FINUFFT_PLAN_T &plan;
  const Subgrid &sub;
  TF *du;
  UBIGINT M;
  const TF *kx;
  const TF *dd;
  template<int NS, int NC>
  int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>())
      return finufft::spreadinterp::report_invalid_kernel_params(NS, NC);
    else {
      plan.template spread_subproblem_1d_kernel<NS, NC>(sub.off1, du, M, kx, dd);
      return 0;
    }
  }
};

template<typename TF> struct FINUFFT_PLAN_T<TF>::SpreadSubproblem2dCaller {
  const FINUFFT_PLAN_T &plan;
  const Subgrid &sub;
  TF *du;
  UBIGINT M;
  const TF *kx;
  const TF *ky;
  const TF *dd;
  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>())
      return finufft::spreadinterp::report_invalid_kernel_params(NS, NC);
    else {
      plan.template spread_subproblem_2d_kernel<NS, NC>(
          sub.off1, sub.off2, sub.padded_size1, du, M, kx, ky, dd);
      return 0;
    }
  }
};

template<typename TF> struct FINUFFT_PLAN_T<TF>::SpreadSubproblem3dCaller {
  const FINUFFT_PLAN_T &plan;
  const Subgrid &sub;
  TF *du;
  UBIGINT M;
  const TF *kx;
  const TF *ky;
  const TF *kz;
  const TF *dd;
  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>())
      return finufft::spreadinterp::report_invalid_kernel_params(NS, NC);
    else {
      plan.template spread_subproblem_3d_kernel<NS, NC>(sub.off1, sub.off2, sub.off3,
                                                        sub.padded_size1, sub.size2, du,
                                                        M, kx, ky, kz, dd);
      return 0;
    }
  }
};

// ---------- FINUFFT_PLAN_T spread-subproblem method definitions ----------
// FINUFFT_PLAN_T is already defined via the transitive include chain:
//   simd.hpp -> finufft/plan.hpp

template<typename TF>
void FINUFFT_PLAN_T<TF>::spread_subproblem_1d(
    const Subgrid &sub, TF *FINUFFT_RESTRICT du, UBIGINT M, const TF *kx,
    [[maybe_unused]] const TF *ky, [[maybe_unused]] const TF *kz,
    const TF *dd) const noexcept
// Spreads the M NU points (kx, dd) into the subgrid du. Uses plan members
// spopts.nspread, nc and horner_coeffs for the kernel dispatch; previous args (opts,
// horner_coeffs_ptr, nc) are now those plan members.
{
  using namespace finufft::spreadinterp;
  using namespace finufft::common;
  SpreadSubproblem1dCaller caller{*this, sub, du, M, kx, dd};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{m.spopts.nspread},
                                         poet::dispatch_param<NcSeq>{m.nc}));
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::spread_subproblem_2d(
    const Subgrid &sub, TF *FINUFFT_RESTRICT du, UBIGINT M, const TF *kx, const TF *ky,
    [[maybe_unused]] const TF *kz, const TF *dd) const noexcept
// 2D version of spread_subproblem_1d.
{
  using namespace finufft::spreadinterp;
  using namespace finufft::common;
  SpreadSubproblem2dCaller caller{*this, sub, du, M, kx, ky, dd};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{m.spopts.nspread},
                                         poet::dispatch_param<NcSeq>{m.nc}));
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::spread_subproblem_3d(const Subgrid &sub, TF *FINUFFT_RESTRICT du,
                                              UBIGINT M, const TF *kx, const TF *ky,
                                              const TF *kz, const TF *dd) const noexcept
// 3D version of spread_subproblem_1d.
{
  using namespace finufft::spreadinterp;
  using namespace finufft::common;
  SpreadSubproblem3dCaller caller{*this, sub, du, M, kx, ky, kz, dd};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{m.spopts.nspread},
                                         poet::dispatch_param<NcSeq>{m.nc}));
}
