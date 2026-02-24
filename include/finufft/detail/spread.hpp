#pragma once

#include <finufft/detail/simd_helpers.hpp>

namespace finufft::spreadinterp {

template<typename T, int ns, int nc>
FINUFFT_NEVER_INLINE void spread_subproblem_1d_kernel(
    const BIGINT off1, const UBIGINT size1, T *FINUFFT_RESTRICT du, const UBIGINT M,
    const T *const kx, const T *const dd, const T *horner_coeffs_ptr) noexcept {
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
     du (length size1 complex, interleaved) - preallocated uniform subgrid array

     The reason periodic wrapping is avoided in subproblems is speed: avoids
     conditionals, indirection (pointers), and integer mod. Originally 2017.
     Kernel eval mods by Ludvig al Klinteberg.
     Fixed so rounding to integer grid consistent w/ get_subgrid, prevents
     chance of segfault when epsmach*N1>O(1), assuming max() and ceil() commute.
     This needed off1 as extra arg. AHB 11/30/20.
     Vectorized using xsimd by M. Barbone 06/24.
  */
  using simd_type                 = PaddedSIMD<T, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto padding   = get_padding<T, 2 * ns>();
  static constexpr auto alignment = arch_t::alignment();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto ns2       = ns * T(0.5); // half spread width
  // something weird here. Reversing ker{0} and std fill causes ker
  // to be zeroed inside the loop GCC uses AVX, clang AVX2
  alignas(alignment) std::array<T, MAX_NSPREAD> ker{0};
  std::fill(du, du + 2 * size1, 0); // zero output
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
    evaluate_kernel_vector<ns, nc, T, simd_type>(ker.data(), horner_coeffs_ptr, x1);
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
    static constexpr auto regular_part = (2 * ns + padding) & (-(2 * simd_size));
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
    static_assert(regular_part + simd_size >= 2 * ns);
    // case where the 2*ns is not a multiple of 2*simd_size
    // checking 2*ns instead of 2*ns+padding as we do not need to compute useless zeros...
    if constexpr (regular_part < 2 * ns) {
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

namespace {

template<typename T> struct SpreadSubproblem1dCaller {
  BIGINT off1;
  UBIGINT size1;
  T *du;
  UBIGINT M;
  const T *kx;
  const T *dd;
  const T *horner_coeffs_ptr;

  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>()) {
      return report_invalid_kernel_params(NS, NC);
    } else {
      spread_subproblem_1d_kernel<T, NS, NC>(off1, size1, du, M, kx, dd,
                                             horner_coeffs_ptr);
      return 0;
    }
  }
};

} // namespace

template<typename T>
static void spread_subproblem_1d(BIGINT off1, UBIGINT size1, T *du, UBIGINT M, T *kx,
                                 T *dd, const finufft_spread_opts &opts,
                                 const T *horner_coeffs_ptr, int nc) noexcept {
  SpreadSubproblem1dCaller<T> caller{off1, size1, du, M, kx, dd, horner_coeffs_ptr};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  using NcSeq = make_range<MIN_NC, MAX_NC>;
  auto params =
      std::make_tuple(DispatchParam<NsSeq>{opts.nspread}, DispatchParam<NcSeq>{nc});
  dispatch(caller, params);
}

template<typename T, int ns, int nc>
FINUFFT_NEVER_INLINE static void spread_subproblem_2d_kernel(
    const BIGINT off1, const BIGINT off2, const UBIGINT size1, const UBIGINT size2,
    T *FINUFFT_RESTRICT du, const UBIGINT M, const T *kx, const T *ky, const T *dd,
    const T *horner_coeffs_ptr) noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
   For algoritmic details see spread_subproblem_1d_kernel.
*/
{
  using simd_type                 = PaddedSIMD<T, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto padding   = get_padding<T, 2 * ns>();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  static constexpr auto ns2 = ns * T(0.5); // half spread width
  alignas(alignment) std::array<T, 2 * MAX_NSPREAD> kernel_values{0};
  std::fill(du, du + 2 * size1 * size2, 0); // initialized to 0 due to the padding
  for (uint64_t pt = 0; pt < M; pt++) {
    // loop over NU pts
    const auto dd_pt = initialize_complex_register<simd_type>(dd[pt * 2], dd[pt * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT)std::ceil(kx[pt] - ns2); // fine grid start indices
    const auto i2 = (BIGINT)std::ceil(ky[pt] - ns2);
    const auto x1 = (T)std::ceil(kx[pt] - ns2) - kx[pt];
    const auto x2 = (T)std::ceil(ky[pt] - ns2) - ky[pt];
    evaluate_kernel_vector<ns, nc, T, simd_type>(kernel_values.data(), horner_coeffs_ptr,
                                                 x1, x2);
    const auto *ker1 = kernel_values.data();
    const auto *ker2 = kernel_values.data() + MAX_NSPREAD;
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    static constexpr uint8_t kerval_vectors = (2 * ns + padding) / simd_size;
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
    for (auto dy = 0; dy < ns; ++dy) {
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

namespace {

template<typename T> struct SpreadSubproblem2dCaller {
  BIGINT off1, off2;
  UBIGINT size1, size2;
  T *du;
  UBIGINT M;
  const T *kx;
  const T *ky;
  const T *dd;
  const T *horner_coeffs_ptr;

  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>()) {
      return report_invalid_kernel_params(NS, NC);
    } else {
      spread_subproblem_2d_kernel<T, NS, NC>(off1, off2, size1, size2, du, M, kx, ky, dd,
                                             horner_coeffs_ptr);
      return 0;
    }
  }
};

} // namespace

template<typename T>
static void spread_subproblem_2d(
    BIGINT off1, BIGINT off2, UBIGINT size1, UBIGINT size2, T *FINUFFT_RESTRICT du,
    UBIGINT M, const T *kx, const T *ky, const T *dd, const finufft_spread_opts &opts,
    const T *horner_coeffs_ptr, int nc) noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
   For algoritmic details see spread_subproblem_1d_kernel.
*/
{
  SpreadSubproblem2dCaller<T> caller{off1, off2, size1, size2, du,
                                     M, kx, ky, dd, horner_coeffs_ptr};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  using NcSeq = make_range<MIN_NC, MAX_NC>;
  auto params =
      std::make_tuple(DispatchParam<NsSeq>{opts.nspread}, DispatchParam<NcSeq>{nc});
  dispatch(caller, params);
}

template<typename T, int ns, int nc>
FINUFFT_NEVER_INLINE void spread_subproblem_3d_kernel(
    const BIGINT off1, const BIGINT off2, const BIGINT off3, const UBIGINT size1,
    const UBIGINT size2, const UBIGINT size3, T *FINUFFT_RESTRICT du, const UBIGINT M,
    const T *kx, const T *ky, const T *kz, const T *dd,
    const T *horner_coeffs_ptr) noexcept {
  using simd_type                 = PaddedSIMD<T, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto padding   = get_padding<T, 2 * ns>();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();

  static constexpr auto ns2 = ns * T(0.5); // half spread width
  alignas(alignment) std::array<T, 3 * MAX_NSPREAD> kernel_values{0};
  std::fill(du, du + 2 * size1 * size2 * size3, 0);

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

    evaluate_kernel_vector<ns, nc, T, simd_type>(kernel_values.data(), horner_coeffs_ptr,
                                                 x1, x2, x3);
    const auto *ker1 = kernel_values.data();
    const auto *ker2 = kernel_values.data() + MAX_NSPREAD;
    const auto *ker3 = kernel_values.data() + 2 * MAX_NSPREAD;
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    // kerval_vectors is the number of SIMD iterations needed to compute all the elements
    static constexpr uint8_t kerval_vectors = (2 * ns + padding) / simd_size;
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
    for (uint8_t dz{0}; dz < ns; ++dz) {
      const auto oz = size1 * size2 * (i3 - off3 + dz); // offset due to z
      for (uint8_t dy{0}; dy < ns; ++dy) {
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

namespace {

template<typename T> struct SpreadSubproblem3dCaller {
  BIGINT off1, off2, off3;
  UBIGINT size1, size2, size3;
  T *du;
  UBIGINT M;
  T *kx;
  T *ky;
  T *kz;
  T *dd;
  const T *horner_coeffs_ptr;

  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>()) {
      return report_invalid_kernel_params(NS, NC);
    } else {
      spread_subproblem_3d_kernel<T, NS, NC>(off1, off2, off3, size1, size2, size3, du, M,
                                             kx, ky, kz, dd, horner_coeffs_ptr);
      return 0;
    }
  }
};

} // namespace

template<typename T>
static void spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT size1,
                                 UBIGINT size2, UBIGINT size3, T *du, UBIGINT M, T *kx,
                                 T *ky, T *kz, T *dd, const finufft_spread_opts &opts,
                                 const T *horner_coeffs_ptr, int nc) noexcept
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
See above docs/notes for spread_subproblem_2d.
kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
dd (size M complex) are complex source strengths
du (size size1*size2*size3) is uniform complex output array
*/
{
  SpreadSubproblem3dCaller<T> caller{off1, off2, off3, size1, size2, size3,
                                     du, M, kx, ky, kz, dd,
                                     horner_coeffs_ptr};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  using NcSeq = make_range<MIN_NC, MAX_NC>;
  auto params =
      std::make_tuple(DispatchParam<NsSeq>{opts.nspread}, DispatchParam<NcSeq>{nc});
  dispatch(caller, params);
}

template<typename T, bool thread_safe>
static void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
                                UBIGINT padded_size1, UBIGINT size1, UBIGINT size2,
                                UBIGINT size3, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                                T *FINUFFT_RESTRICT data_uniform, const T *du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   padded_size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe variant of the above routine,
   using atomic writes (R Blackwell, Nov 2020).
   Merged the thread_safe and the not thread_safe version of the function into one
   (M. Barbone 06/24).
*/
{
  std::vector<BIGINT> o2(size2), o3(size3);
  static auto accumulate = [](T &a, T b) {
    if constexpr (thread_safe) { // NOLINT(*-branch-clone)
#pragma omp atomic
      a += b;
    } else {
      a += b;
    }
  };

  BIGINT y = offset2, z = offset3; // fill wrapped ptr lists in slower dims y,z...
  for (UBIGINT i = 0; i < size2; ++i) {
    if (y < 0) y += BIGINT(N2);
    if (y >= BIGINT(N2)) y -= BIGINT(N2);
    o2[i] = y++;
  }
  for (UBIGINT i = 0; i < size3; ++i) {
    if (z < 0) z += BIGINT(N3);
    if (z >= BIGINT(N3)) z -= BIGINT(N3);
    o3[i] = z++;
  }
  UBIGINT nlo = (offset1 < 0) ? -offset1 : 0; // # wrapping below in x
  UBIGINT nhi = (offset1 + size1 > N1) ? offset1 + size1 - N1 : 0; // " above in x
  // this triple loop works in all dims
  for (UBIGINT dz = 0; dz < size3; dz++) {
    // use ptr lists in each axis
    const auto oz = N1 * N2 * o3[dz]; // offset due to z (0 in <3D)
    for (UBIGINT dy = 0; dy < size2; dy++) {
      const auto oy = N1 * o2[dy] + oz; // off due to y & z (0 in 1D)
      auto *FINUFFT_RESTRICT out = data_uniform + 2 * oy;
      const auto in = du0 + 2 * padded_size1 * (dy + size2 * dz); // ptr to subgrid array
      auto o = 2 * (offset1 + N1); // 1d offset for output
      for (UBIGINT j = 0; j < 2 * nlo; j++) {
        // j is really dx/2 (since re,im parts)
        accumulate(out[j + o], in[j]);
      }
      o = 2 * offset1;
      for (auto j = 2 * nlo; j < 2 * (size1 - nhi); j++) {
        accumulate(out[j + o], in[j]);
      }
      o = 2 * (offset1 - N1);
      for (auto j = 2 * (size1 - nhi); j < 2 * size1; j++) {
        accumulate(out[j + o], in[j]);
      }
    }
  }
}

template<typename T>
static void bin_sort_singlethread(std::vector<BIGINT> &ret, UBIGINT M, const T *kx,
                                  const T *ky, const T *kz, UBIGINT N1, UBIGINT N2,
                                  UBIGINT N3, double bin_size_x, double bin_size_y,
                                  double bin_size_z, int debug [[maybe_unused]])
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D. Single-threaded version
 *
 * This is achieved by binning into cuboids (of given bin_size within the
 * overall box domain), then reading out the indices within
 * these bins in a Cartesian cuboid ordering (x fastest, y med, z slowest).
 * Finally the permutation is inverted, so that the good ordering is: the
 * NU pt of index ret[0], the NU pt of index ret[1],..., NU pt of index ret[M-1]
 *
 * Inputs: M - number of input NU points.
 *         kx,ky,kz - length-M arrays of real coords of NU pts in [-pi, pi).
 *                    Points outside this range are folded into it.
 *         N1,N2,N3 - integer sizes of overall box (N2=N3=1 for 1D, N3=1 for 2D)
 *         bin_size_x,y,z - what binning box size to use in each dimension
 *                    (in rescaled coords where ranges are [0,Ni] ).
 *                    For 1D, only bin_size_x is used; for 2D, it & bin_size_y.
 * Output:
 *         writes to ret a vector list of indices, each in the range 0,..,M-1.
 *         Thus, ret must have been preallocated for M BIGINTs.
 *
 * Notes: I compared RAM usage against declaring an internal vector and passing
 * back; the latter used more RAM and was slower.
 * Avoided the bins array, as in JFM's spreader of 2016,
 * tidied up, early 2017, Barnett.
 * Timings (2017): 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
 * Simplified by Martin Reinecke, 6/19/23 (no apparent effect on speed).
 */
{
  const auto isky = (N2 > 1), iskz = (N3 > 1); // ky,kz avail? (cannot access if not)
  // here the +1 is needed to allow round-off error causing i1=N1/bin_size_x,
  // for kx near +pi, ie foldrescale gives N1 (exact arith would be 0 to N1-1).
  // Note that round-off near kx=-pi stably rounds negative to i1=0.
  const auto nbins1         = BIGINT(T(N1) / bin_size_x + 1);
  const auto nbins2         = isky ? BIGINT(T(N2) / bin_size_y + 1) : 1;
  const auto nbins3         = iskz ? BIGINT(T(N3) / bin_size_z + 1) : 1;
  const auto nbins          = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x = T(1.0 / bin_size_x);
  const auto inv_bin_size_y = T(1.0 / bin_size_y);
  const auto inv_bin_size_z = T(1.0 / bin_size_z);
  // count how many pts in each bin
  std::vector<BIGINT> counts(nbins, 0);

  for (UBIGINT i = 0; i < M; i++) {
    // find the bin index in however many dims are needed
    const auto i1  = BIGINT(fold_rescale<T>(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale<T>(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale<T>(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  // compute the offsets directly in the counts array (no offset array)
  BIGINT current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    BIGINT tmp = counts[i];
    counts[i]  = current_offset; // Reinecke's cute replacement of counts[i]
    current_offset += tmp;
  } // (counts now contains the index offsets for each bin)

  for (UBIGINT i = 0; i < M; i++) {
    // find the bin index (again! but better than using RAM)
    const auto i1    = BIGINT(fold_rescale<T>(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale<T>(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale<T>(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i); // fill the inverse map on the fly
    ++counts[bin]; // update the offsets
  }
}

template<typename T>
static void bin_sort_multithread(std::vector<BIGINT> &ret, UBIGINT M, const T *kx,
                                 const T *ky, const T *kz, UBIGINT N1, UBIGINT N2,
                                 UBIGINT N3, double bin_size_x, double bin_size_y,
                                 double bin_size_z, int debug [[maybe_unused]], int nthr)
/* Mostly-OpenMP'ed version of bin_sort.
   For documentation see: bin_sort_singlethread.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Originally by Barnett 2/8/18
   Explicit #threads control argument 7/20/20.
   Improved by Martin Reinecke, 6/19/23 (up to 50% faster at 1 thr/core).
   Todo: if debug, print timing breakdowns.
 */
{
  bool isky      = (N2 > 1), iskz = (N3 > 1); // ky,kz avail? (cannot access if not)
  UBIGINT nbins1 = N1 / bin_size_x + 1, nbins2, nbins3; // see above note on why +1
  nbins2         = isky ? N2 / bin_size_y + 1 : 1;
  nbins3         = iskz ? N3 / bin_size_z + 1 : 1;
  UBIGINT nbins  = nbins1 * nbins2 * nbins3;
  if (nthr == 0) // should never happen in spreadinterp use
    fprintf(stderr, "[%s] nthr (%d) must be positive!\n", __func__, nthr);
  int nt = std::min(M, UBIGINT(nthr)); // handle case of less points than threads
  std::vector<UBIGINT> brk(nt + 1); // list of start NU pt indices per thread

  // distribute the NU pts to threads once & for all...
  for (int t = 0; t <= nt; ++t)
    brk[t]   = (UBIGINT)(0.5 + M * t / (double)nt); // start index for t'th chunk

  // set up 2d array (nthreads * nbins), just its pointers for now
  // (sub-vectors will be initialized later)
  std::vector<std::vector<UBIGINT>> counts(nt);

#pragma omp parallel num_threads(nt)
  {
    // parallel binning to each thread's count. Block done once per thread
    int t = MY_OMP_GET_THREAD_NUM(); // (we assume all nt threads created)
    auto &my_counts(counts[t]); // name for counts[t]
    my_counts.resize(nbins, 0); // allocate counts[t], now in parallel region
    for (auto i = brk[t]; i < brk[t + 1]; i++) {
      // find the bin index in however many dims are needed
      BIGINT i1 = fold_rescale<T>(kx[i], N1) / bin_size_x, i2 = 0, i3 = 0;
      if (isky) i2 = fold_rescale<T>(ky[i], N2) / bin_size_y;
      if (iskz) i3 = fold_rescale<T>(kz[i], N3) / bin_size_z;
      const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
      ++my_counts[bin]; // no clash btw threads
    }
  }

  // inner sum along both bin and thread (inner) axes to get global offsets
  UBIGINT current_offset = 0;
  for (UBIGINT b = 0; b < nbins; ++b) // (not worth omp)
    for (int t = 0; t < nt; ++t) {
      UBIGINT tmp  = counts[t][b];
      counts[t][b] = current_offset;
      current_offset += tmp;
    } // counts[t][b] is now the index offset as if t ordered fast, b slow

#pragma omp parallel num_threads(nt)
  {
    int t = MY_OMP_GET_THREAD_NUM();
    auto &my_counts(counts[t]);
    for (UBIGINT i = brk[t]; i < brk[t + 1]; i++) {
      // find the bin index (again! but better than using RAM)
      UBIGINT i1 = fold_rescale<T>(kx[i], N1) / bin_size_x, i2 = 0, i3 = 0;
      if (isky) i2 = fold_rescale<T>(ky[i], N2) / bin_size_y;
      if (iskz) i3 = fold_rescale<T>(kz[i], N3) / bin_size_z;
      UBIGINT bin         = i1 + nbins1 * (i2 + nbins2 * i3);
      ret[my_counts[bin]] = i; // inverse is offset for this NU pt and thread
      ++my_counts[bin]; // update the offsets; no thread clash
    }
  }
}

template<typename T>
static void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3,
                        BIGINT &padded_size1, BIGINT &size1, BIGINT &size2, BIGINT &size3,
                        UBIGINT M, const T *kx, const T *ky, const T *kz, int ns,
                        int ndims)
/* Writes out the integer offsets and sizes of a "subgrid" (cuboid subset of
   Z^ndims) large enough to enclose all of the nonuniform points with
   (non-periodic) padding of half the kernel width ns to each side in
   each relevant dimension.

 Inputs:
   M - number of nonuniform points, ie, length of kx array (and ky if ndims>1,
       and kz if ndims>2)
   kx,ky,kz - coords of nonuniform points (ky only read if ndims>1,
              kz only read if ndims>2). To be useful for spreading, they are
              assumed to be in [0,Nj] for dimension j=1,..,ndims.
   ns - (positive integer) spreading kernel width.
   ndims - space dimension (1,2, or 3).

 Outputs:
   offset1,2,3 - left-most coord of cuboid in each dimension (up to ndims)
   padded_size1,2,3   - size of cuboid in each dimension.
                 Thus the right-most coord of cuboid is offset+size-1.
   Returns offset 0 and size 1 for each unused dimension (ie when ndims<3);
   this is required by the calling code.

 Example:
      inputs:
          ndims=1, M=2, kx[0]=0.2, ks[1]=4.9, ns=3
      outputs:
          offset1=-1 (since kx[0] spreads to {-1,0,1}, and -1 is the min)
          padded_size1=8 (since kx[1] spreads to {4,5,6}, so subgrid is {-1,..,6}
                   hence 8 grid points).
 Notes:
   1) Works in all dims 1,2,3.
   2) Rounding of the kx (and ky, kz) to the grid is tricky and must match the
   rounding step used in spread_subproblem_{1,2,3}d. Namely, the ceil of
   (the NU pt coord minus ns/2) gives the left-most index, in each dimension.
   This being done consistently is crucial to prevent segfaults in subproblem
   spreading. This assumes that max() and ceil() commute in the floating pt
   implementation.
   Originally by J Magland, 2017. AHB realised the rounding issue in
   6/16/17, but only fixed a rounding bug causing segfault in (highly
   inaccurate) single-precision with N1>>1e7 on 11/30/20.
   3) Requires O(M) RAM reads to find the k array bnds. Almost negligible in
   tests.
*/
{
  T ns2 = (T)ns / 2;
  T min_kx, max_kx; // 1st (x) dimension: get min/max of nonuniform points
  arrayrange(M, kx, &min_kx, &max_kx);
  offset1      = (BIGINT)std::ceil(min_kx - ns2); // min index touched by kernel
  size1        = (BIGINT)std::ceil(max_kx - ns2) - offset1 + ns; // int(ceil) first!
  padded_size1 = size1 + get_padding<T>(2 * ns) / 2;
  if (ndims > 1) {
    T min_ky, max_ky; // 2nd (y) dimension: get min/max of nonuniform points
    arrayrange(M, ky, &min_ky, &max_ky);
    offset2 = (BIGINT)std::ceil(min_ky - ns2);
    size2   = (BIGINT)std::ceil(max_ky - ns2) - offset2 + ns;
  } else {
    offset2 = 0;
    size2   = 1;
  }
  if (ndims > 2) {
    T min_kz, max_kz; // 3rd (z) dimension: get min/max of nonuniform points
    arrayrange(M, kz, &min_kz, &max_kz);
    offset3 = (BIGINT)std::ceil(min_kz - ns2);
    size3   = (BIGINT)std::ceil(max_kz - ns2) - offset3 + ns;
  } else {
    offset3 = 0;
    size3   = 1;
  }
}

// --------------------------------------------------------------------------
template<typename T>
int spreadSorted(
    const std::vector<BIGINT> &sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
    T *FINUFFT_RESTRICT data_uniform, UBIGINT M, const T *FINUFFT_RESTRICT kx,
    const T *FINUFFT_RESTRICT ky, const T *FINUFFT_RESTRICT kz, const T *data_nonuniform,
    const finufft_spread_opts &opts, int did_sort, const T *horner_coeffs_ptr, int nc)
// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
{
  CNTime timer{};
  const auto ndims = ndims_from_Ns(N1, N2, N3);
  const auto N     = N1 * N2 * N3; // output array size
  const auto ns    = opts.nspread; // abbrev. for w, kernel width
  auto nthr        = MY_OMP_GET_MAX_THREADS(); // guess # threads to use to spread
  if (opts.nthreads > 0) nthr = opts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1; // single-threaded lib must override user
#endif
  if (opts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr);
  timer.start();
  std::fill(data_uniform, data_uniform + 2 * N, 0.0); // zero the output array
  if (opts.debug) printf("\tzero output array\t%.3g s\n", timer.elapsedsec());
  if (M == 0) // no NU pts, we're done
    return 0;

  auto spread_single = (nthr == 1) || (M * 100 < N); // low-density heuristic?
  spread_single      = false; // for now
  timer.start();
  if (spread_single) {
    // ------- Basic single-core t1 spreading ------
    for (UBIGINT j = 0; j < M; j++) {
      // *** todo, not urgent
      // ... (question is: will the index wrapping per NU pt slow it down?)
    }
    if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n", timer.elapsedsec());
  } else {
    // ------- Fancy multi-core blocked t1 spreading ----
    // Splits sorted inds (jfm's advanced2), could double RAM.
    // choose nb (# subprobs) via used nthreads:
    auto nb = std::min((UBIGINT)nthr, M); // simply split one subprob per thr...
    if (nb * (BIGINT)opts.max_subproblem_size < M) {
      // ...or more subprobs to cap size
      nb = 1 + (M - 1) / opts.max_subproblem_size; // int div does
      // ceil(M/opts.max_subproblem_size)
      if (opts.debug)
        printf("\tcapping subproblem sizes to max of %d\n", opts.max_subproblem_size);
    }
    if (M * 1000 < N) {
      // low-density heuristic: one thread per NU pt!
      nb = M;
      if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr == 1) {
      nb = 1;
      if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }
    if (opts.debug && nthr > opts.atomic_threshold)
      printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");

    std::vector<UBIGINT> brk(nb + 1); // NU index breakpoints defining nb subproblems
    for (UBIGINT p = 0; p <= nb; ++p) brk[p] = (M * p + nb - 1) / nb;

#pragma omp parallel num_threads(nthr)
    {
      // local copies of NU pts and data for each subproblem
      std::vector<T> kx0{}, ky0{}, kz0{}, dd0{}, du0{};
#pragma omp for schedule(dynamic, 1)                     // each is big
      for (BIGINT isub = 0; isub < BIGINT(nb); isub++) {
        // Main loop through the
        // subproblems
        const auto M0 = brk[isub + 1] - brk[isub]; // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        kx0.resize(M0);
        ky0.resize(M0 * (N2 > 1));
        kz0.resize(M0 * (N3 > 1));
        dd0.resize(2 * M0); // complex strength data
        for (UBIGINT j = 0; j < M0; j++) {
          // todo: can avoid this copying?
          const auto kk = sort_indices[j + brk[isub]]; // NU pt from subprob index list
          kx0[j]        = fold_rescale<T>(kx[kk], N1);
          if (N2 > 1) ky0[j] = fold_rescale<T>(ky[kk], N2);
          if (N3 > 1) kz0[j] = fold_rescale<T>(kz[kk], N3);
          dd0[j * 2]     = data_nonuniform[kk * 2]; // real part
          dd0[j * 2 + 1] = data_nonuniform[kk * 2 + 1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        // get_subgrid sets
        BIGINT offset1, offset2, offset3, padded_size1, size1, size2, size3;
        // sets offsets and sizes
        get_subgrid(offset1, offset2, offset3, padded_size1, size1, size2, size3, M0,
                    kx0.data(), ky0.data(), kz0.data(), ns, ndims);
        if (opts.debug > 1) {
          print_subgrid_info(ndims, offset1, offset2, offset3, padded_size1, size1, size2,
                             size3, M0);
        }
        // allocate output data for this subgrid
        du0.resize(2 * padded_size1 * size2 * size3); // complex
        // Spread to subgrid without need for bounds checking or wrapping
        if (ndims == 1)
          spread_subproblem_1d(offset1, padded_size1, du0.data(), M0, kx0.data(),
                               dd0.data(), opts, horner_coeffs_ptr, nc);
        else if (ndims == 2)
          spread_subproblem_2d(offset1, offset2, padded_size1, size2, du0.data(), M0,
                               kx0.data(), ky0.data(), dd0.data(), opts,
                               horner_coeffs_ptr, nc);
        else
          spread_subproblem_3d(offset1, offset2, offset3, padded_size1, size2, size3,
                               du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               dd0.data(), opts, horner_coeffs_ptr, nc);

        // add subgrid to output (always do this); atomic vs critical chosen
        if (nthr > opts.atomic_threshold) {
          // see above for debug reporting
          add_wrapped_subgrid<T, true>(offset1, offset2, offset3, padded_size1, size1,
                                       size2, size3, N1, N2, N3, data_uniform,
                                       du0.data()); // R Blackwell's atomic version
        } else {
#pragma omp critical
          add_wrapped_subgrid<T, false>(offset1, offset2, offset3, padded_size1, size1,
                                        size2, size3, N1, N2, N3, data_uniform,
                                        du0.data());
        }
      } // end main loop over subprobs
    }
    if (opts.debug)
      printf("\tt1 fancy spread: \t%.3g s (%" PRIu64 " subprobs)\n", timer.elapsedsec(),
             nb);
  } // end of choice of which t1 spread type to use
  return 0;
};

} // namespace finufft::spreadinterp
