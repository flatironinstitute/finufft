#pragma once

#include <finufft/plan.hpp>
#include <finufft/simd.hpp>
#include <finufft/utils.hpp>

#include <array>
#include <cassert>

#include <poet/poet.hpp> // poet::dispatch / inclusive_range / dispatch_param

namespace finufft::spreadinterp {

template<typename T, uint8_t ns, class simd_type = PaddedSIMD<T, 2 * ns>>
void interp_line(T *FINUFFT_RESTRICT target, const T *du, const T *ker, BIGINT i1)
/* 1D interpolate complex values from size-ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   1d kernel evaluation list ker1.
   Inputs:
   du : input regular subgrid (alternating real,imag), row stride padded past the
        widest SIMD read below (see Subgrid::set_size1)
   ker1 : length-ns real array of 1d kernel evaluations, zero on its padding lanes
   i1 : start (left-most) x-coord index to read du from; the caller guarantees the
        kernel support lies inside the subgrid, so no index wraps
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Internally, dx indices into ker array, j is index in complex du array.
   Barnett 6/16/17.
   M. Barbone July 2024: using explicit SIMD vectorization to overcome the out[2]
                         array limitation
   Wrapped fallback deleted with the tiled subgrid gather, M. Barbone 8/25/26: the
   subgrid pad absorbs the SIMD overshoot and the zeroed ker lanes its values.
*/
{
  using arch_t                       = typename simd_type::arch_type;
  static constexpr auto padding      = get_padding<T, 2 * ns>();
  static constexpr auto simd_size    = simd_type::size;
  static constexpr auto regular_part =
      finufft::utils::round_down<2 * simd_size>(std::size_t(2 * ns + padding));
  std::array<T, 2> out{0};
  const auto j = i1;
  // logic largely similar to spread 1D kernel, please see the explanation there
  // for the first part of this code
  const auto res = [du, j, ker]() constexpr noexcept {
    const auto du_ptr = du + 2 * j;
    simd_type res_low{0}, res_hi{0};
    // narrow ns can leave regular_part == 0: then the tail block below does it all
    if constexpr (regular_part > 0) {
      for (uint8_t dx{0}; dx < regular_part; dx += 2 * simd_size) {
        const auto ker_v   = simd_type::load_aligned(ker + dx / 2);
        const auto du_pt0  = simd_type::load_unaligned(du_ptr + dx);
        const auto du_pt1  = simd_type::load_unaligned(du_ptr + dx + simd_size);
        const auto ker0low = xsimd::swizzle(ker_v, zip_low_index<arch_t, T>);
        const auto ker0hi  = xsimd::swizzle(ker_v, zip_hi_index<arch_t, T>);
        res_low            = xsimd::fma(ker0low, du_pt0, res_low);
        res_hi             = xsimd::fma(ker0hi, du_pt1, res_hi);
      }
    }

    // sanity check at compile time that all the elements are computed
    static_assert(regular_part + simd_size >= 2 * ns);
    if constexpr (regular_part < 2 * ns) {
      const auto ker0    = simd_type::load_unaligned(ker + (regular_part / 2));
      const auto du_pt   = simd_type::load_unaligned(du_ptr + regular_part);
      const auto ker0low = xsimd::swizzle(ker0, zip_low_index<arch_t, T>);
      res_low            = xsimd::fma(ker0low, du_pt, res_low);
    }

    // lambda here to limit the scope of temporary variables and have the compiler
    // optimize the code better
    return res_low + res_hi;
  }();
  // interpolator does an extra horizontal-sum step reducing the SIMD
  // accumulator down to a single {re, im} pair (see complex_hadd).
  const auto c = complex_hadd(res);
  out[0] += c[0];
  out[1] += c[1];
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type = PaddedSIMD<T, 2 * ns>>
void interp_square(T *FINUFFT_RESTRICT target, const T *du, const T *ker1, const T *ker2,
                   BIGINT i1, BIGINT i2, UBIGINT N1)
/* 2D interpolate complex values from a ns*ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   ns*ns outer product of the 1d kernel lists ker1 and ker2.
   Inputs:
   du : input regular subgrid (alternating real,imag), rows N1 apart, padded as in
        interp_line
   ker1, ker2 : length-ns real arrays of 1d kernel evaluations, zero padding lanes
   i1 : start (left-most) x-coord index to read du from
   i2 : start (bottom) y-coord index to read du from; the caller guarantees the
        kernel support lies inside the subgrid, so no index wraps
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Internally, dx,dy indices into ker array, l indices the 2*ns interleaved
   line array, j is index in complex du array.
   Barnett 6/16/17.
   Sped up for FMA/SIMD by Martin Reinecke 6/19/23, with this note:
   "It reduces the number of arithmetic operations per "iteration" in the
   innermost loop from 2.5 to 2, and these two can be converted easily to a
   fused multiply-add instruction (potentially vectorized). Also the strides
   of all invoved arrays in this loop are now 1, instead of the mixed 1 and 2
   before. Also the accumulation onto a double[2] is limiting the vectorization
   pretty badly. I think this is now much more analogous to the way the spread
   operation is implemented, which has always been much faster when I tested
   it."
   M. Barbone July 2024: using explicit SIMD vectorization to overcome the out[2]
                         array limitation
   Wrapped fallback deleted with the tiled subgrid gather, M. Barbone 8/25/26.
   The code is largely similar to 1D interpolation, please see the explanation there
*/
{
  std::array<T, 2> out{0};
  using arch_t                          = typename simd_type::arch_type;
  static constexpr auto padding         = get_padding<T, 2 * ns>();
  static constexpr auto simd_size       = simd_type::size;
  static constexpr uint8_t line_vectors = (2 * ns + padding) / simd_size;
  const auto line                       = [du, N1, i1 = UBIGINT(i1), i2 = UBIGINT(i2),
                                           ker2]() constexpr noexcept {
    // new array du_pts to store the du values for the current y line
    std::array<simd_type, line_vectors> line{0};
    // block for first y line, to avoid explicitly initializing line with zeros
    // add remaining const-y lines to the line (expensive inner loop)
    for (uint8_t dy{0}; dy < ns; dy++) {
      const auto l_ptr = du + 2 * (N1 * (i2 + dy) + i1); // (see above)
      const simd_type ker2_v{ker2[dy]};
      for (uint8_t l{0}; l < line_vectors; ++l) {
        const auto du_pt = simd_type::load_unaligned(l * simd_size + l_ptr);
        line[l]          = xsimd::fma(ker2_v, du_pt, line[l]);
      }
    }
    return line;
  }();
  // This is the same as 1D interpolation
  // using lambda to limit the scope of the temporary variables
  const auto res = [ker1, &line]() constexpr noexcept {
    // apply x kernel to the (interleaved) line and add together
    simd_type res_low{0}, res_hi{0};
    // Start the loop from the second iteration
    for (uint8_t i{0}; i < (line_vectors & ~1); // NOLINT(*-too-small-loop-variable)
         i += 2) {
      const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
      const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>);
      const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t, T>);
      res_low            = xsimd::fma(ker1low, line[i], res_low);
      res_hi             = xsimd::fma(ker1hi, line[i + 1], res_hi);
    }
    if constexpr (line_vectors % 2) {
      const auto ker1_v =
          simd_type::load_aligned(ker1 + (line_vectors - 1) * simd_size / 2);
      const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>);
      res_low            = xsimd::fma(ker1low, line.back(), res_low);
    }
    return res_low + res_hi;
  }();
  const auto c = complex_hadd(res);
  out[0] += c[0];
  out[1] += c[1];
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type = PaddedSIMD<T, 2 * ns>>
void interp_cube(T *FINUFFT_RESTRICT target, const T *du, const T *ker1, const T *ker2,
                 const T *ker3, BIGINT i1, BIGINT i2, BIGINT i3, UBIGINT N1, UBIGINT N2)
/* 3D interpolate complex values from a ns*ns*ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   ns*ns*ns outer product of the 1d kernel lists ker1, ker2, and ker3.
   Inputs:
   du : input regular subgrid (alternating real,imag), rows N1 apart and planes
        N1*N2 apart, padded as in interp_line
   ker1, ker2, ker3 : length-ns real arrays of 1d kernel evaluations, zero padding
        lanes
   i1 : start (left-most) x-coord index to read du from
   i2 : start (bottom) y-coord index to read du from.
   i3 : start (lowest) z-coord index to read du from; the caller guarantees the
        kernel support lies inside the subgrid, so no index wraps
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Internally, dx,dy,dz indices into ker array, l indices the 2*ns interleaved
   line array, j is index in complex du array.
   Barnett 6/16/17.
   Sped up for FMA/SIMD by Reinecke 6/19/23 (see above note in interp_square)
   Barbone July 2024: using explicit SIMD vectorization to overcome the out[2]
                      array limitation
   Wrapped fallback deleted with the tiled subgrid gather, M. Barbone 8/25/26.
   The code is largely similar to 2D and 1D interpolation, please see the explanation
   there
*/
{
  using arch_t                          = typename simd_type::arch_type;
  static constexpr auto padding         = get_padding<T, 2 * ns>();
  static constexpr auto simd_size       = simd_type::size;
  static constexpr uint8_t line_vectors = (2 * ns + padding) / simd_size;
  std::array<T, 2> out{0};
  const auto line = [N1, N2, i1 = UBIGINT(i1), i2 = UBIGINT(i2), i3 = UBIGINT(i3), ker2,
                     ker3, du]() constexpr noexcept {
    std::array<simd_type, line_vectors> line{0};
    // co-add y and z contributions to line in x; do not apply x kernel yet
    for (uint8_t dz{0}; dz < ns; ++dz) {
      const auto oz = N1 * N2 * (i3 + dz);                      // offset due to z
      for (uint8_t dy{0}; dy < ns; ++dy) {                      // expensive inner loop
        const auto l_ptr = du + 2 * (oz + N1 * (i2 + dy) + i1); // ptr start of line
        const simd_type ker23{ker2[dy] * ker3[dz]};
        for (uint8_t l{0}; l < line_vectors; ++l) {
          const auto du_pt = simd_type::load_unaligned(l * simd_size + l_ptr);
          line[l]          = xsimd::fma(ker23, du_pt, line[l]);
        }
      }
    }
    return line;
  }();
  const auto res = [ker1, &line]() constexpr noexcept {
    // apply x kernel to the (interleaved) line and add together
    simd_type res_low{0}, res_hi{0};
    // Start the loop from the second iteration
    for (uint8_t i{0}; i < (line_vectors & ~1); // NOLINT(*-too-small-loop-variable)
         i += 2) {
      const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
      const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>);
      const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t, T>);
      res_low            = xsimd::fma(ker1low, line[i], res_low);
      res_hi             = xsimd::fma(ker1hi, line[i + 1], res_hi);
    }
    if constexpr (line_vectors % 2) {
      const auto ker1_v =
          simd_type::load_aligned(ker1 + (line_vectors - 1) * simd_size / 2);
      const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t, T>);
      res_low            = xsimd::fma(ker1low, line.back(), res_low);
    }
    return res_low + res_hi;
  }();
  const auto c = complex_hadd(res);
  out[0] += c[0];
  out[1] += c[1];
  target[0] = out[0];
  target[1] = out[1];
}

} // namespace finufft::spreadinterp

// ---------- FINUFFT_PLAN_T interp_subproblem_kernel method definition ----------
// The mirror of spread_subproblem_*d_kernel: one subproblem's points, read out of the
// subgrid that subproblem holds. Barbone 8/25/26.

template<typename TF>
template<int NS, int NC, int NDIMS>
FINUFFT_NEVER_INLINE void FINUFFT_PLAN_T<TF>::interp_subproblem_kernel(
    BIGINT off1, [[maybe_unused]] BIGINT off2, [[maybe_unused]] BIGINT off3,
    UBIGINT padded_size1, [[maybe_unused]] UBIGINT size2, [[maybe_unused]] UBIGINT size3,
    const TF *du, UBIGINT M, const TF *kx, [[maybe_unused]] const TF *ky,
    [[maybe_unused]] const TF *kz, const BIGINT *idx,
    TF *FINUFFT_RESTRICT dd) const noexcept
/* Interpolate M NU points (kx, ky, kz, already folded into the fine grid box)
   out of subgrid du into strengths dd.
   The subgrid's lowest corner sits at off1,2,3 in fine-grid coords.
   Rows are padded_size1 apart; the extent in the second and third dimensions is size2,3.
   idx[j] is the pre-sort index of point j; the write to dd goes to dd + 2*idx[j]
   directly, so no gathered buffer and copy is needed. A point's kernel support lies
   inside the subgrid its own subproblem built, so no interp kernel ever wraps: the
   subgrid pad absorbs the SIMD overshoot past a row's end (see the caller's assert).
*/
{
  using namespace finufft::spreadinterp;
  using KBL                   = KernelBufferLayout<TF, NS>;
  using simd_type             = typename KBL::simd_type;
  static constexpr auto ns2   = NS * TF(0.5);
  const TF *horner_coeffs_ptr = m.horner_coeffs.data();
  alignas(KBL::alignment) std::array<TF, 3 * KBL::stride> kernel_values{0};
  const auto *ker1                  = kernel_values.data();
  [[maybe_unused]] const auto *ker2 = kernel_values.data() + KBL::stride;
  [[maybe_unused]] const auto *ker3 = kernel_values.data() + 2 * KBL::stride;
  for (UBIGINT j = 0; j < M; ++j) {
    const auto xj = kx[j];
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = BIGINT(std::ceil(xj - ns2));
    const auto x1 = std::ceil(xj - ns2) - xj;
    if constexpr (NDIMS == 1) {
      evaluate_kernel_vector<NS, NC, TF, simd_type>(kernel_values.data(),
                                                    horner_coeffs_ptr, x1);
      interp_line<TF, NS, simd_type>(dd + 2 * idx[j], du, ker1, i1 - off1);
    } else if constexpr (NDIMS == 2) {
      const auto yj = ky[j];
      const auto i2 = BIGINT(std::ceil(yj - ns2));
      const auto x2 = std::ceil(yj - ns2) - yj;
      evaluate_kernel_vector<NS, NC, TF, simd_type>(kernel_values.data(),
                                                    horner_coeffs_ptr, x1, x2);
      interp_square<TF, NS, simd_type>(dd + 2 * idx[j], du, ker1, ker2, i1 - off1,
                                       i2 - off2, padded_size1);
    } else {
      const auto yj = ky[j];
      const auto zj = kz[j];
      const auto i2 = BIGINT(std::ceil(yj - ns2));
      const auto i3 = BIGINT(std::ceil(zj - ns2));
      const auto x2 = std::ceil(yj - ns2) - yj;
      const auto x3 = std::ceil(zj - ns2) - zj;
      evaluate_kernel_vector<NS, NC, TF, simd_type>(kernel_values.data(),
                                                    horner_coeffs_ptr, x1, x2, x3);
      interp_cube<TF, NS, simd_type>(dd + 2 * idx[j], du, ker1, ker2, ker3, i1 - off1,
                                     i2 - off2, i3 - off3, padded_size1, size2);
    }
  }
}

// ---------- FINUFFT_PLAN_T interp-subproblem nested caller definitions ----------
// Out-of-class definitions of the nested types declared in plan.hpp: member function
// templates are not allowed in local classes (GCC restriction).

template<typename TF> struct FINUFFT_PLAN_T<TF>::InterpSubproblem1dCaller {
  const FINUFFT_PLAN_T &plan;
  const Subgrid &sub;
  const TF *du;
  UBIGINT M;
  const TF *kx;
  const BIGINT *idx;
  TF *dd;
  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>())
      return finufft::spreadinterp::report_invalid_kernel_params(NS, NC);
    else {
      // the row pad must absorb the widest SIMD read past a row's size1 cells; the
      // runtime tail get_subgrid passes to set_size1 covers this compile-time padding
      assert(sub.padded_size1 >=
             sub.size1 + BIGINT(finufft::spreadinterp::get_padding<TF, 2 * NS>() / 2));
      plan.template interp_subproblem_kernel<NS, NC, 1>(
          sub.off1, sub.off2, sub.off3, sub.padded_size1, sub.size2, sub.size3, du, M, kx,
          nullptr, nullptr, idx, dd);
      return 0;
    }
  }
};

template<typename TF> struct FINUFFT_PLAN_T<TF>::InterpSubproblem2dCaller {
  const FINUFFT_PLAN_T &plan;
  const Subgrid &sub;
  const TF *du;
  UBIGINT M;
  const TF *kx;
  const TF *ky;
  const BIGINT *idx;
  TF *dd;
  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>())
      return finufft::spreadinterp::report_invalid_kernel_params(NS, NC);
    else {
      assert(sub.padded_size1 >=
             sub.size1 + BIGINT(finufft::spreadinterp::get_padding<TF, 2 * NS>() / 2));
      plan.template interp_subproblem_kernel<NS, NC, 2>(
          sub.off1, sub.off2, sub.off3, sub.padded_size1, sub.size2, sub.size3, du, M, kx,
          ky, nullptr, idx, dd);
      return 0;
    }
  }
};

template<typename TF> struct FINUFFT_PLAN_T<TF>::InterpSubproblem3dCaller {
  const FINUFFT_PLAN_T &plan;
  const Subgrid &sub;
  const TF *du;
  UBIGINT M;
  const TF *kx;
  const TF *ky;
  const TF *kz;
  const BIGINT *idx;
  TF *dd;
  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>())
      return finufft::spreadinterp::report_invalid_kernel_params(NS, NC);
    else {
      assert(sub.padded_size1 >=
             sub.size1 + BIGINT(finufft::spreadinterp::get_padding<TF, 2 * NS>() / 2));
      plan.template interp_subproblem_kernel<NS, NC, 3>(
          sub.off1, sub.off2, sub.off3, sub.padded_size1, sub.size2, sub.size3, du, M, kx,
          ky, kz, idx, dd);
      return 0;
    }
  }
};

// ---------- FINUFFT_PLAN_T interp-subproblem method definitions ----------
// One per dimension, so the per-dimension TUs (spreadinterp_1d/2d/3d.cpp) each
// instantiate one dimension without pulling in the others.

template<typename TF>
void FINUFFT_PLAN_T<TF>::interp_subproblem_1d(const Subgrid &sub, const TF *du, UBIGINT M,
                                              const TF *kx, [[maybe_unused]] const TF *ky,
                                              [[maybe_unused]] const TF *kz,
                                              const BIGINT *idx, TF *dd) const noexcept
// Interpolates the M NU points (kx, idx, dd) out of the subgrid du. Uses plan members
// spopts.nspread, nc and horner_coeffs for the kernel dispatch.
{
  using namespace finufft::spreadinterp;
  using namespace finufft::common;
  InterpSubproblem1dCaller caller{*this, sub, du, M, kx, idx, dd};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{m.spopts.nspread},
                                         poet::dispatch_param<NcSeq>{m.nc}));
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::interp_subproblem_2d(
    const Subgrid &sub, const TF *du, UBIGINT M, const TF *kx, const TF *ky,
    [[maybe_unused]] const TF *kz, const BIGINT *idx, TF *dd) const noexcept
// 2D version of interp_subproblem_1d.
{
  using namespace finufft::spreadinterp;
  using namespace finufft::common;
  InterpSubproblem2dCaller caller{*this, sub, du, M, kx, ky, idx, dd};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{m.spopts.nspread},
                                         poet::dispatch_param<NcSeq>{m.nc}));
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::interp_subproblem_3d(const Subgrid &sub, const TF *du, UBIGINT M,
                                              const TF *kx, const TF *ky, const TF *kz,
                                              const BIGINT *idx, TF *dd) const noexcept
// 3D version of interp_subproblem_1d.
{
  using namespace finufft::spreadinterp;
  using namespace finufft::common;
  InterpSubproblem3dCaller caller{*this, sub, du, M, kx, ky, kz, idx, dd};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{m.spopts.nspread},
                                         poet::dispatch_param<NcSeq>{m.nc}));
}
