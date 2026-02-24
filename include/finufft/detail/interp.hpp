#pragma once

#include <finufft/detail/simd_helpers.hpp>

namespace finufft::spreadinterp {

template<typename T, uint8_t ns>
static void interp_line_wrap(T *FINUFFT_RESTRICT target, const T *du, const T *ker,
                             const BIGINT i1, const UBIGINT N1) {
  /* This function is called when the kernel wraps around the grid. It is
     slower than interp_line.
     M. Barbone July 2024: - moved the logic to a separate function
                           - using fused multiply-add (fma) for better performance
     */
  std::array<T, 2> out{0};
  BIGINT j = i1;
  if (i1 < 0) {
    // wraps at left
    j += BIGINT(N1);
    for (uint8_t dx = 0; dx < -i1; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
    j -= BIGINT(N1);
    for (uint8_t dx = -i1; dx < ns; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
  } else if (i1 + ns >= BIGINT(N1)) {
    // wraps at right
    for (uint8_t dx = 0; dx < N1 - i1; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
    j -= BIGINT(N1);
    for (uint8_t dx = N1 - i1; dx < ns; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
  } else {
    // padding is okay for ker, but it might spill over du array
    // so this checks for that case and does not explicitly vectorize
    for (uint8_t dx = 0; dx < ns; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type = PaddedSIMD<T, 2 * ns>>
static void interp_line(T *FINUFFT_RESTRICT target, const T *du, const T *ker, BIGINT i1,
                        UBIGINT N1) {
  /* 1D interpolate complex values from size-ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   1d kernel evaluation list ker1.
   Inputs:
   du : input regular grid of size 2*N1 (alternating real,imag)
   ker1 : length-ns real array of 1d kernel evaluations
   i1 : start (left-most) x-coord index to read du from, where the indices
        of du run from 0 to N1-1, and indices outside that range are wrapped.
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Periodic wrapping in the du array is applied, assuming N1>=ns.
   Internally, dx indices into ker array j is index in complex du array.
   Barnett 6/16/17.
    M. Barbone July 2024: - moved wrapping logic to interp_line_wrap
                          - using explicit SIMD vectorization to overcome the out[2] array
                            limitation
*/
  using arch_t                       = typename simd_type::arch_type;
  static constexpr auto padding      = get_padding<T, 2 * ns>();
  static constexpr auto simd_size    = simd_type::size;
  static constexpr auto regular_part = (2 * ns + padding) & (-(2 * simd_size));
  std::array<T, 2> out{0};
  const auto j = i1;
  // removing the wrapping leads up to 10% speedup in certain cases
  // moved the wrapping to another function to reduce instruction cache pressure
  if (i1 < 0 || i1 + ns >= BIGINT(N1) || i1 + ns + (padding + 1) / 2 >= BIGINT(N1)) {
    return interp_line_wrap<T, ns>(target, du, ker, i1, N1);
  } else {
    // doesn't wrap
    // logic largely similar to spread 1D kernel, please see the explanation there
    // for the first part of this code
    const auto res = [du, j, ker]() constexpr noexcept {
      const auto du_ptr = du + 2 * j;
      simd_type res_low{0}, res_hi{0};
      for (uint8_t dx{0}; dx < regular_part; dx += 2 * simd_size) {
        const auto ker_v   = simd_type::load_aligned(ker + dx / 2);
        const auto du_pt0  = simd_type::load_unaligned(du_ptr + dx);
        const auto du_pt1  = simd_type::load_unaligned(du_ptr + dx + simd_size);
        const auto ker0low = xsimd::swizzle(ker_v, zip_low_index<arch_t, T>);
        const auto ker0hi  = xsimd::swizzle(ker_v, zip_hi_index<arch_t, T>);
        res_low            = xsimd::fma(ker0low, du_pt0, res_low);
        res_hi             = xsimd::fma(ker0hi, du_pt1, res_hi);
      }

      if constexpr (regular_part < 2 * ns) {
        const auto ker0    = simd_type::load_unaligned(ker + (regular_part / 2));
        const auto du_pt   = simd_type::load_unaligned(du_ptr + regular_part);
        const auto ker0low = xsimd::swizzle(ker0, zip_low_index<arch_t, T>);
        res_low            = xsimd::fma(ker0low, du_pt, res_low);
      }

      // This does a horizontal sum using a loop instead of relying on SIMD instructions
      // this is faster than the code below but less elegant.
      // lambdas here to limit the scope of temporary variables and have the compiler
      // optimize the code better
      return res_low + res_hi;
    }();
    const auto res_array = xsimd_to_array(res);
    for (uint8_t i{0}; i < simd_size; i += 2) {
      out[0] += res_array[i];
      out[1] += res_array[i + 1];
    }
    // this is where the code differs from spread_kernel, the interpolator does an extra
    // reduction step to SIMD elements down to 2 elements
    // This is known as horizontal sum in SIMD terminology

    // This does a horizontal sum using vector instruction,
    // is slower than summing and looping
    // clang-format off
    // const auto res_real = xsimd::shuffle(res_low, res_hi, select_even_mask<arch_t>);
    // const auto res_imag = xsimd::shuffle(res_low, res_hi, select_odd_mask<arch_t>);
    // out[0]              = xsimd::reduce_add(res_real);
    // out[1]              = xsimd::reduce_add(res_imag);
    // clang-format on
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type>
static void interp_square_wrap(T *FINUFFT_RESTRICT target, const T *du, const T *ker1,
                               const T *ker2, const BIGINT i1, const BIGINT i2,
                               const UBIGINT N1, const UBIGINT N2) {
  /*
   * This function is called when the kernel wraps around the grid. It is slower than
   * the non wrapping version.
   * There is an extra case for when ker is padded and spills over the du array.
   * In this case uses the old non wrapping version.
   */
  std::array<T, 2> out{0};
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto alignment = arch_t::alignment();
  if (i1 >= 0 && i1 + ns <= BIGINT(N1) && i2 >= 0 && i2 + ns <= BIGINT(N2)) {
    // store a horiz line (interleaved real,imag)
    alignas(alignment) std::array<T, 2 * ns> line{0};
    // add remaining const-y lines to the line (expensive inner loop)
    for (uint8_t dy{0}; dy < ns; ++dy) {
      const auto *l_ptr = du + 2 * (N1 * (i2 + dy) + i1); // (see above)
      for (uint8_t l{0}; l < 2 * ns; ++l) {
        line[l] = std::fma(ker2[dy], l_ptr[l], line[l]);
      }
    }
    // apply x kernel to the (interleaved) line and add together
    for (uint8_t dx{0}; dx < ns; dx++) {
      out[0] = std::fma(line[2 * dx], ker1[dx], out[0]);
      out[1] = std::fma(line[2 * dx + 1], ker1[dx], out[1]);
    }
  } else {
    std::array<UBIGINT, ns> j1{}, j2{}; // 1d ptr lists
    auto x = i1, y = i2; // initialize coords
    for (uint8_t d{0}; d < ns; d++) {
      // set up ptr lists
      if (x < 0) x += BIGINT(N1);
      if (x >= BIGINT(N1)) x -= BIGINT(N1);
      j1[d] = x++;
      if (y < 0) y += BIGINT(N2);
      if (y >= BIGINT(N2)) y -= BIGINT(N2);
      j2[d] = y++;
    }
    for (uint8_t dy{0}; dy < ns; dy++) {
      // use the pts lists
      const UBIGINT oy = N1 * j2[dy]; // offset due to y
      for (uint8_t dx{0}; dx < ns; dx++) {
        const auto k    = ker1[dx] * ker2[dy];
        const UBIGINT j = oy + j1[dx];
        out[0] += du[2 * j] * k;
        out[1] += du[2 * j + 1] * k;
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type = PaddedSIMD<T, 2 * ns>>
static void interp_square(T *FINUFFT_RESTRICT target, const T *du, const T *ker1,
                          const T *ker2, BIGINT i1, BIGINT i2, UBIGINT N1, UBIGINT N2)
/* 2D interpolate complex values from a ns*ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   ns*ns outer product of the 1d kernel lists ker1 and ker2.
   Inputs:
   du : input regular grid of size 2*N1*N2 (alternating real,imag)
   ker1, ker2 : length-ns real arrays of 1d kernel evaluations
   i1 : start (left-most) x-coord index to read du from, where the indices
        of du run from 0 to N1-1, and indices outside that range are wrapped.
   i2 : start (bottom) y-coord index to read du from.
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Periodic wrapping in the du array is applied, assuming N1,N2>=ns.
   Internally, dx,dy indices into ker array, l indices the 2*ns interleaved
   line array, j is index in complex du array.
   Barnett 6/16/17.
   No-wrap case sped up for FMA/SIMD by Martin Reinecke 6/19/23, with this note:
   "It reduces the number of arithmetic operations per "iteration" in the
   innermost loop from 2.5 to 2, and these two can be converted easily to a
   fused multiply-add instruction (potentially vectorized). Also the strides
   of all invoved arrays in this loop are now 1, instead of the mixed 1 and 2
   before. Also the accumulation onto a double[2] is limiting the vectorization
   pretty badly. I think this is now much more analogous to the way the spread
   operation is implemented, which has always been much faster when I tested
   it."
   M. Barbone July 2024: - moved the wrapping logic to interp_square_wrap
                         - using explicit SIMD vectorization to overcome the out[2] array
                           limitation
   The code is largely similar to 1D interpolation, please see the explanation there
*/
{
  std::array<T, 2> out{0};
  // no wrapping: avoid ptrs
  using arch_t                          = typename simd_type::arch_type;
  static constexpr auto padding         = get_padding<T, 2 * ns>();
  static constexpr auto simd_size       = simd_type::size;
  static constexpr uint8_t line_vectors = (2 * ns + padding) / simd_size;
  if (i1 >= 0 && i1 + ns <= BIGINT(N1) && i2 >= 0 && i2 + ns <= BIGINT(N2) &&
      (i1 + ns + (padding + 1) / 2 < BIGINT(N1))) {
    const auto line = [du, N1, i1 = UBIGINT(i1), i2 = UBIGINT(i2),
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
    const auto res_array = xsimd_to_array(res);
    for (uint8_t i{0}; i < simd_size; i += 2) {
      out[0] += res_array[i];
      out[1] += res_array[i + 1];
    }
  } else {
    // wraps somewhere: use ptr list
    // this is slower than above, but occurs much less often, with fractional
    // rate O(ns/min(N1,N2)). Thus this code doesn't need to be so optimized.
    return interp_square_wrap<T, ns, simd_type>(target, du, ker1, ker2, i1, i2, N1, N2);
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type>
static void interp_cube_wrapped(T *FINUFFT_RESTRICT target, const T *du, const T *ker1,
                                const T *ker2, const T *ker3, const BIGINT i1,
                                const BIGINT i2, const BIGINT i3, const UBIGINT N1,
                                const UBIGINT N2, const UBIGINT N3) {
  /*
   * This function is called when the kernel wraps around the cube.
   * Similarly to 2D and 1D wrapping, this is slower than the non wrapping version.
   */
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto alignment = arch_t::alignment();
  const auto in_bounds_1          = (i1 >= 0) & (i1 + ns <= BIGINT(N1));
  const auto in_bounds_2          = (i2 >= 0) & (i2 + ns <= BIGINT(N2));
  const auto in_bounds_3          = (i3 >= 0) & (i3 + ns <= BIGINT(N3));
  std::array<T, 2> out{0};
  // case no wrapping needed but padding spills over du array.
  // Hence, no explicit vectorization but the code is still faster
  if (FINUFFT_LIKELY(in_bounds_1 && in_bounds_2 && in_bounds_3)) {
    // no wrapping: avoid ptrs (by far the most common case)
    // store a horiz line (interleaved real,imag)
    // initialize line with zeros; hard to avoid here, but overhead small in 3D
    alignas(alignment) std::array<T, 2 * ns> line{0};
    // co-add y and z contributions to line in x; do not apply x kernel yet
    // This is expensive innermost loop
    for (uint8_t dz{0}; dz < ns; ++dz) {
      const auto oz = N1 * N2 * (i3 + dz); // offset due to z
      for (uint8_t dy{0}; dy < ns; ++dy) {
        const auto l_ptr = du + 2 * (oz + N1 * (i2 + dy) + i1); // ptr start of line
        const auto ker23 = ker2[dy] * ker3[dz];
        for (uint8_t l{0}; l < 2 * ns; ++l) {
          // loop over ns interleaved (R,I) pairs
          line[l] = std::fma(l_ptr[l], ker23, line[l]);
        }
      }
    }
    // apply x kernel to the (interleaved) line and add together (cheap)
    for (uint8_t dx{0}; dx < ns; ++dx) {
      out[0] = std::fma(line[2 * dx], ker1[dx], out[0]);
      out[1] = std::fma(line[2 * dx + 1], ker1[dx], out[1]);
    }
  } else {
    // ...can be slower since this case only happens with probability
    // O(ns/min(N1,N2,N3))
    alignas(alignment) std::array<UBIGINT, ns> j1{}, j2{}, j3{}; // 1d ptr lists
    auto x = i1, y = i2, z = i3; // initialize coords
    for (uint8_t d{0}; d < ns; d++) {
      // set up ptr lists
      if (x < 0) x += BIGINT(N1);
      if (x >= BIGINT(N1)) x -= BIGINT(N1);
      j1[d] = x++;
      if (y < 0) y += BIGINT(N2);
      if (y >= BIGINT(N2)) y -= BIGINT(N2);
      j2[d] = y++;
      if (z < 0) z += BIGINT(N3);
      if (z >= BIGINT(N3)) z -= BIGINT(N3);
      j3[d] = z++;
    }
    for (uint8_t dz{0}; dz < ns; dz++) {
      // use the pts lists
      const auto oz = N1 * N2 * j3[dz]; // offset due to z
      for (uint8_t dy{0}; dy < ns; dy++) {
        const auto oy    = oz + N1 * j2[dy]; // offset due to y & z
        const auto ker23 = ker2[dy] * ker3[dz];
        for (uint8_t dx{0}; dx < ns; dx++) {
          const auto k = ker1[dx] * ker23;
          const auto j = oy + j1[dx];
          out[0] += du[2 * j] * k;
          out[1] += du[2 * j + 1] * k;
        }
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, uint8_t ns, class simd_type = PaddedSIMD<T, 2 * ns>>
static void interp_cube(T *FINUFFT_RESTRICT target, const T *du, const T *ker1,
                        const T *ker2, const T *ker3, BIGINT i1, BIGINT i2, BIGINT i3,
                        UBIGINT N1, UBIGINT N2, UBIGINT N3)
/* 3D interpolate complex values from a ns*ns*ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   ns*ns*ns outer product of the 1d kernel lists ker1, ker2, and ker3.
   Inputs:
   du : input regular grid of size 2*N1*N2*N3 (alternating real,imag)
   ker1, ker2, ker3 : length-ns real arrays of 1d kernel evaluations
   i1 : start (left-most) x-coord index to read du from, where the indices
        of du run from 0 to N1-1, and indices outside that range are wrapped.
   i2 : start (bottom) y-coord index to read du from.
   i3 : start (lowest) z-coord index to read du from.
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
   Internally, dx,dy,dz indices into ker array, l indices the 2*ns interleaved
   line array, j is index in complex du array.

   Internally, dx,dy,dz indices into ker array, j index in complex du array.
   Barnett 6/16/17.
   No-wrap case sped up for FMA/SIMD by Reinecke 6/19/23
   (see above note in interp_square)
   Barbone July 2024: - moved wrapping logic to interp_cube_wrapped
                      - using explicit SIMD vectorization to overcome the out[2] array
                        limitation
   The code is largely similar to 2D and 1D interpolation, please see the explanation
   there
*/
{
  using arch_t                          = typename simd_type::arch_type;
  static constexpr auto padding         = get_padding<T, 2 * ns>();
  static constexpr auto simd_size       = simd_type::size;
  static constexpr uint8_t line_vectors = (2 * ns + padding) / simd_size;
  const auto in_bounds_1                = (i1 >= 0) & (i1 + ns <= BIGINT(N1));
  const auto in_bounds_2                = (i2 >= 0) & (i2 + ns <= BIGINT(N2));
  const auto in_bounds_3                = (i3 >= 0) & (i3 + ns <= BIGINT(N3));
  std::array<T, 2> out{0};
  if (in_bounds_1 && in_bounds_2 && in_bounds_3 &&
      (i1 + ns + (padding + 1) / 2 < BIGINT(N1))) {
    const auto line = [N1, N2, i1 = UBIGINT(i1), i2 = UBIGINT(i2), i3 = UBIGINT(i3), ker2,
          ker3, du]() constexpr noexcept {
          std::array<simd_type, line_vectors> line{0};
          for (uint8_t dz{0}; dz < ns; ++dz) {
            const auto oz = N1 * N2 * (i3 + dz); // offset due to z
            for (uint8_t dy{0}; dy < ns; ++dy) {
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
    const auto res_array = xsimd_to_array(res);
    for (uint8_t i{0}; i < simd_size; i += 2) {
      out[0] += res_array[i];
      out[1] += res_array[i + 1];
    }
  } else {
    return interp_cube_wrapped<T, ns, simd_type>(target, du, ker1, ker2, ker3, i1, i2, i3,
                                                 N1, N2, N3);
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename T, int ns, int nc>
FINUFFT_NEVER_INLINE static int interpSorted_kernel(
    const std::vector<BIGINT> &sort_indices, const UBIGINT N1, const UBIGINT N2,
    const UBIGINT N3, const T *data_uniform, const UBIGINT M,
    const T *FINUFFT_RESTRICT kx, const T *FINUFFT_RESTRICT ky,
    const T *FINUFFT_RESTRICT kz, T *FINUFFT_RESTRICT data_nonuniform,
    const finufft_spread_opts &opts, const T *horner_coeffs_ptr)
// Interpolate to NU pts in sorted order from a uniform grid.
// See spreadinterp() for doc.
{
  using simd_type = PaddedSIMD<T, 2 * ns>;
  using arch_t = typename simd_type::arch_type;
  static constexpr auto alignment = arch_t::alignment();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto ns2 = ns * T(0.5); // half spread width, used as stencil shift

  CNTime timer{};
  const auto ndims = ndims_from_Ns(N1, N2, N3);
  auto nthr        = MY_OMP_GET_MAX_THREADS(); // guess # threads to use to interp
  if (opts.nthreads > 0) nthr = opts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1; // single-threaded lib must override user
#endif
  if (opts.debug)
    printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr);
  timer.start();
#pragma omp parallel num_threads(nthr)
  {
    static constexpr auto CHUNKSIZE = simd_size; // number of targets per chunk
    alignas(alignment) UBIGINT jlist[CHUNKSIZE];
    alignas(alignment) T xjlist[CHUNKSIZE], yjlist[CHUNKSIZE], zjlist[CHUNKSIZE];
    alignas(alignment) T outbuf[2 * CHUNKSIZE];
    // Kernels: static alloc is faster, so we do it for up to 3D...
    alignas(alignment) std::array<T, 3 * MAX_NSPREAD> kernel_values{0};
    auto *FINUFFT_RESTRICT ker1 = kernel_values.data();
    auto *FINUFFT_RESTRICT ker2 = kernel_values.data() + MAX_NSPREAD;
    auto *FINUFFT_RESTRICT ker3 = kernel_values.data() + 2 * MAX_NSPREAD;

    // Loop over interpolation chunks
    // main loop over NU trgs, interp each from U
    // (note: windows omp doesn't like unsigned loop vars)
#pragma omp for schedule(dynamic, 1000) // assign threads to NU targ pts:
    for (BIGINT i = 0; i < BIGINT(M); i += CHUNKSIZE) {
      // Setup buffers for this chunk
      const UBIGINT bufsize = (i + CHUNKSIZE > M) ? M - i : CHUNKSIZE;
      for (UBIGINT ibuf = 0; ibuf < bufsize; ibuf++) {
        UBIGINT j    = sort_indices[i + ibuf];
        jlist[ibuf]  = j;
        xjlist[ibuf] = fold_rescale<T>(kx[j], N1);
        if (ndims >= 2) yjlist[ibuf] = fold_rescale<T>(ky[j], N2);
        if (ndims == 3) zjlist[ibuf] = fold_rescale<T>(kz[j], N3);
      }

      // Loop over targets in chunk
      for (UBIGINT ibuf = 0; ibuf < bufsize; ibuf++) {
        const auto xj = xjlist[ibuf];
        const auto yj = (ndims > 1) ? yjlist[ibuf] : 0;
        const auto zj = (ndims > 2) ? zjlist[ibuf] : 0;

        auto *FINUFFT_RESTRICT target = outbuf + 2 * ibuf;

        // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
        const auto i1 = BIGINT(std::ceil(xj - ns2)); // leftmost grid index
        const auto i2 = (ndims > 1) ? BIGINT(std::ceil(yj - ns2)) : 0; // min y grid index
        const auto i3 = (ndims > 2) ? BIGINT(std::ceil(zj - ns2)) : 0; // min z grid index

        const auto x1 = std::ceil(xj - ns2) - xj; // shift of ker center, in [-w/2,-w/2+1]
        const auto x2 = (ndims > 1) ? std::ceil(yj - ns2) - yj : 0;
        const auto x3 = (ndims > 2) ? std::ceil(zj - ns2) - zj : 0;

        // eval kernel values patch and use to interpolate from uniform data...
        switch (ndims) {
        case 1:
          evaluate_kernel_vector<ns, nc, T, simd_type>(kernel_values.data(),
                                                       horner_coeffs_ptr, x1);
          interp_line<T, ns, simd_type>(target, data_uniform, ker1, i1, N1);
          break;
        case 2:
          evaluate_kernel_vector<ns, nc, T, simd_type>(kernel_values.data(),
                                                       horner_coeffs_ptr, x1, x2);
          interp_square<T, ns, simd_type>(target, data_uniform, ker1, ker2, i1, i2, N1,
                                          N2);
          break;
        case 3:
          evaluate_kernel_vector<ns, nc, T, simd_type>(kernel_values.data(),
                                                       horner_coeffs_ptr, x1, x2, x3);
          interp_cube<T, ns, simd_type>(target, data_uniform, ker1, ker2, ker3, i1, i2,
                                        i3, N1, N2, N3);
          break;
        default: // can't get here
          FINUFFT_UNREACHABLE;
          break;
        }
      } // end loop over targets in chunk

      // Copy result buffer to output array
      for (UBIGINT ibuf = 0; ibuf < bufsize; ibuf++) {
        const UBIGINT j            = jlist[ibuf];
        data_nonuniform[2 * j]     = outbuf[2 * ibuf];
        data_nonuniform[2 * j + 1] = outbuf[2 * ibuf + 1];
      }

    } // end NU targ loop
  } // end parallel section
  if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n", timer.elapsedsec());
  return 0;
}

namespace {

template<typename T> struct InterpSortedCaller {
  const std::vector<BIGINT> &sort_indices;
  UBIGINT N1, N2, N3;
  T *data_uniform;
  UBIGINT M;
  const T *kx;
  const T *ky;
  const T *kz;
  T *data_nonuniform;
  const finufft_spread_opts &opts;
  const T *horner_coeffs_ptr;

  template<int NS, int NC> int operator()() const {
    if constexpr (!::finufft::kernel::ValidKernelParams<NS, NC>()) {
      return report_invalid_kernel_params(NS, NC);
    } else {
      return interpSorted_kernel<T, NS, NC>(sort_indices, N1, N2, N3, data_uniform, M, kx,
                                            ky, kz, data_nonuniform, opts,
                                            horner_coeffs_ptr);
    }
  }
};

} // namespace

template<typename T>
int interpSorted(
    const std::vector<BIGINT> &sort_indices, const UBIGINT N1, const UBIGINT N2,
    const UBIGINT N3, T *FINUFFT_RESTRICT data_uniform, const UBIGINT M,
    const T *FINUFFT_RESTRICT kx, const T *FINUFFT_RESTRICT ky,
    const T *FINUFFT_RESTRICT kz, T *FINUFFT_RESTRICT data_nonuniform,
    const finufft_spread_opts &opts, const T *horner_coeffs_ptr, int nc) {
  InterpSortedCaller<T> caller{
      sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts,
      horner_coeffs_ptr};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  using NcSeq = make_range<MIN_NC, MAX_NC>;
  auto params =
      std::make_tuple(DispatchParam<NsSeq>{opts.nspread}, DispatchParam<NcSeq>{nc});
  return dispatch(caller, params);
}

} // namespace finufft::spreadinterp
