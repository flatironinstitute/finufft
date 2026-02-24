#pragma once

#include <finufft/detail/spread.hpp>
#include <finufft/detail/interp.hpp>

namespace finufft::spreadinterp {

template<typename T>
int indexSort(std::vector<BIGINT> &sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
              UBIGINT M, const T *kx, const T *ky, const T *kz,
              const finufft_spread_opts &opts)
/* This makes a decision whether or not to sort the NU pts (influenced by
   opts.sort), and if yes, calls either single- or multi-threaded bin sort,
   writing reordered index list to sort_indices. If decided not to sort, the
   identity permutation is written to sort_indices.
   The permutation is designed to make RAM access close to contiguous, to
   speed up spreading/interpolation, in the case of disordered NU points.

   Inputs:
    M        - number of input NU points.
    kx,ky,kz - length-M arrays of real coords of NU pts. Domain is [-pi, pi),
                points outside are folded in.
               (only kz used in 1D, only kx and ky used in 2D.)
    N1,N2,N3 - integer sizes of overall box (set N2=N3=1 for 1D, N3=1 for 2D).
               1 = x (fastest), 2 = y (medium), 3 = z (slowest).
    opts     - spreading options struct,
                see ../include/finufft_common/spread_opts.h
   Outputs:
    sort_indices - a good permutation of NU points. (User must preallocate
                   to length M.) Ie, kx[sort_indices[j]], j=0,..,M-1, is a good
                   ordering for the x-coords of NU pts, etc.
    returned value - whether a sort was done (1) or not (0).

   Barnett 2017; split out by Melody Shih, Jun 2018. Barnett nthr logic 2024.
*/
{
  CNTime timer{};
  uint8_t ndims = ndims_from_Ns(N1, N2, N3);
  auto N        = N1 * N2 * N3; // U grid (periodic box) sizes

  // heuristic binning box size for U grid... affects performance:
  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
  // put in heuristics based on cache sizes (only useful for single-thread) ?

  int better_to_sort =
      !(ndims == 1 && (opts.spread_direction == 2 || (M > 1000 * N1))); // 1D small-N or
  // dir=2 case:
  // don't sort

  timer.start(); // if needed, sort all the NU pts...
  int did_sort = 0;
  auto maxnthr = MY_OMP_GET_MAX_THREADS(); // used if both below opts default
  if (opts.nthreads > 0)
    maxnthr = opts.nthreads; // user nthreads overrides, without limit
  if (opts.sort_threads > 0)
    maxnthr = opts.sort_threads; // high-priority override, also no limit
  // At this point: maxnthr = the max threads sorting could use
  // (we don't print warning here, since: no showwarn in spread_opts, and finufft
  // already warned about it. spreadinterp-only advanced users will miss a warning)
  if (opts.sort == 1 || (opts.sort == 2 && better_to_sort)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_debug = (opts.debug >= 2); // show timing output?
    int sort_nthr  = opts.sort_threads; // 0, or user max # threads for sort
#ifndef _OPENMP
    sort_nthr = 1; // if single-threaded lib, override user
#endif
    if (sort_nthr == 0) // multithreaded auto choice: when N>>M, one thread is better!
      sort_nthr = (10 * M > N) ? maxnthr : 1; // heuristic
    if (sort_nthr == 1)
      bin_sort_singlethread(sort_indices, M, kx, ky, kz, N1, N2, N3, bin_size_x,
                            bin_size_y, bin_size_z, sort_debug);
    else // sort_nthr>1, user fixes # threads (>=2)
      bin_sort_multithread(sort_indices, M, kx, ky, kz, N1, N2, N3, bin_size_x,
                           bin_size_y, bin_size_z, sort_debug, sort_nthr);
    if (opts.debug)
      printf("\tsorted (%d threads):\t%.3g s\n", sort_nthr, timer.elapsedsec());
    did_sort = 1;
  } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static, 1000000)
    for (BIGINT i     = 0; i < BIGINT(M); i++) // here omp helps xeon, hinders i7
      sort_indices[i] = i; // the identity permutation
    if (opts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n", (int)opts.sort, timer.elapsedsec());
  }
  return did_sort;
}

template<typename T>
int spreadinterpSorted(
    const std::vector<BIGINT> &sort_indices, const UBIGINT N1, const UBIGINT N2,
    const UBIGINT N3, T *data_uniform, const UBIGINT M, const T *FINUFFT_RESTRICT kx,
    const T *FINUFFT_RESTRICT ky, const T *FINUFFT_RESTRICT kz,
    T *FINUFFT_RESTRICT data_nonuniform, const finufft_spread_opts &opts, int did_sort,
    bool adjoint, const T *horner_coeffs, int nc)
/* ------------Spreader/interpolator for 1, 2, or 3 dimensions --------------
  The concrete operation performed depends on both `opts.spread_direction`
  and the `adjoint` flag passed by the caller. When `adjoint` is false the
  function implements the semantics of `opts.spread_direction` directly; when
  `adjoint` is true the semantics are transposed (spread <-> interp swapped).

  For illustration, in the 1D case with `adjoint==false`:

  - If opts.spread_direction==1, the implemented operation is

                 N1-1
    data_nonuniform[j] =  SUM phi(kx[j] - n) data_uniform[n],   for j=0...M-1
                 n=0

  - If opts.spread_direction==2, the implemented operation is the transpose

               M-1
    data_uniform[n] =  SUM phi(kx[j] - n) data_nonuniform[j],   for n=0...N1-1
               j=0

  When `adjoint==true` the two formulas above are swapped (i.e. the
  function dispatches to the interpolation implementation when the pair
  `(opts.spread_direction, adjoint)` indicates it).

   In each case phi is the spreading kernel, which has support
   [-opts.nspread/2,opts.nspread/2]. In 2D or 3D, the generalization with
   product of 1D kernels is performed.
   For 1D set N2=N3=1; for 2D set N3=1; for 3D set N1,N2,N3>1.

   Notes:
   No particular normalization of the spreading kernel is assumed.
   Uniform (U) points are centered at coords
   [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
   fastest, y medium, z slowest ordering, up to however many
   dimensions are relevant; note that this is Fortran-style ordering for an
   array f(x,y,z), but C style for f[z][y][x]. This is to match the Fortran
   interface of the original CMCL libraries.
  Non-uniform (NU) points kx,ky,kz are real and are folded into the uniform
  grid period by the internal `fold_rescale` helper. Historically the code
  expected points within the central three periods, but `fold_rescale` now
  accepts arbitrary real inputs and reduces them to the canonical period;
  very large magnitudes can, however, suffer numerical inaccuracy in the
  folding operation.
   The finufft_spread_opts struct must have been set up already by calling setup_kernel.
  The caller must ensure the grid is large enough for spreading: in normal
  use this is checked by `spreadcheck(...)` (called from `setpts`) which
  enforces `2*opts.nspread < min(N1,N2,N3)`. If that condition is violated
  the check returns an error and spreading must not proceed.

   Inputs:
  sort_indices - length-M permutation giving the order in which nonuniform
            points should be processed (typically produced by
            indexSort/bin_sort). The implementation accesses
            kx[sort_indices[j]], ky[sort_indices[j]], etc. If no sort
            was performed, sort_indices should contain the identity
            permutation (0..M-1).
  did_sort     - integer flag (0/1) indicating whether a sort was actually
            performed. This can affect subproblem splitting heuristics
            inside the routine; callers should pass the return value of
            indexSort here.
  adjoint      - if true, request the adjoint (transpose) operation. Concretely
            this function dispatches to the two internal routines as follows:
            it calls `spreadSorted(...)` when (opts.spread_direction == 1) !=
            adjoint, and calls `interpSorted(...)` otherwise. In short,
            `adjoint` flips whether spreading (uniform->nonuniform) or
            interpolation (nonuniform->uniform) is performed relative to the
            meaning of `opts.spread_direction`.
  horner_coeffs - pointer to Horner-format kernel coefficients (may be
            nullptr if not used); used when kerevalmeth selects Horner
            evaluation.

   N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
              If N2==1, 1D spreading is done. If N3==1, 2D spreading.
          Otherwise, 3D.
   M - number of NU pts.
   kx, ky, kz - length-M real arrays of NU point coordinates (only kx read in
                1D, only kx and ky read in 2D).

        These should lie in the box -pi<=kx<=pi. Points outside this domain are also
        correctly folded back into this domain.
   opts - spread/interp options struct; see ../include/finufft_common/spread_opts.h

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
  Returned value:
  This wrapper (`spreadinterpSorted`) always returns 0 after dispatching to
  the sorted spread or interp implementations; input validation and any
  errors (for example the box-too-small condition or failures in sorting)
  are performed earlier (see `spreadcheck` called from `setpts` and
  `indexSort`/`bin_sort`). See ../docs/error.rst and `include/finufft_errors.h`
  for the global error codes that higher-level callers may receive.

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
   No separate subprob indices in t-1 2/11/18.
   sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
   kereval, kerpad 4/24/18
   Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
   this routine just a caller to them. Name change, Barnett 7/27/18
   Tidy, Barnett 5/20/20. Tidy doc, Barnett 10/22/20.
*/
{
  if ((opts.spread_direction == 1) != adjoint) // ========= direction 1 (spreading)
    // =======
    spreadSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform,
                 opts, did_sort, horner_coeffs, nc);

  else // ================= direction 2 (interpolation) ===========
    interpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform,
                 opts, horner_coeffs, nc);

  return 0;
}

///////////////////////////////////////////////////////////////////////////

template<typename T>
T evaluate_kernel_runtime(T x, int ns, int nc, const T *horner_coeffs_ptr,
                          [[maybe_unused]] const finufft_spread_opts &spopts) {
  /* Simple runtime spreading kernel evaluator for a single argument.
    Uses the precomputed piecewise polynomial coeffs (degree nc-1, where
    nc = number of coeffs per panel), for the ns panels covering its support.
    Returns phi(2x/w), where standard kernel phi has support [-1,1].
    Need not be fast, but must match the output of the above
    evaluate_kernel_vector(), which evaluates a set of ns kernel values at once,
    for the corresponding ordinate.
    Is used by numerical Fourier transform in finufft_core:onedim_*.
    Coefficients are stored as horner_coeffs[j * padded_ns + i], where padded_ns
    is rounded up to SIMD alignment which *must* be consistent with that used
    in both evaluate_kernel_vector above, and precompute_horner_coeffs.
    Barbone (Dec/25). Fixed Lu 12/23/25.
    Simplified spopts, removed redundant |x|>=ns/2 exit point, Barnett 1/15/26.
  */
  const auto simd_size            = GetPaddedSIMDWidth<T>(2 * ns);
  const int padded_ns             = (ns + simd_size - 1) & -simd_size;
  const T ns2                     = ns / T(2.0); // half width w/2, in grid point units
  T res = (T)0.0;
  for (int i = 0; i < ns; ++i) {             // check if x falls into any piecewise panels
    if (x > -ns2 + i && x <= -ns2 + i + 1) { // if so, eval that Horner polynomial
      T z = std::fma((T)2.0, x - (T)i, (T)(ns - 1)); // maps panel to z in [-1,1]
      for (int j = 0; j < nc; ++j) { // Horner loop (highest to lowest order)...
        res = std::fma(res, z, horner_coeffs_ptr[j * padded_ns + i]);
      }
      break;
    }
  }
  return res;
}

} // namespace finufft::spreadinterp
