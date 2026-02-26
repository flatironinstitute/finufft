#pragma once

// Defines interface to spreading/interpolation code.

/* Devnotes: see plan.hpp for definition of MAX_NSPREAD
    RESCALE macro moved to spreadinterp.cpp, 7/15/20.
    finufft_spread_opts renamed 6/7/22.
    Note as of v2.5 (Dec 2025):
    legacy TF_OMIT_* timing flags were removed. Timing helpers
    previously controlled by these flags have been purged from the codebase.
    The kerevalmeth/kerpad knobs remain in the public API structs solely for
    ABI compatibility and are ignored by the implementation (Horner is always
    used).
    1/9/26: setup_spreadinterp() is a private method on FINUFFT_PLAN_T, defined
    in makeplan.hpp.
*/

#include <finufft/interp.hpp>
#include <finufft/spread.hpp>
#include <finufft/plan.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <inttypes.h>
#include <vector>

// ---------- FINUFFT_PLAN_T method definitions ----------

template<typename TF>
int FINUFFT_PLAN_T<TF>::spreadcheck() const
/* Input checking and reporting for the spreader. Reads nfdim[0..2] and spopts
   from the plan.
   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
   Marco Barbone 5.8.24 removed bounds check as new foldrescale is not limited to
   [-3pi,3pi)
   Converted to class member, Barbone 2/26/26.
*/
{
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  const UBIGINT N1 = (UBIGINT)nfdim[0], N2 = (UBIGINT)nfdim[1], N3 = (UBIGINT)nfdim[2];
  UBIGINT minN = UBIGINT(2 * spopts.nspread);
  if (N1 < minN || (N2 > 1 && N2 < minN) || (N3 > 1 && N3 < minN)) {
    fprintf(stderr,
            "%s error: one or more non-trivial box dims is less than 2.nspread!\n",
            __func__);
    return FINUFFT_ERR_SPREAD_BOX_SMALL;
  }
  if (spopts.spread_direction != 1 && spopts.spread_direction != 2) {
    fprintf(stderr, "%s error: opts.spread_direction must be 1 or 2!\n", __func__);
    return FINUFFT_ERR_SPREAD_DIR;
  }
  return 0;
}

template<typename TF>
TF FINUFFT_PLAN_T<TF>::evaluate_kernel_runtime(TF x) const
/* Simple runtime spreading kernel evaluator for a single argument.
   Uses the precomputed piecewise polynomial coeffs (degree nc-1, where
   nc = number of coeffs per panel), for the ns panels covering its support.
   Returns phi(2x/w), where standard kernel phi has support [-1,1].
   Need not be fast, but must match the output of evaluate_kernel_vector(),
   which evaluates a set of ns kernel values at once, for the corresponding ordinate.
   Is used by numerical Fourier transform in onedim_fseries_kernel and
   Kernel_onedim_FT.
   Coefficients are stored as horner_coeffs[j * padded_ns + i], where padded_ns
   is rounded up to SIMD alignment which *must* be consistent with that used
   in both evaluate_kernel_vector and precompute_horner_coeffs.
   Reads spopts.nspread, nc, padded_ns, horner_coeffs from the plan.
   Barbone (Dec/25). Fixed Lu 12/23/25.
   Simplified spopts, removed redundant |x|>=ns/2 exit point, Barnett 1/15/26.
   Previous args (x, ns, nc, horner_coeffs_ptr, spopts) are now plan members
   (spopts.nspread, nc, horner_coeffs, padded_ns).
   Converted to class member, Barbone 2/24/26.
*/
{
  const int ns    = spopts.nspread;
  const TF ns2    = ns / TF(2.0); // half width w/2, in grid point units
  const TF *coefs = horner_coeffs.data();
  TF res          = TF(0.0);
  for (int i = 0; i < ns; ++i) {             // check if x falls into any piecewise panels
    if (x > -ns2 + i && x <= -ns2 + i + 1) { // if so, eval that Horner polynomial
      TF z = std::fma(TF(2.0), x - TF(i), TF(ns - 1)); // maps panel to z in [-1,1]
      for (int j = 0; j < nc; ++j) // Horner loop (highest to lowest order)...
        res = std::fma(res, z, coefs[j * padded_ns + i]);
      break;
    }
  }
  return res;
}

/*
  Approximates exact 1D Fourier transform of spreadinterp's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi, pi),
  for a kernel with x measured in grid-spacings. (See onedim_fseries_kernel
  for FT definition.) Note: old (pre-2025) name was: onedim_nuft_kernel().

  operator()(k) evaluates the Fourier transform of the kernel at a single
  frequency k, using the z and f arrays from creation time.
    Input: k - frequency, dual to the kernel's natural argument, ie exp(i.k.z)
    Output: phihat - real Fourier transform evaluated at freq k

  Barnett 2/8/17. openmp since cos slow 2/9/17.
  11/25/25, replaced kernel_definition by evaluate_kernel_runtime, so that
  the FT of the piecewise poly approximant (not "exact" kernel) is computed.
  Converted to nested class of FINUFFT_PLAN_T, Barbone 2/24/26.
  Previous constructor args (spopts, horner_coeffs_ptr, nc) are now read from
  the plan reference.
*/
template<typename TF>
FINUFFT_PLAN_T<TF>::Kernel_onedim_FT::Kernel_onedim_FT(const FINUFFT_PLAN_T &plan) {
  // Creator: uses slow kernel evals to initialize z and f arrays.
  using finufft::common::gaussquad;
  TF J2 = plan.spopts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 2.0 * J2); // > pi/2 ratio.  cannot exceed MAX_NQUAD
  if (plan.spopts.debug) printf("q (# ker FT quadr pts) = %d\n", q);
  std::vector<double> Z(2 * q), W(2 * q);
  gaussquad(2 * q, Z.data(), W.data()); // only half the nodes used, for (0,1)
  z.resize(q);
  f.resize(q);
  for (int n = 0; n < q; ++n) {
    z[n] = TF(Z[n] * J2); // quadr nodes for [0,J/2] with weights J2 * w
    f[n] = J2 * TF(W[n]) * plan.evaluate_kernel_runtime(z[n]);
  }
}

template<typename TF>
void FINUFFT_PLAN_T<TF>::indexSort()
/* Decides whether or not to sort the NU pts (influenced by spopts.sort),
   and if yes, calls either single- or multi-threaded bin sort, writing
   reordered index list to sortIndices. If decided not to sort, the
   identity permutation is written to sortIndices. Sets didSort accordingly.
   The permutation is designed to make RAM access close to contiguous, to
   speed up spreading/interpolation, in the case of disordered NU points.
   Ie, XYZ[0][sortIndices[j]], j=0,..,nj-1, is a good ordering for the
   x-coords of NU pts, etc.

   The following args from the old free-function interface are now read/written
   as plan members:
    nj           - number of input NU points.
    XYZ          - pointers to length-nj arrays of real coords of NU pts.
                   Domain is [-pi, pi), points outside are folded in.
                   (only XYZ[2] used in 1D, only XYZ[0] and XYZ[1] in 2D.)
    nfdim        - integer sizes of overall box (nfdim[1]=nfdim[2]=1 for 1D,
                   nfdim[2]=1 for 2D).
                   0 = x (fastest), 1 = y (medium), 2 = z (slowest).
    spopts       - spreading options struct,
                   see finufft_common/spread_opts.h
   Outputs (plan members):
    sortIndices  - a good permutation of NU points. (Preallocated to length nj.)
    didSort      - whether a sort was done (true) or not (false).

   Barnett 2017; split out by Melody Shih, Jun 2018. Barnett nthr logic 2024.
   Previous args (M, kx, ky, kz, N1, N2, N3, opts) are now plan members
   (nj, XYZ, nfdim, spopts). Output sortIndices and didSort are plan members.
   Converted to class member, Barbone 2/24/26.
*/
{
  using namespace finufft::spreadinterp;
  using finufft::utils::CNTime;
  CNTime timer{};
  const UBIGINT N1 = nfdim[0], N2 = nfdim[1], N3 = nfdim[2];
  const UBIGINT M = nj;

  // heuristic binning box size for U grid... affects performance:
  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;

  int better_to_sort =
      !(dim == 1 && (spopts.spread_direction == 2 || (M > 1000 * N1))); // 1D small-N or
  // dir=2 case:
  // don't sort

  timer.start(); // if needed, sort all the NU pts...
  didSort      = false;
  auto maxnthr = MY_OMP_GET_MAX_THREADS(); // used if both below opts default
  if (spopts.nthreads > 0)
    maxnthr = spopts.nthreads;             // user nthreads overrides, without limit
  if (spopts.sort_threads > 0)
    maxnthr = spopts.sort_threads;         // high-priority override, also no limit
  if (spopts.sort == 1 || (spopts.sort == 2 && better_to_sort)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_nthr = spopts.sort_threads; // 0, or user max # threads for sort
#ifndef _OPENMP
    sort_nthr = 1; // if single-threaded lib, override user
#endif
    auto grid_N = N1 * N2 * N3;
    if (sort_nthr == 0) // multithreaded auto choice: when N>>M, one thread is better!
      sort_nthr = (10 * M > grid_N) ? maxnthr : 1; // heuristic
    if (sort_nthr == 1)
      bin_sort_singlethread(bin_size_x, bin_size_y, bin_size_z);
    else // sort_nthr>1, user fixes # threads (>=2)
      bin_sort_multithread(bin_size_x, bin_size_y, bin_size_z, sort_nthr);
    if (spopts.debug)
      printf("\tsorted (%d threads):\t%.3g s\n", sort_nthr, timer.elapsedsec());
    didSort = true;
  } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static, 1000000)
    for (BIGINT i = 0; i < BIGINT(M); i++) // here omp helps xeon, hinders i7
      sortIndices[i] = i;                  // the identity permutation
    if (spopts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n", (int)spopts.sort, timer.elapsedsec());
  }
}

template<typename TF>
int FINUFFT_PLAN_T<TF>::spreadinterpSorted(TF *data_uniform, TF *data_nonuniform,
                                           bool adjoint) const
/* ------------Spreader/interpolator for 1, 2, or 3 dimensions --------------
  The concrete operation performed depends on both `spopts.spread_direction`
  and the `adjoint` flag passed by the caller. When `adjoint` is false the
  function implements the semantics of `spopts.spread_direction` directly; when
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
   The finufft_spread_opts struct must have been set up already by calling
   setup_spreadinterp.
  The caller must ensure the grid is large enough for spreading: in normal
  use this is checked by `spreadcheck(...)` (called from `setpts`) which
  enforces `2*opts.nspread < min(N1,N2,N3)`. If that condition is violated
  the check returns an error and spreading must not proceed.

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
   adjoint      - if true, request the adjoint (transpose) operation. Concretely
            this function dispatches to the two internal routines as follows:
            it calls `spreadSorted(...)` when (spopts.spread_direction == 1) !=
            adjoint, and calls `interpSorted(...)` otherwise. In short,
            `adjoint` flips whether spreading (uniform->nonuniform) or
            interpolation (nonuniform->uniform) is performed relative to the
            meaning of `spopts.spread_direction`.

  The following args from the old free-function interface are now read as plan
  members:
  sortIndices  - length-nj permutation giving the order in which nonuniform
            points should be processed (typically produced by indexSort).
            If no sort was performed, contains the identity permutation.
  didSort      - bool indicating whether a sort was actually performed.
            This can affect subproblem splitting heuristics inside the
            routine.
  nfdim        - grid sizes in x (fastest), y (medium), z (slowest) respectively.
            If nfdim[1]==1, 1D spreading is done. If nfdim[2]==1, 2D.
  nj           - number of NU pts.
  XYZ          - pointers to length-nj real arrays of NU point coordinates
            (only XYZ[0] read in 1D, only XYZ[0] and XYZ[1] read in 2D).
            These should lie in the box -pi<=kx<=pi. Points outside this
            domain are also correctly folded back into this domain.
  spopts       - spread/interp options struct; see finufft_common/spread_opts.h
  horner_coeffs - Horner kernel coefficients.
  nc           - number of Horner coefficients per panel.

  Returned value:
  This wrapper always returns 0 after dispatching to the sorted spread or
  interp implementations; input validation and any errors (for example the
  box-too-small condition or failures in sorting) are performed earlier (see
  `spreadcheck` called from `setpts` and `indexSort`). See ../docs/error.rst
  and `include/finufft_errors.h` for the global error codes that higher-level
  callers may receive.

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
   No separate subprob indices in t-1 2/11/18.
   sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
   kereval, kerpad 4/24/18
   Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
   this routine just a caller to them. Name change, Barnett 7/27/18
   Tidy, Barnett 5/20/20. Tidy doc, Barnett 10/22/20.
   Previous args (sort_indices, N1, N2, N3, M, kx, ky, kz, opts, did_sort,
   horner_coeffs, nc) are now plan members.
   Converted to class member, Barbone 2/24/26.
*/
{
  if ((spopts.spread_direction == 1) != adjoint) // ========= direction 1 (spreading)
    spreadSorted(data_uniform, data_nonuniform);
  else // ================= direction 2 (interpolation) ===========
    interpSorted(data_uniform, data_nonuniform);
  return 0;
}

template<typename TF>
int FINUFFT_PLAN_T<TF>::spreadSorted(TF *FINUFFT_RESTRICT data_uniform,
                                     const TF *data_nonuniform) const
/* Spread NU pts (in sort order) to a uniform grid. See spreadinterpSorted() for doc.
   Plan members used in place of the former free-function arguments:
   sortIndices, nfdim[0..2], nj, XYZ[0..2], spopts, didSort, horner_coeffs, nc.
   Instantiated in src/spread.cpp; extern template in execute.cpp suppresses
   re-instantiation there.
*/
{
  using namespace finufft::spreadinterp;
  using finufft::utils::CNTime;
  // Alias plan members to local names matching the original algorithm.
  const auto N1                 = (UBIGINT)nfdim[0];
  const auto N2                 = (UBIGINT)nfdim[1];
  const auto N3                 = (UBIGINT)nfdim[2];
  const auto M                  = (UBIGINT)nj;
  const auto *kx                = XYZ[0];
  const auto *ky                = XYZ[1];
  const auto *kz                = XYZ[2];
  const auto did_sort = (int)didSort;
  CNTime timer{};
  const auto ndims = ndims_from_Ns(N1, N2, N3);
  const auto N     = N1 * N2 * N3; // output array size
  auto nthr        = MY_OMP_GET_MAX_THREADS(); // guess # threads to use to spread
  if (spopts.nthreads > 0) nthr = spopts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1; // single-threaded lib must override user
#endif
  if (spopts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr);
  timer.start();
  std::fill(data_uniform, data_uniform + 2 * N, 0.0); // zero the output array
  if (spopts.debug) printf("\tzero output array\t%.3g s\n", timer.elapsedsec());
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
    if (spopts.debug) printf("\tt1 simple spreading:\t%.3g s\n", timer.elapsedsec());
  } else {
    // ------- Fancy multi-core blocked t1 spreading ----
    // Splits sorted inds (jfm's advanced2), could double RAM.
    // choose nb (# subprobs) via used nthreads:
    auto nb = std::min((UBIGINT)nthr, M); // simply split one subprob per thr...
    if (nb * (BIGINT)spopts.max_subproblem_size < M) {
      // ...or more subprobs to cap size
      nb = 1 + (M - 1) / spopts.max_subproblem_size; // int div does
      // ceil(M/spopts.max_subproblem_size)
      if (spopts.debug)
        printf("\tcapping subproblem sizes to max of %d\n", spopts.max_subproblem_size);
    }
    if (M * 1000 < N) {
      // low-density heuristic: one thread per NU pt!
      nb = M;
      if (spopts.debug) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr == 1) {
      nb = 1;
      if (spopts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }
    if (spopts.debug && nthr > spopts.atomic_threshold)
      printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");

    std::vector<UBIGINT> brk(nb + 1); // NU index breakpoints defining nb subproblems
    for (UBIGINT p = 0; p <= nb; ++p) brk[p] = (M * p + nb - 1) / nb;

#pragma omp parallel num_threads(nthr)
    {
      // local copies of NU pts and data for each subproblem
      std::vector<TF> kx0{}, ky0{}, kz0{}, dd0{}, du0{};
#pragma omp for schedule(dynamic, 1)                     // each is big
      for (BIGINT isub = 0; isub < BIGINT(nb); isub++) { // Main loop through subproblems
        const auto M0 = brk[isub + 1] - brk[isub];      // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        kx0.resize(M0);
        ky0.resize(M0 * (N2 > 1));
        kz0.resize(M0 * (N3 > 1));
        dd0.resize(2 * M0); // complex strength data
        for (UBIGINT j = 0; j < M0; j++) {
          // todo: can avoid this copying?
          const auto kk = sortIndices[j + brk[isub]]; // NU pt from subprob index list
          kx0[j]        = fold_rescale<TF>(kx[kk], N1);
          if (N2 > 1) ky0[j] = fold_rescale<TF>(ky[kk], N2);
          if (N3 > 1) kz0[j] = fold_rescale<TF>(kz[kk], N3);
          dd0[j * 2]     = data_nonuniform[kk * 2];     // real part
          dd0[j * 2 + 1] = data_nonuniform[kk * 2 + 1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        BIGINT offset1, offset2, offset3, padded_size1, size1, size2, size3;
        get_subgrid(offset1, offset2, offset3, padded_size1, size1, size2, size3, M0,
                    kx0.data(), ky0.data(), kz0.data());
        if (spopts.debug > 1) {
          print_subgrid_info(ndims, offset1, offset2, offset3, padded_size1, size1, size2,
                             size3, M0);
        }
        // allocate output data for this subgrid
        du0.resize(2 * padded_size1 * size2 * size3); // complex
        // Spread to subgrid without need for bounds checking or wrapping
        if (ndims == 1)
          spread_subproblem_1d(offset1, padded_size1, du0.data(), M0, kx0.data(),
                               dd0.data());
        else if (ndims == 2)
          spread_subproblem_2d(offset1, offset2, padded_size1, size2, du0.data(), M0,
                               kx0.data(), ky0.data(), dd0.data());
        else
          spread_subproblem_3d(offset1, offset2, offset3, padded_size1, size2, size3,
                               du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               dd0.data());
        // add subgrid to output (always do this); atomic vs critical chosen
        if (nthr > spopts.atomic_threshold) {
          add_wrapped_subgrid<true>(offset1, offset2, offset3, padded_size1, size1, size2,
                                   size3, data_uniform,
                                   du0.data()); // R Blackwell's atomic version
        } else {
#pragma omp critical
          add_wrapped_subgrid<false>(offset1, offset2, offset3, padded_size1, size1, size2,
                                    size3, data_uniform, du0.data());
        }
      } // end main loop over subprobs
    }
    if (spopts.debug)
      printf("\tt1 fancy spread: \t%.3g s (%" PRIu64 " subprobs)\n", timer.elapsedsec(),
             nb);
  } // end of choice of which t1 spread type to use
  return 0;
}
