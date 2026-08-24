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
#include <finufft/plan.hpp>
#include <finufft/spread.hpp>
#include <finufft/utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <inttypes.h>
#include <vector>

// ---------- FINUFFT_PLAN_T method definitions ----------

template<typename TF>
void FINUFFT_PLAN_T<TF>::spreadcheck() const
/* Input checking and reporting for the spreader. Reads nfdim[0..2] and spopts
   from the plan. Throws finufft::exception on error.
   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
   Marco Barbone 5.8.24 removed bounds check as new foldrescale is not limited to
   [-3pi,3pi)
   Converted to class member, Barbone 2/26/26.
*/
{
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  const UBIGINT N1 = (UBIGINT)m.nfdim[0], N2 = (UBIGINT)m.nfdim[1],
                N3 = (UBIGINT)m.nfdim[2];
  UBIGINT minN = UBIGINT(2 * m.spopts.nspread);
  if (N1 < minN || (N2 > 1 && N2 < minN) || (N3 > 1 && N3 < minN)) {
    fprintf(stderr,
            "%s error: one or more non-trivial box dims is less than 2.nspread!\n",
            __func__);
    throw finufft::exception(FINUFFT_ERR_SPREAD_BOX_SMALL);
  }
  if (m.spopts.spread_direction != 1 && m.spopts.spread_direction != 2) {
    fprintf(stderr, "%s error: opts.spread_direction must be 1 or 2!\n", __func__);
    throw finufft::exception(FINUFFT_ERR_SPREAD_DIR);
  }
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
  const int ns    = m.spopts.nspread;
  const TF ns2    = ns / TF(2.0); // half width w/2, in grid point units
  const TF *coefs = m.horner_coeffs.data();
  TF res          = TF(0.0);
  // Invariant: m.padded_ns is the runtime mirror of
  // finufft::spreadinterp::KernelBufferLayout<TF, NS>::stride; both are produced
  // by kernel_buffer_stride_runtime<TF>(ns) in precompute_horner_coeffs.
  for (int i = 0; i < ns; ++i) {             // check if x falls into any piecewise panels
    if (x > -ns2 + i && x <= -ns2 + i + 1) { // if so, eval that Horner polynomial
      TF z = std::fma(TF(2.0), x - TF(i), TF(ns - 1)); // maps panel to z in [-1,1]
      for (int j = 0; j < m.nc; ++j) // Horner loop (highest to lowest order)...
        res = std::fma(res, z, coefs[j * m.padded_ns + i]);
      break;
    }
  }
  return res;
}

/*
  1D Fourier transform of spreadinterp's real symmetric kernel, evaluated at a
  set of arbitrary freqs k in [-pi, pi), for a kernel with x measured in
  grid-spacings. (See onedim_fseries_kernel for FT definition.) Uses the
  analytic PSWF self-FT (one Horner eval per freq; see pswf_selfft_params), the
  prolate being an eigenfunction of the finite FT. Old name: onedim_nuft_kernel.

  operator()(k) returns the kernel FT phihat at a single frequency k.
    Input: k - frequency, dual to the kernel's natural argument, ie exp(i.k.z)
    Output: phihat - real Fourier transform evaluated at freq k

  Barnett 2/8/17. Converted to nested class, Barbone 2/24/26.
  Analytic PSWF self-FT replacing per-point cosine quadrature, Barbone 7/23/26.
*/
template<typename TF>
FINUFFT_PLAN_T<TF>::Kernel_onedim_FT::Kernel_onedim_FT(const FINUFFT_PLAN_T &plan)
    : plan_ptr(&plan) {
  const auto [gs, pf] = finufft::kernel::pswf_selfft_params(
      plan.m.spopts.nspread, plan.m.spopts.beta, plan.m.horner_coeffs.data(), plan.m.nc,
      int(plan.m.padded_ns));
  grid_scale = TF(gs);
  prefac = TF(pf);
}

template<typename TF> int FINUFFT_PLAN_T<TF>::tile_doublings(int cell) const noexcept {
  using finufft::spreadinterp::ndims_from_Ns;
  return spread_tile_doublings(cell, ndims_from_Ns(m.nfdim[0], m.nfdim[1], m.nfdim[2]),
                               m.spopts.nspread, double(m.nj) / double(grid_size()));
}

template<typename TF>
SpreadSchedule FINUFFT_PLAN_T<TF>::make_schedule(int nthr, int batchSize) const {
  return spread_schedule(m.tiles, UBIGINT(m.nj), grid_size(), nthr, batchSize);
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
  const UBIGINT N1 = m.nfdim[0], N2 = m.nfdim[1], N3 = m.nfdim[2];
  const UBIGINT M = m.nj;

  // Both directions bin into cubic cache tiles and take each tile as one subproblem, so
  // the subgrids are cache-sized cuboids, not x-saturated slabs. Skipping the sort leaves
  // one tile, the whole fine grid, which is the box an unsorted subproblem spans anyway.
  const int ndims = ndims_from_Ns(N1, N2, N3);

  timer.start(); // if needed, sort all the NU pts...
  m.didSort    = false;
  m.tiles.clear();
  auto maxnthr = MY_OMP_GET_MAX_THREADS(); // used if both below opts default
  if (m.spopts.nthreads > 0)
    maxnthr = m.spopts.nthreads;           // user nthreads overrides, without limit
  // the threads that will spread; the sort override below must not reach this
  const int spreading_threads = int(maxnthr);
  if (m.spopts.sort_threads > 0)
    maxnthr = m.spopts.sort_threads;       // high-priority override, also no limit
  // A sort costs one pass over the points and buys every later pass a fine grid that
  // stays in cache, so it pays wherever the grid does not already fit: one thread
  // writing a grid inside L2 revisits nothing the cache lost, and there the pass is all
  // cost.
  const bool cache_resident =
      spreading_threads == 1 &&
      2 * sizeof(TF) * N1 * N2 * N3 <= UBIGINT(finufft::utils::getL2CacheSize());
  if (m.spopts.sort == 1 || (m.spopts.sort == 2 && !cache_resident)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_nthr = m.spopts.sort_threads; // 0, or user max # threads for sort
#ifndef _OPENMP
    sort_nthr = 1; // if single-threaded lib, override user
#endif
    auto grid_N = N1 * N2 * N3;
    if (sort_nthr == 0) // multithreaded auto choice: when N>>M, one thread is better!
      sort_nthr = (10 * M > grid_N) ? maxnthr : 1; // heuristic
    // Tile the fine grid into cubic cells, grouped 2^doublings cells per tile edge.
    const auto [cell, doublings] = [&]() -> std::pair<int, int> {
      // A coarser cell is a cheaper sort, and L1 bounds it: the cell's kernel writes
      // cover (cell+ns)^ndims and every point of the cell revisits them. Grow from 4
      // while the doubled cell's write set fits L1 and the tile keeps two cells per edge.
      int edge              = 4;
      int shift             = tile_doublings(edge);
      // complex fine-grid elements L1 holds, the unit a cell's write set is counted in
      const double l1_cells = double(finufft::utils::getL1CacheSize()) / (2 * sizeof(TF));
      while (shift > 1 &&
             spread_pow_ndims(2.0 * edge + m.spopts.nspread, ndims) <= l1_cells) {
        edge *= 2;
        --shift;
      }
      return {edge, shift};
    }();
    if (m.spopts.debug)
      printf("\tspread tiles of %d grid pts per edge, cells of %d\n", cell << doublings,
             cell);
    if (sort_nthr == 1)
      bin_sort_singlethread(cell, doublings, m.tiles);
    else // sort_nthr>1, user fixes # threads (>=2)
      bin_sort_multithread(cell, sort_nthr, doublings, m.tiles);
    if (m.spopts.debug)
      printf("\tsorted (%d threads):\t%.3g s\n", sort_nthr, timer.elapsedsec());
    m.didSort = true;
  } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static, 1000000)
    for (BIGINT i = 0; i < BIGINT(M); i++) // here omp helps xeon, hinders i7
      m.sortIndices[i] = i;                // the identity permutation
    if (m.spopts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n", (int)m.spopts.sort, timer.elapsedsec());
  }
}

/* ------------Spreader/interpolator for 1, 2, or 3 dimensions --------------
  `spreadSorted` and `interpSorted` below implement the two directions of
  `spopts.spread_direction`; `spreadinterpSortedBatch` (execute.hpp) picks one.

  For illustration, in the 1D case:

  - Spreading (direction 1) computes

               M-1
    data_uniform[n] =  SUM phi(kx[j] - n) data_nonuniform[j],   for n=0...N1-1
               j=0

  - Interpolation (direction 2) computes the transpose

                 N1-1
    data_nonuniform[j] =  SUM phi(kx[j] - n) data_uniform[n],   for j=0...M-1
                 n=0

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

  The following args from the old free-function interface are now read as plan members:
  sortIndices  - length-nj permutation giving the order in which nonuniform points
            should be processed (produced by indexSort). If no sort was performed,
            contains the identity permutation.
  didSort      - bool indicating whether a sort was actually performed.
  tiles        - the tile layout that sort binned into, empty unless tile-binned. The
            subproblems come off its offsets, so it selects the subproblem splitting
            (see spread_schedule).
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

  Both functions always return 0; input validation and any errors (for example
  the box-too-small condition or failures in sorting) are performed earlier
  (see `spreadcheck` called from `setpts` and `indexSort`). See
  ../docs/error.rst and `include/finufft_errors.h` for the global error codes
  that higher-level callers may receive.

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
   Converted to class members, Barbone 2/24/26.
*/

template<typename TF>
int FINUFFT_PLAN_T<TF>::spreadSorted(TF *FINUFFT_RESTRICT data_uniform,
                                     const TF *data_nonuniform, int batchSize) const
/* Spread NU pts (in sort order) to a uniform grid. See the block comment above.
   The sorted points split into cache-sized subproblems; each spreads into its own
   padded subgrid and adds that subgrid back into the fine grid. batchSize vectors,
   laid out one after another in both arrays, share one pass: the (vector, subproblem)
   pairs are what threads draw from.
   Plan members used in place of the former free-function arguments:
   sortIndices, nfdim[0..2], nj, XYZ[0..2], spopts, tiles, horner_coeffs, nc.
   Instantiated in src/spreadinterp.cpp; extern template in execute.cpp suppresses
   re-instantiation there.
   Magland Dec 2016; history in the block comment above.
   Tiled subproblems off the bin sort, M. Barbone 8/25/26.
*/
{
  using namespace finufft::spreadinterp;
  using finufft::utils::CNTime;
  // Alias plan members to local names matching the original algorithm.
  const auto N1  = (UBIGINT)m.nfdim[0];
  const auto N2  = (UBIGINT)m.nfdim[1];
  const auto N3  = (UBIGINT)m.nfdim[2];
  const auto M   = (UBIGINT)m.nj;
  const auto *kx = m.XYZ[0];
  const auto *ky = m.XYZ[1];
  const auto *kz = m.XYZ[2];
  CNTime timer{};
  const auto ndims     = ndims_from_Ns(N1, N2, N3);
  const auto N         = N1 * N2 * N3; // fine grid size
  const auto stride_u  = 2 * N;        // per-vector strides through the batch arrays
  const auto stride_nu = 2 * M;
  auto nthr            = MY_OMP_GET_MAX_THREADS();
  if (m.spopts.nthreads > 0) nthr = m.spopts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1; // single-threaded lib must override user
#endif
  if (m.spopts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d, batch=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr, batchSize);
  timer.start();
  // zero the whole batch, split by thread rather than by vector: the fill is
  // bandwidth-bound and batchSize can be smaller than nthr
  const auto ntot = BIGINT(batchSize) * BIGINT(stride_u);
#pragma omp parallel for num_threads(nthr) schedule(static)
  for (int t = 0; t < nthr; ++t)
    std::fill(data_uniform + ntot * t / nthr, data_uniform + ntot * (t + 1) / nthr,
              TF(0));
  if (m.spopts.debug) printf("\tzero output array\t%.3g s\n", timer.elapsedsec());
  if (M == 0) // no NU pts, we're done
    return 0;

  timer.start();
  // Skipping the sort leaves no tiles, and that list is one tile spanning the whole fine
  // grid, so one call cuts the points either way.
  const SpreadSchedule sched = make_schedule(nthr, batchSize);
  const auto &bounds         = sched.bounds;
  const UBIGINT nb           = bounds.size() - 1;
  if (m.spopts.debug)
    printf("\tcache tiles: %lld subprobs over %lld tiles, cap %lld pts\n", (long long)nb,
           (long long)(m.tiles.empty() ? 1 : m.tiles.starts.size() - 1),
           (long long)sched.points_per_subproblem);

  // The loop below collapses the rectangular (vector, subproblem) space and every thread
  // draws pairs out of it. Neighbouring tiles' halos overlap on one vector's fine grid,
  // so the add back needs a guard; a single pair per vector owns its grid and needs none.
  // The queue advances roughly in order, so the threads in flight on vector ib sit inside
  // its window of nb iterations, at most min(nthr, nb) of them, and that count picks the
  // guard: a lock on that vector below spopts.atomic_threshold, atomic writes above it.

  // TODO: colouring the tiles so no two neighbours run together would drop both guards.
  const bool needs_guard = nb > 1 && nthr > 1;
  const bool use_atomic =
      needs_guard && std::min((UBIGINT)nthr, nb) > (UBIGINT)m.spopts.atomic_threshold;
  if (m.spopts.debug && use_atomic)
    printf("\tup to %d writers per output: add_wrapped switching to atomic (!)\n",
           (int)std::min((UBIGINT)nthr, nb));
  std::vector<my_omp_lock_t> locks(needs_guard && !use_atomic ? batchSize : 0);
  for (auto &l : locks) MY_OMP_INIT_LOCK(&l);

#pragma omp parallel num_threads(nthr)
  {
    // local copies of NU pts and data for each subproblem
    std::vector<TF> kx0{}, ky0{}, kz0{}, dd0{}, du0{};
#pragma omp for collapse(2) schedule(dynamic, 1) // each is big
    for (int ib = 0; ib < batchSize; ib++)
      for (BIGINT isub = 0; isub < BIGINT(nb); isub++) {
        // one subproblem of one vector: spread it into its own padded subgrid and add
        // that subgrid back into the fine grid
        TF *FINUFFT_RESTRICT grid = data_uniform + ib * stride_u;
        const TF *nu              = data_nonuniform + ib * stride_nu;
        const auto M0             = bounds[isub + 1] - bounds[isub]; // # NU pts here
        // copy the location and data vectors for the nonuniform points
        kx0.resize(M0);
        ky0.resize(M0 * (N2 > 1));
        kz0.resize(M0 * (N3 > 1));
        dd0.resize(2 * M0); // complex strength data
        for (UBIGINT j = 0; j < M0; j++) {
          // todo: can avoid this copying?
          const auto kk = m.sortIndices[j + bounds[isub]]; // NU pt from the index list
          kx0[j]        = fold_rescale<TF>(kx[kk], N1);
          if (N2 > 1) ky0[j] = fold_rescale<TF>(ky[kk], N2);
          if (N3 > 1) kz0[j] = fold_rescale<TF>(kz[kk], N3);
          dd0[j * 2]     = nu[kk * 2];     // real part
          dd0[j * 2 + 1] = nu[kk * 2 + 1]; // imag part
        }
        // The subproblem spreads into the padded subgrid around its own points; the add
        // back confines wrapping to walk_wrapped_subgrid, so no kernel wraps.
        const Subgrid sub = get_subgrid(M0, kx0.data(), ky0.data(), kz0.data());
        if (m.spopts.debug > 1)
          print_subgrid_info(ndims, sub.off1, sub.off2, sub.off3, sub.padded_size1,
                             sub.size1, sub.size2, sub.size3, M0);
        du0.resize(2 * sub.cells()); // complex
        if (ndims == 1)
          spread_subproblem_1d(sub, du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               dd0.data());
        else if (ndims == 2)
          spread_subproblem_2d(sub, du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               dd0.data());
        else
          spread_subproblem_3d(sub, du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               dd0.data());
        // add the subgrid back into the fine grid, under the guard the thread count chose
        if (use_atomic) {
          add_wrapped_subgrid<true>(sub, grid, du0.data());
        } else {
          if (needs_guard) MY_OMP_SET_LOCK(&locks[ib]);
          add_wrapped_subgrid<false>(sub, grid, du0.data());
          if (needs_guard) MY_OMP_UNSET_LOCK(&locks[ib]);
        }
      } // end main loop over subprobs
  }
  for (auto &l : locks) MY_OMP_DESTROY_LOCK(&l);
  if (m.spopts.debug)
    printf("\tt1 spread:\t\t%.3g s (%" PRIu64 " subprobs x %d elems)\n",
           timer.elapsedsec(), nb, batchSize);
  return 0;
}

template<typename TF>
int FINUFFT_PLAN_T<TF>::interpSorted(TF *FINUFFT_RESTRICT data_uniform,
                                     TF *FINUFFT_RESTRICT data_nonuniform,
                                     int batchSize) const
/* Interpolate NU pts (in sort order) off a uniform grid. See the block comment above
   spreadSorted. The same subproblem cut as spreadSorted: each subproblem copies its
   padded subgrid out of the fine grid and every point interpolates out of that
   cache-sized copy. Interpolation only reads the grid, so it needs no lock; the strengths
   go straight back to each point's pre-sort index, so no gathered strength buffer either.
   Plan members used in place of the former free-function arguments:
   sortIndices, nfdim[0..2], nj, XYZ[0..2], spopts, tiles, horner_coeffs, nc.
   Instantiated in src/spreadinterp.cpp; extern template in execute.cpp suppresses
   re-instantiation there.
   Magland Dec 2016; history in the block comment above spreadSorted.
   Tiled subgrid gather, M. Barbone 8/25/26.
*/
{
  using namespace finufft::spreadinterp;
  using finufft::utils::CNTime;
  // Alias plan members to local names matching the original algorithm.
  const auto N1  = (UBIGINT)m.nfdim[0];
  const auto N2  = (UBIGINT)m.nfdim[1];
  const auto N3  = (UBIGINT)m.nfdim[2];
  const auto M   = (UBIGINT)m.nj;
  const auto *kx = m.XYZ[0];
  const auto *ky = m.XYZ[1];
  const auto *kz = m.XYZ[2];
  CNTime timer{};
  const auto ndims     = ndims_from_Ns(N1, N2, N3);
  const auto N         = N1 * N2 * N3; // fine grid size
  const auto stride_u  = 2 * N;        // per-vector strides through the batch arrays
  const auto stride_nu = 2 * M;
  auto nthr            = MY_OMP_GET_MAX_THREADS();
  if (m.spopts.nthreads > 0) nthr = m.spopts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1; // single-threaded lib must override user
#endif
  if (m.spopts.debug)
    printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d, batch=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr, batchSize);
  if (M == 0) // no NU pts, we're done
    return 0;

  timer.start();
  // Skipping the sort leaves no tiles, and that list is one tile spanning the whole fine
  // grid, so one call cuts the points either way.
  const SpreadSchedule sched = make_schedule(nthr, batchSize);
  const auto &bounds         = sched.bounds;
  const UBIGINT nb           = bounds.size() - 1;
  if (m.spopts.debug)
    printf("\tcache tiles: %lld subprobs over %lld tiles, cap %lld pts\n", (long long)nb,
           (long long)(m.tiles.empty() ? 1 : m.tiles.starts.size() - 1),
           (long long)sched.points_per_subproblem);

#pragma omp parallel num_threads(nthr)
  {
    // local copies of NU pts and the gathered subgrid for each subproblem
    std::vector<TF> kx0{}, ky0{}, kz0{}, du0{};
#pragma omp for collapse(2) schedule(dynamic, 1) // each is big
    for (int ib = 0; ib < batchSize; ib++)
      for (BIGINT isub = 0; isub < BIGINT(nb); isub++) {
        // one subproblem of one vector: read the subgrid out of the fine grid and
        // interpolate the points from it
        const TF *grid          = data_uniform + ib * stride_u;
        TF *FINUFFT_RESTRICT nu = data_nonuniform + ib * stride_nu;
        const auto M0           = bounds[isub + 1] - bounds[isub]; // # NU pts here
        // copy the location vectors for the nonuniform points
        kx0.resize(M0);
        ky0.resize(M0 * (N2 > 1));
        kz0.resize(M0 * (N3 > 1));
        for (UBIGINT j = 0; j < M0; j++) {
          // todo: can avoid this copying?
          const auto kk = m.sortIndices[j + bounds[isub]]; // NU pt from the index list
          kx0[j]        = fold_rescale<TF>(kx[kk], N1);
          if (N2 > 1) ky0[j] = fold_rescale<TF>(ky[kk], N2);
          if (N3 > 1) kz0[j] = fold_rescale<TF>(kz[kk], N3);
        }
        // read the subgrid out of the fine grid, so that every point of this subproblem
        // interpolates out of one cache-sized copy of it. The copy confines wrapping to
        // walk_wrapped_subgrid, so no kernel wraps.
        const Subgrid sub = get_subgrid(M0, kx0.data(), ky0.data(), kz0.data());
        if (m.spopts.debug > 1)
          print_subgrid_info(ndims, sub.off1, sub.off2, sub.off3, sub.padded_size1,
                             sub.size1, sub.size2, sub.size3, M0);
        du0.resize(2 * sub.cells()); // complex
        copy_wrapped_subgrid(sub, grid, du0.data());
        // The strengths go straight back to the index each point held before the sort.
        const BIGINT *idx = m.sortIndices.data() + bounds[isub];
        if (ndims == 1)
          interp_subproblem_1d(sub, du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               idx, nu);
        else if (ndims == 2)
          interp_subproblem_2d(sub, du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               idx, nu);
        else
          interp_subproblem_3d(sub, du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                               idx, nu);
      } // end main loop over subprobs
  }
  if (m.spopts.debug)
    printf("\tt2 interp:\t\t%.3g s (%" PRIu64 " subprobs x %d elems)\n",
           timer.elapsedsec(), nb, batchSize);
  return 0;
}
